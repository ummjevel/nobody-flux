"""Phase 4: the conversation state machine, and barge-in that works during
generation.

What was wrong before
---------------------

The turn loop used to be a sequence of blocking calls on one thread::

    utterance = vad.listen_for_utterance(...)   # blocks until the user stops
    produce(utterance)                          # blocks through LLM + TTS

Capture therefore only happened inside the first call. For the entire duration
of the second -- the whole LLM generation, which is the single longest stretch
of a turn -- nothing was reading the microphone at all. An interruption during
generation was not "handled late"; it was not *observed*, because the frames it
arrived in were never pulled from the device. Only once generation finished and
the loop came back around to listening could speech register, by which point
the user had usually given up and repeated themselves.

``scripts/talk.py`` documented this honestly as a known gap. Closing it is what
Phase 4 is.

The fix: capture is not a phase
-------------------------------

Capture moves onto its own thread and runs continuously, for the whole session,
regardless of what the conversation is doing. It reads frames from the audio
session, drives ``VadStream``, and publishes events. The main thread consumes
those events and runs the state machine.

Two things follow immediately. Barge-in is detected during generation, because
frames keep being read during generation. And a turn the user starts while the
reply is still playing is already being captured when the reply is cut -- so
the next turn does not begin with its first syllable missing.

The three states
----------------

``IDLE``
    Nothing playing, nobody speaking. Waiting.
``LISTENING``
    Speech is in progress and being captured.
``RESPONDING``
    A reply is being generated and/or played. Capture continues throughout, and
    this is the only state in which a confirmed barge-in means anything.

Making the state explicit is not bookkeeping for its own sake -- the barge-in
rule *is* a statement about state ("a confirmed barge-in cancels the reply if
and only if we are RESPONDING"), and previously that rule was implicit in which
callback happened to be installed at the time.

Threading
---------

One capture thread, one main thread, and a strict rule about which touches
what:

* The capture thread owns ``VadStream`` and the streaming transcriber. Nothing
  else may touch them. Both are single-consumer by design.
* Completed turns cross to the main thread through a ``queue.Queue``.
* Cancellation crosses back through a ``threading.Event``.

Nothing else is shared. In particular the pipeline, the conversation store and
the LLM history are only ever touched from the main thread, which is what keeps
SQLite's single-thread affinity satisfied without a lock.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from .vad import Utterance, VadEvent, VoiceActivityDetector


class TurnState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RESPONDING = "responding"


class CaptureFailed(RuntimeError):
    """The capture thread died and no further turns can ever arrive.

    Raised from ``next_turn`` (on the main thread) rather than logged from the
    capture thread, because the failure mode it replaces was the worst kind:
    the capture loop returned silently, the main loop kept polling an empty
    queue forever, and a session with a dead microphone was indistinguishable
    from a user who simply stopped talking.
    """


@dataclass
class CapturedTurn:
    """One complete user turn, handed from the capture thread to the main one."""

    utterance: Utterance
    index: int
    # Text from the live streaming recognizer, when one is attached. None means
    # no streaming ASR was running, and the caller must transcribe the audio the
    # ordinary way. When present, recognition already happened *while the user
    # was speaking*, so the caller can skip the ASR stage entirely -- that is
    # the whole point of Phase 3, realized here.
    streamed_text: str | None = None

    @property
    def duration_s(self) -> float:
        """Full capture-buffer length, pre-roll included."""
        return self.utterance.duration_s

    @property
    def speech_duration_s(self) -> float:
        """VAD-measured speech length, excluding pre-roll padding.

        Use this for anything that judges "how long did the user speak" --
        the backchannel gate above all. ``duration_s`` is inflated by
        ``pre_roll_ms`` and exists for buffer-sized concerns.
        """
        return self.utterance.speech_duration_s


@dataclass
class TurnController:
    """Drives a live conversation: continuous capture, explicit state, and
    cancellation that reaches into generation.

    Typical use::

        controller = TurnController(vad=registry.build_vad(), frame_source=...)
        controller.start()
        try:
            while True:
                turn = controller.next_turn()
                player = controller.begin_response()
                try:
                    ...  # generate + enqueue chunks, polling controller.cancelled
                finally:
                    controller.finish_response()
        finally:
            controller.stop()

    The caller keeps ownership of the pipeline, storage and logging. This class
    owns only turn-taking -- when to listen, when a turn ended, and when a reply
    should be abandoned.
    """

    vad: VoiceActivityDetector
    # Returns the next mono float32 frame of FRAME_SAMPLES. Normally
    # AudioSession.read_frame, which supplies echo-cancelled audio from the
    # duplex stream; any callable with that shape works, which is what makes
    # this testable without a device.
    frame_source: Callable[[], np.ndarray]
    # Builds the player for each reply (audio.player.StreamPlayer or
    # SessionPlayer). Injected rather than constructed here so this module needs
    # no knowledge of which audio backend is in use.
    player_factory: Callable[[], object] | None = None
    # Optional Smart Turn v3 endpoint detection -- extends a turn past a
    # mid-thought pause instead of cutting it.
    turn_detector: object | None = None
    # Optional stage.asr_stream.StreamingTranscriber. When set, frames are fed
    # to it as they arrive and CapturedTurn.streamed_text is populated.
    transcriber: object | None = None
    # False reproduces the old sequential behaviour: a confirmed barge-in is
    # observed and logged, but does not cancel the reply.
    allow_barge_in: bool = True
    # Called from the CAPTURE THREAD for every VAD event. Intended for logging
    # only -- anything heavier will delay frame reads and, through that, every
    # timing decision this class makes.
    on_event: Callable[[VadEvent, TurnState], None] | None = None

    # -- internal state ----------------------------------------------------
    _state: TurnState = field(init=False, default=TurnState.IDLE)
    _state_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _turns: queue.Queue = field(init=False, default_factory=queue.Queue)
    _cancel: threading.Event = field(init=False, default_factory=threading.Event)
    _shutdown: threading.Event = field(init=False, default_factory=threading.Event)
    # Set by the capture thread when it dies with an exception; read by
    # next_turn on the main thread, which turns it into a CaptureFailed raise.
    _capture_failed: threading.Event = field(init=False, default_factory=threading.Event)
    _capture_error: BaseException | None = field(init=False, default=None)
    _thread: threading.Thread | None = field(init=False, default=None)
    _turn_index: int = field(init=False, default=0)
    _barge_in_count: int = field(init=False, default=0)
    # The player for the reply currently being spoken, or None. Written by the
    # main thread in begin_response/finish_response and read by the capture
    # thread on barge-in. Unguarded on purpose: attribute assignment is atomic
    # under the GIL, and the only possible race -- reading a player that was
    # cleared a moment ago -- resolves to stopping an already-finished reply,
    # which is a no-op. A lock here would sit in the frame-read path for no
    # correctness gain.
    _current_player: object | None = field(init=False, default=None)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Begin capturing. Returns immediately; capture runs until stop()."""
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="turn-capture", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop capturing and wait for the capture thread to finish.

        The thread may be blocked inside ``frame_source()`` when this is
        called, and there is no portable way to interrupt a blocking device
        read. So the shutdown flag is checked after each frame returns, and the
        join is bounded: worst case this waits one frame period, and if the
        device has wedged entirely the daemon thread is abandoned rather than
        hanging the process on the way out.
        """
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> TurnState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: TurnState) -> None:
        with self._state_lock:
            self._state = state

    @property
    def cancelled(self) -> bool:
        """True once a barge-in has been confirmed against the current reply.

        Pass ``lambda: controller.cancelled`` as ``should_cancel`` to
        ``pipeline.run_streaming``: that is the path by which an interruption
        detected on the capture thread stops generation on the main one.
        """
        return self._cancel.is_set()

    @property
    def barge_in_count(self) -> int:
        """How many replies have been interrupted this session. Useful when
        tuning ``barge_in_confirm_ms`` -- a count far above what actually
        happened is the signature of the reply's own echo tripping the VAD."""
        return self._barge_in_count

    # -- main-thread API ---------------------------------------------------

    @property
    def capture_failed(self) -> bool:
        """True once the capture thread has died. See ``CaptureFailed``."""
        return self._capture_failed.is_set()

    def next_turn(self, timeout: float | None = None) -> CapturedTurn | None:
        """Block until a complete user turn is available, or ``timeout``.

        Returns None on timeout. Turns are queued rather than dropped, so one
        captured while a reply was still playing is delivered on the next call
        instead of being lost to the race between "reply ended" and "user
        started".

        Raises ``CaptureFailed`` once the capture thread has died AND the queue
        is drained -- turns captured before the failure are still delivered
        first, then the caller learns the session cannot continue. Without the
        raise, a dead microphone left the main loop polling an empty queue
        forever while the session looked merely idle.
        """
        try:
            return self._turns.get(timeout=timeout)
        except queue.Empty:
            if self._capture_failed.is_set():
                raise CaptureFailed(
                    "capture thread died; the microphone/VAD path is no longer running"
                ) from self._capture_error
            return None

    def begin_response(self):
        """Enter RESPONDING and return a fresh player for this reply.

        Clearing the cancellation flag here, at the start of a reply, is what
        scopes it to that reply: a barge-in that arrived while the previous one
        was playing must not immediately cancel this one.
        """
        self._cancel.clear()
        player = self.player_factory() if self.player_factory is not None else None
        if player is not None:
            player.start()
        self._current_player = player
        self._set_state(TurnState.RESPONDING)
        return player

    def finish_response(self) -> None:
        """Leave RESPONDING.

        Returns to IDLE rather than LISTENING even if the user is mid-utterance:
        the capture thread will move the state to LISTENING on its own at the
        next speech event, and having one owner for that transition is simpler
        than trying to reconcile two.
        """
        self._current_player = None
        self._set_state(TurnState.IDLE)

    # -- capture thread ----------------------------------------------------

    def _capture_loop(self) -> None:
        """Read frames forever, drive the VAD, publish turns and cancellations.

        This is the only thread that touches the VAD stream or the transcriber.
        It must stay cheap: every microsecond spent here is a microsecond the
        next frame waits, and frames arriving late shift every duration this
        class measures.

        Any exception -- a device read failing, the transcriber rejecting a
        frame, the VAD runtime throwing -- ends capture, and *must not* end it
        silently: the previous version bare-returned on frame_source errors and
        let everything else kill the thread with nothing but an unhandled
        traceback on stderr, leaving the main loop polling forever. The error
        is stored and surfaced through next_turn as CaptureFailed instead.
        """
        try:
            stream = self.vad.open_stream(turn_detector=self.turn_detector)
            if self.transcriber is not None:
                self.transcriber.reset()

            while not self._shutdown.is_set():
                frame = self.frame_source()

                if self.transcriber is not None:
                    # Fed unconditionally, not only while the VAD reports speech.
                    # A streaming transducer needs the silence around an utterance
                    # to establish context and to run its own endpoint rules --
                    # gating it on the VAD would degrade both.
                    self.transcriber.accept_frame(frame)

                for event in stream.push(frame):
                    self._handle_event(event, stream)
        except Exception as exc:
            if self._shutdown.is_set():
                # A device torn down by stop() may fail its in-flight read;
                # that is shutdown working, not the microphone dying.
                return
            self._capture_error = exc
            self._capture_failed.set()

    def _handle_event(self, event: VadEvent, stream) -> None:
        if self.on_event is not None:
            self.on_event(event, self.state)

        if event is VadEvent.SPEECH_STARTED:
            # Only IDLE advances to LISTENING. From RESPONDING it stays put:
            # speech during a reply is not yet a turn, it is a candidate
            # interruption, and it is BARGE_IN_CONFIRMED that decides.
            if self.state is TurnState.IDLE:
                self._set_state(TurnState.LISTENING)
            return

        if event is VadEvent.BARGE_IN_CONFIRMED:
            self._on_barge_in()
            return

        if event is VadEvent.UTTERANCE_READY:
            utterance = stream.take_utterance()
            if utterance is None:
                return
            self._publish(utterance)

    def _on_barge_in(self) -> None:
        """Speech has continued long enough to count as a real interruption."""
        if not self.allow_barge_in or self.state is not TurnState.RESPONDING:
            # Not responding: this is simply the user taking their turn, and
            # there is nothing to interrupt.
            return

        self._barge_in_count += 1
        # Order matters. Set the flag first so that generation -- which may be
        # mid-token on the main thread right now -- sees it at its very next
        # poll. Stopping playback second means the silence the user hears
        # arrives no later than the moment the pipeline stops working.
        self._cancel.set()
        player = self._current_player
        if player is not None:
            player.stop()

    def _publish(self, utterance: Utterance) -> None:
        """Hand a finished utterance to the main thread."""
        streamed_text = None
        if self.transcriber is not None:
            # finalize() flushes the decoder and returns the complete text, then
            # reset() prepares it for the next turn. Both happen here, on the
            # capture thread, because it is the transcriber's sole owner.
            streamed_text = self.transcriber.finalize()
            self.transcriber.reset()

        self._turn_index += 1
        self._turns.put(
            CapturedTurn(
                utterance=utterance,
                index=self._turn_index,
                streamed_text=streamed_text or None,
            )
        )
        if self.state is TurnState.LISTENING:
            self._set_state(TurnState.IDLE)
