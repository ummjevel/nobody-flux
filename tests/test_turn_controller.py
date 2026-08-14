"""TurnController's state machine, with every dependency faked.

Every branch here corresponds to a distinct user-visible failure if it
regresses (docs/code-review-20260814.md, section B): barge-in outside
RESPONDING must be a no-op, begin_response must scope cancellation to its own
reply, a dead frame source must surface as CaptureFailed rather than an
eternally idle-looking session, and so on. The controller takes everything by
injection, so no thread, device, or model is needed -- _capture_loop is run
synchronously on the test thread.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

from src.nobody_flux.turn.controller import CaptureFailed, TurnController, TurnState
from src.nobody_flux.turn.vad import FRAME_SAMPLES, Utterance, VadEvent


class FakeStream:
    """Scripted VadStream: push() replays a queue of event lists, one per
    frame; take_utterance() hands out queued utterances."""

    def __init__(self):
        self.scripted_events: list[list[VadEvent]] = []
        self.utterances: list[Utterance | None] = []

    def push(self, frame):
        events = self.scripted_events.pop(0) if self.scripted_events else []
        yield from events

    def take_utterance(self):
        return self.utterances.pop(0) if self.utterances else None


class FakeVad:
    def __init__(self, stream: FakeStream):
        self.stream = stream

    def open_stream(self, turn_detector=None):
        return self.stream


class FakePlayer:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeTranscriber:
    def __init__(self, text: str = ""):
        self.text = text
        self.calls: list[str] = []

    def reset(self):
        self.calls.append("reset")

    def accept_frame(self, frame):
        self.calls.append("accept")

    def finalize(self) -> str:
        self.calls.append("finalize")
        return self.text


def make_controller(stream: FakeStream | None = None, **kwargs) -> TurnController:
    stream = stream if stream is not None else FakeStream()
    defaults = dict(
        vad=FakeVad(stream),
        frame_source=lambda: np.zeros(FRAME_SAMPLES, dtype=np.float32),
        player_factory=FakePlayer,
    )
    defaults.update(kwargs)
    return TurnController(**defaults)


def utterance(seconds: float = 1.0) -> Utterance:
    return Utterance(audio=np.zeros(int(16_000 * seconds), dtype=np.float32))


# -- state transitions -------------------------------------------------------


def test_speech_started_moves_idle_to_listening():
    controller = make_controller()
    controller._handle_event(VadEvent.SPEECH_STARTED, FakeStream())
    assert controller.state is TurnState.LISTENING


def test_speech_started_ignored_while_responding():
    controller = make_controller()
    controller.begin_response()
    controller._handle_event(VadEvent.SPEECH_STARTED, FakeStream())
    assert controller.state is TurnState.RESPONDING


def test_finish_response_returns_to_idle():
    controller = make_controller()
    controller.begin_response()
    controller.finish_response()
    assert controller.state is TurnState.IDLE
    assert controller._current_player is None


# -- barge-in ----------------------------------------------------------------


def test_barge_in_outside_responding_is_noop():
    controller = make_controller()
    controller._handle_event(VadEvent.BARGE_IN_CONFIRMED, FakeStream())
    assert not controller.cancelled
    assert controller.barge_in_count == 0


def test_barge_in_while_responding_cancels_and_stops_player():
    controller = make_controller()
    player = controller.begin_response()
    controller._handle_event(VadEvent.BARGE_IN_CONFIRMED, FakeStream())
    assert controller.cancelled
    assert controller.barge_in_count == 1
    assert player.stopped


def test_barge_in_disabled_observes_but_does_not_cancel():
    controller = make_controller(allow_barge_in=False)
    player = controller.begin_response()
    controller._handle_event(VadEvent.BARGE_IN_CONFIRMED, FakeStream())
    assert not controller.cancelled
    assert controller.barge_in_count == 0
    assert not player.stopped


def test_begin_response_scopes_cancellation_to_its_own_reply():
    """If this regresses, every reply after the first barge-in dies at birth."""
    controller = make_controller()
    controller.begin_response()
    controller._handle_event(VadEvent.BARGE_IN_CONFIRMED, FakeStream())
    controller.finish_response()
    assert controller.cancelled  # still set from the interrupted reply...
    controller.begin_response()
    assert not controller.cancelled  # ...but cleared for the new one


# -- publishing turns --------------------------------------------------------


def test_utterance_ready_queues_turn_with_incrementing_index():
    stream = FakeStream()
    stream.utterances = [utterance(), utterance()]
    controller = make_controller(stream)

    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)

    first = controller.next_turn(timeout=0)
    second = controller.next_turn(timeout=0)
    assert (first.index, second.index) == (1, 2)
    assert controller.next_turn(timeout=0) is None


def test_utterance_ready_with_nothing_pending_publishes_nothing():
    stream = FakeStream()
    stream.utterances = [None]
    controller = make_controller(stream)
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    assert controller.next_turn(timeout=0) is None


def test_publish_during_responding_keeps_state_and_queues():
    """A turn captured while the reply still plays is delivered later, not
    dropped, and must not yank the state out of RESPONDING."""
    stream = FakeStream()
    stream.utterances = [utterance()]
    controller = make_controller(stream)
    controller.begin_response()
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    assert controller.state is TurnState.RESPONDING
    assert controller.next_turn(timeout=0) is not None


def test_publish_from_listening_returns_to_idle():
    stream = FakeStream()
    stream.utterances = [utterance()]
    controller = make_controller(stream)
    controller._handle_event(VadEvent.SPEECH_STARTED, stream)
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    assert controller.state is TurnState.IDLE


def test_publish_finalizes_then_resets_transcriber():
    stream = FakeStream()
    stream.utterances = [utterance()]
    transcriber = FakeTranscriber(text="안녕하세요")
    controller = make_controller(stream, transcriber=transcriber)
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    # finalize() must come before reset(), or the text is thrown away.
    assert transcriber.calls == ["finalize", "reset"]
    assert controller.next_turn(timeout=0).streamed_text == "안녕하세요"


def test_empty_streamed_text_becomes_none():
    stream = FakeStream()
    stream.utterances = [utterance()]
    controller = make_controller(stream, transcriber=FakeTranscriber(text=""))
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    assert controller.next_turn(timeout=0).streamed_text is None


# -- speech_duration plumbing (code-review #1) --------------------------------


def test_captured_turn_exposes_speech_duration():
    stream = FakeStream()
    stream.utterances = [
        Utterance(audio=np.zeros(16_000, dtype=np.float32), speech_samples=8_000)
    ]
    controller = make_controller(stream)
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)
    turn = controller.next_turn(timeout=0)
    assert turn.duration_s == 1.0
    assert turn.speech_duration_s == 0.5


# -- capture failure propagation (code-review #3) ------------------------------


def test_frame_source_failure_surfaces_as_capture_failed():
    boom = RuntimeError("device unplugged")

    def dying_source():
        raise boom

    controller = make_controller(frame_source=dying_source)
    controller._capture_loop()  # synchronously, on this thread

    assert controller.capture_failed
    with pytest.raises(CaptureFailed) as excinfo:
        controller.next_turn(timeout=0)
    assert excinfo.value.__cause__ is boom


def test_transcriber_failure_surfaces_as_capture_failed():
    class ExplodingTranscriber(FakeTranscriber):
        def accept_frame(self, frame):
            raise ValueError("rate mismatch")

    controller = make_controller(transcriber=ExplodingTranscriber())
    controller._capture_loop()
    assert controller.capture_failed
    with pytest.raises(CaptureFailed):
        controller.next_turn(timeout=0)


def test_queued_turns_are_delivered_before_the_failure_raises():
    stream = FakeStream()
    stream.utterances = [utterance()]
    controller = make_controller(stream)
    controller._handle_event(VadEvent.UTTERANCE_READY, stream)

    controller._capture_error = RuntimeError("late failure")
    controller._capture_failed.set()

    assert controller.next_turn(timeout=0) is not None  # buffered turn first
    with pytest.raises(CaptureFailed):
        controller.next_turn(timeout=0)


def test_failure_during_shutdown_is_not_an_error():
    """A device torn down by stop() may fail its in-flight read; that is
    shutdown working, not the microphone dying -- no CaptureFailed."""
    controller = None

    def dying_source():
        # Simulates stop() landing while the read is in flight.
        controller._shutdown.set()
        raise RuntimeError("stream closed by teardown")

    controller = make_controller(frame_source=dying_source)
    controller._capture_loop()
    assert not controller.capture_failed
    assert controller.next_turn(timeout=0) is None
