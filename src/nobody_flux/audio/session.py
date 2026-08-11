"""Unified duplex audio: one owner for mic capture AND speaker playback, so the
reply being played is available as the echo-cancellation reference for the mic
(see aec.py). This is the Phase 1.5 foundation.

Why one owner / one stream: scripts/talk.py used to open the mic (vad.py's
sd.InputStream) and the speaker (sd.play) as two independent streams. On
macOS/CoreAudio opening both at once fails with PaMacCore err -50 and corrupts
capture -- which is why talk.py has a --no-barge-in sequential fallback. A single
duplex sounddevice Stream is ONE stream doing both, so it sidesteps err -50
entirely *and* gives the AEC its reference for free. That combination -- fix the
duplex conflict and cancel echo in the same layer -- is the whole point.

Backends (all expose the same AudioSession interface: read_frame() for the VAD,
play()/stop_playback() for replies):

  - SharedStreamSession  -- portable primary. One duplex Stream at 16k; the
    output it's about to play is fed to an EchoCanceller (aec.py) alongside the
    mic frame. Works on macOS (single stream, no err -50), Linux/WSL, and the
    CM4 target. Pair it with SpeexEchoCanceller for real AEC or ReferenceGate
    for the dependency-free suppression gate.
  - OsEchoCancelSession  -- Linux/CM4 optimization. Delegates to a
    SharedStreamSession pointed at PipeWire/PulseAudio's module-echo-cancel
    virtual source (the OS does WebRTC AEC), so the in-process canceller is just
    PassThrough. Near-zero app cost when the OS module is available.
  - CoreAudioVpioSession -- macOS optimization hook. Apple's VoiceProcessingIO
    audio unit does AEC+NS+AGC in the OS and is itself a single duplex unit
    (would also fix err -50). PortAudio/sounddevice don't expose it, so this
    needs a native (pyobjc) binding; until that's wired it raises with guidance
    and callers fall back to SharedStreamSession, which already works on Mac.

On Windows, SharedStreamSession is the backend, with one important extra step:
host API selection (see SharedStreamSession.hostapi). PortAudio exposes several
Windows audio APIs, and the one it picks by default -- MME -- carries something
like 100-200ms of buffering in each direction. That is not a quality problem,
it is a *turn-taking* problem: it lands directly on top of every latency number
this project tunes, and it delays the microphone's view of an interruption past
the barge-in confirmation window. Selecting WASAPI instead brings the round trip
down to the tens of milliseconds, which is what makes the Windows box usable as
the turn-parameter measurement rig it exists to be.

build_audio_session() (see registry.build_audio_session) auto-selects by
platform + installed libs, overridable via configs/audio.yaml / --aec.
"""

from __future__ import annotations

import collections
import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from .aec import EchoCanceller, PassThrough, ReferenceGate, SpeexEchoCanceller
from .resample import resample_to
from ..platform_support import IS_MACOS, IS_WINDOWS

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480, matches turn/vad.py

# Capture frames thrown away when a stream opens. Starting an input stream
# produces a transient: measured on this project's Windows USB microphone, the
# first ~150ms came back at rms 0.27 with a full-scale peak, against a room
# floor of rms 0.003. The VAD reads that as speech, so without this the session
# opens by inventing an utterance out of a device click -- the assistant's first
# act is answering a pop. 0.5s covers the measured transient with margin and is
# paid once per session, inside model loading time.
WARMUP_FRAMES = int(500 / FRAME_MS)

__all__ = [
    "AudioSession",
    "SharedStreamSession",
    "OsEchoCancelSession",
    "CoreAudioVpioSession",
    "build_session",
    "select_backend",
    "make_echo_canceller",
    "skip_warmup",
    "resample_to",
    "SAMPLE_RATE",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "WARMUP_FRAMES",
]


def skip_warmup(read_frame, frames: int = WARMUP_FRAMES):
    """Wrap a frame source so the first ``frames`` frames are discarded.

    Discarded, not zeroed. Digital silence is not silence as far as TEN-VAD is
    concerned -- feeding it exact zeros produces a degenerate feature vector and
    the model reports speech through it (measured with
    ``scripts/_calibrate_vad_threshold.py``). Dropping the frames entirely means
    the VAD's first input is real room tone, which is the only thing it was
    trained to reason about.

    Shared by both capture paths -- the duplex session below and the private
    input stream ``scripts/talk.py`` opens when AEC is off -- so a session's
    first moments behave the same either way.
    """
    remaining = frames

    def read():
        nonlocal remaining
        while remaining > 0:
            remaining -= 1
            read_frame()
        return read_frame()

    return read


class AudioSession:
    """Interface. read_frame() returns one echo-cancelled 16k mono frame
    (FRAME_SAMPLES long) for the VAD; play()/stop_playback() drive the speaker
    and, in duplex backends, supply the AEC reference."""

    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES

    def start(self) -> None:  # pragma: no cover - device dependent
        raise NotImplementedError

    def read_frame(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def play(self, samples: np.ndarray, sample_rate: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def stop_playback(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def playback_active(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class SharedStreamSession(AudioSession):
    """Portable duplex backend (see module docstring). One sounddevice.Stream at
    16k; its callback both captures the mic and fills the speaker from the
    playback buffer, then hands (mic, reference) to the echo canceller and
    queues the cleaned frame for read_frame().

    The callback is intentionally a plain method (_process_block) that takes and
    returns numpy arrays, so its logic is testable without opening a device."""

    echo_canceller: EchoCanceller = field(default_factory=ReferenceGate)
    # Speaker->mic acoustic round-trip, in 30ms frames, used to align the
    # reference for gate-style cancellers (Speex models the delay itself within
    # its filter tail, so this mostly matters for ReferenceGate). 4 frames =
    # 120ms is a loose default; scripts/_calibrate_aec_delay.py measures the
    # real value into configs/audio.yaml.
    delay_frames: int = 4
    input_device: int | str | None = None
    output_device: int | str | None = None
    # Preferred PortAudio host API, by substring match against its reported
    # name (e.g. "WASAPI", "ALSA", "Core Audio"). Only consulted when
    # input_device/output_device are left as None -- an explicit device already
    # implies its host API, and second-guessing that would be wrong.
    #
    # None means "resolve per platform" (see _preferred_hostapi), which on
    # Windows means WASAPI. This matters more than it looks: PortAudio's
    # default Windows host API is MME, whose buffering adds roughly 100-200ms
    # in each direction. That inflates measured turn latency, and worse, it
    # delays the mic's view of an interruption -- barge_in_confirm_ms is 250ms,
    # so an MME round trip is the same order as the entire decision window it
    # is supposed to be measured against.
    #
    # Set to "" to disable host API preference entirely and take PortAudio's
    # default, which is the escape hatch if WASAPI misbehaves with a particular
    # device.
    hostapi: str | None = None

    # Rate to open the *device* at. None negotiates: 16kHz if the device will
    # take it, otherwise the device's own default, with conversion to and from
    # 16kHz happening inside the callback.
    #
    # This is not a nicety. WASAPI in shared mode only opens at the rate the
    # Windows mix format is set to -- both devices on the measurement box report
    # 48kHz and nothing else -- so a hard-coded 16kHz duplex stream fails with
    # "Invalid sample rate [PaErrorCode -9997]" and the entire AEC path is
    # unreachable on Windows. The same applies to any USB microphone that does
    # not implement 16kHz, which is most of them; it simply had not been hit
    # before, because the duplex path had never run on real hardware.
    #
    # 16kHz stays the internal contract regardless: the VAD, both recognizers
    # and the echo cancellers are all 16kHz, and pushing the conversion to the
    # device edge keeps it that way.
    stream_rate: int | None = None

    def __post_init__(self):
        self._lock = threading.Lock()
        self._play_buf = np.zeros(0, dtype=np.float32)
        self._captured: queue.Queue = queue.Queue()
        # Ring of recently-played frames; the one delay_frames back is the
        # reference the mic is hearing now.
        self._ref_ring: collections.deque = collections.deque(maxlen=self.delay_frames + 1)
        self._active = threading.Event()
        self._stream = None
        # Resolved by start(); until then the device rate is assumed to be the
        # internal one, which makes _to_device a no-op if anything reaches it
        # early.
        self._device_rate = SAMPLE_RATE
        self._device_frame = FRAME_SAMPLES
        # The queue's own getter, wrapped so the device's start-up transient
        # never reaches the VAD. Re-wrapped by start(), since the transient
        # belongs to opening the device rather than to this object's lifetime --
        # a session that is stopped and started again gets a fresh window.
        self._read = skip_warmup(self._captured.get)

    def _preferred_hostapi(self) -> str:
        """Host API substring to prefer when no explicit device was given.

        Windows is the only platform where the default is actively bad (MME --
        see the `hostapi` field). Linux and macOS each expose one obvious
        choice that PortAudio already defaults to, so they opt out by returning
        an empty string rather than pinning a name that might not exist on a
        given build.
        """
        if self.hostapi is not None:
            return self.hostapi
        return "WASAPI" if IS_WINDOWS else ""

    def _resolve_devices(self) -> tuple[int | str | None, int | str | None]:
        """Pick (input, output) device indices, honouring the host API
        preference.

        Returns whatever was explicitly configured, untouched. Otherwise looks
        for the default input/output device *belonging to* the preferred host
        API and returns those indices; if the host API is not present on this
        machine, or has no usable default device, falls back to None so
        PortAudio applies its own default. Failing soft is deliberate -- a
        missing host API should cost latency, not the ability to run at all.
        """
        if self.input_device is not None or self.output_device is not None:
            return self.input_device, self.output_device

        wanted = self._preferred_hostapi()
        if not wanted:
            return None, None

        import sounddevice as sd

        for api in sd.query_hostapis():
            if wanted.lower() not in api["name"].lower():
                continue
            # PortAudio reports -1 for "this host API has no default device of
            # that direction", which is not a valid index to hand back.
            capture = api.get("default_input_device", -1)
            playback = api.get("default_output_device", -1)
            if capture < 0 or playback < 0:
                continue
            return capture, playback
        return None, None

    def _negotiate_rate(self, input_device, output_device) -> int:
        """Pick a rate the device will actually open a duplex stream at.

        16kHz first, because matching the internal rate means no conversion at
        all. Falling back to the device's own default rather than to a list of
        guesses: whatever it reports is what its driver is configured for, and
        on the host APIs that are picky (WASAPI shared mode) that is the only
        rate that can succeed.

        Both directions are checked, since a duplex stream needs one rate that
        satisfies both, and a machine can easily have a 48kHz microphone and a
        44.1kHz output.
        """
        import sounddevice as sd

        def works(rate: float) -> bool:
            try:
                sd.check_input_settings(
                    device=input_device, samplerate=rate, channels=1, dtype="float32"
                )
                sd.check_output_settings(
                    device=output_device, samplerate=rate, channels=1, dtype="float32"
                )
                return True
            except Exception:
                return False

        if works(SAMPLE_RATE):
            return SAMPLE_RATE

        for device, kind in ((input_device, "input"), (output_device, "output")):
            try:
                info = sd.query_devices(device, kind) if device is None else sd.query_devices(device)
            except Exception:
                continue
            rate = int(info.get("default_samplerate") or 0)
            if rate and works(rate):
                return rate

        # Nothing negotiated. Return the internal rate and let sd.Stream raise
        # its own error, which names the device and the rate -- more useful than
        # anything this function could invent.
        return SAMPLE_RATE

    def start(self) -> None:
        import sounddevice as sd

        input_device, output_device = self._resolve_devices()
        rate = self.stream_rate or self._negotiate_rate(input_device, output_device)
        self._device_rate = rate
        # One frame's worth at the device's rate, so each callback still maps to
        # exactly one FRAME_MS frame after conversion and the frame cadence the
        # VAD sees is unchanged.
        self._device_frame = int(round(rate * FRAME_MS / 1000))
        self._read = skip_warmup(self._captured.get)
        self._stream = sd.Stream(
            samplerate=rate,
            blocksize=self._device_frame,
            channels=1,
            dtype="float32",
            device=(input_device, output_device),
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, outdata, frames, time_info, status):  # pragma: no cover - device
        mic = np.array(indata[:, 0], dtype=np.float32, copy=True)
        # Down to the internal rate first: everything below this line -- the
        # echo canceller, the reference ring, the queue the VAD reads -- is
        # 16kHz, and keeping the conversion at this single boundary is what
        # allows that to stay true.
        mic = resample_to(mic, self._device_rate, SAMPLE_RATE)
        played = self._fill_output(len(mic))
        outdata[:, 0] = self._to_device(played, frames)
        self._captured.put(self._process_block(mic, played))

    def _to_device(self, played: np.ndarray, frames: int) -> np.ndarray:
        """The 16kHz frame just consumed, at the device's rate and exactly
        ``frames`` long.

        The length is forced rather than trusted: rounding in the rate
        conversion can leave the result a sample short or long, and PortAudio
        requires the output buffer filled exactly. A one-sample correction at a
        30ms cadence is inaudible; a shape mismatch is a hard error in the
        audio callback, where exceptions are unrecoverable.
        """
        out = resample_to(played, SAMPLE_RATE, self._device_rate)
        if len(out) == frames:
            return out
        fitted = np.zeros(frames, dtype=np.float32)
        n = min(frames, len(out))
        fitted[:n] = out[:n]
        return fitted

    def _fill_output(self, frames: int) -> np.ndarray:
        """Pop up to `frames` samples off the playback buffer, zero-padded."""
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            n = min(frames, len(self._play_buf))
            if n:
                out[:n] = self._play_buf[:n]
                self._play_buf = self._play_buf[n:]
            if len(self._play_buf) == 0:
                self._active.clear()
        return out

    def _process_block(self, mic: np.ndarray, played: np.ndarray) -> np.ndarray:
        """Given the captured mic frame and the frame just played, return the
        echo-cancelled mic frame. Reference is the played frame delayed by
        delay_frames (for gate-style cancellers)."""
        self._ref_ring.append(played)
        if len(self._ref_ring) > self.delay_frames:
            ref = self._ref_ring[0]
        else:
            ref = played
        if len(ref) != len(mic):
            ref = np.resize(ref, len(mic))
        return self.echo_canceller.process(mic, ref)

    def read_frame(self) -> np.ndarray:
        return self._read()

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        buf = resample_to(samples, sample_rate, SAMPLE_RATE)
        with self._lock:
            self._play_buf = np.concatenate([self._play_buf, buf])
            if len(self._play_buf):
                self._active.set()

    def stop_playback(self) -> None:
        with self._lock:
            self._play_buf = np.zeros(0, dtype=np.float32)
            self._active.clear()
        self.echo_canceller.reset()

    def playback_active(self) -> bool:
        return self._active.is_set()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


@dataclass
class OsEchoCancelSession(AudioSession):
    """Linux/CM4 backend: let the OS cancel the echo. PipeWire/PulseAudio's
    module-echo-cancel exposes a virtual capture source whose echo is already
    removed (WebRTC AEC under the hood); we just capture from it. Implemented as
    a SharedStreamSession pointed at that source with a PassThrough canceller --
    reusing all the duplex/playback machinery, only the echo work moves into the
    OS.

    The module must be loaded (once, at the system/image level -- for the CM4
    target that's part of the device image; on a dev box:
    `pactl load-module module-echo-cancel aec_method=webrtc source_name=echocancel`).
    This backend does NOT load/unload it -- managing global audio server modules
    from an app process is fragile and would fight other users of the audio
    server. It only selects the source by name; if it's absent, start() raises
    with that exact command so the fix is obvious."""

    source_name: str = "echocancel"
    output_device: int | str | None = None
    _inner: SharedStreamSession = field(init=False, default=None)

    def start(self) -> None:
        import sounddevice as sd

        # Resolve the named echo-cancel source to a device index sounddevice can
        # open; fail loudly (not silently onto the raw mic) if it isn't there.
        match = None
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and self.source_name in dev["name"]:
                match = idx
                break
        if match is None:
            raise RuntimeError(
                f"module-echo-cancel source '{self.source_name}' not found. Load it "
                f"with: pactl load-module module-echo-cancel aec_method=webrtc "
                f"source_name={self.source_name}  (or use --aec speex for the "
                f"in-process canceller)."
            )
        self._inner = SharedStreamSession(
            echo_canceller=PassThrough(),
            input_device=match,
            output_device=self.output_device,
        )
        self._inner.start()

    def read_frame(self) -> np.ndarray:
        return self._inner.read_frame()

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self._inner.play(samples, sample_rate)

    def stop_playback(self) -> None:
        self._inner.stop_playback()

    def playback_active(self) -> bool:
        return self._inner.playback_active()

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()


@dataclass
class CoreAudioVpioSession(AudioSession):
    """macOS optimization hook: Apple's VoiceProcessingIO audio unit does
    AEC+NS+AGC in the OS and is a single duplex unit (also sidesteps err -50).
    PortAudio/sounddevice don't expose VPIO, so a real implementation needs a
    native binding (pyobjc + AudioUnit, subclassing to feed read_frame()/play()).
    That's not wired yet -- selecting this backend raises with guidance rather
    than pretending. On macOS the auto-selector uses SharedStreamSession, which
    already gives AEC + duplex there; VPIO is a later quality upgrade."""

    def start(self) -> None:
        raise NotImplementedError(
            "CoreAudio VoiceProcessingIO backend is not wired yet (needs a native "
            "pyobjc/AudioUnit binding). Use --aec speex (SharedStreamSession) on "
            "macOS -- a single duplex stream already cancels echo and avoids the "
            "err -50 conflict."
        )

    def read_frame(self) -> np.ndarray:
        raise NotImplementedError

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        raise NotImplementedError

    def stop_playback(self) -> None:
        raise NotImplementedError

    def playback_active(self) -> bool:
        return False

    def close(self) -> None:
        pass


def _speexdsp_available() -> bool:
    try:
        import speexdsp  # noqa: F401

        return True
    except ImportError:
        return False


def make_echo_canceller(kind: str) -> EchoCanceller:
    """kind -> canceller. 'refgate' (dep-free gate), 'speex' (real AEC), 'off'
    (passthrough). Kept separate from the session so a session backend can be
    paired with any canceller."""
    if kind == "off":
        return PassThrough()
    if kind == "refgate":
        return ReferenceGate()
    if kind == "speex":
        return SpeexEchoCanceller(frame_size=FRAME_SAMPLES)
    raise ValueError(f"Unknown echo canceller '{kind}'. Use off/refgate/speex.")


def select_backend(prefer: str) -> str:
    """Resolve ``prefer='auto'`` to a concrete backend name for this platform,
    or pass an already-concrete choice straight through. Concrete names:
    ``shared-speex``, ``shared-refgate``, ``os-echocancel``, ``vpio``, ``off``.

    The auto policy is the same shape everywhere -- always a single duplex
    stream, then the best echo canceller actually installed -- because the
    duplex part is what fixes the device conflicts, and the canceller is the
    part that varies by what wheels exist:

    ``macOS``
        speexdsp if present, else the reference gate. Either way one stream,
        which is what avoids the CoreAudio err -50 conflict. VoiceProcessingIO
        would be better still but is not wired (see CoreAudioVpioSession).
    ``Linux/WSL2``
        speexdsp if present, else the reference gate. ``os-echocancel`` (the
        PipeWire/Pulse WebRTC module) beats both, but it requires a
        system-level module to be loaded first, so it stays opt-in rather than
        being auto-selected into a failure.
    ``Windows``
        the reference gate. speexdsp publishes no Windows wheel, so testing for
        it would only ever be a wasted import; naming that here beats letting
        the generic fallback make it look accidental. Windows does have an OS
        canceller (the Voice Capture DSP APO), but PortAudio exposes no way to
        request it -- the same situation as VPIO on macOS. It is not a gap that
        hurts much here: the duplex session already holds a *digital* copy of
        exactly what it sent to the speaker, which is a cleaner reference than
        anything a loopback capture would recover.
    """
    if prefer != "auto":
        return prefer
    if IS_WINDOWS:
        return "shared-refgate"
    if _speexdsp_available():
        return "shared-speex"
    return "shared-refgate"


def build_session(backend: str, delay_frames: int = 4) -> AudioSession:
    """Instantiate a resolved (non-``'auto'``) backend name. See
    select_backend, which is what turns a preference into one of these names."""
    if backend == "off":
        return SharedStreamSession(echo_canceller=PassThrough(), delay_frames=delay_frames)
    if backend == "shared-refgate":
        return SharedStreamSession(echo_canceller=ReferenceGate(), delay_frames=delay_frames)
    if backend == "shared-speex":
        return SharedStreamSession(
            echo_canceller=SpeexEchoCanceller(frame_size=FRAME_SAMPLES), delay_frames=delay_frames
        )
    if backend == "os-echocancel":
        if IS_WINDOWS or IS_MACOS:
            # Caught here rather than inside OsEchoCancelSession.start(), where
            # it would surface as "no device named 'echocancel'" -- technically
            # true, but it sends the reader looking for a missing device
            # instead of telling them the backend cannot exist on this OS.
            raise ValueError(
                "The 'os-echocancel' backend is Linux-only (it captures from "
                "PipeWire/PulseAudio's module-echo-cancel source). Use "
                "--aec auto, or --aec refgate for the dependency-free gate."
            )
        return OsEchoCancelSession()
    if backend == "vpio":
        return CoreAudioVpioSession()
    raise ValueError(
        f"Unknown audio backend '{backend}'. "
        "Options: off, shared-refgate, shared-speex, os-echocancel, vpio, auto."
    )
