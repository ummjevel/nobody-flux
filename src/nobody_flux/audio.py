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

build_audio_session() (see registry.build_audio_session) auto-selects by
platform + installed libs, overridable via configs/audio.yaml / --aec.
"""

from __future__ import annotations

import collections
import queue
import sys
import threading
from dataclasses import dataclass, field

import numpy as np

from .aec import EchoCanceller, PassThrough, ReferenceGate, SpeexEchoCanceller

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480, matches vad.py


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolate mono audio to sr_out. Cheap and dependency-free
    (numpy only) -- adequate for feeding a 16k duplex stream from 22k/48k TTS
    output, where the consumer is a voice pipeline, not a mastering chain. The
    16k output ceiling is a deliberate simplicity tradeoff of the shared-stream
    backend (the OS backends keep native quality)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or len(x) == 0:
        return np.ascontiguousarray(x, dtype=np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(x), endpoint=False)
    dst = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(dst, src, x).astype(np.float32)


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

    def __post_init__(self):
        self._lock = threading.Lock()
        self._play_buf = np.zeros(0, dtype=np.float32)
        self._captured: queue.Queue = queue.Queue()
        # Ring of recently-played frames; the one delay_frames back is the
        # reference the mic is hearing now.
        self._ref_ring: collections.deque = collections.deque(maxlen=self.delay_frames + 1)
        self._active = threading.Event()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="float32",
            device=(self.input_device, self.output_device),
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, outdata, frames, time_info, status):  # pragma: no cover - device
        mic = np.array(indata[:, 0], dtype=np.float32, copy=True)
        played = self._fill_output(frames)
        outdata[:, 0] = played
        self._captured.put(self._process_block(mic, played))

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
        return self._captured.get()

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
    """Resolve prefer='auto' to a concrete backend name for this platform, or
    pass a concrete choice through. Concrete names: shared-speex, shared-refgate,
    os-echocancel, vpio, off.

    auto policy: macOS -> shared-speex if speexdsp present else shared-refgate
    (single duplex stream fixes err -50 either way); Linux -> shared-speex if
    speexdsp present else shared-refgate (OS module-echo-cancel is better still
    but must be set up explicitly, so it's opt-in, not auto); anything else ->
    shared-refgate."""
    if prefer != "auto":
        return prefer
    if _speexdsp_available():
        return "shared-speex"
    return "shared-refgate"


def build_session(backend: str, delay_frames: int = 4) -> AudioSession:
    """Instantiate a resolved (non-'auto') backend name. See select_backend."""
    if backend == "off":
        return SharedStreamSession(echo_canceller=PassThrough(), delay_frames=delay_frames)
    if backend == "shared-refgate":
        return SharedStreamSession(echo_canceller=ReferenceGate(), delay_frames=delay_frames)
    if backend == "shared-speex":
        return SharedStreamSession(
            echo_canceller=SpeexEchoCanceller(frame_size=FRAME_SAMPLES), delay_frames=delay_frames
        )
    if backend == "os-echocancel":
        return OsEchoCancelSession()
    if backend == "vpio":
        return CoreAudioVpioSession()
    raise ValueError(
        f"Unknown audio backend '{backend}'. "
        "Options: off, shared-refgate, shared-speex, os-echocancel, vpio, auto."
    )


def is_macos() -> bool:
    return sys.platform == "darwin"
