"""Frame-level echo cancellers for the barge-in mic loop.

The problem (documented in scripts/talk.py's "No echo cancellation" note): when
the mic is open while nobody's reply plays out of the same device's speaker, the
reply bleeds back into the mic. Without cancellation, TEN-VAD can mistake that
bleed for the user speaking and fire a false barge-in (nobody interrupting
itself). On the CM4 target -- speaker and mic in one enclosure -- this is not
hypothetical the way it is on the WSL2 dev box (separate logical devices).

Two backends, same tiny interface (process(mic_frame, ref_frame) -> cleaned
frame, both 16k mono float32, equal length, ref already delay-aligned to mic by
the caller -- see audio.AudioSession and scripts/_calibrate_aec_delay.py):

  - ReferenceGate: no dependencies, no adaptive filter. During playback it just
    decides "is this mic frame mostly the reference echoing back?" (high
    normalized cross-correlation with the aligned reference) and, if so,
    attenuates it toward silence so the VAD doesn't trip. It does NOT clean the
    audio for ASR -- but the barge-in path only needs to *detect* real user
    speech during playback (the reply is then clipped and a fresh turn starts),
    so suppression is enough and far cheaper than true AEC. This is the "lighter
    than AEC" universal fallback and the always-available default.

  - SpeexEchoCanceller: real AEC (SpeexDSP's MDF frequency-domain adaptive
    filter, C, ARM/x86, import-guarded on the optional `speexdsp` package). Use
    when a cleaned mic signal is actually wanted. Handles double-talk itself.

OS-level cancellers (macOS VoiceProcessingIO, PipeWire module-echo-cancel) don't
fit this per-frame shape -- they clean the capture stream inside the OS -- so
they live in audio.py as AudioSession backends, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
_EPS = 1e-9


class EchoCanceller:
    """Interface: turn a mic frame into a frame with the speaker echo removed or
    suppressed, given the reference (what was playing) aligned to it. Subclasses
    must not change the frame length."""

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any adaptive state at a turn boundary. No-op for stateless
        backends."""


class PassThrough(EchoCanceller):
    """No echo handling -- returns the mic frame untouched. The explicit
    '--aec off' backend and the default on setups where playback can't reach the
    mic (separate devices), so nothing is spent pretending otherwise."""

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return mic


@dataclass
class ReferenceGate(EchoCanceller):
    """Reference-gated suppression (see module docstring). Cheap: one dot product
    per frame. Stateless."""

    # Normalized cross-correlation above which the mic frame is treated as echo
    # of the reference and suppressed. 1.0 = mic is a scaled copy of ref, 0.0 =
    # uncorrelated. 0.7 leaves headroom for the room/codec coloring the echo
    # (it's never a perfect copy) while staying well above the correlation an
    # independent voice has with the reference. Tuning knob -- confirm against a
    # real speaker/mic (scripts/_calibrate_aec_delay.py records the pair).
    corr_threshold: float = 0.7
    # What a suppressed frame is multiplied by. Not exactly 0 so a genuine
    # double-talk frame (user speaking *over* the reply -- correlated with ref
    # but with extra energy) isn't perfectly silenced; the residual still lets a
    # sufficiently loud real interruption push VAD over its threshold.
    attenuation: float = 0.1
    # Below this reference energy nothing is playing loudly enough to echo, so
    # the frame passes through untouched (avoids gating on quiet tails/noise).
    ref_energy_floor: float = 1e-4

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        ref_energy = float(np.dot(ref, ref)) / max(len(ref), 1)
        if ref_energy < self.ref_energy_floor:
            return mic
        mic_energy = float(np.dot(mic, mic))
        if mic_energy < _EPS:
            return mic
        corr = float(np.dot(mic, ref)) / (np.sqrt(mic_energy * float(np.dot(ref, ref))) + _EPS)
        if abs(corr) >= self.corr_threshold:
            return mic * self.attenuation
        return mic


@dataclass
class SpeexEchoCanceller(EchoCanceller):
    """True AEC via SpeexDSP (optional `speexdsp` package). Frames are converted
    to the int16 PCM SpeexDSP expects and back. filter_length is the echo tail
    the adaptive filter models -- 2048 samples ~= 128ms at 16k, comfortably past
    typical speaker->mic acoustic delay."""

    frame_size: int
    filter_length: int = 2048
    _ec: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Import-guarded: speexdsp is an optional dep (not every platform ships a
        # wheel; the CM4/Linux + macOS targets do). Failure here is actionable,
        # not a mystery deep in process().
        try:
            from speexdsp import EchoCanceller as _SpeexEC
        except ImportError as exc:
            raise ImportError(
                "SpeexEchoCanceller needs the 'speexdsp' package "
                "(pip install speexdsp). Use --aec refgate for the dependency-free "
                "reference gate instead."
            ) from exc
        self._ec = _SpeexEC.create(self.frame_size, self.filter_length)

    @staticmethod
    def _to_i16(frame: np.ndarray) -> bytes:
        clipped = np.clip(frame, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def _from_i16(raw: bytes) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        out = self._ec.process(self._to_i16(mic), self._to_i16(ref))
        cleaned = self._from_i16(out)
        # Length is preserved by SpeexDSP, but guard against a short final frame
        # so callers can rely on the contract.
        if len(cleaned) != len(mic):
            cleaned = np.resize(cleaned, len(mic))
        return np.ascontiguousarray(cleaned, dtype=np.float32)

    def reset(self) -> None:
        """Rebuild the adaptive filter. The base-class reset() was a silent
        no-op here (code-review #5's footnote): AudioSession.stop_playback()
        was 'resetting' an object that never dropped its state. A filter that
        adapted onto a reply mid-echo diverges when playback cuts abruptly;
        starting clean at the turn boundary reconverges in well under a
        second, which is cheaper than dragging a diverged tail into the next
        reply."""
        from speexdsp import EchoCanceller as _SpeexEC

        self._ec = _SpeexEC.create(self.frame_size, self.filter_length)
