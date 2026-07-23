"""Energy-based voice activity detection: decides when one spoken turn starts
and ends, without a wakeword or full streaming ASR.

Deliberately simple: RMS energy over short frames, compared against two
thresholds (start/silence), gated by minimum speech/silence durations. This
is *not* a real VAD model (no ML, no noise-robustness beyond the thresholds
below) -- it's the smallest thing that turns "record for N seconds" into
"record until the person stops talking," which is what makes talk.py feel
like a conversation instead of a walkie-talkie.

Known limits (document, don't silently paper over):
  - A single fixed energy threshold doesn't adapt to room noise floor or mic
    gain -- SILENCE_THRESHOLD/START_THRESHOLD below WILL need retuning per
    microphone/environment. Override via the constructor, not by editing
    these constants.
  - No barge-in: recording only happens between turns, not during TTS
    playback. Full-duplex is out of scope for this pass (see pipeline.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# RMS (on float32 samples in [-1, 1]) above which a frame counts as "speech."
# Typical quiet-room speech at a normal mic distance lands well above this;
# typical room noise floor lands well below it -- but "typical" is doing a
# lot of work in that sentence, hence the module docstring's warning.
START_THRESHOLD = 0.02
SILENCE_THRESHOLD = 0.012

START_MS = 150  # must see speech-level energy for this long to start recording
SILENCE_MS = 800  # must see silence for this long (after speech) to end the turn
MAX_UTTERANCE_MS = 20_000  # hard cap so a stuck-open mic can't hang the loop forever


@dataclass
class Utterance:
    audio: np.ndarray  # float32, mono, SAMPLE_RATE
    sample_rate: int = SAMPLE_RATE


@dataclass
class VoiceActivityDetector:
    start_threshold: float = START_THRESHOLD
    silence_threshold: float = SILENCE_THRESHOLD
    start_ms: int = START_MS
    silence_ms: int = SILENCE_MS
    max_utterance_ms: int = MAX_UTTERANCE_MS

    _start_frames_needed: int = field(init=False)
    _silence_frames_needed: int = field(init=False)
    _max_frames: int = field(init=False)

    def __post_init__(self):
        self._start_frames_needed = max(1, self.start_ms // FRAME_MS)
        self._silence_frames_needed = max(1, self.silence_ms // FRAME_MS)
        self._max_frames = max(1, self.max_utterance_ms // FRAME_MS)

    def listen_for_utterance(self) -> Utterance:
        """Block until one spoken turn has been captured, then return it.

        State machine: IDLE (waiting for start_frames_needed consecutive
        loud frames) -> RECORDING (buffering, waiting for silence_frames_needed
        consecutive quiet frames after having recorded anything) -> return.
        """
        frames: list[np.ndarray] = []
        loud_streak = 0
        quiet_streak = 0
        recording = False

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
        ) as stream:
            while True:
                block, _overflowed = stream.read(FRAME_SAMPLES)
                block = block[:, 0]
                energy = float(np.sqrt(np.mean(np.square(block))))

                if not recording:
                    if energy >= self.start_threshold:
                        loud_streak += 1
                    else:
                        loud_streak = 0
                    if loud_streak >= self._start_frames_needed:
                        recording = True
                        frames.extend([block] * loud_streak)  # don't clip the onset
                    continue

                frames.append(block)

                if energy < self.silence_threshold:
                    quiet_streak += 1
                else:
                    quiet_streak = 0

                if quiet_streak >= self._silence_frames_needed:
                    break
                if len(frames) >= self._max_frames:
                    break

        audio = np.concatenate(frames) if frames else np.zeros(0, dtype="float32")
        return Utterance(audio=audio, sample_rate=SAMPLE_RATE)
