"""VadStream + Utterance duration semantics, driven by a fake sherpa VAD.

The headline regression here is code-review-20260814 #1: Utterance.duration_s
includes pre_roll_ms of padding, so when pre-roll grew 300->500ms (afc0df8)
every capture measured >= ~0.65s and the 0.6s backchannel duration gate could
never pass -- the entire stage-2 lexical check became dead code. The fix is
speech_duration_s, measured from the VAD's own segment lengths, and these
tests pin both that value and the gate actually firing on it.

VadStream reads its sherpa VAD off the config object (config._vad), which is
what makes this drivable with a fake: no ONNX model, no audio device.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.nobody_flux.turn.backchannel import is_backchannel
from src.nobody_flux.turn.vad import FRAME_SAMPLES, SAMPLE_RATE, Utterance, VadEvent, VadStream


class FakeSegment:
    def __init__(self, start: int, length: int):
        self.start = start
        self.samples = np.zeros(length, dtype=np.float32)


class FakeSherpaVad:
    """Scriptable stand-in for sherpa_onnx.VoiceActivityDetector: the test sets
    `speech` and queues segments; VadStream reads them exactly as it would from
    the real one."""

    def __init__(self):
        self.speech = False
        self.segments: list[FakeSegment] = []

    def accept_waveform(self, frame):
        pass

    def is_speech_detected(self):
        return self.speech

    def empty(self):
        return not self.segments

    @property
    def front(self):
        return self.segments[0]

    def pop(self):
        self.segments.pop(0)

    def reset(self):
        self.speech = False
        self.segments.clear()


def make_stream(fake_vad, *, pre_roll_ms=500, turn_detector=None, grace_frames=10):
    config = SimpleNamespace(
        _vad=fake_vad,
        pre_roll_ms=pre_roll_ms,
        barge_in_confirm_ms=250,
        max_speech_duration=20.0,
        endpoint_grace_ms=800,
        grace_frames_for_prob=lambda prob: grace_frames,
    )
    return VadStream(config, turn_detector=turn_detector)


def push_frames(stream, count: int) -> list[VadEvent]:
    events = []
    frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    for _ in range(count):
        events.extend(stream.push(frame))
    return events


def capture_utterance(speech_frames: int, pre_roll_ms: int = 500) -> Utterance:
    """Silence, then `speech_frames` frames of speech, then the segment
    finalizes -- the shape of every ordinary single-segment turn."""
    fake = FakeSherpaVad()
    stream = make_stream(fake, pre_roll_ms=pre_roll_ms)

    silence_frames = 20
    push_frames(stream, silence_frames)

    fake.speech = True
    push_frames(stream, speech_frames)

    fake.speech = False
    speech_start = silence_frames * FRAME_SAMPLES
    fake.segments.append(FakeSegment(speech_start, speech_frames * FRAME_SAMPLES))
    events = push_frames(stream, 1)
    assert VadEvent.UTTERANCE_READY in events
    utterance = stream.take_utterance()
    assert utterance is not None
    return utterance


def test_speech_duration_excludes_pre_roll():
    utterance = capture_utterance(speech_frames=7)  # 210ms of speech
    assert utterance.speech_duration_s == (7 * FRAME_SAMPLES) / SAMPLE_RATE
    # The full buffer includes the 500ms pre-roll on top.
    assert utterance.duration_s > utterance.speech_duration_s
    assert abs(utterance.duration_s - (0.5 + utterance.speech_duration_s)) < 0.031


def test_backchannel_gate_fires_on_speech_duration():
    """THE regression: a 210ms '응' must be judged by its speech length."""
    utterance = capture_utterance(speech_frames=7)
    assert is_backchannel("응", utterance.speech_duration_s) is True
    # The pre-bug-fix input -- the pre-roll-inflated buffer length -- fails the
    # gate, which is exactly how the feature died. Kept as documentation.
    assert is_backchannel("응", utterance.duration_s) is False


def test_long_utterance_is_not_backchannel():
    utterance = capture_utterance(speech_frames=30)  # 900ms
    assert is_backchannel("응", utterance.speech_duration_s) is False


def test_speech_duration_falls_back_to_buffer_length():
    # Capture paths that don't measure speech (or old callers) still work.
    utterance = Utterance(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert utterance.speech_duration_s == utterance.duration_s == 1.0


def test_barge_in_confirm_fires_once_after_threshold():
    fake = FakeSherpaVad()
    stream = make_stream(fake)
    push_frames(stream, 5)

    fake.speech = True
    # 250ms confirm = 4000 samples = 8.33 frames -> fires on the 9th frame.
    events = push_frames(stream, 8)
    assert events.count(VadEvent.SPEECH_STARTED) == 1
    assert VadEvent.BARGE_IN_CONFIRMED not in events
    events = push_frames(stream, 1)
    assert events == [VadEvent.BARGE_IN_CONFIRMED]
    # Once per turn, no matter how long speech continues.
    assert VadEvent.BARGE_IN_CONFIRMED not in push_frames(stream, 20)


def test_speech_duration_accumulates_across_carried_segments():
    """A turn the endpoint detector extends over two segments counts both
    segments' speech -- and only the segments, not the silence between them."""
    fake = FakeSherpaVad()
    detector = SimpleNamespace(verdicts=[(False, 0.2), (True, 0.9)])
    detector.predict = lambda audio, sr: detector.verdicts.pop(0)
    stream = make_stream(fake, turn_detector=detector, grace_frames=10)

    push_frames(stream, 20)
    fake.speech = True
    push_frames(stream, 10)
    fake.speech = False
    fake.segments.append(FakeSegment(20 * FRAME_SAMPLES, 10 * FRAME_SAMPLES))
    events = push_frames(stream, 1)
    assert VadEvent.UTTERANCE_READY not in events  # judged incomplete, carried

    # The user resumes: 5 more frames of speech, then the segment finalizes.
    # (The fake VAD's segment start is counted from its reset, which the carry
    # path performs -- mirror that by restarting absolute positions.)
    push_frames(stream, 3)
    fake.speech = True
    push_frames(stream, 5)
    fake.speech = False
    fake.segments.append(FakeSegment(4 * FRAME_SAMPLES, 5 * FRAME_SAMPLES))
    events = push_frames(stream, 1)
    assert VadEvent.UTTERANCE_READY in events

    utterance = stream.take_utterance()
    assert utterance.speech_samples == 15 * FRAME_SAMPLES


def test_grace_timeout_returns_carried_audio_with_its_speech_duration():
    fake = FakeSherpaVad()
    detector = SimpleNamespace(predict=lambda audio, sr: (False, 0.2))
    stream = make_stream(fake, turn_detector=detector, grace_frames=5)

    push_frames(stream, 20)
    fake.speech = True
    push_frames(stream, 10)
    fake.speech = False
    fake.segments.append(FakeSegment(20 * FRAME_SAMPLES, 10 * FRAME_SAMPLES))
    push_frames(stream, 1)  # incomplete -> carried

    # No resume: after grace_frames of silence the verdict is overruled and
    # the carried audio comes back as the turn.
    events = push_frames(stream, 5)
    assert VadEvent.UTTERANCE_READY in events
    utterance = stream.take_utterance()
    assert utterance.speech_samples == 10 * FRAME_SAMPLES


def test_take_utterance_resets_turn_state():
    first = capture_utterance(speech_frames=7)
    assert first.speech_samples == 7 * FRAME_SAMPLES

    # A second, independent capture must not inherit the first's speech count.
    second = capture_utterance(speech_frames=10)
    assert second.speech_samples == 10 * FRAME_SAMPLES
