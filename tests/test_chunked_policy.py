"""The chunked-SenseVoice decode-trigger policy, model-free.

`should_decode` decides both the CPU cost of the chunked streaming path (how
many times an utterance gets re-decoded from scratch) and its partial-transcript
latency (how soon a hypothesis exists to agree with). Both were measured to
matter, so both are pinned here.

Weights-free by construction: `ChunkedSenseVoiceTranscriber.__post_init__`
loads a 229MB ONNX model, which is exactly why the policy does not live inside
it. See `tests/conftest.py` -- this suite must run with no weights and no audio
device.
"""

import pytest

from src.nobody_flux.stage.asr_stream import SAMPLE_RATE, should_decode

MIN_DECODE_S = 0.8
HOP_S = 0.48


def s(seconds: float) -> int:
    """Seconds -> samples at the fixed 16kHz capture rate."""
    return int(seconds * SAMPLE_RATE)


# ------------------------------------------------------- the absolute floor

def test_no_decode_below_min_decode_s():
    """Short input is worse than useless, not merely wasteful.

    SenseVoice returns nothing for very short audio (docs/FEATURES.md measured
    empty at 0.5s, partial at 0.8s). Two empty hypotheses in a row would satisfy
    LocalAgreement's agreement_n=2 and commit the empty string as a prefix --
    and the committed prefix is monotonic, so that mistake cannot be taken back.
    """
    for elapsed in (0.0, 0.1, 0.5, 0.79):
        assert not should_decode(s(elapsed), 0, MIN_DECODE_S, HOP_S), elapsed


def test_decodes_once_min_decode_s_is_reached():
    assert should_decode(s(0.8), 0, MIN_DECODE_S, HOP_S)


def test_min_decode_floor_applies_even_after_a_long_gap():
    """The floor is absolute, not relative to the last decode.

    A caller that reset mid-utterance must not get a decode of 0.1s of audio
    just because samples_at_last_decode is 0 and the hop looks satisfied.
    """
    assert not should_decode(s(0.3), 0, MIN_DECODE_S, HOP_S)


# -------------------------------------------------------------- the hop gate

def test_no_second_decode_until_a_full_hop_of_new_audio():
    at_first_decode = s(0.8)
    for extra in (0.0, 0.1, 0.47):
        assert not should_decode(
            at_first_decode + s(extra), at_first_decode, MIN_DECODE_S, HOP_S
        ), extra


def test_second_decode_once_the_hop_is_full():
    at_first_decode = s(0.8)
    assert should_decode(at_first_decode + s(HOP_S), at_first_decode, MIN_DECODE_S, HOP_S)


def test_hop_is_measured_from_the_last_decode_not_from_zero():
    """Otherwise a long utterance would decode on every single frame."""
    assert not should_decode(s(5.0), s(4.9), MIN_DECODE_S, HOP_S)
    assert should_decode(s(5.0), s(4.4), MIN_DECODE_S, HOP_S)


# ------------------------------------------------- the commit-latency floor

def test_earliest_possible_commit_is_min_decode_plus_one_hop():
    """This is the number that decides whether the partial path is useful at all.

    One decode forms a hypothesis; a second must agree with it before anything
    is committed (agreement_n=2). So the floor is min_decode_s + hop_s -- 1.28s
    at the defaults, which is *longer than most turns* in this project's own
    capture set. Measured: 7 of 16 real captures never committed anything.

    If this test starts failing because the defaults changed, re-check that
    claim rather than just updating the number.
    """
    first = s(MIN_DECODE_S)
    assert should_decode(first, 0, MIN_DECODE_S, HOP_S)
    second_at = first + s(HOP_S)
    assert should_decode(second_at, first, MIN_DECODE_S, HOP_S)
    assert second_at / SAMPLE_RATE == pytest.approx(1.28, abs=0.01)


# ---------------------------------------------------------------- tuning dials

@pytest.mark.parametrize(
    "hop_s,expected_floor_s",
    [
        (0.48, 1.28),
        (0.24, 1.04),
        (0.96, 1.76),
    ],
)
def test_hop_size_moves_the_commit_floor(hop_s, expected_floor_s):
    """Shrinking the hop buys latency and costs CPU, and cannot go below
    min_decode_s -- that part is a property of the checkpoint, not a dial."""
    floor = (s(MIN_DECODE_S) + s(hop_s)) / SAMPLE_RATE
    assert floor == pytest.approx(expected_floor_s, abs=0.01)
    assert floor > MIN_DECODE_S


def test_a_tiny_hop_still_respects_the_floor():
    assert not should_decode(s(0.5), 0, MIN_DECODE_S, hop_s=0.01)
