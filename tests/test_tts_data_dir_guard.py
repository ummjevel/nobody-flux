"""SherpaMatchaTts's espeak-ng data_dir guard.

Testable without weights only because the guard runs first in __post_init__ and
raises before any model file is opened -- which is also the reason it is placed
there. If someone moves the check below the OfflineTts construction, these tests
start needing a 74MB acoustic model and will say so loudly.

Why the guard exists at all (measured, scripts/_probe_espeak_dependency.py):
espeak-ng falls back to a compiled-in /usr/share/espeak-ng-data when the path it
is handed is not a directory. On this dev box that path does not exist and espeak
aborts the process -- no Python exception. On a Linux host with a system
espeak-ng it would instead succeed silently against the wrong phoneme tables.
"""

from types import SimpleNamespace

import pytest

from src.nobody_flux.stage.tts import SherpaMatchaTts

ESPEAK_FILES = SherpaMatchaTts._ESPEAK_DATA_FILES


def _espeak_dir(tmp_path, *, omit=()):
    d = tmp_path / "espeak-ng-data"
    d.mkdir()
    for f in ESPEAK_FILES:
        if f not in omit:
            (d / f).write_bytes(b"\x00")
    return d


def _build(data_dir):
    return SherpaMatchaTts(data_dir=data_dir)


def test_missing_directory_raises_here_not_in_espeak(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        _build(tmp_path / "nope")
    assert "not found" in str(e.value)


def test_empty_string_data_dir_is_rejected(tmp_path):
    """`data_dir=""` resolves to the cwd, which is a directory, so the
    is_dir() check passes and only the file check catches it. espeak's own
    guard stops at is_dir() and would proceed to die on phontab."""
    with pytest.raises(FileNotFoundError) as e:
        _build("")
    assert "phontab" in str(e.value)


def test_existing_but_empty_directory_is_rejected(tmp_path):
    """The case espeak-ng's check_data_path() cannot see: a real directory with
    none of the phoneme tables in it."""
    d = tmp_path / "espeak-ng-data"
    d.mkdir()
    with pytest.raises(FileNotFoundError) as e:
        _build(d)
    for f in ESPEAK_FILES:
        assert f in str(e.value)


@pytest.mark.parametrize("omitted", ESPEAK_FILES)
def test_each_required_table_is_checked_individually(tmp_path, omitted):
    """A partial bundle is the likely real-world failure -- an interrupted
    download, a truncated image -- and the error has to name what is missing."""
    with pytest.raises(FileNotFoundError) as e:
        _build(_espeak_dir(tmp_path, omit=(omitted,)))
    msg = str(e.value)
    assert omitted in msg
    assert all(f not in msg for f in ESPEAK_FILES if f != omitted)


def test_a_directory_named_like_a_table_does_not_satisfy_the_check(tmp_path):
    """is_file(), not exists(): a directory called phontab is not phontab."""
    d = tmp_path / "espeak-ng-data"
    d.mkdir()
    for f in ESPEAK_FILES:
        (d / f).mkdir()
    with pytest.raises(FileNotFoundError):
        _build(d)


def test_error_points_at_the_notices_file(tmp_path):
    """The GPL-3.0 situation behind this dependency is not obvious from the
    stack trace, so the message carries the pointer."""
    with pytest.raises(FileNotFoundError) as e:
        _build(tmp_path / "nope")
    assert "THIRD-PARTY-NOTICES.md" in str(e.value)


def test_a_complete_bundle_passes_the_guard(tmp_path):
    """The guard must not be the thing that fails on a well-formed bundle.

    Called on a duck-typed stand-in rather than through the constructor on
    purpose. Going through __post_init__ here would get past the guard and hand
    espeak-ng a directory of zero-byte files named phontab and friends, and
    espeak parses those in C: the first version of this test aborted the whole
    pytest process with no failure report, only truncated output and a nonzero
    exit. The guard only reads self.data_dir, so this exercises exactly it.
    """
    stub = SimpleNamespace(
        data_dir=_espeak_dir(tmp_path), _ESPEAK_DATA_FILES=ESPEAK_FILES
    )
    assert SherpaMatchaTts._check_data_dir(stub) is None
