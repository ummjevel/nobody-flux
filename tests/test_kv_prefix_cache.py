"""Unit tests for the prompt-prefix KV cache keying and pruning in stage/llm.py.

Neither function loads a model, which is the point of them being free functions:
the risky part of the disk-backed warm_up() is deciding *whether* a snapshot may
be restored, and that decision has to be testable without a 484MB GGUF. The
restore path itself is covered by scripts/_verify_kv_prefix.py, because its
failure mode (KV and token bookkeeping disagreeing) only shows up as wrong
generated text and needs a real model to see.
"""

import os

import pytest

from src.nobody_flux.stage.llm import (
    KV_CACHE_FILE_PREFIX,
    kv_prefix_cache_key,
    prune_stale_kv_snapshots,
)


def _snapshot(d, key, content=b"x"):
    p = d / ("%s%s.bin" % (KV_CACHE_FILE_PREFIX, key))
    p.write_bytes(content)
    return p


# ------------------------------------------------------------------ cache key

def test_key_is_stable_for_the_same_inputs(tmp_path):
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    assert kv_prefix_cache_key(m, [1, 2, 3]) == kv_prefix_cache_key(m, [1, 2, 3])


def test_key_changes_when_the_prefix_changes(tmp_path):
    """A persona or few-shot edit shifts the tokens, and must miss the cache.

    This is the case that would otherwise restore a KV state belonging to a
    prompt the model is no longer being given -- which does not raise, it
    generates from a state that never existed.
    """
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    assert kv_prefix_cache_key(m, [1, 2, 3]) != kv_prefix_cache_key(m, [1, 2, 4])


def test_key_is_order_sensitive(tmp_path):
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    assert kv_prefix_cache_key(m, [1, 2]) != kv_prefix_cache_key(m, [2, 1])


def test_key_distinguishes_prefixes_that_share_a_flattening(tmp_path):
    """[1, 23] and [12, 3] must not collide.

    They would if the tokens were joined without a separator. Cheap to get
    wrong, and the consequence is restoring the wrong snapshot silently.
    """
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    assert kv_prefix_cache_key(m, [1, 23]) != kv_prefix_cache_key(m, [12, 3])


def test_key_changes_when_the_model_file_size_changes(tmp_path):
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    before = kv_prefix_cache_key(m, [1, 2, 3])
    m.write_bytes(b"weights-but-longer")
    assert kv_prefix_cache_key(m, [1, 2, 3]) != before


def test_key_changes_when_the_model_mtime_changes(tmp_path):
    """A redownloaded GGUF of identical name and size is not necessarily the
    same bytes, so mtime is part of the key on purpose."""
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    before = kv_prefix_cache_key(m, [1, 2, 3])
    st = m.stat()
    os.utime(m, (st.st_atime, st.st_mtime + 120))
    assert kv_prefix_cache_key(m, [1, 2, 3]) != before


def test_key_falls_back_instead_of_raising_when_the_model_is_absent(tmp_path):
    """warm_up() swallows failures, so this must not be the thing that raises."""
    missing = tmp_path / "gone.gguf"
    key = kv_prefix_cache_key(missing, [1, 2, 3])
    assert isinstance(key, str) and key
    assert key == kv_prefix_cache_key(missing, [1, 2, 3])


def test_key_differs_between_two_absent_models(tmp_path):
    a = kv_prefix_cache_key(tmp_path / "a.gguf", [1])
    b = kv_prefix_cache_key(tmp_path / "b.gguf", [1])
    assert a != b


def test_empty_prefix_still_produces_a_key(tmp_path):
    m = tmp_path / "model.gguf"
    m.write_bytes(b"weights")
    assert kv_prefix_cache_key(m, [])


# -------------------------------------------------------------------- pruning

def test_prune_removes_superseded_snapshots(tmp_path):
    """Each persona edit strands another ~75MB file; pruning is required, not tidy."""
    keep = _snapshot(tmp_path, "aaaa")
    stale1 = _snapshot(tmp_path, "bbbb")
    stale2 = _snapshot(tmp_path, "cccc")
    removed = prune_stale_kv_snapshots(tmp_path, keep=keep)
    assert set(removed) == {stale1, stale2}
    assert keep.exists()
    assert not stale1.exists() and not stale2.exists()


def test_prune_with_no_keep_removes_everything(tmp_path):
    a = _snapshot(tmp_path, "aaaa")
    b = _snapshot(tmp_path, "bbbb")
    assert set(prune_stale_kv_snapshots(tmp_path)) == {a, b}
    assert not a.exists() and not b.exists()


def test_prune_ignores_files_that_are_not_snapshots(tmp_path):
    """Narrow by design -- this points at a directory under data/."""
    keep = _snapshot(tmp_path, "aaaa")
    bystanders = [tmp_path / "notes.txt", tmp_path / "prefix-aaaa.bin.tmp", tmp_path / "other.bin"]
    for f in bystanders:
        f.write_bytes(b"keep me")
    prune_stale_kv_snapshots(tmp_path, keep=keep)
    assert all(f.exists() for f in bystanders)


def test_prune_does_not_recurse(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    buried = _snapshot(sub, "bbbb")
    prune_stale_kv_snapshots(tmp_path)
    assert buried.exists()


def test_prune_skips_directories_named_like_snapshots(tmp_path):
    d = tmp_path / ("%saaaa.bin" % KV_CACHE_FILE_PREFIX)
    d.mkdir()
    assert prune_stale_kv_snapshots(tmp_path) == []
    assert d.is_dir()


def test_prune_on_missing_directory_is_a_noop(tmp_path):
    """First run ever: warm_up() calls this before the directory can exist."""
    assert prune_stale_kv_snapshots(tmp_path / "never-created") == []


def test_prune_keeps_a_file_it_was_told_to_keep_even_if_named_differently(tmp_path):
    """keep is compared by resolved path, not by name, so a relative path works."""
    keep = _snapshot(tmp_path, "aaaa")
    stale = _snapshot(tmp_path, "bbbb")
    removed = prune_stale_kv_snapshots(tmp_path, keep=str(keep))
    assert removed == [stale]
    assert keep.exists()
