"""ConversationStore's memory queries against a real (temp) SQLite file: the
NULL-inversion in ORDER BY and the ROW_NUMBER dedup are exactly the SQL nobody
re-derives when recall starts returning plausibly-wrong rows.

_now() is monkeypatched to a deterministic counter where recency matters --
two inserts inside the same microsecond would otherwise tie unpredictably.
"""

from __future__ import annotations

import itertools

import pytest

from src.nobody_flux import storage
from src.nobody_flux.storage import ConversationStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(
        storage, "_now", lambda: f"2026-08-14T00:00:{next(ticks):02d}.000000+00:00"
    )
    store = ConversationStore(db_path=tmp_path / "test.db")
    yield store
    store.close()


def test_recent_memories_orders_by_confidence_nulls_last(store):
    sid = store.start_session()
    store.save_memory(sid, "identity", "이름", "민준", confidence=None)
    store.save_memory(sid, "interest", "취미", "러닝", confidence=0.5)
    store.save_memory(sid, "context", "직업", "교사", confidence=0.9)

    rows = store.recent_memories()
    assert [r[1] for r in rows] == ["직업", "취미", "이름"]


def test_recent_memories_respects_limit(store):
    sid = store.start_session()
    for i in range(5):
        store.save_memory(sid, "interest", f"항목{i}", f"값{i}", confidence=0.5)
    assert len(store.recent_memories(limit=3)) == 3


def test_recent_memories_dedupes_by_category_key(store):
    sid = store.start_session()
    store.save_memory(sid, "identity", "이름", "민준", confidence=0.5)
    store.save_memory(sid, "identity", "이름", "민준이", confidence=0.9)

    rows = store.recent_memories()
    assert len(rows) == 1
    assert rows[0][2] == "민준이"  # higher confidence wins


def test_recent_memories_dedup_tie_prefers_most_recent(store):
    sid = store.start_session()
    store.save_memory(sid, "context", "사는 곳", "서울", confidence=0.8)
    store.save_memory(sid, "context", "사는 곳", "부산", confidence=0.8)
    rows = store.recent_memories()
    assert len(rows) == 1
    assert rows[0][2] == "부산"


def test_same_key_different_category_not_merged(store):
    sid = store.start_session()
    store.save_memory(sid, "interest", "취미", "러닝", confidence=0.8)
    store.save_memory(sid, "preference", "취미", "아침 러닝", confidence=0.8)
    assert len(store.recent_memories()) == 2


def test_memories_for_consolidation_returns_winning_row_ids(store):
    sid = store.start_session()
    losing_id = store.save_memory(sid, "identity", "이름", "민준", confidence=0.5)
    winning_id = store.save_memory(sid, "identity", "이름", "민준이", confidence=0.9)

    rows = store.memories_for_consolidation()
    assert len(rows) == 1
    assert rows[0]["id"] == winning_id != losing_id
    assert rows[0]["value"] == "민준이"


def test_update_memory_rewrites_and_wins_recency(store):
    sid = store.start_session()
    target = store.save_memory(sid, "context", "사는 곳", "서울", confidence=0.8)
    store.save_memory(sid, "context", "사는 곳", "제주", confidence=0.8)  # newer row

    # UPDATE means "current truth": the updated row must out-rank the newer
    # insert on the refreshed created_at, same confidence.
    store.update_memory(target, "부산", confidence=0.8)
    rows = store.recent_memories()
    assert len(rows) == 1
    assert rows[0][2] == "부산"


def test_turns_roundtrip_for_session(store):
    sid = store.start_session()
    store.log_turn(sid, 1, "안녕", "응 안녕", asr_preset="a", llm_preset="l", tts_preset="t")
    store.log_turn(sid, 2, "잘 지냈어?", "응 너는?")
    turns = store.turns_for_session(sid)
    assert [(t[3], t[4]) for t in turns] == [("안녕", "응 안녕"), ("잘 지냈어?", "응 너는?")]


def test_cancelled_turns_are_logged_and_flagged(store):
    """code-review #8: an interrupted reply is real history -- it must land in
    the table (memory extraction reads it) with the cancelled flag set."""
    sid = store.start_session()
    store.log_turn(sid, 1, "얘기해줘", "얘기하자면 말이야", cancelled=True)
    store.log_turn(sid, 2, "계속해", "응 계속할게")

    flags = store._conn.execute(
        "SELECT turn_index, cancelled FROM turns WHERE session_id = ? ORDER BY turn_index",
        (sid,),
    ).fetchall()
    assert flags == [(1, 1), (2, 0)]
    # And the transcript view still sees both turns.
    assert len(store.turns_for_session(sid)) == 2


def test_pre_cancelled_schema_is_migrated(tmp_path):
    """A conversations.db created before the cancelled column existed must be
    ALTERed on open, not crash log_turn."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, ended_at TEXT
        );
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            turn_index INTEGER NOT NULL, ts TEXT NOT NULL,
            user_text TEXT NOT NULL, reply_text TEXT NOT NULL,
            user_wav_path TEXT, reply_wav_path TEXT,
            asr_preset TEXT, llm_preset TEXT, tts_preset TEXT,
            asr_ms INTEGER, llm_ms INTEGER, tts_ms INTEGER
        );
        """
    )
    conn.commit()
    conn.close()

    migrated = ConversationStore(db_path=db)
    try:
        sid = migrated.start_session()
        migrated.log_turn(sid, 1, "안녕", "응", cancelled=True)
        row = migrated._conn.execute("SELECT cancelled FROM turns").fetchone()
        assert row == (1,)
    finally:
        migrated.close()
