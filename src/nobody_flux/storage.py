"""Conversation storage: SQLite, stdlib-only.

SQLite over anything else here on purpose: it's a single file, no server
process, works identically on the RTX 5090/H100 dev boxes and the eventual
CM4 target, and needs zero extra dependency (Python ships sqlite3). That's
the literal definition of "on-device database."

Three tables:
  - sessions: one row per talk.py run (or scripted session)
  - turns: one row per exchange, including which ASR/LLM/TTS preset produced
    it and how long each stage took -- this doubles as a benchmark log once
    more presets exist in configs/models.yaml (see registry.py)
  - memories: schema only for now. Nothing writes to this table yet -- see
    docs/memory-design.md for the extraction design this is waiting on.

One connection per ConversationStore, opened once and kept for the instance's
lifetime (not reopened + re-CREATE-TABLE'd on every call) -- talk.py logs a
turn every exchange, and open/close-per-call is real I/O overhead to repeat on
every turn, particularly on the eventual SD-card-backed CM4 target.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "conversations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    turn_index INTEGER NOT NULL,
    ts TEXT NOT NULL,
    user_text TEXT NOT NULL,
    reply_text TEXT NOT NULL,
    user_wav_path TEXT,
    reply_wav_path TEXT,
    asr_preset TEXT,
    llm_preset TEXT,
    tts_preset TEXT,
    asr_ms INTEGER,
    llm_ms INTEGER,
    tts_ms INTEGER
);

-- Schema only -- see docs/memory-design.md. Nothing populates this table yet;
-- it exists so the extraction pass (when built) has somewhere to write
-- without a migration.
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationStore:
    db_path: Path = DEFAULT_DB_PATH
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def start_session(self) -> int:
        """Open a new session row, return its id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (_now(),)
            )
            return cur.lastrowid

    def end_session(self, session_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id)
            )

    def log_turn(
        self,
        session_id: int,
        turn_index: int,
        user_text: str,
        reply_text: str,
        *,
        user_wav_path: str | None = None,
        reply_wav_path: str | None = None,
        asr_preset: str | None = None,
        llm_preset: str | None = None,
        tts_preset: str | None = None,
        asr_ms: int | None = None,
        llm_ms: int | None = None,
        tts_ms: int | None = None,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO turns (
                    session_id, turn_index, ts, user_text, reply_text,
                    user_wav_path, reply_wav_path,
                    asr_preset, llm_preset, tts_preset,
                    asr_ms, llm_ms, tts_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_index,
                    _now(),
                    user_text,
                    reply_text,
                    user_wav_path,
                    reply_wav_path,
                    asr_preset,
                    llm_preset,
                    tts_preset,
                    asr_ms,
                    llm_ms,
                    tts_ms,
                ),
            )
            return cur.lastrowid

    def turns_by_preset(self, session_id: int) -> list[tuple]:
        """One row per distinct (asr_preset, llm_preset, tts_preset) combo
        logged under session_id, with turn count and average per-stage ms --
        the aggregation scripts/benchmark.py needs to turn a session's worth
        of individual turns into a comparison table. Not needed by talk.py/
        run_pipeline.py (they only ever log, never read back), which is why
        this lives here instead of as a bare SQL string duplicated in
        benchmark.py.
        """
        return self._conn.execute(
            """
            SELECT asr_preset, llm_preset, tts_preset,
                   COUNT(*), AVG(asr_ms), AVG(llm_ms), AVG(tts_ms),
                   AVG(asr_ms + llm_ms + tts_ms)
            FROM turns
            WHERE session_id = ?
            GROUP BY asr_preset, llm_preset, tts_preset
            ORDER BY AVG(asr_ms + llm_ms + tts_ms)
            """,
            (session_id,),
        ).fetchall()

    def turns_for_session(self, session_id: int) -> list[tuple]:
        """Every turn logged under session_id, in order -- backs
        scripts/benchmark.py's --verbose transcript listing, and
        talk.py's session-end memory extraction (see memory.py)."""
        return self._conn.execute(
            """
            SELECT asr_preset, llm_preset, tts_preset, user_text, reply_text
            FROM turns WHERE session_id = ? ORDER BY turn_index
            """,
            (session_id,),
        ).fetchall()

    def save_memory(
        self,
        session_id: int,
        category: str,
        key: str,
        value: str,
        confidence: float | None = None,
    ) -> int:
        """Writes one extracted fact -- see docs/memory-design.md for the
        category vocabulary and src/nobody_flux/memory.py for what calls
        this (talk.py's session-end extraction pass)."""
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO memories (session_id, category, key, value, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, category, key, value, confidence, _now()),
            )
            return cur.lastrowid

    def recent_memories(self, limit: int = 10) -> list[tuple]:
        """Highest-confidence memories first (rows with no confidence score
        sort last, not first -- SQLite's default is NULL-first on ASC, so
        `confidence IS NULL` as the primary sort key inverts that), most
        recent as the tiebreaker. Capped at `limit` -- see
        docs/memory-design.md's "다음 세션에 어떻게 반영할까" for why this is
        bounded rather than injecting every fact ever extracted into the
        system prompt.

        Deduped across sessions by (category, key): the same fact can get
        re-extracted session after session (e.g. "이름: 지수" said again in
        a later conversation) since memory.py's extraction only ever sees
        one session's turns and has no way to know it already saved this --
        without dedup here, `limit` recent-but-repetitive rows could crowd
        out other distinct facts entirely. Uses a window function (SQLite
        3.25+, bundled with Python 3.11+'s stdlib sqlite3) to keep, per
        (category, key), only the highest-confidence/most-recent row before
        applying ORDER BY/LIMIT -- same tie-break rule as the outer query,
        just scoped per key first. Doesn't delete the superseded rows (this
        is a read-time view, not a write-time merge) -- `memories` stays a
        full history; only what gets recalled into the next session's
        prompt is collapsed.
        """
        return self._conn.execute(
            """
            SELECT category, key, value, confidence FROM (
                SELECT category, key, value, confidence, created_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY category, key
                           ORDER BY confidence IS NULL, confidence DESC, created_at DESC
                       ) AS rn
                FROM memories
            )
            WHERE rn = 1
            ORDER BY confidence IS NULL, confidence DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def memories_for_consolidation(self) -> list[dict]:
        """The current canonical (deduped) memory set WITH row ids, for
        memory.py's Mem0-style consolidation to diff a new session's facts
        against. Same per-(category, key) dedup as recent_memories (so the
        LLM sees one row per fact, not the full history), but returns dicts
        with the winning row's id so an UPDATE op can target the exact row to
        rewrite. Uncapped (unlike recent_memories' LIMIT) -- consolidation
        should compare against everything known, not just the top-N that get
        recalled into a prompt.
        """
        rows = self._conn.execute(
            """
            SELECT id, category, key, value, confidence FROM (
                SELECT id, category, key, value, confidence, created_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY category, key
                           ORDER BY confidence IS NULL, confidence DESC, created_at DESC
                       ) AS rn
                FROM memories
            )
            WHERE rn = 1
            ORDER BY confidence IS NULL, confidence DESC, created_at DESC
            """
        ).fetchall()
        return [
            {"id": r[0], "category": r[1], "key": r[2], "value": r[3], "confidence": r[4]}
            for r in rows
        ]

    def update_memory(self, memory_id: int, value: str, confidence: float | None = None) -> None:
        """Rewrite a stored memory's value/confidence in place (Mem0-style
        UPDATE op). Also refreshes created_at, so the updated fact sorts as
        the most recent for recent_memories' dedup/recency ordering -- an
        UPDATE means "this is the current truth," which should win over any
        older row for the same (category, key)."""
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET value = ?, confidence = ?, created_at = ? WHERE id = ?",
                (value, confidence, _now(), memory_id),
            )
