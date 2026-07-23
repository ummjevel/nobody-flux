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
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "conversations.db"

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


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


@dataclass
class ConversationStore:
    db_path: Path = DEFAULT_DB_PATH

    def __post_init__(self):
        self.db_path = Path(self.db_path)

    def start_session(self) -> int:
        """Open a new session row, return its id."""
        with closing(_connect(self.db_path)) as conn, conn:
            cur = conn.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (_now(),)
            )
            return cur.lastrowid

    def end_session(self, session_id: int) -> None:
        with closing(_connect(self.db_path)) as conn, conn:
            conn.execute(
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
        with closing(_connect(self.db_path)) as conn, conn:
            cur = conn.execute(
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
