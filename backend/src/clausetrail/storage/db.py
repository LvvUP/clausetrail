import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, path TEXT NOT NULL, pages INTEGER NOT NULL,
    chunks INTEGER NOT NULL, page_types TEXT NOT NULL, parsers TEXT NOT NULL, status TEXT NOT NULL,
    kb_version INTEGER NOT NULL, ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
    summarized_turns INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    question TEXT NOT NULL, standalone_query TEXT NOT NULL, topic TEXT NOT NULL, answer TEXT NOT NULL,
    status TEXT NOT NULL, refusal_reason TEXT, citations TEXT NOT NULL, used_citations TEXT NOT NULL,
    cache_hit INTEGER NOT NULL, timings TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, created_at);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('kb_version', '0')")
            # 每个库一个随机 epoch，参与缓存 key；删库重建后版本号从 0 重来也不会撞上旧缓存
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('kb_epoch', ?)", (uuid.uuid4().hex,))
            self._conn.commit()

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
            self._conn.commit()
            return rows

    def kb_version(self) -> int:
        return int(self._exec("SELECT value FROM meta WHERE key = 'kb_version'")[0]["value"])

    # 加一和读回在同一条语句里完成：分成两次加锁时，中间别的线程再加一次，返回的就不是自己加出来的版本
    def bump_kb_version(self) -> int:
        rows = self._exec("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'kb_version' RETURNING value")
        return int(rows[0]["value"])

    def get_meta(self, key: str) -> str | None:
        rows = self._exec("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def delete_meta(self, key: str) -> None:
        self._exec("DELETE FROM meta WHERE key = ?", (key,))

    # 按前缀列出 meta；用 substr 比较，不用 LIKE，前缀里的 _ 不会被当成通配符
    def meta_items(self, prefix: str) -> dict[str, str]:
        rows = self._exec("SELECT key, value FROM meta WHERE substr(key, 1, ?) = ?", (len(prefix), prefix))
        return {r["key"]: r["value"] for r in rows}

    def upsert_document(self, doc: dict[str, Any]) -> None:
        self._exec(
            """INSERT INTO documents VALUES (:doc_id, :file_name, :path, :pages, :chunks, :page_types, :parsers,
                                              :status, :kb_version, :ingested_at)
               ON CONFLICT(doc_id) DO UPDATE SET file_name=excluded.file_name, path=excluded.path, pages=excluded.pages,
                 chunks=excluded.chunks, page_types=excluded.page_types, parsers=excluded.parsers,
                 status=excluded.status,
                 kb_version=excluded.kb_version, ingested_at=excluded.ingested_at""",
            {**doc, "page_types": json.dumps(doc["page_types"]), "parsers": json.dumps(doc["parsers"])},
        )

    def kb_epoch(self) -> str:
        return self.get_meta("kb_epoch") or ""

    @staticmethod
    def _document(r: sqlite3.Row) -> dict[str, Any]:
        return {**dict(r), "page_types": json.loads(r["page_types"]), "parsers": json.loads(r["parsers"])}

    def list_documents(self) -> list[dict[str, Any]]:
        return [self._document(r) for r in self._exec("SELECT * FROM documents ORDER BY ingested_at DESC")]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return self._document(rows[0]) if rows else None

    # 检索时用：每份可用文档当前指向哪个版本的片段
    def ready_versions(self) -> dict[str, int]:
        rows = self._exec("SELECT doc_id, kb_version FROM documents WHERE status = 'ready'")
        return {r["doc_id"]: int(r["kb_version"]) for r in rows}

    def count_documents(self) -> int:
        return int(self._exec("SELECT COUNT(*) AS n FROM documents")[0]["n"])

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def create_session(self, title: str) -> str:
        session_id = uuid.uuid4().hex[:16]
        ts = now_iso()
        self._exec(
            "INSERT INTO sessions(session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, ts, ts),
        )
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return dict(rows[0]) if rows else None

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        return [dict(r) for r in self._exec("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,))]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def set_summary(self, session_id: str, summary: str, summarized_turns: int) -> None:
        self._exec(
            "UPDATE sessions SET summary = ?, summarized_turns = ? WHERE session_id = ?",
            (summary, summarized_turns, session_id),
        )

    def add_turn(self, session_id: str, turn: dict[str, Any], turn_id: str | None = None) -> str:
        turn_id = turn_id or uuid.uuid4().hex[:16]
        ts = now_iso()
        self._exec(
            """INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                turn_id,
                session_id,
                turn["question"],
                turn["standalone_query"],
                turn["topic"],
                turn["answer"],
                turn["status"],
                turn.get("refusal_reason"),
                json.dumps(turn["citations"], ensure_ascii=False),
                json.dumps(turn["used_citations"]),
                int(turn["cache_hit"]),
                json.dumps(turn["timings"]),
                ts,
            ),
        )
        self._exec("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (ts, session_id))
        return turn_id

    def turns(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._exec("SELECT * FROM turns WHERE session_id = ? ORDER BY created_at, rowid", (session_id,))
        return [
            {
                **dict(r),
                "citations": json.loads(r["citations"]),
                "used_citations": json.loads(r["used_citations"]),
                "timings": json.loads(r["timings"]),
                "cache_hit": bool(r["cache_hit"]),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
