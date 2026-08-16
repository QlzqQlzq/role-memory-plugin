"""角色记忆与关系状态的 SQLite 存储。"""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import MemoryRecord, RelationshipState


SCHEMA_VERSION = 1
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_QUERY_STOPWORDS = {"什么", "怎么", "为什么", "其实", "这个", "那个", "告诉", "不用", "愿意", "时候"}
_QUERY_STOP_PHRASES = ("没关系", "不用告诉我", "等你愿意", "我理解你")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _search_terms(query: str) -> list[tuple[str, float]]:
    """Build weighted terms that also work for natural Chinese sentences."""

    weighted: dict[str, float] = {}
    normalized_query = query.casefold()
    for phrase in _QUERY_STOP_PHRASES:
        normalized_query = normalized_query.replace(phrase, " ")
    for token in _TOKEN_RE.findall(normalized_query):
        if len(token) < 2:
            continue
        if token not in _QUERY_STOPWORDS:
            weighted[token] = max(weighted.get(token, 0.0), 1.0)
        if _CJK_RE.fullmatch(token) and len(token) > 2:
            for size, weight in ((2, 0.42), (3, 0.58)):
                for index in range(len(token) - size + 1):
                    gram = token[index : index + size]
                    if gram in _QUERY_STOPWORDS:
                        continue
                    weighted[gram] = max(weighted.get(gram, 0.0), weight)
    return sorted(weighted.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:80]


class MemoryStore:
    """小型单角色知识库；所有运行态数据都保存在 Host 分配的数据目录。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        connection = sqlite3.connect(self.path, check_same_thread=False)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._connection = connection
            self._initialize()
        except Exception:
            connection.close()
            raise

    @classmethod
    def open_resilient(cls, path: Path) -> tuple["MemoryStore", Path | None]:
        """Open a store, preserving an unreadable database before rebuilding it."""

        path = Path(path)
        try:
            return cls(path), None
        except (OSError, RuntimeError, sqlite3.DatabaseError):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = path.with_name(f"{path.name}.corrupt-{timestamp}")
            for suffix in ("", "-wal", "-shm"):
                source = Path(f"{path}{suffix}")
                target = Path(f"{backup}{suffix}")
                if source.exists():
                    source.replace(target)
            return cls(path), backup

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    timeline TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                CREATE INDEX IF NOT EXISTS idx_memories_timeline ON memories(timeline);
                CREATE TABLE IF NOT EXISTS memory_fts (
                    memory_id TEXT PRIMARY KEY,
                    title,
                    content,
                    tags
                );
                CREATE TABLE IF NOT EXISTS relationship_states (
                    subject_key TEXT PRIMARY KEY,
                    familiarity REAL NOT NULL DEFAULT 0,
                    trust REAL NOT NULL DEFAULT 0,
                    closeness REAL NOT NULL DEFAULT 0,
                    respect REAL NOT NULL DEFAULT 50,
                    wariness REAL NOT NULL DEFAULT 50,
                    tension REAL NOT NULL DEFAULT 0,
                    openness REAL NOT NULL DEFAULT 20,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    deltas_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_state_events_subject ON state_events(subject_key, created_at);
                """
            )
            current_version = self._connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if current_version is not None:
                try:
                    parsed_version = int(current_version[0])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"invalid role memory schema: {current_version[0]}") from exc
                if parsed_version != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"unsupported role memory schema: {parsed_version} (expected {SCHEMA_VERSION})"
                    )
            self._connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @property
    def memory_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def import_dossier(self, dossier: dict[str, Any], persona_text: str = "", source_file: str = "") -> int:
        rows = self._flatten_dossier(dossier)
        if persona_text.strip():
            rows.insert(
                0,
                {
                    "memory_id": "persona_baseline",
                    "kind": "persona",
                    "title": "MaiBot 基础人设补充",
                    "content": persona_text.strip(),
                    "tags": ["核心人设", "固定底座"],
                    "source_refs": ["persona_file"],
                    "timeline": "post-ending",
                    "importance": 1.0,
                },
            )
        unique_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            content_key = " ".join(str(row["content"]).split()).casefold()
            unique_rows.setdefault((str(row["kind"]), content_key), row)
        rows = list(unique_rows.values())
        now = _now()
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM memories")
            self._connection.execute("DELETE FROM memory_fts")
            for row in rows:
                tags = _as_list(row.get("tags"))
                refs = _as_list(row.get("source_refs"))
                self._connection.execute(
                    """
                    INSERT INTO memories(memory_id, kind, title, content, tags_json, source_refs_json, timeline, importance, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["memory_id"],
                        row["kind"],
                        row["title"],
                        row["content"],
                        json.dumps(tags, ensure_ascii=False),
                        json.dumps(refs, ensure_ascii=False),
                        str(row.get("timeline") or "post-ending"),
                        float(row.get("importance") or 0.5),
                        now,
                    ),
                )
                self._connection.execute(
                    "INSERT INTO memory_fts(memory_id, title, content, tags) VALUES(?, ?, ?, ?)",
                    (row["memory_id"], row["title"], row["content"], " ".join(tags)),
                )
            self._connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('source_file', ?)", (source_file,))
            self._connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('imported_at', ?)", (now,))
            self._connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('character', ?)",
                (str(dossier.get("identity", {}).get("summary", "")).split("，", 1)[0],),
            )
        return len(rows)

    def _flatten_dossier(self, dossier: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        identity = dossier.get("identity") if isinstance(dossier.get("identity"), dict) else {}
        summary = str(identity.get("summary") or "").strip()
        if summary:
            rows.append(self._row("identity", "身份概述", summary, ["身份", "背景"], [] , 0.98))
        for index, item in enumerate(identity.get("important_background", []) or []):
            rows.append(self._row("memory", f"重要背景 {index + 1}", str(item), ["背景", "经历"], [], 0.92))
        aliases = _as_list(identity.get("aliases"))
        if aliases:
            rows.append(self._row("alias", "角色别名", "、".join(aliases), ["别名", *aliases], [], 0.9))

        sections: Iterable[tuple[str, str, str, str, float]] = (
            ("core_traits", "trait", "核心性格", "trait", 0.9),
            ("values_and_motivations", "value", "价值与动机", "value", 0.9),
            ("decision_patterns", "behavior", "决策模式", "situation", 0.9),
            ("relationships", "relationship", "重要关系", "person", 0.88),
            ("emotional_dynamics", "emotion", "情绪机制", "trigger", 0.86),
            ("development", "development", "成长与变化", "", 0.84),
        )
        for key, kind, label, title_key, importance in sections:
            values = dossier.get(key) or []
            if isinstance(values, dict):
                values = [values]
            elif not isinstance(values, (list, tuple)):
                values = [values]
            for index, item in enumerate(values):
                if isinstance(item, dict):
                    title = str(item.get(title_key) or item.get("trait") or item.get("value") or item.get("trigger") or f"{label} {index + 1}")
                    bits = [f"{k}：{v}" for k, v in item.items() if k not in {"evidence_refs", "source_ref"} and v]
                    content = "；".join(bits)
                    refs = _as_list(item.get("evidence_refs") or item.get("source_ref"))
                    tags = [label, title]
                    timeline = "post-ending"
                    for marker in ("Act01", "Act02", "真结局", "终局"):
                        if marker in content:
                            timeline = marker
                            tags.append(marker)
                    rows.append(self._row(kind, title, content, tags, refs, importance, timeline))
                elif str(item).strip():
                    rows.append(self._row(kind, f"{label} {index + 1}", str(item), [label], [], importance))

        speech = dossier.get("speech_style") if isinstance(dossier.get("speech_style"), dict) else {}
        for key, value in speech.items():
            values = _as_list(value)
            for index, item in enumerate(values):
                rows.append(self._row("style", f"表达方式：{key} {index + 1}", item, ["表达方式", key], [], 0.86))
        for index, item in enumerate(dossier.get("representative_quotes", []) or []):
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            if not quote:
                continue
            content = "；".join(f"{k}：{v}" for k, v in item.items() if v and k not in {"source_ref"})
            rows.append(self._row("quote", f"代表台词 {index + 1}", content, ["台词", "表达样本"], _as_list(item.get("source_ref")), 0.8))
        for index, item in enumerate(dossier.get("contradictions_and_limits", []) or []):
            if str(item).strip():
                rows.append(self._row("constraint", f"矛盾与边界 {index + 1}", str(item), ["边界", "矛盾"], [], 0.9))
        for index, item in enumerate(dossier.get("uncertainties", []) or []):
            if str(item).strip():
                rows.append(self._row("uncertainty", f"待确认事项 {index + 1}", str(item), ["不确定"], [], 0.4))
        return rows

    @staticmethod
    def _row(kind: str, title: str, content: str, tags: list[str], refs: list[str], importance: float, timeline: str = "post-ending") -> dict[str, Any]:
        digest = hashlib.sha256(f"{kind}\x00{title}\x00{content}".encode("utf-8")).hexdigest()[:16]
        memory_id = f"{kind}_{digest}"
        return {
            "memory_id": memory_id,
            "kind": kind,
            "title": title,
            "content": content,
            "tags": tags,
            "source_refs": refs,
            "timeline": timeline,
            "importance": importance,
        }

    def search(self, query: str, limit: int = 8, kinds: set[str] | None = None) -> list[MemoryRecord]:
        query = " ".join(str(query or "").split()).strip()
        if not query:
            return []
        terms = _search_terms(query)
        with self._lock:
            rows = self._connection.execute("SELECT * FROM memories").fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if kinds and str(row["kind"]) not in kinds:
                continue
            try:
                tags = json.loads(row["tags_json"] or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                tags = []
            if not isinstance(tags, list):
                tags = []
            normalized_tags = [str(tag) for tag in tags if str(tag).strip()]
            title_tags = " ".join([str(row["title"]), *normalized_tags]).casefold()
            content = str(row["content"])
            content_folded = content.casefold()
            haystack = f"{title_tags} {content_folded}"
            score = float(row["importance"] or 0.5) * 0.2
            if query.casefold() in haystack:
                score += 2.5
            term_score = sum(
                weight * (1.35 if term in title_tags else 0.48)
                for term, weight in terms
                if term in haystack
            )
            score += min(term_score, 4.0)
            if str(row["kind"]) == "persona":
                score -= 1.2
            if len(content) > 1200:
                score -= min(1.0, (len(content) - 1200) / 4000)
            if str(row["kind"]) == "style" and any(word in query for word in ("怎么说", "语气", "表达", "回复", "说话")):
                score += 0.7
            if score >= 0.65:
                scored.append((score, row))
        scored.sort(key=lambda item: (item[0], float(item[1]["importance"] or 0.5)), reverse=True)
        return [MemoryRecord.from_row(row) for _, row in scored[: max(1, min(limit, 32))]]

    def get_state(self, subject_key: str) -> RelationshipState:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM relationship_states WHERE subject_key = ?", (subject_key,)
            ).fetchone()
            if row is None:
                state = RelationshipState(subject_key=subject_key, updated_at=_now())
                self._connection.execute(
                    """
                    INSERT INTO relationship_states(subject_key, familiarity, trust, closeness, respect, wariness, tension, openness, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (subject_key, state.familiarity, state.trust, state.closeness, state.respect, state.wariness, state.tension, state.openness, state.updated_at),
                )
                self._connection.commit()
                return state
            return RelationshipState(**dict(row))

    def update_state(
        self,
        subject_key: str,
        deltas: dict[str, float],
        event_type: str,
        reason: str,
        confidence: float,
        *,
        record_event: bool = True,
        event_limit: int = 200,
    ) -> RelationshipState:
        with self._lock, self._connection:
            current = self.get_state(subject_key)
            values = {field: getattr(current, field) for field in ("familiarity", "trust", "closeness", "respect", "wariness", "tension", "openness")}
            for field, delta in deltas.items():
                if field in values:
                    values[field] = max(-100.0, min(100.0, float(delta))) + values[field]
            state = RelationshipState(subject_key=subject_key, updated_at=_now(), **values).bounded()
            self._connection.execute(
                """
                INSERT INTO relationship_states(subject_key, familiarity, trust, closeness, respect, wariness, tension, openness, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_key) DO UPDATE SET familiarity=excluded.familiarity, trust=excluded.trust,
                  closeness=excluded.closeness, respect=excluded.respect, wariness=excluded.wariness,
                  tension=excluded.tension, openness=excluded.openness, updated_at=excluded.updated_at
                """,
                (state.subject_key, state.familiarity, state.trust, state.closeness, state.respect, state.wariness, state.tension, state.openness, state.updated_at),
            )
            if record_event:
                self._connection.execute(
                    "INSERT INTO state_events(subject_key, event_type, reason, deltas_json, confidence, created_at) VALUES(?, ?, ?, ?, ?, ?)",
                    (subject_key, event_type, reason, json.dumps(deltas, ensure_ascii=False), max(0.0, min(1.0, confidence)), state.updated_at),
                )
                keep = max(20, min(int(event_limit), 2000))
                self._connection.execute(
                    """
                    DELETE FROM state_events
                    WHERE subject_key = ? AND event_id NOT IN (
                        SELECT event_id FROM state_events
                        WHERE subject_key = ? ORDER BY event_id DESC LIMIT ?
                    )
                    """,
                    (subject_key, subject_key, keep),
                )
            return state

    def has_recent_event(self, subject_key: str, event_type: str, within_seconds: int) -> bool:
        if within_seconds <= 0:
            return False
        with self._lock:
            row = self._connection.execute(
                """
                SELECT created_at FROM state_events
                WHERE subject_key = ? AND event_type = ?
                ORDER BY event_id DESC LIMIT 1
                """,
                (subject_key, event_type),
            ).fetchone()
        if row is None:
            return False
        try:
            created_at = datetime.fromisoformat(str(row["created_at"]))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
        return 0 <= age_seconds <= within_seconds

    def decay_state(self, subject_key: str, tension_factor: float = 0.92) -> RelationshipState:
        with self._lock, self._connection:
            current = self.get_state(subject_key)
            state = RelationshipState(
                subject_key=subject_key,
                familiarity=current.familiarity,
                trust=current.trust,
                closeness=current.closeness,
                respect=current.respect,
                wariness=current.wariness,
                tension=current.tension * tension_factor,
                openness=current.openness,
                updated_at=_now(),
            ).bounded()
            self._connection.execute(
                "UPDATE relationship_states SET tension=?, updated_at=? WHERE subject_key=?",
                (state.tension, state.updated_at, subject_key),
            )
            return state
