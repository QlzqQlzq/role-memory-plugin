"""数据模型：原作记忆与关系状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _json_string_tuple(value: Any) -> tuple[str, ...]:
    import json

    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded if str(item).strip())


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    kind: str
    title: str
    content: str
    tags: tuple[str, ...]
    source_refs: tuple[str, ...]
    timeline: str
    importance: float

    @classmethod
    def from_row(cls, row: Any) -> "MemoryRecord":
        return cls(
            memory_id=str(row["memory_id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            content=str(row["content"]),
            tags=_json_string_tuple(row["tags_json"]),
            source_refs=_json_string_tuple(row["source_refs_json"]),
            timeline=str(row["timeline"] or ""),
            importance=float(row["importance"] or 0.5),
        )


@dataclass(frozen=True)
class RelationshipState:
    DIMENSIONS = ("familiarity", "trust", "closeness", "respect", "wariness", "tension", "openness")
    subject_key: str
    familiarity: float = 0.0
    trust: float = 0.0
    closeness: float = 0.0
    respect: float = 50.0
    wariness: float = 50.0
    tension: float = 0.0
    openness: float = 20.0
    updated_at: str = ""

    def bounded(self) -> "RelationshipState":
        values = {
            "familiarity": self.familiarity,
            "trust": self.trust,
            "closeness": self.closeness,
            "respect": self.respect,
            "wariness": self.wariness,
            "tension": self.tension,
            "openness": self.openness,
        }
        values = {key: max(0.0, min(100.0, float(value))) for key, value in values.items()}
        return RelationshipState(subject_key=self.subject_key, updated_at=self.updated_at, **values)

    @staticmethod
    def band(value: float) -> str:
        if value < 25:
            return "低"
        if value < 60:
            return "中等"
        if value < 80:
            return "较高"
        return "很高"

    def to_prompt_lines(self, include_scores: bool = True, weights: dict[str, float] | None = None) -> list[str]:
        weights = weights or {}
        def label(name: str, value: float) -> str:
            effective = max(0.0, min(100.0, value * float(weights.get(name, 1.0))))
            text = self.band(effective)
            return f"{effective:.0f} / {text}" if include_scores else text

        return [
            f"熟悉程度：{label('familiarity', self.familiarity)}",
            f"信任程度：{label('trust', self.trust)}；仍需根据实际行动判断对方动机",
            f"亲近程度：{label('closeness', self.closeness)}",
            f"认可程度：{label('respect', self.respect)}",
            f"戒备程度：{label('wariness', self.wariness)}",
            f"当前紧张度：{label('tension', self.tension)}",
            f"主动袒露内心的意愿：{label('openness', self.openness)}",
        ]
