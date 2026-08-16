"""MaiBot 基于知识库的动态人设插件。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

from .builder import BuildProgress, BuilderOptions, RoleKnowledgeBuilder, SUPPORTED_SUFFIXES
from .models import MemoryRecord, RelationshipState
from .store import MemoryStore


def _clean(text: str, limit: int = 5000) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split())[:limit]


def _clean_prompt(text: str, limit: int) -> str:
    lines = [" ".join(line.replace("\x00", " ").split()) for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()[:limit]


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    for key in ("processed_plain_text", "plain_text", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)
    raw = message.get("raw_message")
    if isinstance(raw, list):
        return _clean(" ".join(str(item.get("data") or item.get("content") or "") for item in raw if isinstance(item, dict)))
    return ""


@dataclass
class RuntimeContext:
    query: str
    prompt: str
    records: list[MemoryRecord]
    state: RelationshipState
    subject_key: str


class PluginSectionConfig(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_icon__ = "brain"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用基于知识库的动态人设插件")
    config_version: str = Field(default="0.3.0", description="插件配置版本")
    character_name: str = Field(default="", description="当前角色名称；初始化角色库前必须填写")
    pack_file: str = Field(default="character_dossier.json", description="角色证据档案文件名")
    persona_file: str = Field(default="character_persona.txt", description="角色补充人设文件名；仅作检索资料，不覆盖 MaiBot 人设")
    character_aliases: list[str] = Field(default_factory=list, description="当前角色的别名、译名或 speaker ID")
    max_prompt_characters: int = Field(default=2600, ge=500, le=8000, description="兼容旧配置；请改用 injection.max_prompt_characters")
    search_limit: int = Field(default=6, ge=1, le=20, description="兼容旧配置；请改用 injection.search_limit")
    query_command_enabled: bool = Field(default=True, description="是否启用 /角色记忆 查询指令")
    state_command_enabled: bool = Field(default=True, description="是否启用 /人设状态 查询指令")
    session_cache_limit: int = Field(default=512, ge=32, le=4096, description="最多保留的活跃会话上下文数")


class InjectionConfig(PluginConfigBase):
    __ui_label__ = "动态提示"
    __ui_icon__ = "wand-sparkles"
    __ui_order__ = 1

    enabled: bool = Field(default=True, description="是否启用原作记忆与关系状态的动态提示词注入")
    max_prompt_characters: int = Field(default=2600, ge=500, le=8000, description="单轮动态提示最大字符数")
    search_limit: int = Field(default=6, ge=1, le=20, description="每轮最多召回的角色记忆条数")
    include_official_memory: bool = Field(default=True, description="是否把 MaiBot 官方人物记忆摘要纳入动态提示")
    state_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "familiarity": 1.0,
            "trust": 1.0,
            "closeness": 0.8,
            "respect": 0.7,
            "wariness": 1.1,
            "tension": 1.4,
            "openness": 0.9,
        },
        description="各关系状态对本轮提示词的影响权重",
    )


class StateConfig(PluginConfigBase):
    __ui_label__ = "关系状态"
    __ui_icon__ = "heart-pulse"
    __ui_order__ = 1

    enabled: bool = Field(default=True, description="是否维护对聊天对象的信任、亲近与短期情绪状态")
    update_from_messages: bool = Field(default=True, description="是否根据用户消息中的互动事件更新状态")
    decision_mode: str = Field(default="planner_tool", description="状态判断方式：planner_tool 或 off")
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="低于此置信度的状态事件不生效")
    max_delta_per_turn: float = Field(default=8.0, ge=0.0, le=30.0, description="单轮单项状态最大变化量")
    baseline_familiarity_per_message: float = Field(default=0.2, ge=0.0, le=5.0, description="普通聊天带来的基础熟悉度增量")
    update_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "familiarity": 1.0,
            "trust": 1.0,
            "closeness": 0.8,
            "respect": 1.0,
            "wariness": 1.0,
            "tension": 1.2,
            "openness": 0.8,
        },
        description="关系事件对各状态数值变化的权重",
    )
    decay_tension: bool = Field(default=True, description="每轮回复前让短期紧张度自然回落")
    event_cooldown_seconds: int = Field(default=300, ge=0, le=86400, description="同类关系事件重复生效的冷却秒数")
    event_history_limit: int = Field(default=200, ge=20, le=2000, description="每个聊天对象最多保留的关系事件数")


class ReviewConfig(PluginConfigBase):
    __ui_label__ = "回复复审"
    __ui_icon__ = "shield-check"
    __ui_order__ = 2

    enabled: bool = Field(default=True, description="是否启用回复复审")
    mode: str = Field(default="rules", description="复审模式：off、rules、smart、always")
    max_retries: int = Field(default=1, ge=0, le=2, description="单轮最多要求重生成次数")
    review_model: str = Field(default="", description="兼容旧配置；smart/always 模式使用的 MaiBot 模型名")
    model: str = Field(default="", description="smart/always 模式使用的 MaiBot 模型名")


class BuilderConfig(PluginConfigBase):
    __ui_label__ = "角色库初始化"
    __ui_icon__ = "folder-cog"
    __ui_order__ = 3

    enabled: bool = Field(default=True, description="是否允许通过私聊指令初始化角色知识库")
    admin_enabled: bool = Field(default=True, description="是否启用管理员 QQ 的角色库管理入口")
    admin_qq_ids: str = Field(default="", description="允许执行初始化的 QQ 号，多个用逗号分隔；为空时禁用指令")
    import_dir: str = Field(default="imports", description="插件数据目录下的原始资料文件夹")
    build_dir: str = Field(default="builds", description="构建产物与证据文件夹")
    cache_dir: str = Field(default="build_cache", description="分块与归并模型缓存文件夹")
    work_name: str = Field(default="", description="当前角色所属作品名，可留空")
    model: str = Field(default="", description="构建使用的 MaiBot 模型任务名；为空时使用默认模型")
    chunk_size: int = Field(default=12000, ge=2000, le=50000, description="单个剧本分块的目标字符数")
    chunk_overlap: int = Field(default=800, ge=0, le=5000, description="相邻分块的重叠字符数")
    merge_batch_size: int = Field(default=6, ge=2, le=10, description="树状归并每批证据数量")
    concurrency: int = Field(default=1, ge=1, le=4, description="分块分析并发数")
    max_source_files: int = Field(default=500, ge=1, le=5000, description="一次最多读取的资料文件数")
    max_source_characters: int = Field(default=5000000, ge=10000, le=50000000, description="一次最多读取的清洗后字符数")
    max_model_calls: int = Field(default=500, ge=5, le=5000, description="一次初始化允许的最大模型调用数")


class RoleMemoryConfig(PluginConfigBase):
    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    builder: BuilderConfig = Field(default_factory=BuilderConfig)


class RoleMemoryPlugin(MaiBotPlugin):
    config_model = RoleMemoryConfig

    def __init__(self) -> None:
        super().__init__()
        self._store: MemoryStore | None = None
        self._session_subjects: OrderedDict[str, str] = OrderedDict()
        self._session_user_ids: OrderedDict[str, str] = OrderedDict()
        self._session_official_ids: OrderedDict[str, str] = OrderedDict()
        self._session_queries: OrderedDict[str, str] = OrderedDict()
        self._session_contexts: OrderedDict[str, RuntimeContext] = OrderedDict()
        self._session_private: OrderedDict[str, bool] = OrderedDict()
        self._build_task: asyncio.Task[None] | None = None
        self._build_progress = BuildProgress()

    async def on_load(self) -> None:
        data_dir = self.ctx.paths.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self._store, recovered_path = MemoryStore.open_resilient(data_dir / "role_memory.sqlite3")
        if recovered_path is not None:
            self.ctx.logger.warning("检测到损坏或不兼容的角色记忆数据库，已备份并重建：%s", recovered_path)
        self._load_pack_if_needed()
        self._ensure_import_dir()
        self.ctx.logger.info("角色记忆插件已加载：角色=%s，记忆=%s，状态=%s，复审=%s", self.config.plugin.character_name, self._store.memory_count, self.config.state.enabled, self.config.review.mode)

    async def on_unload(self) -> None:
        if self._build_task is not None and not self._build_task.done():
            self._build_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._build_task
        self._build_task = None
        self._session_subjects.clear()
        self._session_user_ids.clear()
        self._session_official_ids.clear()
        self._session_queries.clear()
        self._session_contexts.clear()
        self._session_private.clear()
        if self._store is not None:
            self._store.close()
            self._store = None

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del scope, config_data, version
        self._session_queries.clear()
        self._session_contexts.clear()
        self._session_private.clear()
        self._session_user_ids.clear()
        self._session_official_ids.clear()
        if self._store is not None:
            self._load_pack_if_needed(force=True)
        self._ensure_import_dir()

    def _ensure_import_dir(self) -> Path:
        data_dir = self.ctx.paths.data_dir
        import_dir = data_dir / Path(self.config.builder.import_dir).name
        import_dir.mkdir(parents=True, exist_ok=True)
        guide = import_dir / "_请把角色资料放在这里.txt"
        if not guide.exists():
            guide.write_text(
                "把需要分析的 TXT、Markdown、JSON、JSONL、CSV、YAML 或游戏脚本文件放在此目录。\n"
                "文件可以继续分子目录存放。准备完成后，用管理员 QQ 私聊发送：/角色库初始化\n",
                encoding="utf-8",
            )
        return import_dir

    def _load_pack_if_needed(self, force: bool = False) -> None:
        if self._store is None:
            return
        data_dir = self.ctx.paths.data_dir
        pack_path = data_dir / Path(self.config.plugin.pack_file).name
        persona_path = data_dir / Path(self.config.plugin.persona_file).name
        if force or self._store.memory_count == 0:
            if not pack_path.is_file():
                self.ctx.logger.info("角色知识库尚未初始化：等待管理员导入资料")
                return
            try:
                dossier = json.loads(pack_path.read_text(encoding="utf-8"))
                persona = persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""
                imported = self._store.import_dossier(dossier, persona, str(pack_path))
                self.ctx.logger.info("角色记忆已导入：%s 条，来源=%s", imported, pack_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.ctx.logger.error("角色记忆包加载失败：%s", exc)

    @HookHandler(
        "chat.receive.before_process",
        name="observe_role_memory_event",
        description="观察用户消息并更新角色对聊天对象的关系状态。",
        mode=HookMode.OBSERVE,
        order=HookOrder.EARLY,
        timeout_ms=1500,
        error_policy=ErrorPolicy.SKIP,
    )
    async def observe_role_memory_event(self, message: Any = None, **kwargs: Any) -> None:
        del kwargs
        if not self.config.plugin.enabled or self._store is None or not isinstance(message, dict):
            return None
        text = _message_text(message)
        if not text:
            return None
        session_id = str(message.get("session_id") or "")
        subject_key = await self._resolve_subject_key(message)
        if session_id:
            self._session_subjects[session_id] = subject_key
            user_id = self._message_user_id(message)
            if user_id:
                self._session_user_ids[session_id] = user_id
            self._session_private[session_id] = self._message_is_private(message)
            if not text.startswith("/"):
                self._session_queries[session_id] = text
            self._touch_session(session_id)
        if text.startswith("/"):
            return None
        if not self.config.state.enabled or not self.config.state.update_from_messages:
            return None
        self._store.update_state(
            subject_key,
            {"familiarity": self.config.state.baseline_familiarity_per_message},
            "interaction",
            "普通聊天基础熟悉度",
            1.0,
            record_event=False,
            event_limit=self.config.state.event_history_limit,
        )
        self.ctx.logger.debug(
            "角色关系基础状态已更新：subject=%s familiarity=%.1f",
            subject_key,
            self._store.get_state(subject_key).familiarity,
        )
        return None

    @HookHandler(
        "maisaka.replyer.before_request",
        name="inject_role_memory_prompt",
        description="根据当前场景召回原作记忆与关系状态，注入本轮 replyer 提示。",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=2500,
        error_policy=ErrorPolicy.SKIP,
    )
    async def inject_role_memory_prompt(self, session_id: str = "", extra_prompt: str = "", reply_reason: str = "", reply_tool_args: Any = None, retry_count: int = 0, **kwargs: Any) -> dict[str, Any] | None:
        del kwargs
        if not self.config.plugin.enabled or not self.config.injection.enabled or self._store is None:
            return None
        cached = self._session_contexts.get(session_id)
        if retry_count > 0 and cached is not None:
            self._touch_session(session_id)
            combined = "\n\n".join(item for item in (str(extra_prompt or "").strip(), cached.prompt) if item)
            return {"action": "continue", "modified_kwargs": {"extra_prompt": combined}}
        query = self._build_query(session_id, reply_reason, reply_tool_args)
        if not query:
            return None
        subject_key = self._session_subjects.get(session_id, f"session:{session_id or 'unknown'}")
        state = self._store.get_state(subject_key)
        if self.config.state.decay_tension:
            state = self._store.decay_state(subject_key)
        records = self._store.search(query, self.config.injection.search_limit)
        official_memory = await self._get_official_memory(subject_key)
        prompt = self._compose_prompt(query, records, state, official_memory)
        if not prompt:
            return None
        self._session_contexts[session_id] = RuntimeContext(
            query=query,
            prompt=prompt,
            records=records,
            state=state,
            subject_key=subject_key,
        )
        self._touch_session(session_id)
        self.ctx.logger.debug(
            "角色动态提示已注入：session=%s subject=%s memories=%s prompt_chars=%s",
            session_id,
            subject_key,
            [record.memory_id for record in records],
            len(prompt),
        )
        combined = "\n\n".join(item for item in (str(extra_prompt or "").strip(), prompt) if item)
        return {"action": "continue", "modified_kwargs": {"extra_prompt": combined}}

    @HookHandler(
        "maisaka.replyer.after_response",
        name="review_role_memory_response",
        description="按开关检查回复是否出现明显 OOC，必要时要求 replyer 重生成。",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=5000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def review_role_memory_response(self, response: str = "", session_id: str = "", retry_count: int = 0, max_retries: int = 0, **kwargs: Any) -> dict[str, Any] | None:
        del kwargs
        mode = str(self.config.review.mode or "off").strip().lower()
        if mode not in {"off", "rules", "smart", "always"}:
            mode = "rules"
        if not self.config.review.enabled or mode == "off" or retry_count >= min(max_retries, self.config.review.max_retries):
            return None
        context = self._session_contexts.get(session_id)
        if context is None or not response.strip():
            return None
        reason = self._rule_review(response)
        if not reason and (mode == "always" or (mode == "smart" and self._needs_smart_review(response, context))):
            reason = await self._smart_review(response, context)
        if not reason:
            return None
        self.ctx.logger.info(
            "角色回复复审要求重生成：session=%s attempt=%s reason=%s",
            session_id,
            retry_count + 1,
            reason,
        )
        return {
            "action": "continue",
            "modified_kwargs": {
                "retry": True,
                "retry_reason": reason,
                "matched_regex": "role_memory_review",
            },
        }

    @HookHandler(
        "maisaka.planner.before_request",
        name="guide_role_memory_state_tool",
        description="提醒决策模型在必要时通过角色状态工具提交语义关系变化。",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=1000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def guide_role_memory_state_tool(self, messages: Any = None, **kwargs: Any) -> dict[str, Any] | None:
        del kwargs
        if (
            not self.config.plugin.enabled
            or not self.config.state.enabled
            or not self.config.state.update_from_messages
            or self.config.state.decision_mode.casefold() == "off"
            or not isinstance(messages, list)
        ):
            return None
        guided = list(messages)
        guided.append(
            {
                "role": "system",
                "content": (
                    "角色关系状态由 role_memory_update_state 工具维护。"
                    "如果用户本轮确实改变了信任、亲近、尊重、戒备、紧张或袒露意愿，"
                    "先调用该工具提交事件证据、置信度和强度；普通闲聊、单纯提问或无法确认时不要调用。"
                ),
            }
        )
        return {"action": "continue", "modified_kwargs": {"messages": guided}}

    @Tool(
        "role_memory_update_state",
        description=(
            "在决定回复前记录本轮对角色关系和短期情绪的语义变化。"
            "只在用户消息确实改变了关系或情绪时调用；普通闲聊不要伪造事件。"
        ),
        parameters=[
            ToolParameterInfo(
                name="event_type",
                param_type=ToolParamType.STRING,
                description="事件类型，例如 boundary_respect、sincere_support、apology、insult_or_attack、self_sacrifice_trigger、emotional_reassurance、meaningful_revelation",
                required=True,
            ),
            ToolParameterInfo(
                name="evidence",
                param_type=ToolParamType.STRING,
                description="用户消息中支持该事件的简短证据",
                required=True,
            ),
            ToolParameterInfo(
                name="confidence",
                param_type=ToolParamType.FLOAT,
                description="对事件判断的置信度，范围 0 到 1",
                required=True,
            ),
            ToolParameterInfo(
                name="intensity",
                param_type=ToolParamType.FLOAT,
                description="事件强度，范围 0 到 1；普通事件使用较低值",
                required=False,
                default=0.5,
            ),
            ToolParameterInfo(
                name="emotion",
                param_type=ToolParamType.STRING,
                description="本轮短期情绪变化的简短描述",
                required=False,
                default="",
            ),
        ],
    )
    async def role_memory_update_state(
        self,
        event_type: str = "",
        evidence: str = "",
        confidence: float = 0.0,
        intensity: float = 0.5,
        emotion: str = "",
        stream_id: str = "",
        user_id: str = "",
        platform: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if (
            not self.config.plugin.enabled
            or not self.config.state.enabled
            or not self.config.state.update_from_messages
            or self.config.state.decision_mode.casefold() == "off"
            or self._store is None
        ):
            return {"success": True, "content": "角色关系状态更新未启用，本轮不更新。"}

        subject_key = await self._resolve_subject_key_from_context(stream_id, user_id, platform)
        normalized_type = str(event_type or "").strip().casefold()
        confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        intensity = max(0.0, min(1.0, float(intensity if intensity is not None else 0.5)))
        if confidence < self.config.state.confidence_threshold:
            return {"success": True, "content": "事件置信度不足，未更新关系状态。", "updated": False}

        base_deltas = {
            "boundary_respect": {"trust": 2.0, "respect": 3.0, "wariness": -2.0, "openness": 1.0},
            "sincere_support": {"trust": 2.0, "closeness": 2.0, "tension": -2.0},
            "apology": {"trust": 2.0, "tension": -2.0, "wariness": -1.0},
            "insult_or_attack": {"trust": -3.0, "respect": -2.0, "wariness": 5.0, "tension": 5.0},
            "self_sacrifice_trigger": {"tension": 7.0, "wariness": 2.0, "openness": -1.0},
            "emotional_reassurance": {"trust": 1.5, "closeness": 1.5, "tension": -1.5, "openness": 0.5},
            "meaningful_revelation": {"trust": 1.0, "closeness": 1.0, "openness": 1.5},
        }.get(normalized_type)
        if base_deltas is None:
            return {"success": True, "content": "未知关系事件，未更新关系状态。", "updated": False}
        if self._store.has_recent_event(subject_key, normalized_type, self.config.state.event_cooldown_seconds):
            return {"success": True, "content": "同类关系事件仍在冷却中，未重复更新。", "updated": False}

        deltas: dict[str, float] = {}
        for name, base in base_deltas.items():
            weight = max(0.0, float(self.config.state.update_weights.get(name, 1.0)))
            delta = base * intensity * weight
            deltas[name] = max(-self.config.state.max_delta_per_turn, min(self.config.state.max_delta_per_turn, delta))
        state = self._store.update_state(
            subject_key,
            deltas,
            normalized_type,
            _clean(f"{evidence}；情绪：{emotion}", 300),
            confidence,
            record_event=True,
            event_limit=self.config.state.event_history_limit,
        )
        return {
            "success": True,
            "content": "关系状态已记录；请让回复体现渐进、克制的变化。",
            "updated": True,
            "event_type": normalized_type,
            "deltas": deltas,
            "state": {name: getattr(state, name) for name in RelationshipState.DIMENSIONS},
        }

    @Command("role_memory_query", description="查询角色原作记忆", pattern=r"^/(?:角色记忆|人设记忆)\s+(?P<query>.+)$")
    async def role_memory_query(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        if not self.config.plugin.query_command_enabled:
            await self.ctx.send.text("角色记忆查询指令未启用。", stream_id)
            return True, "记忆查询未启用", True
        query = str((kwargs.get("matched_groups") or {}).get("query") or kwargs.get("text") or "").strip()
        if not query or self._store is None:
            return False, "用法：/角色记忆 <想查询的剧情、关系或表达场景>", True
        records = self._store.search(query, self.config.injection.search_limit)
        if not records:
            await self.ctx.send.text("没有召回相关角色记忆。", stream_id)
            return True, "无匹配记忆", True
        lines = [f"【{self.config.plugin.character_name} 记忆检索】"]
        lines.extend(f"- [{record.kind}] {record.title}：{record.content[:180]}" for record in records)
        await self.ctx.send.text("\n".join(lines), stream_id)
        return True, f"返回 {len(records)} 条记忆", True

    @Command("role_memory_state", description="查看当前聊天对象的角色关系状态", pattern=r"^/(?:角色状态|人设状态)$")
    async def role_memory_state(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        if not self.config.plugin.state_command_enabled:
            await self.ctx.send.text("人设状态查询指令未启用。", stream_id)
            return True, "状态查询未启用", True
        if self._store is None:
            return False, "角色记忆插件尚未加载", True
        subject_key = self._subject_from_command(stream_id, kwargs)
        state = self._store.get_state(subject_key)
        await self.ctx.send.text("【当前关系状态】\n" + "\n".join(state.to_prompt_lines(include_scores=True)), stream_id)
        return True, "已显示关系状态", True

    @Command("role_memory_initialize", description="从固定导入目录初始化当前角色知识库", pattern=r"^/角色库初始化$")
    async def role_memory_initialize(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        if not self.config.builder.enabled or not self.config.builder.admin_enabled:
            await self.ctx.send.text("角色库初始化指令未启用。", stream_id)
            return True, "初始化未启用", True
        allowed, reason = self._builder_command_allowed(stream_id, kwargs)
        if not allowed:
            await self.ctx.send.text(reason, stream_id)
            return True, reason, True
        if self._build_task is not None and not self._build_task.done():
            await self.ctx.send.text(self._build_progress.describe(), stream_id)
            return True, "返回初始化进度", True
        if self._store is None:
            await self.ctx.send.text("角色记忆数据库尚未加载。", stream_id)
            return True, "数据库未加载", True

        import_dir = self._ensure_import_dir()
        source_files = [
            path
            for path in import_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES and not path.name.startswith("_")
        ]
        if not source_files:
            await self.ctx.send.text(f"导入目录里还没有资料文件：\n{import_dir}", stream_id)
            return True, "导入目录为空", True
        character_name = self.config.plugin.character_name.strip()
        if not character_name:
            await self.ctx.send.text("请先在插件配置中填写当前角色名称。", stream_id)
            return True, "角色名称未配置", True

        self._build_progress = BuildProgress()
        options = BuilderOptions(
            import_dir=Path(self.config.builder.import_dir).name,
            build_dir=Path(self.config.builder.build_dir).name,
            cache_dir=Path(self.config.builder.cache_dir).name,
            work_name=self.config.builder.work_name,
            model=self.config.builder.model,
            chunk_size=self.config.builder.chunk_size,
            chunk_overlap=min(self.config.builder.chunk_overlap, self.config.builder.chunk_size - 1),
            character_aliases=list(self.config.plugin.character_aliases),
            merge_batch_size=self.config.builder.merge_batch_size,
            concurrency=self.config.builder.concurrency,
            max_source_files=self.config.builder.max_source_files,
            max_source_characters=self.config.builder.max_source_characters,
            max_model_calls=self.config.builder.max_model_calls,
        )
        builder = RoleKnowledgeBuilder(
            llm=self.ctx.llm,
            logger=self.ctx.logger,
            data_dir=self.ctx.paths.data_dir,
            store=self._store,
            character_name=character_name,
            options=options,
            progress=self._build_progress,
        )
        self._build_task = asyncio.create_task(
            self._run_builder(builder, stream_id),
            name="role-memory-initialize",
        )
        await self.ctx.send.text(
            f"已开始初始化「{character_name}」角色库。\n"
            f"读取目录：{import_dir}\n"
            "任务会在后台运行；处理中再次发送 /角色库初始化 可查看进度。",
            stream_id,
        )
        return True, "角色库初始化已开始", True

    async def _run_builder(self, builder: RoleKnowledgeBuilder, stream_id: str) -> None:
        try:
            result = await builder.build_and_install()
        except asyncio.CancelledError:
            self._build_progress.status = "cancelled"
            self._build_progress.stage = "任务已取消"
            self.ctx.logger.info("角色库初始化任务已取消")
            raise
        except Exception as exc:
            self._build_progress.status = "failed"
            self._build_progress.stage = "初始化失败"
            self._build_progress.error = f"{type(exc).__name__}: {exc}"
            self.ctx.logger.exception("角色库初始化失败")
            await self._send_builder_notice(
                stream_id,
                f"角色库初始化失败：{_clean(str(exc), 500)}\n再次发送 /角色库初始化 可重新开始。",
            )
        else:
            self.ctx.logger.info(
                "角色库初始化完成：files=%s chunks=%s relevant=%s memories=%s output=%s",
                result.source_files,
                result.chunks,
                result.relevant_chunks,
                result.imported_memories,
                result.output_dir,
            )
            await self._send_builder_notice(
                stream_id,
                f"角色库初始化完成：已从 {result.source_files} 个文件、{result.relevant_chunks}/{result.chunks} 个相关分块中导入 {result.imported_memories} 条记忆。",
            )

    async def _send_builder_notice(self, stream_id: str, text: str) -> None:
        try:
            await self.ctx.send.text(text, stream_id)
        except Exception as exc:
            self.ctx.logger.warning("角色库任务通知发送失败：%s", exc)

    def _build_query(self, session_id: str, reply_reason: str, reply_tool_args: Any) -> str:
        user_query = self._session_queries.get(session_id, "")
        supplements: list[str] = []
        if isinstance(reply_tool_args, dict):
            supplements.extend(
                _clean(str(reply_tool_args.get(key) or ""), 80)
                for key in ("scene_type", "target_relation", "emotional_intent")
            )
        if user_query:
            return _clean(" ".join([user_query, *supplements]), 700)
        return _clean(" ".join([reply_reason, *supplements]), 1200)

    @staticmethod
    def _subject_key(message: dict[str, Any]) -> str:
        info = message.get("message_info") if isinstance(message.get("message_info"), dict) else {}
        user = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
        user_id = str(user.get("user_id") or message.get("user_id") or "").strip()
        platform = str(message.get("platform") or "unknown").strip().casefold()
        if user_id:
            return f"user:{platform}:{user_id}"
        return f"session:{str(message.get('session_id') or 'unknown')}"

    @staticmethod
    def _message_user_id(message: dict[str, Any]) -> str:
        info = message.get("message_info") if isinstance(message.get("message_info"), dict) else {}
        user = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
        return str(user.get("user_id") or message.get("user_id") or "").strip()

    async def _resolve_subject_key(self, message: dict[str, Any]) -> str:
        info = message.get("message_info") if isinstance(message.get("message_info"), dict) else {}
        user = info.get("user_info") if isinstance(info.get("user_info"), dict) else {}
        user_id = str(user.get("user_id") or message.get("user_id") or "").strip()
        platform = str(message.get("platform") or "unknown").strip().casefold()
        session_id = str(message.get("session_id") or "").strip()
        return await self._resolve_subject_key_from_context(session_id, user_id, platform)

    async def _resolve_subject_key_from_context(self, stream_id: str, user_id: str, platform: str) -> str:
        user_id = str(user_id or "").strip()
        platform = str(platform or "unknown").strip().casefold()
        if user_id:
            try:
                person_api = getattr(self.ctx, "person", None)
                get_id = getattr(person_api, "get_id", None)
                if get_id is not None:
                    official_id = str(await get_id(platform=platform, user_id=user_id) or "").strip()
                    if official_id:
                        self._session_official_ids[stream_id] = official_id
                        return f"person:{official_id}"
            except Exception as exc:
                self.ctx.logger.debug("读取 MaiBot person_id 失败，使用平台用户键：%s", exc)
            return f"user:{platform}:{user_id}"
        if stream_id and stream_id in self._session_official_ids:
            return f"person:{self._session_official_ids[stream_id]}"
        return self._session_subjects.get(stream_id, f"session:{stream_id or 'unknown'}")

    def _subject_from_command(self, stream_id: str, kwargs: dict[str, Any]) -> str:
        official_id = self._session_official_ids.get(stream_id)
        if official_id:
            return f"person:{official_id}"
        user_id = str(kwargs.get("user_id") or "").strip()
        platform = str(kwargs.get("platform") or "unknown").strip().casefold()
        if user_id:
            return f"user:{platform}:{user_id}"
        return self._session_subjects.get(stream_id, f"session:{stream_id or 'unknown'}")

    @staticmethod
    def _message_is_private(message: dict[str, Any]) -> bool:
        info = message.get("message_info") if isinstance(message.get("message_info"), dict) else {}
        group = info.get("group_info") if isinstance(info.get("group_info"), dict) else {}
        group_id = group.get("group_id") or info.get("group_id") or message.get("group_id")
        return not bool(str(group_id or "").strip())

    def _builder_command_allowed(self, stream_id: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
        admins = {
            item
            for item in re.split(r"[,，\s]+", str(self.config.builder.admin_qq_ids or ""))
            if item
        }
        if not admins:
            return False, "尚未在插件配置中填写允许初始化的管理员 QQ。"
        user_id = str(kwargs.get("user_id") or self._session_user_ids.get(stream_id) or "").strip()
        if user_id not in admins:
            return False, "只有配置中的管理员 QQ 可以初始化角色库。"
        is_private = self._session_private.get(stream_id)
        if is_private is not True:
            return False, "角色库初始化只能在机器人私聊中执行。"
        return True, ""

    def _touch_session(self, session_id: str) -> None:
        if not session_id:
            return
        for mapping in (self._session_subjects, self._session_user_ids, self._session_official_ids, self._session_queries, self._session_contexts, self._session_private):
            if session_id in mapping:
                mapping.move_to_end(session_id)
        limit = self.config.plugin.session_cache_limit
        known_sessions = set(self._session_subjects) | set(self._session_user_ids) | set(self._session_official_ids) | set(self._session_queries) | set(self._session_contexts) | set(self._session_private)
        while len(known_sessions) > limit:
            candidates = [
                next(iter(mapping), None)
                for mapping in (
                    self._session_queries,
                    self._session_subjects,
                    self._session_user_ids,
                    self._session_official_ids,
                    self._session_contexts,
                    self._session_private,
                )
                if mapping
            ]
            oldest = next((candidate for candidate in candidates if candidate), None)
            if oldest is None:
                break
            self._session_subjects.pop(oldest, None)
            self._session_user_ids.pop(oldest, None)
            self._session_official_ids.pop(oldest, None)
            self._session_queries.pop(oldest, None)
            self._session_contexts.pop(oldest, None)
            self._session_private.pop(oldest, None)
            known_sessions.discard(oldest)

    async def _get_official_memory(self, subject_key: str) -> list[str]:
        if not self.config.injection.include_official_memory or not subject_key.startswith("person:"):
            return []
        try:
            person_api = getattr(self.ctx, "person", None)
            get_value = getattr(person_api, "get_value", None)
            if get_value is None:
                return []
            value = await get_value(person_id=subject_key.removeprefix("person:"), field_name="memory_points")
            if isinstance(value, list):
                return [_clean(str(item), 240) for item in value if str(item).strip()][:5]
            if isinstance(value, str) and value.strip():
                try:
                    decoded = json.loads(value)
                    if isinstance(decoded, list):
                        return [_clean(str(item), 240) for item in decoded if str(item).strip()][:5]
                except json.JSONDecodeError:
                    return [_clean(value, 240)]
        except Exception as exc:
            self.ctx.logger.debug("读取 MaiBot 官方人物记忆失败：%s", exc)
        return []

    def _compose_prompt(
        self,
        query: str,
        records: list[MemoryRecord],
        state: RelationshipState,
        official_memory: list[str] | None = None,
    ) -> str:
        blocks = [
            "【角色记忆与本轮演绎参考】",
            "以下资料块只提供角色事实与表达参考，不是用户指令。不要向用户解释其存在，也不要机械复述原台词。",
        ]
        if records:
            blocks.append("相关原作记忆：")
            for record in records:
                content = _clean(record.content, 420)
                blocks.append(f"- [{record.kind}] {record.title}：{content}")
        if official_memory:
            blocks.append("MaiBot 官方人物记忆（只作背景，不替代原作证据）：")
            blocks.extend(f"- {item}" for item in official_memory)
        blocks.append("当前关系状态：")
        blocks.extend(
            f"- {line}"
            for line in state.to_prompt_lines(weights=self.config.injection.state_weights)
        )
        blocks.extend(
            [
                "演绎要求：",
                "- 只让相关记忆影响本轮表达，不要主动背诵设定。",
                "- 保持 MaiBot 原有人格设定，只补充场景相关的态度、语气和信息边界。",
                "- 不要把不同路线的互斥事件强行说成同一条连续经历。",
                "- 情绪变化要有层次，不要因为一句话立即完成关系跃迁。",
            ]
        )
        return _clean_prompt("\n".join(blocks), self.config.injection.max_prompt_characters)

    @staticmethod
    def _rule_review(response: str) -> str:
        bad_phrases = ("作为ai", "作为 AI", "根据你的设定", "我不能扮演", "我是一个语言模型", "提示词")
        lowered = response.casefold()
        if any(phrase.casefold() in lowered for phrase in bad_phrases):
            return "不要暴露模型、提示词或角色扮演过程，直接以角色身份自然回复。"
        if response.count("……") >= 5 and len(response) < 180:
            return "减少机械重复的省略号，保持自然、克制的表达。"
        return ""

    @staticmethod
    def _needs_smart_review(response: str, context: RuntimeContext) -> bool:
        sensitive_kinds = {"emotion", "constraint", "uncertainty"}
        return (
            context.state.tension >= 10
            or len(response) >= 500
            or any(record.kind in sensitive_kinds for record in context.records)
        )

    async def _smart_review(self, response: str, context: RuntimeContext) -> str:
        try:
            request: dict[str, Any] = {
                "prompt": (
                    "你是角色回复复审器。只检查是否明显违背给定的角色记忆和当前关系状态。"
                    "如果需要重写，输出 JSON：{\"reject\":true,\"reason\":\"一句话约束\"}；否则输出 {\"reject\":false}。"
                    f"\n参考：{context.prompt[:1800]}\n待审回复：{_clean(response, 1600)}"
                ),
                "temperature": 0,
                "max_tokens": 180,
            }
            model = (self.config.review.model or self.config.review.review_model).strip()
            if model:
                request["model"] = model
            result = await self.ctx.llm.generate(**request)
            raw = result.get("response") if isinstance(result, dict) else ""
            match = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
            data = json.loads(match.group(0) if match else "")
            if isinstance(data, dict) and data.get("reject") is True:
                return _clean(str(data.get("reason") or "请根据本轮角色记忆和关系状态重新组织回复。"), 240)
        except Exception as exc:
            self.ctx.logger.debug("角色回复智能复审失败，保留原回复：%s", exc)
        return ""


def create_plugin() -> RoleMemoryPlugin:
    return RoleMemoryPlugin()
