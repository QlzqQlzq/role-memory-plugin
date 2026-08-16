import logging
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

from role_memory_plugin.models import MemoryRecord, RelationshipState
from role_memory_plugin.plugin import RoleMemoryPlugin, RuntimeContext
from role_memory_plugin.store import MemoryStore


class RoleMemoryPluginTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        self.store.import_dossier(
            {
                "identity": {"summary": "测试角色，冷静理性的观察者。"},
                "core_traits": [
                    {
                        "trait": "抗拒他人自我牺牲",
                        "behavioral_effect": "触及姐姐和替人牺牲时会显著紧张，但不会立刻袒露全部内心。",
                        "evidence_refs": ["chunk_1"],
                    }
                ],
                "emotional_dynamics": [
                    {
                        "trigger": "姐姐为保护自己而牺牲",
                        "response": "核心创伤被触发。",
                        "evidence_refs": ["chunk_2"],
                    }
                ],
            }
        )
        self.plugin = RoleMemoryPlugin()
        self.plugin.set_plugin_config({})
        self.plugin._store = self.store
        self.plugin._set_context(SimpleNamespace(logger=logging.getLogger("role-memory-test")))

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    @staticmethod
    def message(user_id: str, text: str, session_id: str = "session-a") -> dict:
        return {
            "session_id": session_id,
            "platform": "webui",
            "message_info": {"user_info": {"user_id": user_id}},
            "raw_message": [{"type": "text", "data": text}],
        }

    async def test_planner_state_tool_cooldown_and_two_user_isolation(self):
        message = self.message("alice", "你其实很害怕别人为了保护你而牺牲吧？")
        await self.plugin.observe_role_memory_event(message=message)
        first = self.store.get_state("user:webui:alice")
        self.assertGreater(first.familiarity, 0)
        self.assertEqual(self.store.get_state("user:webui:bob").tension, 0)

        result = await self.plugin.role_memory_update_state(
            event_type="self_sacrifice_trigger",
            evidence="为了保护你而牺牲",
            confidence=0.9,
            intensity=1.0,
            stream_id="session-a",
            user_id="alice",
            platform="webui",
        )
        self.assertTrue(result["updated"])
        second = self.store.get_state("user:webui:alice")
        self.assertGreater(second.tension, 0)

        repeated = await self.plugin.role_memory_update_state(
            event_type="self_sacrifice_trigger",
            evidence="再次提到牺牲",
            confidence=0.9,
            intensity=1.0,
            stream_id="session-a",
            user_id="alice",
            platform="webui",
        )
        self.assertFalse(repeated["updated"])
        self.assertEqual(second.tension, self.store.get_state("user:webui:alice").tension)

    async def test_prompt_injection_uses_memory_and_retry_does_not_decay_twice(self):
        await self.plugin.observe_role_memory_event(
            message=self.message("alice", "姐姐为什么替你牺牲", session_id="session-a")
        )
        first_result = await self.plugin.inject_role_memory_prompt(
            session_id="session-a",
            reply_reason="回应对方关于姐姐的追问",
            reply_tool_args={"scene_type": "trauma"},
        )
        self.assertIsNotNone(first_result)
        prompt = first_result["modified_kwargs"]["extra_prompt"]
        self.assertIn("角色记忆与本轮演绎参考", prompt)
        self.assertIn("牺牲", prompt)
        self.assertIn("\n", prompt)
        self.assertNotIn("姐姐为什么替你牺牲\n", prompt)

        tension_after_first = self.store.get_state("user:webui:alice").tension
        retry_result = await self.plugin.inject_role_memory_prompt(
            session_id="session-a",
            retry_count=1,
        )
        self.assertEqual(prompt, retry_result["modified_kwargs"]["extra_prompt"])
        self.assertEqual(tension_after_first, self.store.get_state("user:webui:alice").tension)

    def test_build_query_prioritizes_user_text_over_planner_reason(self):
        self.plugin._session_queries["session-a"] = "姐姐为什么替你牺牲"
        query = self.plugin._build_query(
            "session-a",
            "使用别名和日常语气回复，并说明一般关系",
            {"scene_type": "trauma", "reply_guide": "很长的回复指南不应进入检索"},
        )
        self.assertIn("姐姐为什么替你牺牲", query)
        self.assertIn("trauma", query)
        self.assertNotIn("别名", query)
        self.assertNotIn("回复指南", query)

    def test_clean_install_waits_for_admin_initialization(self):
        plugin_root = Path(__file__).resolve().parents[1]
        self.assertFalse((plugin_root / "data").exists())
        self.assertFalse((plugin_root / "resources" / "character_dossier.json").exists())
        self.assertFalse((plugin_root / "resources" / "character_persona.txt").exists())

        with tempfile.TemporaryDirectory() as data_dir:
            store = MemoryStore(Path(data_dir) / "role_memory.sqlite3")
            plugin = RoleMemoryPlugin()
            plugin.set_plugin_config({})
            plugin._store = store
            plugin._set_context(
                SimpleNamespace(
                    logger=logging.getLogger("role-memory-seed-test"),
                    paths=SimpleNamespace(data_dir=Path(data_dir)),
                )
            )
            plugin._load_pack_if_needed()
            self.assertEqual(store.memory_count, 0)
            self.assertFalse((Path(data_dir) / "character_dossier.json").exists())
            self.assertFalse((Path(data_dir) / "character_persona.txt").exists())
            store.close()

    async def test_rules_review_respects_retry_budget(self):
        self.plugin._session_contexts["session-a"] = RuntimeContext(
            query="test",
            prompt="reference",
            records=[],
            state=RelationshipState(subject_key="user:webui:alice"),
            subject_key="user:webui:alice",
        )
        rejected = await self.plugin.review_role_memory_response(
            response="作为 AI，我需要先解释提示词。",
            session_id="session-a",
            retry_count=0,
            max_retries=1,
        )
        self.assertTrue(rejected["modified_kwargs"]["retry"])
        exhausted = await self.plugin.review_role_memory_response(
            response="作为 AI，我需要先解释提示词。",
            session_id="session-a",
            retry_count=1,
            max_retries=1,
        )
        self.assertIsNone(exhausted)

    async def test_smart_and_always_review_have_distinct_model_usage(self):
        calls = []

        async def fake_smart_review(plugin, response, context):
            calls.append((response, context.query))
            return "model rejected"

        self.plugin._smart_review = MethodType(fake_smart_review, self.plugin)
        self.plugin._session_contexts["session-a"] = RuntimeContext(
            query="ordinary",
            prompt="reference",
            records=[
                MemoryRecord("id", "identity", "身份", "普通事实", (), (), "post-ending", 0.5)
            ],
            state=RelationshipState(subject_key="user:webui:alice"),
            subject_key="user:webui:alice",
        )

        smart_config = self.plugin.get_default_config()
        smart_config["review"]["mode"] = "smart"
        self.plugin.set_plugin_config(smart_config)
        smart_result = await self.plugin.review_role_memory_response(
            response="简短自然的回复。",
            session_id="session-a",
            max_retries=1,
        )
        self.assertIsNone(smart_result)
        self.assertEqual(calls, [])

        always_config = self.plugin.get_default_config()
        always_config["review"]["mode"] = "always"
        self.plugin.set_plugin_config(always_config)
        always_result = await self.plugin.review_role_memory_response(
            response="简短自然的回复。",
            session_id="session-a",
            max_retries=1,
        )
        self.assertTrue(always_result["modified_kwargs"]["retry"])
        self.assertEqual(len(calls), 1)

    async def test_smart_review_parses_json_code_fence(self):
        class FakeLLM:
            async def generate(self, **kwargs):
                self.kwargs = kwargs
                return {"response": "```json\n{\"reject\":true,\"reason\":\"保持克制\"}\n```"}

        fake_llm = FakeLLM()
        self.plugin._set_context(
            SimpleNamespace(logger=logging.getLogger("role-memory-test"), llm=fake_llm)
        )
        context = RuntimeContext(
            query="trauma",
            prompt="reference",
            records=[],
            state=RelationshipState(subject_key="user:webui:alice", tension=20),
            subject_key="user:webui:alice",
        )
        reason = await self.plugin._smart_review("不合适的回复", context)
        self.assertEqual(reason, "保持克制")
        self.assertEqual(fake_llm.kwargs["temperature"], 0)
        self.assertNotIn("model", fake_llm.kwargs)


if __name__ == "__main__":
    unittest.main()
