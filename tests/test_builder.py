import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from role_memory_plugin.builder import BuildProgress, BuilderOptions, RoleKnowledgeBuilder
from role_memory_plugin.plugin import RoleMemoryPlugin
from role_memory_plugin.store import MemoryStore


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = MemoryStore(self.root / "role_memory.sqlite3")
        self.builder = RoleKnowledgeBuilder(
            llm=SimpleNamespace(),
            logger=logging.getLogger("builder-test"),
            data_dir=self.root,
            store=self.store,
            character_name="测试角色",
            options=BuilderOptions(chunk_size=40, chunk_overlap=5),
            progress=BuildProgress(),
        )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_folder_sources_decode_and_split_with_provenance(self):
        source_dir = self.root / "imports"
        source_dir.mkdir()
        (source_dir / "scene.txt").write_bytes("角色甲：不是害怕。\n".encode("cp932"))
        (source_dir / "notes.md").write_text("姐姐保护她。\n" * 8, encoding="utf-8")

        files = self.builder._source_files(source_dir)
        documents = [self.builder._read_source(path, source_dir) for path in files]
        chunks = self.builder._make_chunks(documents)

        self.assertEqual(len(files), 2)
        self.assertTrue(any(doc[2] == "cp932" for doc in documents))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source_path for chunk in chunks))
        self.assertTrue(all(chunk.chunk_id.startswith("chunk_") for chunk in chunks))

    def test_install_backups_files_and_preserves_relationship_state(self):
        old_dossier = {"identity": {"summary": "旧档案"}}
        self.store.import_dossier(old_dossier)
        self.store.update_state(
            "user:qq:10001",
            {"trust": 4, "tension": 3},
            "interaction",
            "test",
            1.0,
            record_event=False,
            event_limit=20,
        )
        (self.root / "character_dossier.json").write_text('{"old": true}', encoding="utf-8")
        (self.root / "character_persona.txt").write_text("旧人设", encoding="utf-8")
        dossier = {
            "identity": {"summary": "测试角色，冷静而戒备。"},
            "core_traits": [
                {
                    "trait": "戒备",
                    "behavioral_effect": "不轻信他人。",
                    "evidence_refs": ["chunk_0001"],
                }
            ],
        }

        pack, persona, imported = self.builder._install(
            dossier,
            "新版人设",
            datetime.now(timezone.utc),
        )

        self.assertTrue(pack.is_file())
        self.assertTrue(persona.is_file())
        self.assertGreater(imported, 1)
        self.assertTrue(list((self.root / "backups").rglob("character_dossier.json")))
        state = self.store.get_state("user:qq:10001")
        self.assertEqual(state.trust, 4)
        self.assertEqual(state.tension, 3)

    def test_command_requires_private_configured_admin(self):
        plugin = RoleMemoryPlugin()
        plugin.set_plugin_config(
            {
                "plugin": {"config_version": "0.3.0"},
                "builder": {"admin_qq_ids": "10001"},
            }
        )
        plugin._session_subjects["s"] = "user:qq:10001"
        plugin._session_user_ids["s"] = "10001"
        plugin._session_private["s"] = True
        allowed, _ = plugin._builder_command_allowed("s", {})
        self.assertTrue(allowed)

        plugin._session_official_ids["s"] = "person-42"
        allowed, _ = plugin._builder_command_allowed("s", {})
        self.assertTrue(allowed)

        plugin._session_private["s"] = False
        allowed, reason = plugin._builder_command_allowed("s", {})
        self.assertFalse(allowed)
        self.assertIn("私聊", reason)


class BuilderPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_model_builds_and_hot_installs_pack(self):
        class FakeLLM:
            async def generate(self, **kwargs):
                prompt = kwargs["prompt"]
                if "结构侦察器" in prompt:
                    response = {
                        "primary_language": "zh",
                        "script_format": "dialogue",
                        "aliases": ["角色甲"],
                        "speaker_ids": [],
                        "confirmed_name_mappings": [],
                        "dialogue_markers": ["角色：台词"],
                        "scene_markers": [],
                        "analysis_hints": [],
                        "warnings": [],
                    }
                    return {"response": __import__("json").dumps(response, ensure_ascii=False)}
                if "角色证据提取器" in prompt:
                    response = {
                        "relevant": True,
                        "names_seen": ["测试角色"],
                        "dialogues": [
                            {
                                "quote": "不是害怕。",
                                "translation": "不是害怕。",
                                "context": "谈及牺牲",
                                "evidence_type": "direct",
                            }
                        ],
                        "traits": [
                            {
                                "trait": "抗拒牺牲",
                                "evidence": "拒绝他人为自己牺牲",
                                "confidence": "high",
                            }
                        ],
                        "speech_patterns": [],
                        "actions_and_choices": [],
                        "relationships": [],
                        "plot_context": [],
                        "contradictions_or_changes": [],
                        "uncertainties": [],
                    }
                    return {"response": __import__("json").dumps(response, ensure_ascii=False)}
                if "证据档案归并器" in prompt:
                    response = {
                        "identity": {
                            "summary": "测试角色，冷静且抗拒他人为自己牺牲。",
                            "aliases": ["角色甲"],
                            "important_background": [],
                        },
                        "core_traits": [
                            {
                                "trait": "抗拒牺牲",
                                "behavioral_effect": "面对自我牺牲会明显紧张并保持边界。",
                                "confidence": "high",
                                "evidence_refs": ["chunk_0001"],
                            }
                        ],
                        "values_and_motivations": [],
                        "decision_patterns": [],
                        "speech_style": {
                            "tone": ["克制"],
                            "wording": [],
                            "sentence_patterns": ["短句"],
                            "emotional_expression": [],
                            "relationship_differences": [],
                        },
                        "relationships": [],
                        "emotional_dynamics": [],
                        "contradictions_and_limits": [],
                        "development": [],
                        "representative_quotes": [],
                        "uncertainties": [],
                        "source_refs": ["chunk_0001"],
                    }
                    return {"response": __import__("json").dumps(response, ensure_ascii=False)}
                return {"response": "【身份】\n测试角色。\n\n【核心性格】\n冷静、克制。"}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imports = root / "imports"
            imports.mkdir()
            (imports / "scene.txt").write_text(
                "测试角色：不是害怕。\nA：那为什么拒绝别人保护你？\n" * 20,
                encoding="utf-8",
            )
            store = MemoryStore(root / "role_memory.sqlite3")
            progress = BuildProgress()
            builder = RoleKnowledgeBuilder(
                llm=FakeLLM(),
                logger=logging.getLogger("builder-pipeline-test"),
                data_dir=root,
                store=store,
                character_name="测试角色",
                options=BuilderOptions(chunk_size=2000, chunk_overlap=100),
                progress=progress,
            )
            result = await builder.build_and_install()
            self.assertEqual(progress.status, "completed")
            self.assertGreater(result.imported_memories, 2)
            self.assertTrue(result.dossier_path.is_file())
            self.assertTrue(result.persona_path.is_file())
            self.assertTrue((result.output_dir / "构建清单.json").is_file())
            self.assertIn("抗拒牺牲", result.dossier_path.read_text(encoding="utf-8"))
            store.close()


if __name__ == "__main__":
    unittest.main()
