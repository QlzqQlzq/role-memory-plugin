import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from role_memory_plugin.models import RelationshipState
from role_memory_plugin.store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_import_and_search(self):
        dossier = {
            "identity": {"summary": "测试角色，冷静理性的观察者。", "aliases": ["角色甲"]},
            "core_traits": [{"trait": "理性求真", "behavioral_effect": "要求证据", "confidence": "high", "evidence_refs": ["chunk_1"]}],
            "decision_patterns": [{"situation": "面对逻辑漏洞", "usual_response": "要求物证", "evidence_refs": ["chunk_2"]}],
            "relationships": [{"person": "姐姐", "dynamic": "重要的家人关系", "evidence_refs": ["chunk_4"]}],
            "speech_style": {"tone": ["低沉克制"], "wording": ["客观事实"]},
            "representative_quotes": [{"quote": "……拿出证据吧。", "meaning": "要求证据", "source_ref": "chunk_3"}],
        }
        store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        try:
            self.assertGreaterEqual(store.import_dossier(dossier, "固定人设底座"), 6)
            results = store.search("证据", limit=5)
            self.assertTrue(results)
            self.assertTrue(any("证据" in result.content for result in results))
            natural_query_results = store.search("面对逻辑漏洞时为什么要求证据", limit=5)
            self.assertTrue(any("证据" in result.content for result in natural_query_results))
            support_results = store.search("没关系，你不用告诉我，等你愿意的时候再说", limit=5)
            self.assertEqual(support_results, [])
        finally:
            store.close()

    def test_import_deduplicates_repeated_model_evidence(self):
        dossier = {
            "identity": {"summary": "测试角色，保持克制。"},
            "core_traits": ["不会轻易相信别人", "不会轻易相信别人"],
        }
        store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        try:
            imported = store.import_dossier(dossier)
            self.assertEqual(imported, 2)
            self.assertEqual(store.memory_count, 2)
        finally:
            store.close()


    def test_state_updates_are_bounded_and_persisted(self):
        path = Path(self.temp_dir.name) / "memory.sqlite3"
        store = MemoryStore(path)
        try:
            initial = store.get_state("user:1")
            self.assertIsInstance(initial, RelationshipState)
            updated = store.update_state("user:1", {"trust": 200, "wariness": -200}, "test", "unit", 1.0)
            self.assertTrue(0 <= updated.trust <= 100)
            self.assertTrue(0 <= updated.wariness <= 100)
        finally:
            store.close()
        reopened = MemoryStore(path)
        try:
            persisted = reopened.get_state("user:1")
            self.assertEqual(persisted.trust, updated.trust)
            self.assertEqual(persisted.wariness, updated.wariness)
        finally:
            reopened.close()


    def test_dossier_is_json_serializable(self):
        payload = {"identity": {"summary": "test"}}
        self.assertEqual(json.loads(json.dumps(payload, ensure_ascii=False)), payload)

    def test_recent_events_and_user_isolation(self):
        store = MemoryStore(Path(self.temp_dir.name) / "memory.sqlite3")
        try:
            store.update_state("user:webui:a", {"trust": 3}, "support", "unit", 0.9)
            self.assertTrue(store.has_recent_event("user:webui:a", "support", 60))
            self.assertFalse(store.has_recent_event("user:webui:b", "support", 60))
            self.assertEqual(store.get_state("user:webui:b").trust, 0)
        finally:
            store.close()

    def test_corrupt_database_is_preserved_and_rebuilt(self):
        path = Path(self.temp_dir.name) / "memory.sqlite3"
        path.write_bytes(b"not a sqlite database")
        store, backup = MemoryStore.open_resilient(path)
        try:
            self.assertIsNotNone(backup)
            self.assertTrue(backup.exists())
            self.assertEqual(store.memory_count, 0)
        finally:
            store.close()

    def test_incompatible_schema_is_preserved_and_rebuilt(self):
        path = Path(self.temp_dir.name) / "memory.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO meta(key, value) VALUES('schema_version', '999')")
        connection.commit()
        connection.close()

        store, backup = MemoryStore.open_resilient(path)
        try:
            self.assertIsNotNone(backup)
            self.assertTrue(backup.exists())
            self.assertEqual(store.memory_count, 0)
        finally:
            store.close()
