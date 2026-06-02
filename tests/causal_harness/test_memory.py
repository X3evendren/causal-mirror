"""Tests for structured memory system."""
import tempfile

from causal_harness.memory.schema import MemoryType, MemoryItem
from causal_harness.memory.store import MemoryStore
from causal_harness.memory.consolidation import consolidate_episode
from causal_harness.events.schema import EventType, EventOutcome, new_event
from causal_harness.events.adapters import tool_event, error_event


class TestMemoryItem:
    def test_auto_timestamps(self):
        item = MemoryItem(memory_id="1", memory_type=MemoryType.IDENTITY)
        assert item.created_at
        assert item.updated_at
        assert item.is_active()

    def test_touch_updates_access(self):
        item = MemoryItem(memory_id="1", memory_type=MemoryType.PREFERENCES)
        item.touch()
        assert item.access_count == 1
        assert item.last_accessed

    def test_supersede(self):
        item = MemoryItem(memory_id="1", memory_type=MemoryType.SKILLS)
        item.supersede("2")
        assert not item.is_active()
        assert item.superseded_by == "2"


class TestMemoryStore:
    def test_add_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            mid = store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.PREFERENCES,
                content="Use short responses", tags=["style"],
            ))
            assert mid

            results = store.retrieve("short", memory_type=MemoryType.PREFERENCES)
            assert len(results) == 1
            assert "short" in results[0].content

    def test_supersede_marks_old_inactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            mid = store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.PROJECT_STATE,
                content="Old state",
            ))
            replacement = MemoryItem(
                memory_id="", memory_type=MemoryType.PROJECT_STATE,
                content="New state",
            )
            store.supersede(mid, MemoryType.PROJECT_STATE, replacement)

            old = store.get(mid, MemoryType.PROJECT_STATE)
            assert not old.is_active()

            active = store.list_active(MemoryType.PROJECT_STATE)
            assert len(active) == 1
            assert "New" in active[0].content

    def test_retrieve_by_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.SKILLS,
                content="Skill A", source_events=["evt_1"],
            ))
            store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.SKILLS,
                content="Skill B", source_events=["evt_2"],
            ))
            results = store.retrieve_by_source("evt_1")
            assert len(results) == 1
            assert "Skill A" in results[0].content

    def test_decay_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.PROJECT_STATE,
                content="Stale item", confidence=1.0,
            ))
            # Mark as old by setting last_accessed far in the past
            for item in store.list_active(MemoryType.PROJECT_STATE):
                item.last_accessed = "2020-01-01T00:00:00+00:00"

            count = store.decay_stale(MemoryType.PROJECT_STATE, max_age_days=1)
            assert count == 1
            active = store.list_active(MemoryType.PROJECT_STATE)
            assert active[0].confidence < 1.0

    def test_tag_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.SKILLS,
                content="Pattern X", tags=["coding", "python"],
            ))
            store.add(MemoryItem(
                memory_id="", memory_type=MemoryType.SKILLS,
                content="Pattern Y", tags=["coding", "javascript"],
            ))
            results = store.retrieve(tags=["python"])
            assert len(results) == 1
            assert "Pattern X" in results[0].content


class TestConsolidation:
    def test_extract_files_from_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            events = [
                tool_event("read_file", {"file_path": "a.py"}, "ep1", 1),
                tool_event("edit_file", {"file_path": "a.py"}, "ep1", 1),
                tool_event("read_file", {"file_path": "b.py"}, "ep1", 2),
                tool_event("edit_file", {"file_path": "b.py"}, "ep1", 2),
                new_event(EventType.EPISODE_COMPLETED, "ep1", 2,
                          outcome=EventOutcome.SUCCESS),
            ]
            created = consolidate_episode(events, store)
            assert len(created) >= 1  # at least project_state for files

            files_mem = store.retrieve(tags=["files"])
            assert len(files_mem) >= 1

    def test_extract_skill_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            events = [
                tool_event("read_file", {"file_path": "x.py"}, "ep1", 1),
                tool_event("edit_file", {"file_path": "x.py"}, "ep1", 1),
                tool_event("shell", {"command": "pytest"}, "ep1", 2),
                new_event(EventType.EPISODE_COMPLETED, "ep1", 2,
                          outcome=EventOutcome.SUCCESS),
            ]
            created = consolidate_episode(events, store)
            skills = store.retrieve(memory_type=MemoryType.SKILLS)
            assert len(skills) >= 1
            assert "read_file" in skills[0].content
            assert "edit_file" in skills[0].content

    def test_extract_errors_as_risks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            events = [
                error_event("ep1", 1, "timeout", "Connection timed out"),
            ]
            consolidate_episode(events, store)
            risks = store.retrieve(tags=["risk", "timeout"])
            assert len(risks) == 1
