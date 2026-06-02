"""Tests for the event ledger system."""
import json
import tempfile
from pathlib import Path

from causal_harness.events.schema import (
    Event, EventType, EventOutcome, new_event,
)
from causal_harness.events.ledger import EventLedger
from causal_harness.events.episode import Episode, EpisodeManager, EpisodeStatus


class TestEventSchema:
    def test_new_event_defaults(self):
        evt = new_event(EventType.TURN_STARTED, episode_id="ep1", turn_id=1)
        assert evt.event_id
        assert len(evt.event_id) == 12
        assert evt.episode_id == "ep1"
        assert evt.turn_id == 1
        assert evt.actor == "agent"
        assert evt.event_type == EventType.TURN_STARTED
        assert evt.outcome == EventOutcome.UNKNOWN
        assert evt.risk_before is None
        assert evt.risk_after is None
        assert evt.payload == {}

    def test_new_event_with_payload(self):
        evt = new_event(
            EventType.TOOL_CALLED, "ep1", 2,
            actor="executor",
            payload={"tool": "read_file", "path": "/tmp/x.txt"},
            risk_before=0.3,
            risk_after=0.5,
            outcome=EventOutcome.SUCCESS,
        )
        assert evt.payload["tool"] == "read_file"
        assert evt.risk_before == 0.3
        assert evt.risk_after == 0.5
        assert evt.outcome == EventOutcome.SUCCESS

    def test_roundtrip_dict(self):
        evt = new_event(
            EventType.ERROR_DETECTED, "ep1", 3,
            payload={"error_class": "assertion_failure", "message": "boom"},
            outcome=EventOutcome.FAILURE,
        )
        d = evt.to_dict()
        loaded = Event.from_dict(d)
        assert loaded.event_id == evt.event_id
        assert loaded.event_type == evt.event_type
        assert loaded.outcome == evt.outcome
        assert loaded.payload == evt.payload

    def test_all_event_types_present(self):
        from causal_harness.events.schema import EVENT_TYPES
        assert EventType.TURN_STARTED in EVENT_TYPES
        assert EventType.EPISODE_STARTED in EVENT_TYPES
        assert EventType.EPISODE_COMPLETED in EVENT_TYPES
        assert len(EVENT_TYPES) == 14


class TestEventLedger:
    def test_append_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            e1 = new_event(EventType.EPISODE_STARTED, "ep1", 0)
            e2 = new_event(EventType.TURN_STARTED, "ep1", 1)
            e3 = new_event(EventType.TURN_COMPLETED, "ep1", 1)
            ledger.append(e1)
            ledger.append(e2)
            ledger.append(e3)

            events = list(ledger.replay())
            assert len(events) == 3
            assert events[0].event_type == EventType.EPISODE_STARTED
            assert events[2].event_type == EventType.TURN_COMPLETED

    def test_filter_by_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            ledger.append(new_event(EventType.TURN_STARTED, "ep1", 1))
            ledger.append(new_event(EventType.TURN_STARTED, "ep2", 1))
            ledger.append(new_event(EventType.TURN_COMPLETED, "ep1", 1))

            events = list(ledger.replay(episode_id="ep1"))
            assert len(events) == 2
            assert all(e.episode_id == "ep1" for e in events)

    def test_filter_by_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            ledger.append(new_event(EventType.TOOL_CALLED, "ep1", 1))
            ledger.append(new_event(EventType.TOOL_RESULT, "ep1", 1))
            ledger.append(new_event(EventType.TOOL_CALLED, "ep1", 1))

            tools = list(ledger.replay(event_type=EventType.TOOL_CALLED))
            assert len(tools) == 2

    def test_replay_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            for i in range(5):
                ledger.append(new_event(EventType.TURN_STARTED, "ep_a", i))

            events = ledger.replay_episode("ep_a")
            assert len(events) == 5

    def test_corrupted_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            ledger.append(new_event(EventType.TURN_STARTED, "ep1", 1))
            # Manually corrupt the file with a non-JSON line between valid events
            current = ledger._current_path
            with open(current, "a", encoding="utf-8") as f:
                f.write("this is not json\n")
                f.flush()
            # Append a valid event to the same file (bypass rotation check)
            ledger.append(new_event(EventType.TURN_COMPLETED, "ep1", 1))

            events = list(ledger.replay())
            # Corrupted line should be skipped, both valid events retained
            assert len(events) == 2, f"Expected 2 events, got {len(events)}: {[e.event_type.value for e in events]}"

    def test_file_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp, max_file_mb=0)  # force rotation every write
            ledger.append(new_event(EventType.TURN_STARTED, "ep1", 1))
            ledger.append(new_event(EventType.TURN_COMPLETED, "ep1", 1))

            paths = ledger._sorted_ledger_paths()
            assert len(paths) >= 2
            events = list(ledger.replay())
            assert len(events) == 2


class TestEpisode:
    def test_episode_lifecycle(self):
        ep = Episode(episode_id="ep1", goal="Solve math problem")
        assert ep.status == EpisodeStatus.PENDING

        start = ep.start()
        assert ep.status == EpisodeStatus.RUNNING
        assert start.event_type == EventType.EPISODE_STARTED

        assert ep.next_turn() == 1
        assert ep.next_turn() == 2

        end = ep.complete(EventOutcome.SUCCESS)
        assert ep.status == EpisodeStatus.COMPLETED
        assert ep.turn_count == 2
        assert end.event_type == EventType.EPISODE_COMPLETED

    def test_episode_failure(self):
        ep = Episode(episode_id="ep1", goal="task")
        ep.start()
        end = ep.complete(EventOutcome.FAILURE)
        assert ep.status == EpisodeStatus.FAILED


class TestEpisodeManager:
    def test_begin_and_end(self):
        mgr = EpisodeManager()
        assert mgr.active is None

        ep, start_event, auto = mgr.begin_episode("Task A")
        assert auto is None  # no prior episode to auto-complete
        assert mgr.active is ep
        assert start_event.event_type == EventType.EPISODE_STARTED

        end_event = mgr.end_episode(EventOutcome.SUCCESS)
        assert mgr.active is None
        assert end_event.event_type == EventType.EPISODE_COMPLETED
        assert end_event.outcome == EventOutcome.SUCCESS

    def test_auto_complete_prior_episode(self):
        mgr = EpisodeManager()
        ep1, _, auto1 = mgr.begin_episode("Task A")
        assert auto1 is None
        ep2, _, auto2 = mgr.begin_episode("Task B")
        # auto2 is the completion event from ep1 (auto-closed with UNKNOWN)
        assert auto2 is not None
        assert auto2.event_type == EventType.EPISODE_COMPLETED
        assert auto2.episode_id == ep1.episode_id

        assert mgr.active is ep2
        assert ep1.status == EpisodeStatus.FAILED  # auto-closed with UNKNOWN → FAILED
        assert mgr.get(ep1.episode_id) is ep1
        assert mgr.get(ep2.episode_id) is ep2

    def test_all_episode_ids(self):
        mgr = EpisodeManager()
        mgr.begin_episode("Task A")
        mgr.end_episode(EventOutcome.SUCCESS)
        mgr.begin_episode("Task B")

        ids = mgr.all_episode_ids()
        assert len(ids) == 2
