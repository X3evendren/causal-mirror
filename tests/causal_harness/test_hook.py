"""Tests for CausalMirrorHook with mocked nanobot contexts."""
import tempfile
import asyncio
from dataclasses import dataclass, field
from typing import Any

from causal_harness.events.ledger import EventLedger
from causal_harness.events.schema import EventType
from causal_harness.integrations.nanobot import CausalMirrorHook
from causal_harness.integrations.nanobot.event_adapters import (
    tool_name_to_symbol,
    map_tool_call_to_event,
    map_tool_result_to_event,
)


# -- Mock nanobot context objects --

@dataclass
class MockToolCall:
    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class MockToolResult:
    is_error: bool = False
    error_type: str = ""
    error_message: str = ""
    content: str = ""


@dataclass
class MockAgentContext:
    iteration: int = 0
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    tool_events: list = field(default_factory=list)
    error: Any = None
    final_content: str = ""


class TestToolAdapters:
    def test_tool_name_to_symbol_known(self):
        assert tool_name_to_symbol("read_file") == "READ_FILE"
        assert tool_name_to_symbol("edit_file") == "EDIT_FILE"
        assert tool_name_to_symbol("web_search") == "OBSERVE"

    def test_tool_name_to_symbol_unknown(self):
        assert tool_name_to_symbol("some_future_tool") == "PLAN"

    def test_map_tool_call_event(self):
        event = map_tool_call_to_event("read_file", {"file_path": "/x.py"}, "ep1", 1)
        assert event.event_type == EventType.TOOL_CALLED
        assert event.payload["tool"] == "read_file"
        assert event.payload["category"] == "READ_FILE"

    def test_map_tool_result_success(self):
        call = map_tool_call_to_event("read_file", {}, "ep1", 1)
        result = map_tool_result_to_event(call, result="file contents", success=True)
        assert result.event_type == EventType.TOOL_RESULT
        assert result.payload["success"] is True

    def test_map_tool_result_error(self):
        call = map_tool_call_to_event("edit_file", {}, "ep1", 1)
        result = map_tool_result_to_event(
            call, success=False, error_type="FileNotFoundError", error_message="no file"
        )
        assert result.payload["success"] is False
        assert result.payload["error_class"] == "missing_file"


class TestCausalMirrorHook:
    def test_start_and_end_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            hook = CausalMirrorHook(ledger, auto_start_episode=False)

            ep_id = hook.start_session(goal="test task")
            assert ep_id
            assert hook.current_episode_id == ep_id

            hook.end_session()
            events = list(ledger.replay())
            assert len(events) == 2
            assert events[0].event_type == EventType.EPISODE_STARTED
            assert events[1].event_type == EventType.EPISODE_COMPLETED

    def test_before_iteration_records_turn_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            hook = CausalMirrorHook(ledger)
            hook.start_session(goal="test")

            ctx = MockAgentContext(iteration=1)
            asyncio.run(hook.before_iteration(ctx))

            events = list(ledger.replay(event_type=EventType.TURN_STARTED))
            assert len(events) == 1

    def test_tool_call_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            hook = CausalMirrorHook(ledger)
            hook.start_session(goal="test")

            # Simulate a full iteration
            ctx = MockAgentContext(
                iteration=1,
                tool_calls=[
                    MockToolCall(name="read_file", arguments={"file_path": "/x.py"}),
                    MockToolCall(name="edit_file", arguments={"file_path": "/x.py"}),
                ],
                tool_results=[
                    MockToolResult(content="content", is_error=False),
                    MockToolResult(content="edited", is_error=False),
                ],
            )

            asyncio.run(hook.before_iteration(ctx))
            asyncio.run(hook.before_execute_tools(ctx))
            asyncio.run(hook.after_iteration(ctx))

            tool_calls = list(ledger.replay(event_type=EventType.TOOL_CALLED))
            tool_results = list(ledger.replay(event_type=EventType.TOOL_RESULT))
            assert len(tool_calls) == 2
            assert len(tool_results) == 2

    def test_error_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            hook = CausalMirrorHook(ledger)
            hook.start_session(goal="test")

            ctx = MockAgentContext(
                iteration=1,
                tool_calls=[
                    MockToolCall(name="shell", arguments={"command": "bad"}),
                ],
                tool_results=[
                    MockToolResult(
                        is_error=True,
                        error_type="RuntimeError",
                        error_message="command failed",
                    ),
                ],
            )

            asyncio.run(hook.before_iteration(ctx))
            asyncio.run(hook.before_execute_tools(ctx))
            asyncio.run(hook.after_iteration(ctx))

            errors = list(ledger.replay(event_type=EventType.ERROR_DETECTED))
            assert len(errors) == 1

    def test_hook_is_safe_on_failure(self):
        """Hook must never crash the agent loop."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            hook = CausalMirrorHook(ledger)

            # before_iteration without start_session should auto-start
            ctx = MockAgentContext(iteration=1)
            asyncio.run(hook.before_iteration(ctx))  # must not raise

            # bad tool_calls should not crash
            ctx.tool_calls = [None]
            asyncio.run(hook.before_execute_tools(ctx))  # must not raise
