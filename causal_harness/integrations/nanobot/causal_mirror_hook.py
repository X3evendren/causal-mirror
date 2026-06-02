"""CausalMirrorHook — observe AND intervene in nanobot agent behavior.

Implements nanobot's AgentHook interface with two phases:
  Phase 2 (observe): Record all tool calls, results, and errors into the
                     causal harness event ledger.
  Phase 4 (intervene): Evaluate proposed actions through the CausalPipeline
                       and modify tool calls based on risk and policy.

When a CausalPipeline is attached, the hook automatically:
  - Feeds events to the pipeline for encoding and CSSR analysis
  - Evaluates each tool call before execution
  - Injects verification steps or blocks dangerous actions

Usage with nanobot (observe-only):
    ledger = EventLedger("sessions/session_01")
    hook = CausalMirrorHook(ledger)

Usage with nanobot (full pipeline):
    from causal_harness.pipeline import CausalPipeline
    pipeline = CausalPipeline(ledger_dir="sessions/session_01")
    hook = CausalMirrorHook(ledger, pipeline=pipeline)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from causal_harness.controller.policy import PolicyResult
from causal_harness.events.schema import (
    Event, EventType, EventOutcome, new_event,
)
from causal_harness.events.ledger import EventLedger
from causal_harness.events.episode import EpisodeManager, Episode
from causal_harness.integrations.nanobot.event_adapters import (
    map_tool_call_to_event,
    map_tool_result_to_event,
    map_error_to_event,
    tool_name_to_symbol,
)

logger = logging.getLogger(__name__)


class CausalMirrorHook:
    """Nanobot AgentHook: observe behavior and optionally intervene.

    Phase 2 (observe-only): Without a pipeline, records all events passively.
    Phase 4 (intervene): With a CausalPipeline attached, evaluates proposed
                         tool calls and can block, verify, or escalate.

    This hook is designed to be safe: it must never raise an exception that
    propagates into the agent loop. All failures are caught and logged.

    Attributes:
        ledger: The event ledger to write to.
        episode_manager: Tracks episode lifecycle across turns.
        pipeline: Optional CausalPipeline for risk-aware intervention.
        current_episode_id: Active episode identifier.
        current_turn: Turn counter within the current episode.
    """

    def __init__(
        self,
        ledger: EventLedger,
        episode_manager: EpisodeManager | None = None,
        auto_start_episode: bool = True,
        pipeline: Any = None,  # CausalPipeline | None (lazy import)
    ):
        self.ledger = ledger
        self.episode_manager = episode_manager or EpisodeManager()
        self.auto_start = auto_start_episode
        self.pipeline = pipeline
        self.current_episode_id: str = ""
        self.current_turn: int = 0
        self._pending_tool_events: list[Event] = []
        self._turn_error_detected: bool = False

    # -- nanobot AgentHook lifecycle --

    def wants_streaming(self) -> bool:
        return False  # we don't need token-level visibility

    async def before_iteration(self, context: Any) -> None:
        """Called before each LLM iteration. Record turn start."""
        try:
            self._ensure_episode()
            self.current_turn += 1
            self._turn_error_detected = False
            self._pending_tool_events.clear()

            self.ledger.append(new_event(
                EventType.TURN_STARTED,
                episode_id=self.current_episode_id,
                turn_id=self.current_turn,
                actor="agent",
            ))
        except Exception as exc:
            logger.warning("CausalMirrorHook.before_iteration failed: %s", exc)

    async def on_stream(self, context: Any, delta: str) -> None:
        pass  # token-level streaming not needed for causal state

    async def on_stream_end(self, context: Any, *, resuming: bool = False) -> None:
        pass

    async def before_execute_tools(self, context: Any) -> None:
        """Record TOOL_CALLED events, then evaluate interventions (Phase 4)."""
        try:
            tool_calls = getattr(context, "tool_calls", None) or []
            recorded_events: list[Event] = []

            for tc in tool_calls:
                name = getattr(tc, "name", "") or tc.get("name", "")
                args = getattr(tc, "arguments", {}) or tc.get("arguments", {})
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                event = map_tool_call_to_event(
                    name, args, self.current_episode_id, self.current_turn
                )
                self._pending_tool_events.append(event)
                self.ledger.append(event)
                recorded_events.append(event)

                # Feed to pipeline if attached
                if self.pipeline is not None:
                    try:
                        self.pipeline.record_event(event)
                    except Exception as exc:
                        logger.debug("Pipeline record_event failed: %s", exc)

            # Phase 4: Evaluate interventions (only with pipeline attached)
            if self.pipeline is not None and tool_calls:
                self._intervene_tool_calls(context, list(tool_calls))

        except Exception as exc:
            logger.warning("CausalMirrorHook.before_execute_tools failed: %s", exc)

    async def after_iteration(self, context: Any) -> None:
        """Record TOOL_RESULT and ERROR_DETECTED events after execution."""
        try:
            tool_results = getattr(context, "tool_results", None) or []
            tool_events = getattr(context, "tool_events", None) or []
            error = getattr(context, "error", None)

            # Record tool results paired to their call events
            for i, result_item in enumerate(tool_results):
                call_event = (
                    self._pending_tool_events[i]
                    if i < len(self._pending_tool_events) else None
                )
                if call_event is None:
                    continue

                # Determine success/failure
                is_error = False
                error_type = ""
                error_msg = ""
                result_text = ""

                if isinstance(result_item, dict):
                    is_error = result_item.get("is_error", False)
                    error_type = result_item.get("error_type", "")
                    error_msg = result_item.get("error_message", "")
                    result_text = result_item.get("content", "")
                elif hasattr(result_item, "is_error"):
                    is_error = result_item.is_error
                    error_type = getattr(result_item, "error_type", "")
                    error_msg = getattr(result_item, "error_message", "")
                    result_text = getattr(result_item, "content", "")
                else:
                    result_text = str(result_item)

                outcome_event = map_tool_result_to_event(
                    call_event,
                    result=result_text,
                    success=not is_error,
                    error_type=error_type,
                    error_message=error_msg,
                )
                self.ledger.append(outcome_event)

                # Feed result to pipeline
                if self.pipeline is not None:
                    try:
                        self.pipeline.record_event(outcome_event)
                    except Exception as exc:
                        logger.debug("Pipeline record_event (result) failed: %s", exc)

                if is_error:
                    self._turn_error_detected = True
                    err_event = map_error_to_event(
                        self.current_episode_id,
                        self.current_turn,
                        error_type,
                        error_msg,
                    )
                    self.ledger.append(err_event)

                    if self.pipeline is not None:
                        try:
                            self.pipeline.record_event(err_event)
                        except Exception as exc:
                            logger.debug("Pipeline record_event (error) failed: %s", exc)

            # If the runner itself had an error
            if error and not self._turn_error_detected:
                self._turn_error_detected = True
                err_event = map_error_to_event(
                    self.current_episode_id,
                    self.current_turn,
                    type(error).__name__ if error else "UnknownError",
                    str(error) if error else "",
                )
                self.ledger.append(err_event)

                if self.pipeline is not None:
                    try:
                        self.pipeline.record_event(err_event)
                    except Exception as exc:
                        logger.debug("Pipeline record_event (runner error) failed: %s", exc)

            self._pending_tool_events.clear()
        except Exception as exc:
            logger.warning("CausalMirrorHook.after_iteration failed: %s", exc)

    def finalize_content(self, context: Any, content: str) -> str:
        """Post-process final response content. Pass-through for observe-only."""
        return content

    # -- public helpers --

    def start_session(self, goal: str = "", metadata: dict | None = None) -> str:
        """Begin a new observation session. Returns episode_id."""
        ep, start_event, auto_complete = self.episode_manager.begin_episode(
            goal, metadata or {}
        )
        # If a previous episode was auto-completed, record its completion event
        if auto_complete is not None:
            self.ledger.append(auto_complete)
        self.current_episode_id = ep.episode_id
        self.current_turn = 0
        self.ledger.append(start_event)
        logger.info("CausalMirrorHook: started session %s", ep.episode_id)
        return ep.episode_id

    def end_session(self, outcome: EventOutcome = EventOutcome.SUCCESS) -> None:
        """End the current observation session."""
        end_event = self.episode_manager.end_episode(outcome)
        if end_event:
            self.ledger.append(end_event)
            logger.info("CausalMirrorHook: ended session %s", self.current_episode_id)

    # -- Phase 4: intervention --

    def _intervene_tool_calls(self, context: Any, tool_calls: list) -> None:
        """Evaluate proposed tool calls through the pipeline and apply interventions.

        Modifies context.tool_calls in-place. The most restrictive intervention
        across all tool calls wins.

        Intervention actions:
          - ALLOW: no change
          - VERIFY_BEFORE: prepend an ask_user verification step
          - INSPECT_FIRST: log warning (agent should read/search first)
          - ASK_USER: replace all tool calls with an ask_user request
          - REPLAN: log warning (agent should reconsider approach)
          - BLOCK: clear all tool calls
        """
        decisions = []
        for tc in tool_calls:
            name = self._extract_tool_name(tc)
            symbol = tool_name_to_symbol(name)
            try:
                decision = self.pipeline.evaluate_action(symbol)
                decisions.append(decision)
                logger.debug(
                    "CausalMirrorHook: %s -> risk=%.2f intervention=%s",
                    name, decision.risk.p_failure,
                    decision.intervention.result.value,
                )
            except Exception as exc:
                logger.debug("Intervention evaluation failed for %s: %s", name, exc)
                continue

        if not decisions:
            return

        # Find the most restrictive intervention
        priority = {
            PolicyResult.BLOCK: 0,
            PolicyResult.ASK_USER: 1,
            PolicyResult.VERIFY_BEFORE: 2,
            PolicyResult.INSPECT_FIRST: 3,
            PolicyResult.REPLAN: 4,
            PolicyResult.ALLOW: 5,
        }
        worst = min(decisions, key=lambda d: priority.get(
            d.intervention.result, 99,
        ))
        result = worst.intervention.result

        if result == PolicyResult.BLOCK:
            context.tool_calls = []
            logger.warning(
                "CausalMirrorHook: BLOCKED %d tool call(s) — %s",
                len(tool_calls), worst.intervention.reason,
            )

        elif result == PolicyResult.VERIFY_BEFORE:
            verify_call = self._make_verify_tool_call(
                f"High risk: {worst.intervention.reason}. "
                f"P(failure)={worst.risk.p_failure:.0%}. Continue?"
            )
            context.tool_calls = [verify_call] + list(tool_calls)
            logger.info(
                "CausalMirrorHook: VERIFY_BEFORE injected for %d tool call(s)",
                len(tool_calls) - 1,
            )

        elif result == PolicyResult.ASK_USER:
            ask_call = self._make_verify_tool_call(
                f"Approval needed: {worst.intervention.reason}. "
                f"P(failure)={worst.risk.p_failure:.0%}. Proceed?"
            )
            context.tool_calls = [ask_call]
            logger.info(
                "CausalMirrorHook: ASK_USER — replaced %d tool call(s)", len(tool_calls),
            )

        elif result in (PolicyResult.INSPECT_FIRST, PolicyResult.REPLAN):
            logger.info(
                "CausalMirrorHook: %s — %s (agent should adapt)",
                result.value, worst.intervention.reason,
            )
            # Passive: allow tool calls but log the recommendation.
            # A more advanced Phase 4 could inject a system message.

    def _make_verify_tool_call(self, question: str) -> Any:
        """Create a minimal ask_user tool call for verification/approval.

        Returns a duck-typed object compatible with nanobot's ToolCallRequest.
        Uses a dict-based fallback that works across nanobot versions.
        """
        verify_id = f"cm_verify_{uuid.uuid4().hex[:8]}"
        try:
            # Try nanobot's native ToolCallRequest
            from nanobot.providers.base import ToolCallRequest
            return ToolCallRequest(
                id=verify_id,
                name="ask_user",
                arguments={"question": question},
            )
        except ImportError:
            pass

        # Fallback: dict-based duck-typed object
        class _ToolCall:
            def __init__(self, call_id, name, arguments):
                self.id = call_id
                self.name = name
                self.arguments = arguments

            def get(self, key, default=None):
                return getattr(self, key, default)

        return _ToolCall(verify_id, "ask_user", {"question": question})

    @staticmethod
    def _extract_tool_name(tc) -> str:
        """Extract tool name from a nanobot ToolCallRequest or dict."""
        return getattr(tc, "name", "") or tc.get("name", "") if hasattr(tc, "get") else ""

    # -- internal --

    def _ensure_episode(self) -> None:
        if self.current_episode_id:
            return
        if self.auto_start:
            self.start_session(goal="nanobot session")
        else:
            raise RuntimeError("No active episode. Call start_session() first.")
