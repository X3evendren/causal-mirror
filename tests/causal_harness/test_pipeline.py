"""Integration tests for the CausalPipeline end-to-end flow.

Tests the complete pipeline: Event -> Encoder -> CSSR -> RiskModel
-> PolicyController -> VerificationGate, plus training extraction
and loop detection integration.
"""

import tempfile
import pytest

from cssr import CSSRConfig
from causal_harness.events.schema import (
    Event, EventType, EventOutcome, new_event,
)
from causal_harness.events.ledger import EventLedger
from causal_harness.events.episode import EpisodeManager
from causal_harness.encoding.encoder import BehaviorEncoderV2
from causal_harness.encoding.symbols import ActionSymbol, BOUNDARY
from causal_harness.controller.policy import PolicyResult
from causal_harness.pipeline.pipeline import (
    CausalPipeline, PipelineDecision, PipelineContext,
)
from causal_harness.pipeline.extractor import TrainingDataExtractor


# ── helpers ────────────────────────────────────────────────────

def _make_event(
    event_type: EventType,
    ep_id: str = "ep1",
    turn: int = 1,
    payload: dict | None = None,
    outcome: EventOutcome = EventOutcome.UNKNOWN,
) -> Event:
    return new_event(event_type, ep_id, turn, payload=payload or {}, outcome=outcome)


def _make_tool_call(name: str, ep_id: str = "ep1", turn: int = 1) -> Event:
    return _make_event(
        EventType.TOOL_CALLED, ep_id, turn,
        payload={"tool": name, "category": ActionSymbol.PLAN.value},
    )


def _make_tool_result(
    name: str, success: bool = True, ep_id: str = "ep1", turn: int = 1,
) -> Event:
    return _make_event(
        EventType.TOOL_RESULT, ep_id, turn,
        payload={"tool": name, "success": success},
        outcome=EventOutcome.SUCCESS if success else EventOutcome.FAILURE,
    )


def _make_error(ep_id: str = "ep1", turn: int = 1) -> Event:
    return _make_event(
        EventType.ERROR_DETECTED, ep_id, turn,
        payload={"error_class": "assertion_failure", "message": "test failed"},
        outcome=EventOutcome.FAILURE,
    )


def _simulate_episode(
    pipeline: CausalPipeline,
    goal: str,
    tool_sequence: list[tuple[str, bool]],  # (tool_name, success)
) -> str:
    """Simulate one episode with a sequence of tool calls + results."""
    ep_id = pipeline.start_episode(goal)

    for i, (tool_name, success) in enumerate(tool_sequence):
        turn = i + 1
        # Tool call
        tc_event = _make_tool_call(tool_name, ep_id, turn)
        pipeline.record_event(tc_event)

        # Tool result
        tr_event = _make_tool_result(tool_name, success, ep_id, turn)
        pipeline.record_event(tr_event)

        if not success:
            err_event = _make_error(ep_id, turn)
            pipeline.record_event(err_event)

    episode_outcome = EventOutcome.SUCCESS
    # Check if any tool failed
    if any(not s for _, s in tool_sequence):
        episode_outcome = EventOutcome.FAILURE

    pipeline.end_episode(episode_outcome)
    return ep_id


# ── tests ──────────────────────────────────────────────────────

class TestPipelineLifecycle:
    def test_start_end_episode(self):
        pipeline = CausalPipeline()
        ep_id = pipeline.start_episode("Test task")
        assert ep_id
        assert len(ep_id) == 12
        assert pipeline._current_episode_id == ep_id

        pipeline.end_episode(EventOutcome.SUCCESS)
        assert pipeline._current_episode_id == ""

    def test_record_event_adds_symbols(self):
        pipeline = CausalPipeline()
        pipeline.start_episode("Test")

        tc = _make_tool_call("read_file")
        sym = pipeline.record_event(tc)
        assert sym is not None
        assert sym == ActionSymbol.READ_FILE.value
        assert len(pipeline._symbol_sequence) == 1

        pipeline.end_episode(EventOutcome.SUCCESS)

    def test_boundary_between_episodes(self):
        pipeline = CausalPipeline()

        pipeline.start_episode("Ep 1")
        pipeline.record_event(_make_tool_call("read_file"))
        pipeline.record_event(_make_tool_result("read_file", True))
        pipeline.end_episode(EventOutcome.SUCCESS)

        pipeline.start_episode("Ep 2")
        pipeline.record_event(_make_tool_call("edit_file"))
        pipeline.record_event(_make_tool_result("edit_file", True))
        pipeline.end_episode(EventOutcome.SUCCESS)

        # BOUNDARY should separate episodes (appended in end_episode)
        assert BOUNDARY in pipeline._symbol_sequence
        symbols = pipeline._symbol_sequence
        # After BOUNDARY following Ep 1, next symbols should include Ep 2's actions
        boundary_idx = symbols.index(BOUNDARY)
        assert ActionSymbol.EDIT_FILE.value in symbols[boundary_idx:]

    def test_multiple_episodes_accumulate(self):
        pipeline = CausalPipeline()
        for ep_name in ["A", "B", "C"]:
            _simulate_episode(pipeline, ep_name, [
                ("read_file", True), ("edit_file", True),
            ])

        # 3 episodes trained
        assert len(pipeline._training_episodes) == 3
        assert pipeline.symbol_count >= 6  # at least 2 symbols per episode


class TestPipelineAnalysis:
    def test_analyze_empty_returns_none(self):
        pipeline = CausalPipeline()
        result = pipeline.analyze()
        assert result is None

    def test_analyze_with_data(self):
        pipeline = CausalPipeline()

        # Simulate a coding episode with varied behavior
        _simulate_episode(pipeline, "Debug auth", [
            ("read_file", True), ("read_file", True),
            ("edit_file", True), ("shell", True),
        ])
        _simulate_episode(pipeline, "Fix tests", [
            ("read_file", True), ("edit_file", False),
            ("edit_file", True), ("shell", True),
        ])

        metrics = pipeline.analyze()
        assert metrics is not None
        assert "statistical_complexity" in metrics
        assert "entropy_rate" in metrics
        assert "n_states" in metrics
        assert metrics["n_states"] >= 1

    def test_analyze_with_cssr_config(self):
        pipeline = CausalPipeline(
            cssr_config=CSSRConfig(L_max=2, alpha=0.05, min_count=1),
        )
        # Need enough symbols (>=10) for CSSR to run.
        # Each tool call produces 1 symbol. 10 calls = 10 symbols.
        _simulate_episode(pipeline, "Task", [
            ("read_file", True), ("read_file", True),
            ("search", True), ("edit_file", True),
            ("shell", True), ("read_file", True),
            ("edit_file", True), ("shell", True),
            ("search", True), ("edit_file", True),
        ])

        metrics = pipeline.analyze()
        assert metrics is not None
        assert metrics["n_states"] >= 1
        assert metrics["alphabet_size"] > 0

    def test_get_current_state_id(self):
        pipeline = CausalPipeline()
        _simulate_episode(pipeline, "Task", [
            ("read_file", True), ("edit_file", True),
            ("shell", True),
        ])

        pipeline.analyze()
        state_id = pipeline.get_current_state_id()

        # Should return a valid state or -1
        assert isinstance(state_id, int)
        if pipeline.n_states > 0:
            assert state_id >= -1


class TestPipelineEvaluation:
    def test_evaluate_untrained_returns_fallback(self):
        pipeline = CausalPipeline()
        pipeline.start_episode("Test")

        decision = pipeline.evaluate_action("EDIT_FILE")
        assert isinstance(decision, PipelineDecision)
        assert decision.action_symbol == "EDIT_FILE"
        assert decision.risk is not None
        # Untrained: global fallback should be ~0.5
        assert 0.0 <= decision.risk.p_failure <= 1.0
        pipeline.end_episode(EventOutcome.SUCCESS)

    def test_evaluate_with_context(self):
        pipeline = CausalPipeline()
        pipeline.start_episode("Test")

        decision = pipeline.evaluate_action(
            "EDIT_FILE",
            extra_context={"risk_tier": "high", "reversible": False},
        )
        assert decision.intervention.result is not None
        pipeline.end_episode(EventOutcome.SUCCESS)

    def test_intervention_on_high_risk(self):
        pipeline = CausalPipeline()
        pipeline.start_episode("Dangerous task")

        # High P(failure) + critical tier should trigger intervention
        decision = pipeline.evaluate_action(
            "EDIT_FILE",
            extra_context={
                "risk_tier": "critical",
                "reversible": False,
                "ambiguity": "high",
            },
        )
        # Either BLOCK or ASK_USER depending on policy priority
        assert decision.intervention.result in (
            PolicyResult.BLOCK,
            PolicyResult.ASK_USER,
            PolicyResult.VERIFY_BEFORE,
        )

        pipeline.end_episode(EventOutcome.SUCCESS)

    def test_pipeline_decision_has_all_fields(self):
        pipeline = CausalPipeline()
        pipeline.start_episode("Test")

        decision = pipeline.evaluate_action("OBSERVE")

        assert decision.intervention is not None
        assert decision.verification is not None
        assert decision.risk is not None
        assert decision.context is not None
        assert decision.action_symbol == "OBSERVE"
        assert decision.verification.require_verification is not None
        assert decision.verification.require_approval is not None

        pipeline.end_episode(EventOutcome.SUCCESS)


class TestPipelineTraining:
    def test_training_extraction(self):
        pipeline = CausalPipeline()

        # Episode 1: success
        _simulate_episode(pipeline, "Success task", [
            ("read_file", True), ("edit_file", True), ("shell", True),
        ])
        # Episode 2: failure
        _simulate_episode(pipeline, "Failed task", [
            ("read_file", True), ("edit_file", False),
        ])

        # Train
        result = pipeline.train()
        assert result is True
        assert pipeline.is_trained

    def test_training_improves_prediction(self):
        pipeline = CausalPipeline()

        # Multiple "edit_file -> failure" patterns
        for i in range(5):
            tool_seq = [("read_file", True), ("edit_file", i % 2 == 0)]
            _simulate_episode(pipeline, f"Task {i}", tool_seq)

        pipeline.train()

        # After training, predictions should be more confident
        decision = pipeline.evaluate_action("EDIT_FILE")
        assert decision.risk.confidence > 0  # not total guesswork

    def test_extractor_from_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EventLedger(tmp)
            encoder = BehaviorEncoderV2()

            # Write events for one episode directly
            ep_id = "ep_test_01"
            ledger.append(_make_event(EventType.EPISODE_STARTED, ep_id, 0))
            ledger.append(_make_tool_call("read_file", ep_id, 1))
            ledger.append(_make_tool_result("read_file", True, ep_id, 1))
            ledger.append(_make_tool_call("edit_file", ep_id, 2))
            ledger.append(_make_tool_result("edit_file", False, ep_id, 2))
            ledger.append(_make_error(ep_id, 2))
            ledger.append(_make_event(
                EventType.EPISODE_COMPLETED, ep_id, 2,
                outcome=EventOutcome.FAILURE,
            ))

            extractor = TrainingDataExtractor()
            episodes, outcomes = extractor.extract(ledger, encoder)

            assert len(episodes) == 1
            assert outcomes[0] == 1  # failure
            assert len(episodes[0]) >= 2  # at least read + edit symbols


class TestLoopDetection:
    def test_loop_detection_in_pipeline(self):
        pipeline = CausalPipeline(
            loop_window=5, loop_threshold=3,
        )
        pipeline.start_episode("Looping task")

        # Repeat the same action
        for i in range(4):
            tc = _make_tool_call("edit_file", turn=i + 1)
            pipeline.record_event(tc)

        assert pipeline.loop_detector.is_looping()
        assert pipeline.loop_detector.loop_count() >= 3

    def test_loop_no_trigger_below_threshold(self):
        pipeline = CausalPipeline(
            loop_window=5, loop_threshold=3,
        )
        pipeline.start_episode("Normal task")

        pipeline.record_event(_make_tool_call("read_file", turn=1))
        pipeline.record_event(_make_tool_call("edit_file", turn=2))

        assert not pipeline.loop_detector.is_looping()


class TestPipelineSummary:
    def test_summary(self):
        pipeline = CausalPipeline()
        _simulate_episode(pipeline, "Task", [
            ("read_file", True), ("edit_file", True),
        ])

        summary = pipeline.summary()
        assert summary["total_symbols"] >= 2
        assert "trained" in summary
        assert "ledger_dir" in summary
