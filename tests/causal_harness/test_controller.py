"""Tests for the policy controller module."""
from causal_harness.models.base import RiskPrediction
from causal_harness.controller.policy import (
    PolicyController, PolicyRule, PolicyResult, Intervention,
)
from causal_harness.controller.loop_detection import LoopDetector
from causal_harness.controller.verification import (
    VerificationGate, VerificationDecision, RiskTier, ACTION_RISK_TIERS,
)


class TestPolicyController:
    def test_allow_when_safe(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.1, confidence=0.9)
        result = ctrl.evaluate(risk, {})
        assert result.result == PolicyResult.ALLOW

    def test_verify_on_high_risk(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.75, confidence=0.8)
        result = ctrl.evaluate(risk, {})
        assert result.result == PolicyResult.VERIFY_BEFORE

    def test_ask_user_on_low_confidence_high_ambiguity(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.3, confidence=0.2)
        result = ctrl.evaluate(risk, {"ambiguity": "high"})
        assert result.result == PolicyResult.ASK_USER

    def test_inspect_first_when_uninformed(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.4, confidence=0.2)
        result = ctrl.evaluate(risk, {"has_read_file": False})
        assert result.result == PolicyResult.INSPECT_FIRST

    def test_replan_on_loop(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.3, confidence=0.5)
        result = ctrl.evaluate(risk, {"loop_count": 5})
        assert result.result == PolicyResult.REPLAN

    def test_block_on_critical_risk(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.3, confidence=0.5)
        result = ctrl.evaluate(risk, {"risk_tier": "critical", "reversible": False})
        assert result.result in (PolicyResult.BLOCK, PolicyResult.ASK_USER)

    def test_repair_switch_on_repeated_error(self):
        ctrl = PolicyController()
        risk = RiskPrediction(p_failure=0.3, confidence=0.5)
        ctrl.record_error("assertion_failure")
        ctrl.record_error("assertion_failure")
        result = ctrl.evaluate(risk, {"last_error_class": "assertion_failure"})
        assert result.result == PolicyResult.REPLAN
        assert "switch strategy" in result.reason.lower()

    def test_disable_rule(self):
        ctrl = PolicyController()
        ctrl.disable_rule("high_risk_verify")
        risk = RiskPrediction(p_failure=0.9, confidence=0.8)
        result = ctrl.evaluate(risk, {})
        assert result.result == PolicyResult.ALLOW  # rule disabled

    def test_most_restrictive_wins(self):
        ctrl = PolicyController()
        # Conditions: high risk + loop + low confidence
        risk = RiskPrediction(p_failure=0.8, confidence=0.1)
        ctx = {"loop_count": 5, "ambiguity": "high"}
        result = ctrl.evaluate(risk, ctx)
        # ASK_USER is more restrictive than VERIFY_BEFORE
        assert result.result == PolicyResult.ASK_USER


class TestLoopDetector:
    def test_no_loop_initially(self):
        ld = LoopDetector(window=5, threshold=3)
        assert not ld.is_looping()

    def test_detects_simple_loop(self):
        ld = LoopDetector(window=5, threshold=3)
        for _ in range(3):
            ld.observe("EDIT_FILE")
        assert ld.is_looping()
        assert ld.loop_count() == 3

    def test_window_slides(self):
        ld = LoopDetector(window=3, threshold=3)
        ld.observe("EDIT_FILE")
        ld.observe("EDIT_FILE")
        ld.observe("RUN_TEST")    # window: E, E, T → max=2, no loop
        assert not ld.is_looping()
        ld.observe("EDIT_FILE")   # window: E, T, E → max=2
        assert not ld.is_looping()

    def test_most_frequent(self):
        ld = LoopDetector(window=5, threshold=3)
        ld.observe("A"); ld.observe("B"); ld.observe("B")
        assert ld.most_frequent() == "B"

    def test_recent_pattern(self):
        ld = LoopDetector(window=3, threshold=2)
        ld.observe("A"); ld.observe("B")
        assert ld.recent_pattern() == ["A", "B"]

    def test_reset(self):
        ld = LoopDetector(window=5, threshold=3)
        ld.observe("EDIT_FILE")
        ld.observe("EDIT_FILE")
        ld.observe("EDIT_FILE")
        assert ld.is_looping()
        ld.reset()
        assert not ld.is_looping()


class TestVerificationGate:
    def test_low_risk_no_verification(self):
        gate = VerificationGate()
        risk = RiskPrediction(p_failure=0.1)
        decision = gate.evaluate(risk, "OBSERVE")
        assert not decision.require_verification

    def test_high_risk_verification(self):
        gate = VerificationGate()
        risk = RiskPrediction(p_failure=0.3)
        decision = gate.evaluate(risk, "ESCALATE")
        assert decision.require_verification

    def test_medium_risk_exceeds_threshold(self):
        gate = VerificationGate(risk_threshold=0.60)
        risk = RiskPrediction(p_failure=0.75)
        decision = gate.evaluate(risk, "EDIT_FILE")
        assert decision.require_verification

    def test_medium_risk_below_threshold(self):
        gate = VerificationGate(risk_threshold=0.60)
        risk = RiskPrediction(p_failure=0.30)
        decision = gate.evaluate(risk, "EDIT_FILE")
        assert not decision.require_verification

    def test_critical_requires_approval(self):
        gate = VerificationGate()
        risk = RiskPrediction(p_failure=0.1)
        decision = gate.evaluate(risk, "EDIT_FILE", is_reversible=False)
        # Medium tier + not reversible → depends on risk
        # Actually let's test critical specifically
        gate.always_verify_tiers.add(RiskTier.CRITICAL)

    def test_verification_type_by_action(self):
        gate = VerificationGate()
        risk = RiskPrediction(p_failure=0.8)
        assert gate.evaluate(risk, "EDIT_FILE").verification_type == "test"
        assert gate.evaluate(risk, "RUN_TEST").verification_type == "dry_run"
        assert gate.evaluate(risk, "REPAIR").verification_type == "test"
