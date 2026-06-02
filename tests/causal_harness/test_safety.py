"""Tests for safety governor module."""
import tempfile

from causal_harness.safety.risk import (
    RiskTier,
    ActionRisk,
    classify_action_risk,
    PREAPPROVED_LOW_RISK,
    ACTION_RISKS,
)
from causal_harness.safety.governor import (
    SafetyGovernor,
    SafetyDecision,
    Decision,
    TransactionRecord,
)


class TestRiskClassification:
    def test_read_file_is_low(self):
        risk = classify_action_risk("read_file")
        assert risk.tier == RiskTier.LOW
        assert risk.reversible

    def test_edit_file_is_medium(self):
        risk = classify_action_risk("edit_file")
        assert risk.tier == RiskTier.MEDIUM
        assert risk.reversible

    def test_exec_is_high(self):
        risk = classify_action_risk("exec")
        assert risk.tier == RiskTier.HIGH

    def test_delete_file_is_critical(self):
        risk = classify_action_risk("delete_file")
        assert risk.tier == RiskTier.CRITICAL

    def test_unknown_tool_defaults_medium(self):
        risk = classify_action_risk("future_tool_v2")
        assert risk.tier == RiskTier.MEDIUM

    def test_preapproved_low_risk(self):
        assert "read_file" in PREAPPROVED_LOW_RISK
        assert "web_search" in PREAPPROVED_LOW_RISK
        assert "exec" not in PREAPPROVED_LOW_RISK


class TestSafetyGovernor:
    def test_low_risk_auto_allowed(self):
        gov = SafetyGovernor()
        result = gov.check("read_file", {"file_path": "test.py"})
        assert result.decision == Decision.ALLOW

    def test_edit_file_without_read_blocks(self):
        gov = SafetyGovernor(require_preconditions=True)
        result = gov.check("edit_file", {"file_path": "new.py"})
        assert result.decision == Decision.BLOCK
        assert "file has been read" in result.reason.lower()

    def test_edit_file_after_read_allows(self):
        gov = SafetyGovernor(require_preconditions=True)
        gov.record_file_read("new.py")
        result = gov.check("edit_file", {"file_path": "new.py"})
        assert result.decision in (Decision.ALLOW, Decision.ALLOW_WITH_LOG)

    def test_critical_requires_approval(self):
        gov = SafetyGovernor()
        result = gov.check("delete_file", {"file_path": "x.py"})
        assert result.decision == Decision.REQUIRE_APPROVAL

    def test_critical_with_approval_allowed(self):
        gov = SafetyGovernor()
        gov.grant_approval("delete_file")
        result = gov.check("delete_file", {"file_path": "x.py"})
        assert result.decision == Decision.ALLOW_WITH_LOG

    def test_transaction_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            gov = SafetyGovernor(transaction_dir=tmp)
            gov.record_file_read("main.py")
            result = gov.check("edit_file", {"file_path": "main.py"})
            assert result.decision == Decision.ALLOW_WITH_LOG
            assert result.transaction_id

            tx = gov.get_transaction(result.transaction_id)
            assert tx is not None
            assert tx.tool_name == "edit_file"
            assert tx.risk_tier == RiskTier.MEDIUM

    def test_approval_policy_ask_medium(self):
        gov = SafetyGovernor(approval_policy="ask_medium")
        gov.record_file_read("x.py")  # satisfy precondition first
        result = gov.check("edit_file", {"file_path": "x.py"})
        # Medium tier with ask_medium policy needs approval
        assert result.decision == Decision.REQUIRE_APPROVAL
