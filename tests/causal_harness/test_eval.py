"""Tests for evaluation system."""
import tempfile

from causal_harness.eval.scenario import (
    Scenario, ScenarioSuite, ScenarioType,
    coding_suite, safety_suite, memory_suite, planning_suite,
    BUILTIN_SUITES,
)
from causal_harness.eval.metrics import (
    ReliabilityMetrics,
    compute_reliability_metrics,
    compare_interventions,
)
from causal_harness.eval.reports import (
    InterventionPair,
    ExperimentReport,
    generate_report,
)


class TestScenarios:
    def test_coding_suite_has_scenarios(self):
        suite = coding_suite()
        assert len(suite) == 3
        assert all(s.scenario_type == ScenarioType.CODING for s in suite)

    def test_safety_suite_has_critical(self):
        suite = safety_suite()
        tiers = [s.risk_tier for s in suite]
        assert "critical" in tiers

    def test_builtin_suites_all_present(self):
        assert "coding" in BUILTIN_SUITES
        assert "safety" in BUILTIN_SUITES
        assert "memory" in BUILTIN_SUITES
        assert "planning" in BUILTIN_SUITES


class TestReliabilityMetrics:
    def test_perfect_run(self):
        results = [
            {"success": True, "silent_failure": False, "recovered": False,
             "has_error": False, "verifier_flagged": False, "verifier_correct": False,
             "high_risk_approved_correctly": False, "unnecessary_escalation": False,
             "loop_detected": False, "rollback_successful": None, "tool_calls": 3},
            {"success": True, "silent_failure": False, "recovered": False,
             "has_error": False, "verifier_flagged": False, "verifier_correct": False,
             "high_risk_approved_correctly": False, "unnecessary_escalation": False,
             "loop_detected": False, "rollback_successful": None, "tool_calls": 4},
        ]
        m = compute_reliability_metrics(results)
        assert m.task_success_rate == 1.0
        assert m.silent_failure_rate == 0.0
        assert m.loop_rate == 0.0

    def test_mixed_run(self):
        results = [
            {"success": True, "silent_failure": False, "recovered": False,
             "has_error": False, "verifier_flagged": False, "verifier_correct": False,
             "high_risk_approved_correctly": False, "unnecessary_escalation": False,
             "loop_detected": False, "rollback_successful": None, "tool_calls": 2},
            {"success": False, "silent_failure": True, "recovered": False,
             "has_error": True, "verifier_flagged": False, "verifier_correct": False,
             "high_risk_approved_correctly": False, "unnecessary_escalation": False,
             "loop_detected": False, "rollback_successful": None, "tool_calls": 5},
        ]
        m = compute_reliability_metrics(results)
        assert m.task_success_rate == 0.5
        assert m.silent_failure_rate == 1.0  # the failure was silent

    def test_recovery_tracked(self):
        results = [
            {"success": True, "silent_failure": False, "recovered": True,
             "has_error": True, "verifier_flagged": True, "verifier_correct": True,
             "high_risk_approved_correctly": False, "unnecessary_escalation": False,
             "loop_detected": True, "rollback_successful": None, "tool_calls": 8},
        ]
        m = compute_reliability_metrics(results)
        assert m.recovery_rate == 1.0
        assert m.loop_rate == 1.0
        assert m.n_recoveries == 1

    def test_empty_no_crash(self):
        m = compute_reliability_metrics([])
        assert m.n_episodes == 0

    def test_compare_interventions(self):
        baseline = ReliabilityMetrics(
            task_success_rate=0.7, silent_failure_rate=0.3,
            recovery_rate=0.4, loop_rate=0.15, n_episodes=100,
        )
        intervention = ReliabilityMetrics(
            task_success_rate=0.85, silent_failure_rate=0.1,
            recovery_rate=0.7, loop_rate=0.05, n_episodes=100,
        )
        result = compare_interventions(baseline, intervention)
        assert abs(result["deltas"]["task_success_rate"] - 0.15) < 0.001
        assert abs(result["deltas"]["silent_failure_rate"] - 0.20) < 0.001
        assert "improved" in result["summary"]


class TestReports:
    def test_generate_correlation_report(self):
        baseline = ReliabilityMetrics(task_success_rate=0.7, n_episodes=50)
        intervention = ReliabilityMetrics(task_success_rate=0.75, n_episodes=50)

        pair = InterventionPair(
            name="cssr_vs_baseline",
            baseline=baseline,
            intervention=intervention,
            claim_type="correlation",
        )
        report = generate_report("exp_01", "CSSR correlation test", [pair])
        assert len(report.correlations) == 1
        assert len(report.causal_claims) == 0
        assert "confounds" in report.overall_summary.lower()

    def test_generate_causal_report(self):
        baseline = ReliabilityMetrics(task_success_rate=0.7, n_episodes=50)
        intervention = ReliabilityMetrics(task_success_rate=0.82, n_episodes=50)

        pair = InterventionPair(
            name="policy_vs_observe",
            baseline=baseline, intervention=intervention,
            claim_type="causal",
        )
        report = generate_report("exp_02", "Policy control A/B", [pair])
        assert len(report.causal_claims) == 1
        assert "success" in report.causal_claims[0] and "+12" in report.causal_claims[0]

    def test_report_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = ExperimentReport(experiment_id="exp_03", description="Test")
            report.causal_claims.append("Claim 1")
            path = f"{tmp}/report.json"
            report.save(path)

            loaded = ExperimentReport.load(path)
            assert loaded.experiment_id == "exp_03"
            assert loaded.causal_claims == ["Claim 1"]
