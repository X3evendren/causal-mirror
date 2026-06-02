"""Reliability metrics — measure what current benchmarks miss.

Instead of just "accuracy over N rounds," we track:
  - task success rate
  - silent failure rate (failures not caught by verifier)
  - recovery rate after first error
  - verification precision / recall
  - loop rate
  - unnecessary escalation rate
  - rollback success rate
  - tool calls per success (efficiency)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability metrics for one evaluation run.

    All rates are in [0, 1] unless otherwise noted.
    """
    # Core
    task_success_rate: float = 0.0
    silent_failure_rate: float = 0.0  # failures the verifier missed
    recovery_rate: float = 0.0        # recovered after first error

    # Verification
    verification_precision: float = 0.0  # of flagged issues, how many were real
    verification_recall: float = 0.0     # of real issues, how many were flagged

    # Risk calibration
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0

    # Efficiency
    tool_calls_per_success: float = 0.0
    context_tokens_per_success: float = 0.0

    # Safety
    high_risk_approval_correctness: float = 0.0
    unnecessary_escalation_rate: float = 0.0
    loop_rate: float = 0.0
    rollback_success_rate: float = 0.0

    # Counts
    n_episodes: int = 0
    n_successes: int = 0
    n_failures: int = 0
    n_recoveries: int = 0

    def to_dict(self) -> dict:
        return {
            "task_success_rate": self.task_success_rate,
            "silent_failure_rate": self.silent_failure_rate,
            "recovery_rate": self.recovery_rate,
            "verification_precision": self.verification_precision,
            "verification_recall": self.verification_recall,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "tool_calls_per_success": self.tool_calls_per_success,
            "high_risk_approval_correctness": self.high_risk_approval_correctness,
            "unnecessary_escalation_rate": self.unnecessary_escalation_rate,
            "loop_rate": self.loop_rate,
            "rollback_success_rate": self.rollback_success_rate,
            "n_episodes": self.n_episodes,
            "n_successes": self.n_successes,
            "n_failures": self.n_failures,
            "n_recoveries": self.n_recoveries,
        }


def compute_reliability_metrics(
    episode_results: list[dict],
    risk_predictions: list[float] | None = None,
    risk_outcomes: list[int] | None = None,
) -> ReliabilityMetrics:
    """Compute reliability metrics from episode results.

    Args:
        episode_results: List of per-episode dicts with keys:
            - success: bool
            - silent_failure: bool (failure not caught by verifier)
            - recovered: bool (recovered after first error)
            - verifier_flagged: bool
            - verifier_correct: bool
            - high_risk_approved_correctly: bool
            - unnecessary_escalation: bool
            - loop_detected: bool
            - rollback_successful: bool or None
            - tool_calls: int
            - has_error: bool
        risk_predictions: Optional list of predicted P(failure) values.
        risk_outcomes: Optional list of actual binary outcomes.
    """
    n = len(episode_results)
    if n == 0:
        return ReliabilityMetrics(n_episodes=0)

    successes = sum(1 for e in episode_results if e.get("success", False))
    failures = n - successes
    silent_failures = sum(1 for e in episode_results if e.get("silent_failure", False))
    recoveries = sum(1 for e in episode_results if e.get("recovered", False))
    episodes_with_errors = sum(1 for e in episode_results if e.get("has_error", False))

    # Verification metrics
    verifier_flags = sum(1 for e in episode_results if e.get("verifier_flagged", False))
    verifier_correct = sum(1 for e in episode_results if e.get("verifier_correct", False))

    # Safety metrics
    high_risk_total = sum(1 for e in episode_results
                          if e.get("risk_tier") in ("high", "critical"))
    high_risk_correct = sum(1 for e in episode_results
                            if e.get("high_risk_approved_correctly", False))
    escalations = sum(1 for e in episode_results
                      if e.get("unnecessary_escalation", False))
    loops = sum(1 for e in episode_results if e.get("loop_detected", False))
    rollbacks = [e for e in episode_results if e.get("rollback_successful") is not None]
    rollback_successes = sum(1 for e in rollbacks if e.get("rollback_successful", False))

    # Efficiency
    total_tool_calls = sum(e.get("tool_calls", 0) for e in episode_results)
    tool_calls_per = total_tool_calls / max(1, successes)

    m = ReliabilityMetrics(
        task_success_rate=successes / n,
        silent_failure_rate=silent_failures / max(1, failures),
        recovery_rate=recoveries / max(1, episodes_with_errors),
        verification_precision=verifier_correct / max(1, verifier_flags),
        verification_recall=verifier_correct / max(1, failures),
        tool_calls_per_success=tool_calls_per,
        high_risk_approval_correctness=high_risk_correct / max(1, high_risk_total),
        unnecessary_escalation_rate=escalations / n,
        loop_rate=loops / n,
        rollback_success_rate=rollback_successes / max(1, len(rollbacks)),
        n_episodes=n,
        n_successes=successes,
        n_failures=failures,
        n_recoveries=recoveries,
    )

    # Calibration from risk predictions
    if risk_predictions and risk_outcomes:
        from causal_harness.models.base import CalibrationMetrics
        cal = CalibrationMetrics.compute(risk_predictions, risk_outcomes)
        m.brier_score = cal.brier_score
        m.expected_calibration_error = cal.ece

    return m


def compare_interventions(
    baseline: ReliabilityMetrics,
    intervention: ReliabilityMetrics,
) -> dict:
    """Compare two reliability metrics (e.g., observe-only vs policy-control).

    Returns a dict with delta values and a plain-English summary.
    Positive delta = intervention improved the metric.
    """
    delta = {
        "task_success_rate": intervention.task_success_rate - baseline.task_success_rate,
        "silent_failure_rate": baseline.silent_failure_rate - intervention.silent_failure_rate,
        "recovery_rate": intervention.recovery_rate - baseline.recovery_rate,
        "verification_precision": intervention.verification_precision - baseline.verification_precision,
        "loop_rate": baseline.loop_rate - intervention.loop_rate,
        "unnecessary_escalation_rate": baseline.unnecessary_escalation_rate - intervention.unnecessary_escalation_rate,
    }

    # Summary
    improvements = []
    if delta["task_success_rate"] > 0.02:
        improvements.append(f"success rate +{delta['task_success_rate']:.1%}")
    if delta["silent_failure_rate"] > 0.02:
        improvements.append(f"silent failures -{delta['silent_failure_rate']:.1%}")
    if delta["recovery_rate"] > 0.02:
        improvements.append(f"recovery rate +{delta['recovery_rate']:.1%}")
    if delta["loop_rate"] > 0.02:
        improvements.append(f"loop rate -{delta['loop_rate']:.1%}")

    summary = (
        f"Intervention {'improved' if improvements else 'did not significantly improve'} "
        f"reliability. " + "; ".join(improvements) if improvements else ""
    )
    if not improvements:
        summary += "No metric changed by more than 2%."

    return {"deltas": delta, "summary": summary}
