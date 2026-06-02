"""Experiment report generation — causal claims with ablations.

Reports separate causal claims from correlations. A claim may be called
"causal" only when:
  1. The intervention was defined before the run
  2. The control group is matched
  3. Confounds (report length, extra attention) are controlled
  4. Logs show the intervention changed a concrete action
  5. The changed action plausibly caused the outcome difference
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

from causal_harness.eval.metrics import ReliabilityMetrics


@dataclass
class InterventionPair:
    """One A/B comparison: baseline vs intervention."""
    name: str
    baseline: ReliabilityMetrics
    intervention: ReliabilityMetrics
    baseline_label: str = "observe_only"
    intervention_label: str = "policy_control"
    claim_type: str = "correlation"  # "correlation" or "causal"

    def delta(self, metric: str) -> float:
        b = getattr(self.baseline, metric, 0.0)
        i = getattr(self.intervention, metric, 0.0)
        return i - b


@dataclass
class ExperimentReport:
    """A structured experiment report with claims and ablations."""
    experiment_id: str
    created_at: str = ""
    description: str = ""
    pairs: list[InterventionPair] = field(default_factory=list)
    overall_summary: str = ""
    causal_claims: list[str] = field(default_factory=list)
    correlations: list[str] = field(default_factory=list)
    ablations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def add_pair(self, pair: InterventionPair) -> None:
        self.pairs.append(pair)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "description": self.description,
            "pairs": [
                {
                    "name": p.name,
                    "baseline_label": p.baseline_label,
                    "intervention_label": p.intervention_label,
                    "claim_type": p.claim_type,
                    "baseline": p.baseline.to_dict(),
                    "intervention": p.intervention.to_dict(),
                }
                for p in self.pairs
            ],
            "overall_summary": self.overall_summary,
            "causal_claims": self.causal_claims,
            "correlations": self.correlations,
            "ablations": self.ablations,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentReport":
        with open(path, "r") as f:
            data = json.load(f)
        report = cls(
            experiment_id=data["experiment_id"],
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )
        for p_data in data.get("pairs", []):
            pair = InterventionPair(
                name=p_data["name"],
                baseline=ReliabilityMetrics(**p_data["baseline"]),
                intervention=ReliabilityMetrics(**p_data["intervention"]),
                baseline_label=p_data.get("baseline_label", ""),
                intervention_label=p_data.get("intervention_label", ""),
                claim_type=p_data.get("claim_type", "correlation"),
            )
            report.pairs.append(pair)
        report.overall_summary = data.get("overall_summary", "")
        report.causal_claims = data.get("causal_claims", [])
        report.correlations = data.get("correlations", [])
        report.ablations = data.get("ablations", [])
        return report


def generate_report(
    experiment_id: str,
    description: str,
    pairs: list[InterventionPair],
    ablations: list[str] | None = None,
) -> ExperimentReport:
    """Generate an experiment report with separated causal claims and correlations.

    This is intentionally a structured-data generator, not an LLM prose generator.
    The report can be passed to an LLM for natural-language summarization later.
    """
    report = ExperimentReport(
        experiment_id=experiment_id,
        description=description,
        pairs=pairs,
        ablations=ablations or [],
    )

    for pair in pairs:
        if pair.claim_type == "causal":
            report.causal_claims.append(
                f"[{pair.name}] {pair.intervention_label} vs {pair.baseline_label}: "
                f"success Δ={pair.delta('task_success_rate'):+.2%}, "
                f"silent failure Δ={pair.delta('silent_failure_rate'):+.2%}, "
                f"recovery Δ={pair.delta('recovery_rate'):+.2%}"
            )
        else:
            report.correlations.append(
                f"[{pair.name}] {pair.intervention_label} is correlated with "
                f"success Δ={pair.delta('task_success_rate'):+.2%} "
                f"(confounds not controlled)"
            )

    # Overall summary
    if not report.causal_claims:
        report.overall_summary = (
            "No causal claims are made in this report. "
            "All observed differences may be explained by confounds "
            "(prompt length, extra attention, task ordering). "
            "To make a causal claim, define the intervention before the run, "
            "use matched controls, and control for known confounds."
        )
    else:
        report.overall_summary = (
            f"{len(report.causal_claims)} causal claim(s) and "
            f"{len(report.correlations)} correlation(s) recorded. "
            f"Ablations: {len(report.ablations)}."
        )

    return report
