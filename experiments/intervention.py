"""Targeted Intervention V2 — minimal, format-preserving, data-driven.

Principles:
  1. NEVER touch the system prompt
  2. Intervention goes in USER message (after problem text)
  3. Max 2 sentences, <80 Chinese characters total
  4. Based on real CSSR behavioral analysis
  5. Self-report saved separately to disk

Four intervention types derived from CSSR epsilon-machine analysis:
  - VERIFY_SKIP: skipping verification correlates with errors
  - DOMAIN_WEAK: accuracy <30% in a specific domain
  - HIGH_ENTROPY: scattered reasoning (h_mu > 1.5)
  - NO_INTERVENTION: healthy behavior patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Intervention text templates (short!) ─────────────────────

INTERVENTION_TEXTS: dict[str, str] = {
    "verify": "[提醒] 请在给出答案前验证计算过程。",
    "domain": "[提醒] 此类问题请特别注意边界条件和特殊情况。",
    "slow": "[提醒] 每步推理写出完整过程，不要跳步。",
    "slow+verify": "[提醒] 每步写完整推理；答案前验证计算。",
}


# ── Detection helpers ────────────────────────────────────────

def _detect_verify_skip(machine) -> bool:
    """Check if any causal state shows 'skip verify → error' pattern."""
    if machine is None or not hasattr(machine, "states"):
        return False

    for state in machine.states:
        # Emission probabilities for this state
        probs = state.emission_probs

        # Count verification-skipped probability
        skip_verify = 0.0
        error_or_partial = 0.0

        for sym, prob in probs.items():
            s = str(sym)
            if "SKIPPED" in s:
                skip_verify += prob
            if "INCORRECT" in s or "PARTIAL" in s:
                error_or_partial += prob

        # Pattern: state where skipping occurs AND errors occur
        if skip_verify > 0 and error_or_partial > 0.3:
            return True

    return False


def _detect_domain_weak(domain_accuracy: dict[str, float],
                        problem_domain: str) -> bool:
    """Check if this problem's domain has low accuracy."""
    if not domain_accuracy:
        return False
    acc = domain_accuracy.get(problem_domain, 1.0)
    return acc < 0.3


def _detect_high_entropy(metrics: dict | None) -> bool:
    """Check if entropy rate indicates scattered reasoning."""
    if metrics is None:
        return False
    return metrics.get("entropy_rate", 0) > 1.5


# ── TargetedIntervention ─────────────────────────────────────

@dataclass
class InterventionResult:
    """Output of the intervention generator."""
    text: str = ""                    # The intervention suffix ("" = no intervention)
    triggers: list[str] = field(default_factory=list)
    skip_verify_detected: bool = False
    domain_weak_detected: bool = False
    high_entropy_detected: bool = False


class TargetedIntervention:
    """Generate minimal, targeted interventions from CSSR analysis.

    Usage:
        ti = TargetedIntervention()
        text = ti.generate(metrics, machine, domain_accuracy, problem)
        # text is either a short reminder string or "" (empty = no intervention)
    """

    def generate(
        self,
        metrics: dict | None,
        machine,                          # EpsilonMachine or None
        domain_accuracy: dict[str, float] | None,
        problem: dict | None = None,
    ) -> str:
        """Generate intervention text for one problem.

        Args:
            metrics: CSSR metrics dict (or None if not yet analyzed).
            machine: EpsilonMachine instance (or None).
            domain_accuracy: Per-domain accuracy from last round.
            problem: The problem dict (for domain targeting).

        Returns:
            Short intervention string to append to user message,
            or "" if no intervention is needed.
        """
        result = self.evaluate(metrics, machine, domain_accuracy, problem)
        return result.text

    def evaluate(
        self,
        metrics: dict | None,
        machine,
        domain_accuracy: dict[str, float] | None,
        problem: dict | None = None,
    ) -> InterventionResult:
        """Full evaluation returning InterventionResult with trigger details."""
        da = domain_accuracy or {}
        domain = problem.get("domain", "") if problem else ""

        result = InterventionResult()

        # Rule 1: verify-skip pattern
        if _detect_verify_skip(machine):
            result.skip_verify_detected = True
            result.triggers.append("verify")

        # Rule 2: domain weakness (per-problem targeting)
        if _detect_domain_weak(da, domain):
            result.domain_weak_detected = True
            result.triggers.append("domain")

        # Rule 3: high entropy
        if _detect_high_entropy(metrics):
            result.high_entropy_detected = True
            result.triggers.append("slow")

        # Build text
        result.text = self._build_text(result.triggers)
        return result

    def _build_text(self, triggers: list[str]) -> str:
        """Build minimal intervention text from trigger list.

        Uses pre-composed templates to keep text short (<80 chars).
        Only uses the first 2 triggers (most important ones).
        """
        if not triggers:
            return ""

        # Deduplicate and limit to 2
        unique = list(dict.fromkeys(triggers))[:2]

        if len(unique) == 1:
            return INTERVENTION_TEXTS.get(unique[0], "")

        # Two triggers: use combined template if available
        key = "+".join(sorted(unique))
        combined = INTERVENTION_TEXTS.get(key)
        if combined:
            return combined

        # Fallback: join individual texts with "；"
        parts = [INTERVENTION_TEXTS.get(t, "") for t in unique]
        parts = [p for p in parts if p]
        return "；".join(parts) if parts else ""


# ── Self-report (saved to disk, NOT injected) ────────────────

def generate_self_report(
    metrics,
    machine,
    domain_accuracy: dict[str, float],
    round_num: int,
) -> str:
    """Generate a comprehensive CSSR analysis report for disk storage.

    This is NOT injected into prompts — it's saved to
    experiments/results/report_A_roundN.md for human analysis.
    """
    lines = [
        f"# Causal Mirror 自我认知报告 — Round {round_num}",
        "",
        "## CSSR 度量",
        "",
        f"- 统计复杂度 C_μ: {metrics.statistical_complexity():.4f} bits",
        f"- 熵率 h_μ: {metrics.entropy_rate():.4f} bits",
        f"- 超熵 E: {metrics.excess_entropy():.4f} bits",
        f"- 预测信息 χ: {metrics.predictive_information():.4f} bits",
        f"- 因果状态数: {len(machine.states)}",
        "",
        "## 因果状态结构",
        "",
    ]

    for s in machine.states:
        probs = ", ".join(
            f"{b}: {p:.2f}" for b, p in
            sorted(s.emission_probs.items(), key=lambda x: -x[1])
        )
        lines.append(f"### 状态 {s.state_id} ({len(s.histories)} 个历史模式)")
        lines.append(f"  输出分布: {{{probs}}}")

        # Error analysis per state
        err_probs = {}
        for sym, prob in s.emission_probs.items():
            s_str = str(sym)
            if "INCORRECT" in s_str or "PARTIAL" in s_str:
                err_probs[s_str] = prob
            elif "SKIPPED" in s_str:
                err_probs[s_str] = prob
        if err_probs:
            err_str = ", ".join(f"{k}: {v:.2f}" for k, v in err_probs.items())
            lines.append(f"  风险标记: {{{err_str}}}")

        lines.append("")

    # Domain analysis
    if domain_accuracy:
        lines.append("## 领域准确率")
        lines.append("")
        domain_names = {
            "math": "高等数学", "algo": "算法理论", "logic": "逻辑组合",
        }
        for d, acc in sorted(domain_accuracy.items()):
            name = domain_names.get(d, d)
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            lines.append(f"- {name:8s} [{bar}] {acc:.0%}")

    lines.append("")
    lines.append("## 干预触发")
    lines.append("")

    ti = TargetedIntervention()
    for d, acc in (domain_accuracy or {}).items():
        result = ti.evaluate(
            {"entropy_rate": metrics.entropy_rate()},
            machine, domain_accuracy, {"domain": d},
        )
        if result.triggers:
            lines.append(f"- {d}: {', '.join(result.triggers)} → `{result.text}`")
        else:
            lines.append(f"- {d}: 无干预触发")

    return "\n".join(lines)
