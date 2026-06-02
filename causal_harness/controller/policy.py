"""Policy controller — translates risk predictions into runtime interventions.

The controller does NOT make the final decision about what action to take.
It emits *constraints* and *suggestions* that the agent loop must respect.
This separation ensures safety policy is independent from the LLM.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from causal_harness.models.base import RiskPrediction


class PolicyResult(str, Enum):
    """What the controller recommends for the next action."""
    ALLOW = "allow"               # proceed normally
    VERIFY_BEFORE = "verify"      # run verification before executing
    INSPECT_FIRST = "inspect"     # gather more info before acting
    ASK_USER = "ask_user"         # escalate to user
    REPLAN = "replan"             # abandon current approach, re-think
    BLOCK = "block"               # action is too dangerous, refuse


@dataclass
class Intervention:
    """A constraint or suggestion injected into the agent loop."""
    result: PolicyResult
    reason: str = ""
    required_action: str = ""   # e.g., "run_tests", "read_file", "ask_user"
    risk_threshold_exceeded: bool = False


@dataclass
class PolicyRule:
    """A single policy rule with an opt-out switch for A/B experiments.

    Each rule can be independently enabled/disabled so the controller
    can be configured for ablation studies.
    """
    name: str
    enabled: bool = True
    description: str = ""


class PolicyController:
    """Evaluates risk predictions against policy rules and returns interventions.

    Usage:
        controller = PolicyController()
        intervention = controller.evaluate(risk_prediction, context)
        if intervention.result != PolicyResult.ALLOW:
            agent.inject(intervention.required_action)
    """

    def __init__(self):
        self.rules: dict[str, PolicyRule] = {}
        self._error_class_counts: dict[str, int] = {}
        self._setup_default_rules()

    def evaluate(
        self,
        risk: RiskPrediction,
        context: dict[str, Any] | None = None,
    ) -> Intervention:
        """Evaluate all enabled rules and return the most restrictive intervention.

        Priority: BLOCK > ASK_USER > VERIFY_BEFORE > INSPECT_FIRST > REPLAN > ALLOW
        """
        ctx = context or {}
        interventions: list[Intervention] = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue
            result = self._evaluate_rule(rule, risk, ctx)
            if result is not None:
                interventions.append(result)

        if not interventions:
            return Intervention(result=PolicyResult.ALLOW)

        return _most_restrictive(interventions)

    def enable_rule(self, name: str) -> None:
        if name in self.rules:
            self.rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        if name in self.rules:
            self.rules[name].enabled = False

    def record_error(self, error_class: str) -> None:
        """Record an error occurrence for repair-switching policy."""
        self._error_class_counts[error_class] = (
            self._error_class_counts.get(error_class, 0) + 1
        )

    def reset_error_counts(self) -> None:
        self._error_class_counts.clear()

    # -- rule evaluation --

    def _setup_default_rules(self) -> None:
        self.rules = {
            "high_risk_verify": PolicyRule(
                "high_risk_verify", True,
                "If P(failure) > 0.60, require verification before execution",
            ),
            "repair_switch": PolicyRule(
                "repair_switch", True,
                "If same error class occurs twice, switch repair strategy",
            ),
            "info_gain_first": PolicyRule(
                "info_gain_first", True,
                "If confidence is low (<0.3), inspect or ask before acting",
            ),
            "loop_detect": PolicyRule(
                "loop_detect", True,
                "If same state-action repeats >= 3 times, re-plan",
            ),
            "high_impact_approval": PolicyRule(
                "high_impact_approval", True,
                "If action is high-risk and irreversible, request user approval",
            ),
            "low_confidence_ask": PolicyRule(
                "low_confidence_ask", True,
                "If confidence is low and ambiguity is high, ask user",
            ),
        }

    def _evaluate_rule(
        self, rule: PolicyRule, risk: RiskPrediction, ctx: dict[str, Any]
    ) -> Intervention | None:
        if rule.name == "high_risk_verify":
            return self._rule_high_risk_verify(risk, ctx)

        if rule.name == "repair_switch":
            return self._rule_repair_switch(risk, ctx)

        if rule.name == "info_gain_first":
            return self._rule_info_gain(risk, ctx)

        if rule.name == "loop_detect":
            return self._rule_loop(risk, ctx)

        if rule.name == "high_impact_approval":
            return self._rule_high_impact(risk, ctx)

        if rule.name == "low_confidence_ask":
            return self._rule_low_confidence(risk, ctx)

        return None

    def _rule_high_risk_verify(self, risk: RiskPrediction, ctx: dict) -> Intervention | None:
        if risk.p_failure > 0.60:
            return Intervention(
                result=PolicyResult.VERIFY_BEFORE,
                reason=f"P(failure)={risk.p_failure:.2f} exceeds 0.60 threshold",
                required_action="run_verification",
                risk_threshold_exceeded=True,
            )
        return None

    def _rule_repair_switch(self, risk: RiskPrediction, ctx: dict) -> Intervention | None:
        error_class = ctx.get("last_error_class", "")
        if error_class and self._error_class_counts.get(error_class, 0) >= 2:
            return Intervention(
                result=PolicyResult.REPLAN,
                reason=f"Error class '{error_class}' repeated {self._error_class_counts[error_class]} times — switch strategy",
                required_action="replan_with_different_approach",
            )
        return None

    def _rule_info_gain(self, risk: RiskPrediction, ctx: dict) -> Intervention | None:
        if risk.confidence < 0.3:
            has_read = ctx.get("has_read_file", False)
            has_searched = ctx.get("has_searched", False)
            if not has_read and not has_searched:
                return Intervention(
                    result=PolicyResult.INSPECT_FIRST,
                    reason=f"Confidence={risk.confidence:.2f}, no inspection done yet",
                    required_action="inspect_or_search",
                )
            return Intervention(
                result=PolicyResult.ASK_USER,
                reason=f"Confidence={risk.confidence:.2f} after inspection — ask user",
                required_action="ask_user",
            )
        return None

    def _rule_loop(self, risk: RiskPrediction, ctx: dict) -> Intervention | None:
        loop_count = ctx.get("loop_count", 0)
        if loop_count >= 3:
            return Intervention(
                result=PolicyResult.REPLAN,
                reason=f"State-action loop detected ({loop_count} repetitions)",
                required_action="summarize_and_replan",
            )
        return None

    def _rule_high_impact(self, risk: RiskPrediction, ctx: dict) -> Intervention | None:
        risk_tier = ctx.get("risk_tier", "low")
        is_reversible = ctx.get("reversible", True)
        if risk_tier in ("high", "critical") and not is_reversible:
            return Intervention(
                result=PolicyResult.ASK_USER,
                reason=f"{risk_tier}-risk irreversible action requires approval",
                required_action="request_user_approval",
            )
        if risk_tier == "critical":
            return Intervention(
                result=PolicyResult.BLOCK,
                reason="Critical-risk action blocked by policy",
                required_action="escalate_to_human",
            )
        return None

    def _rule_low_confidence(self, risk: RiskPrediction, ctx: dict) -> Intervention | None:
        ambiguity = ctx.get("ambiguity", "low")
        if risk.confidence < 0.4 and ambiguity == "high":
            return Intervention(
                result=PolicyResult.ASK_USER,
                reason=f"Low confidence ({risk.confidence:.2f}) + high ambiguity — ask user",
                required_action="ask_user_for_clarification",
            )
        return None


def _most_restrictive(interventions: list[Intervention]) -> Intervention:
    priority = {
        PolicyResult.BLOCK: 0,
        PolicyResult.ASK_USER: 1,
        PolicyResult.VERIFY_BEFORE: 2,
        PolicyResult.INSPECT_FIRST: 3,
        PolicyResult.REPLAN: 4,
        PolicyResult.ALLOW: 5,
    }
    return min(interventions, key=lambda iv: priority.get(iv.result, 99))
