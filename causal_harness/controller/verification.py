"""Verification gate — determines whether an action needs pre-execution checks.

The gate uses three signals:
1. Risk prediction: P(failure) from the risk model
2. Action risk tier: low/medium/high/critical
3. Reversibility: whether the action can be rolled back
"""

from dataclasses import dataclass
from enum import Enum

from causal_harness.models.base import RiskPrediction


class RiskTier(str, Enum):
    LOW = "low"           # read, search, list
    MEDIUM = "medium"     # write files, run tests, create branches
    HIGH = "high"         # network calls, credential access, broad edits
    CRITICAL = "critical" # destructive operations, external side effects


ACTION_RISK_TIERS: dict[str, RiskTier] = {
    "OBSERVE": RiskTier.LOW,
    "PLAN": RiskTier.LOW,
    "READ_FILE": RiskTier.LOW,
    "EDIT_FILE": RiskTier.MEDIUM,
    "RUN_TEST": RiskTier.MEDIUM,
    "VERIFY": RiskTier.LOW,
    "ERROR": RiskTier.LOW,
    "REPAIR": RiskTier.MEDIUM,
    "ASK_USER": RiskTier.LOW,
    "ESCALATE": RiskTier.HIGH,
    "COMPLETE": RiskTier.LOW,
}


REVERSIBLE_ACTIONS: set[str] = {
    "EDIT_FILE",
    "RUN_TEST",
}


@dataclass
class VerificationDecision:
    """Output of the verification gate."""
    require_verification: bool
    require_approval: bool
    reason: str = ""
    verification_type: str = ""  # "test", "diff", "dry_run", "user_confirm"


class VerificationGate:
    """Decides whether an action requires pre-execution verification or approval."""

    def __init__(
        self,
        risk_threshold: float = 0.60,
        always_verify_tiers: tuple[RiskTier, ...] = (RiskTier.HIGH,),
    ):
        self.risk_threshold = risk_threshold
        self.always_verify_tiers = set(always_verify_tiers)

    def evaluate(
        self,
        risk: RiskPrediction,
        action_symbol: str,
        is_reversible: bool | None = None,
    ) -> VerificationDecision:
        """Decide whether `action_symbol` needs verification before execution.

        Args:
            risk: The risk model's prediction.
            action_symbol: Layer A behavior symbol.
            is_reversible: Override reversibility; defaults to lookup table.

        Returns:
            A VerificationDecision with required checks.
        """
        tier = ACTION_RISK_TIERS.get(action_symbol, RiskTier.MEDIUM)
        reversible = (
            is_reversible if is_reversible is not None
            else action_symbol in REVERSIBLE_ACTIONS
        )

        # Critical tier: always block without approval
        if tier == RiskTier.CRITICAL:
            return VerificationDecision(
                require_verification=True,
                require_approval=True,
                reason="Critical-risk action — approval required",
                verification_type="user_confirm",
            )

        # High tier: always verify
        if tier in self.always_verify_tiers:
            return VerificationDecision(
                require_verification=True,
                require_approval=not reversible,
                reason=f"High-risk action '{action_symbol}' requires verification",
                verification_type=self._pick_verification(action_symbol),
            )

        # Medium tier: verify if risk model says so
        if tier == RiskTier.MEDIUM and risk.p_failure > self.risk_threshold:
            return VerificationDecision(
                require_verification=True,
                require_approval=False,
                reason=(
                    f"P(failure)={risk.p_failure:.2f} > {self.risk_threshold} "
                    f"for medium-risk action '{action_symbol}'"
                ),
                verification_type=self._pick_verification(action_symbol),
            )

        # Low tier: no verification needed
        return VerificationDecision(
            require_verification=False,
            require_approval=False,
            reason=f"Low-risk action '{action_symbol}' — no verification required",
        )

    def _pick_verification(self, action: str) -> str:
        if action in ("EDIT_FILE", "REPAIR"):
            return "test"
        if action == "RUN_TEST":
            return "dry_run"
        return "diff"
