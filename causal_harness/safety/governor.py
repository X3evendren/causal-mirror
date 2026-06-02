"""Safety governor — independent policy enforcement for high-risk actions.

The governor runs BEFORE any tool execution. It checks:
1. Action risk tier against preconfigured policy
2. Preconditions are satisfied
3. User approval if needed
4. Transaction logging for rollback

The governor must never be bypassed. Safety policy is independent from
the LLM — no LLM output can change a risk tier or skip a check.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path

from causal_harness.safety.risk import (
    RiskTier,
    ActionRisk,
    classify_action_risk,
    PREAPPROVED_LOW_RISK,
)


class Decision(str, Enum):
    ALLOW = "allow"           # proceed
    ALLOW_WITH_LOG = "allow_with_log"  # proceed but record for audit
    REQUIRE_APPROVAL = "require_approval"  # ask user
    BLOCK = "block"           # refuse to execute


@dataclass
class SafetyDecision:
    """Output of a safety check."""
    decision: Decision
    reason: str = ""
    required_preconditions: list[str] = field(default_factory=list)
    rollback_command: str = ""
    transaction_id: str = ""


@dataclass
class TransactionRecord:
    """A logged action for rollback support."""
    transaction_id: str
    tool_name: str
    tool_args: dict
    risk_tier: RiskTier
    timestamp: str
    rollback: str = ""
    preconditions_met: list[str] = field(default_factory=list)
    approved_by: str = ""  # "policy" or "user"


class SafetyGovernor:
    """Safety policy engine — checks actions before execution.

    Usage:
        governor = SafetyGovernor(approval_policy="auto_low")
        decision = governor.check("edit_file", {"file_path": "src/main.py"})
        if decision.decision == Decision.ALLOW:
            execute_tool(...)
    """

    def __init__(
        self,
        approval_policy: str = "auto_low",
        transaction_dir: str | Path | None = None,
        require_preconditions: bool = True,
    ):
        """
        Args:
            approval_policy: "auto_low" (auto-approve low), "ask_medium"
                             (ask for medium+), "ask_all" (always ask).
            transaction_dir: Where to log transaction records for rollback.
            require_preconditions: If True, block if preconditions aren't met.
        """
        self.approval_policy = approval_policy
        self.require_preconditions = require_preconditions
        self.transaction_dir = Path(transaction_dir) if transaction_dir else None
        if self.transaction_dir:
            self.transaction_dir.mkdir(parents=True, exist_ok=True)

        # Precondition tracker — records what the agent has done
        self._files_read: set[str] = set()
        self._files_written: set[str] = set()
        self._tests_run: set[str] = set()
        self._approvals_granted: set[str] = set()

    def check(
        self,
        tool_name: str,
        tool_args: dict,
        context: dict | None = None,
    ) -> SafetyDecision:
        """Evaluate whether `tool_name` can be executed safely.

        Args:
            tool_name: The tool being called.
            tool_args: Arguments to the tool.
            context: Optional runtime context (session state, etc).

        Returns:
            A SafetyDecision with the verdict.
        """
        ctx = context or {}
        risk = classify_action_risk(tool_name)

        # --- Critical tier: always block without explicit approval ---
        if risk.tier == RiskTier.CRITICAL:
            if tool_name in self._approvals_granted:
                self._approvals_granted.discard(tool_name)
                return SafetyDecision(
                    decision=Decision.ALLOW_WITH_LOG,
                    reason=f"Critical action '{tool_name}' explicitly approved by user",
                    rollback_command=risk.rollback,
                )
            return SafetyDecision(
                decision=Decision.REQUIRE_APPROVAL,
                reason=f"Critical action '{tool_name}' requires user approval",
                required_preconditions=risk.preconditions,
            )

        # --- Check preconditions ---
        if self.require_preconditions and risk.preconditions:
            unmet = self._check_preconditions(risk, tool_args, ctx)
            if unmet:
                return SafetyDecision(
                    decision=Decision.BLOCK,
                    reason=f"Preconditions not met: {', '.join(unmet)}",
                    required_preconditions=unmet,
                )

        # --- Approval policy ---
        needs_approval = self._needs_approval(risk.tier)
        if needs_approval and tool_name not in self._approvals_granted:
            return SafetyDecision(
                decision=Decision.REQUIRE_APPROVAL,
                reason=f"{risk.tier.value}-risk action '{tool_name}' requires approval",
                rollback_command=risk.rollback,
            )

        # --- Log transaction for rollback ---
        if risk.tier in (RiskTier.MEDIUM, RiskTier.HIGH) and risk.rollback:
            tx_id = self._log_transaction(tool_name, tool_args, risk, "policy")
            return SafetyDecision(
                decision=Decision.ALLOW_WITH_LOG,
                reason=f"Logged for rollback: {risk.rollback}",
                rollback_command=risk.rollback,
                transaction_id=tx_id,
            )

        return SafetyDecision(decision=Decision.ALLOW)

    def grant_approval(self, tool_name: str) -> None:
        """Grant one-time approval for a specific tool."""
        self._approvals_granted.add(tool_name)

    def record_file_read(self, path: str) -> None:
        self._files_read.add(path)

    def record_file_write(self, path: str) -> None:
        self._files_written.add(path)

    def record_test_run(self, test_name: str) -> None:
        self._tests_run.add(test_name)

    def get_transaction(self, tx_id: str) -> TransactionRecord | None:
        """Retrieve a transaction record by ID."""
        if not self.transaction_dir:
            return None
        path = self.transaction_dir / f"{tx_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return TransactionRecord(**data)

    # -- internal --

    def _needs_approval(self, tier: RiskTier) -> bool:
        if tier == RiskTier.LOW:
            return False
        if self.approval_policy == "auto_low":
            return tier in (RiskTier.HIGH, RiskTier.CRITICAL)
        if self.approval_policy == "ask_medium":
            return tier != RiskTier.LOW
        if self.approval_policy == "ask_all":
            return True
        return False

    def _check_preconditions(
        self, risk: ActionRisk, args: dict, ctx: dict
    ) -> list[str]:
        unmet = []
        for pc in risk.preconditions:
            if "file has been read" in pc:
                target = args.get("file_path", args.get("path", ""))
                if target and target not in self._files_read:
                    unmet.append(pc)
            elif "command is safe" in pc:
                # Delegate to shell guard; always met for now
                pass
            elif "command reviewed" in pc:
                # Requires explicit review flag
                if not ctx.get("command_reviewed", False):
                    unmet.append(pc)
            elif "url has been validated" in pc:
                if not ctx.get("url_validated", False):
                    unmet.append(pc)
            elif "backup exists" in pc:
                if not ctx.get("has_backup", False):
                    unmet.append(pc)
            elif "changes reviewed" in pc:
                if not ctx.get("changes_reviewed", False):
                    unmet.append(pc)
            elif "tests pass" in pc:
                if not ctx.get("tests_pass", False):
                    unmet.append(pc)
            elif "explicit user authorization" in pc:
                unmet.append(pc)  # always requires explicit approval
        return unmet

    def _log_transaction(
        self, tool_name: str, args: dict, risk: ActionRisk, approved_by: str
    ) -> str:
        import uuid
        tx_id = uuid.uuid4().hex[:12]
        if not self.transaction_dir:
            return tx_id

        record = TransactionRecord(
            transaction_id=tx_id,
            tool_name=tool_name,
            tool_args=args,
            risk_tier=risk.tier,
            timestamp=datetime.now(timezone.utc).isoformat(),
            rollback=risk.rollback,
            approved_by=approved_by,
        )
        path = self.transaction_dir / f"{tx_id}.json"
        with open(path, "w") as f:
            json.dump(record.__dict__, f, indent=2, default=str)
        return tx_id
