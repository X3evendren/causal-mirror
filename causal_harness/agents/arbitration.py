"""Arbitration — resolve conflicts between agent roles.

The referee role uses the arbitrator to evaluate conflicting claims.
Key principles:
  - Verifier evidence must be INDEPENDENT (not derived from executor output)
  - Prefer verifier when evidence is supplied
  - Tie-breaking based on evidence quality, not role seniority
"""

from dataclasses import dataclass, field
from enum import Enum

from causal_harness.agents.blackboard import Blackboard, BlackboardEntry, EntryType
from causal_harness.agents.roles import AgentRole


class ArbitrationOutcome(str, Enum):
    ACCEPT = "accept"           # accept the proposed action/result
    REJECT = "reject"           # reject, request re-do
    MODIFY = "modify"           # accept with modifications
    ESCALATE = "escalate"       # needs human judgment
    MORE_EVIDENCE = "more_evidence"  # insufficient data to decide


@dataclass
class ArbitrationResult:
    """The referee's decision on a conflict."""
    outcome: ArbitrationOutcome
    reason: str = ""
    accepted_entries: list[str] = field(default_factory=list)
    rejected_entries: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    confidence: float = 0.5


class Arbitrator:
    """Evaluates conflicting claims using evidence quality and independence."""

    def __init__(self, require_independent_evidence: bool = True):
        self.require_independent = require_independent_evidence

    def resolve(
        self, blackboard: Blackboard
    ) -> ArbitrationResult:
        """Resolve all pending objections on the blackboard.

        Returns the most restrictive ruling needed.
        """
        pending = blackboard.objections_pending()
        if not pending:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.ACCEPT,
                reason="No pending objections",
            )

        for obj in pending:
            result = self._resolve_objection(obj, blackboard)
            if result.outcome != ArbitrationOutcome.ACCEPT:
                return result

        return ArbitrationResult(outcome=ArbitrationOutcome.ACCEPT)

    def resolve_disagreement(
        self,
        claim_a: BlackboardEntry,
        claim_b: BlackboardEntry,
    ) -> ArbitrationResult:
        """Resolve a pair of conflicting entries.

        Rules (in priority order):
        1. If verifier has independent evidence → accept verifier
        2. If both have evidence → accept the one with more independent sources
        3. If neither has evidence → request more information
        4. If executor vs planner → accept executor (closer to ground truth)
        """
        # Rule 1: Independent evidence wins
        a_evidence = self._count_independent(claim_a)
        b_evidence = self._count_independent(claim_b)

        if a_evidence > b_evidence:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.ACCEPT,
                reason=f"Accepting {claim_a.role.value}: more independent evidence",
                accepted_entries=[claim_a.entry_id],
                rejected_entries=[claim_b.entry_id],
                confidence=min(0.9, 0.5 + a_evidence * 0.15),
            )
        elif b_evidence > a_evidence:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.ACCEPT,
                reason=f"Accepting {claim_b.role.value}: more independent evidence",
                accepted_entries=[claim_b.entry_id],
                rejected_entries=[claim_a.entry_id],
                confidence=min(0.9, 0.5 + b_evidence * 0.15),
            )

        # Rule 2: Both have same evidence → verifier preference
        if claim_a.role == AgentRole.VERIFIER:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.ACCEPT,
                reason="Verifier claim preferred (same evidence level)",
                accepted_entries=[claim_a.entry_id],
                confidence=0.6,
            )
        if claim_b.role == AgentRole.VERIFIER:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.ACCEPT,
                reason="Verifier claim preferred (same evidence level)",
                accepted_entries=[claim_b.entry_id],
                confidence=0.6,
            )

        # Rule 3: No evidence → request more
        return ArbitrationResult(
            outcome=ArbitrationOutcome.MORE_EVIDENCE,
            reason="Insufficient independent evidence to resolve conflict",
            required_actions=["gather_evidence", "re_verify"],
            confidence=0.2,
        )

    def _resolve_objection(
        self, objection: BlackboardEntry, blackboard: Blackboard
    ) -> ArbitrationResult:
        """Resolve a single objection."""
        if not objection.references:
            return ArbitrationResult(outcome=ArbitrationOutcome.ACCEPT)

        target_id = objection.references[0]
        target = None
        for e in blackboard.query():
            if e.entry_id == target_id:
                target = e
                break

        if target is None:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.ACCEPT,
                reason="Objection target not found",
            )

        # VERIFIER objection against EXECUTOR result
        if objection.role == AgentRole.VERIFIER and target.role == AgentRole.EXECUTOR:
            return self.resolve_disagreement(objection, target)

        # REFEREE objection (highest authority)
        if objection.role == AgentRole.REFEREE:
            return ArbitrationResult(
                outcome=ArbitrationOutcome.REJECT,
                reason="Referee objection — overriding",
                rejected_entries=[target_id],
                accepted_entries=[objection.entry_id],
                confidence=0.95,
            )

        return ArbitrationResult(outcome=ArbitrationOutcome.ACCEPT)

    def _count_independent(self, entry: BlackboardEntry) -> int:
        """Count pieces of evidence not derived from the same role's prior output."""
        if not entry.evidence:
            return 0
        # Evidence that references external sources (files, tests, URLs) counts as independent
        independent = 0
        for ev in entry.evidence:
            if ev.startswith("/") or ev.startswith("http"):
                independent += 2  # external reference is strong
            elif ev.startswith("test:") or ev.endswith("_test"):
                independent += 1  # test result
            else:
                independent += 0  # self-reference or role-internal
        return independent
