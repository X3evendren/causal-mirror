"""Causal credit assignment — track which role/action changed failure risk.

After each episode, credit is assigned to the roles and actions that
most contributed to the outcome change. This is NOT reward — it's
a diagnostic: "which interventions were followed by risk reduction?"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CreditRecord:
    """A single credit assignment event."""
    record_id: str
    episode_id: str
    role: str           # which role is being credited/blamed
    action: str         # which action was taken
    risk_before: float  # estimated risk before the action
    risk_after: float   # estimated risk after the action
    outcome: int        # 1 = episode ended in failure, 0 = success
    contribution: float # positive = helped, negative = hurt
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class CreditAssigner:
    """Tracks risk changes and assigns credit to roles and actions.

    Credit = -(risk_after - risk_before) * (1 if success else -1)
    Positive credit: action reduced risk AND episode succeeded (or
                    action increased risk AND episode failed → correct diagnosis)
    Negative credit: action reduced risk but episode failed anyway
                    (the fix was insufficient or wrong target)
    """

    def __init__(self):
        self._records: list[CreditRecord] = []
        self._role_totals: dict[str, float] = {}
        self._action_totals: dict[str, float] = {}

    def record(
        self,
        episode_id: str,
        role: str,
        action: str,
        risk_before: float,
        risk_after: float,
        outcome: int,
    ) -> CreditRecord:
        """Record one action and assign credit."""
        delta_risk = risk_after - risk_before
        # Positive: risk decreased. Weight by whether the episode succeeded.
        if outcome == 0:  # success
            contribution = -delta_risk  # risk decrease = positive credit
        else:  # failure
            contribution = delta_risk   # risk decrease that didn't work = negative credit
            # (the action looked good but didn't prevent failure)

        cr = CreditRecord(
            record_id="",
            episode_id=episode_id,
            role=role,
            action=action,
            risk_before=risk_before,
            risk_after=risk_after,
            outcome=outcome,
            contribution=round(contribution, 4),
        )
        self._records.append(cr)
        self._role_totals[role] = self._role_totals.get(role, 0.0) + contribution
        self._action_totals[action] = self._action_totals.get(action, 0.0) + contribution
        return cr

    def role_credit(self, role: str) -> float:
        """Total accumulated credit for a role."""
        return self._role_totals.get(role, 0.0)

    def action_credit(self, action: str) -> float:
        """Total accumulated credit for an action type."""
        return self._action_totals.get(action, 0.0)

    def top_contributors(self, n: int = 5) -> list[tuple[str, float]]:
        """Roles with the highest total credit."""
        ranked = sorted(self._role_totals.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def top_actions(self, n: int = 5) -> list[tuple[str, float]]:
        """Actions with the highest total credit."""
        ranked = sorted(self._action_totals.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def episode_summary(self, episode_id: str) -> dict[str, Any]:
        """Aggregate credit for one episode."""
        ep_records = [r for r in self._records if r.episode_id == episode_id]
        if not ep_records:
            return {}
        by_role: dict[str, float] = {}
        for r in ep_records:
            by_role[r.role] = by_role.get(r.role, 0.0) + r.contribution
        return {
            "episode_id": episode_id,
            "total_actions": len(ep_records),
            "net_credit": sum(r.contribution for r in ep_records),
            "by_role": by_role,
            "outcome": ep_records[0].outcome,
        }

    def reset(self) -> None:
        self._records.clear()
        self._role_totals.clear()
        self._action_totals.clear()
