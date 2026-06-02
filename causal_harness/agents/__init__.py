"""Multi-agent organization — role separation with blackboard coordination."""

from causal_harness.agents.roles import AgentRole, RoleRegistry, ROLE_DEFINITIONS
from causal_harness.agents.blackboard import Blackboard, BlackboardEntry, EntryType
from causal_harness.agents.arbitration import Arbitrator, ArbitrationResult
from causal_harness.agents.credit import CreditAssigner, CreditRecord

__all__ = [
    "AgentRole",
    "RoleRegistry",
    "ROLE_DEFINITIONS",
    "Blackboard",
    "BlackboardEntry",
    "EntryType",
    "Arbitrator",
    "ArbitrationResult",
    "CreditAssigner",
    "CreditRecord",
]
