"""Action risk classification — tiers, preconditions, and reversibility.

Risk tiers are defined independently from the LLM. No LLM output
can override the risk tier of an action.
"""

from dataclasses import dataclass
from enum import Enum


class RiskTier(str, Enum):
    LOW = "low"           # read, search, list, summarize
    MEDIUM = "medium"     # write files, run tests, create branches
    HIGH = "high"         # network calls, credential access, broad edits
    CRITICAL = "critical" # destructive operations, external side effects


@dataclass
class ActionRisk:
    """Risk profile for a single action type."""
    tier: RiskTier
    reversible: bool = True
    preconditions: list[str] = None  # type: ignore
    verification: str = ""          # "test", "diff", "dry_run", "user_confirm"
    rollback: str = ""              # How to undo this action

    def __post_init__(self):
        if self.preconditions is None:
            self.preconditions = []


# --- Action risk registry ---

ACTION_RISKS: dict[str, ActionRisk] = {
    "read_file": ActionRisk(
        RiskTier.LOW, reversible=True,
        preconditions=[],
        verification="",
    ),
    "search": ActionRisk(
        RiskTier.LOW, reversible=True,
    ),
    "list_files": ActionRisk(
        RiskTier.LOW, reversible=True,
    ),
    "web_search": ActionRisk(
        RiskTier.LOW, reversible=True,
    ),
    "web_fetch": ActionRisk(
        RiskTier.MEDIUM, reversible=True,
        preconditions=["url has been validated"],
    ),
    "edit_file": ActionRisk(
        RiskTier.MEDIUM, reversible=True,
        preconditions=["file has been read recently"],
        verification="test",
        rollback="reverse patch",
    ),
    "write_file": ActionRisk(
        RiskTier.MEDIUM, reversible=True,
        preconditions=["target directory exists"],
        verification="test",
        rollback="delete file",
    ),
    "shell": ActionRisk(
        RiskTier.MEDIUM, reversible=False,
        preconditions=["command is safe for filesystem"],
        verification="dry_run",
    ),
    "exec": ActionRisk(
        RiskTier.HIGH, reversible=False,
        preconditions=["command reviewed", "no destructive flags"],
        verification="dry_run",
    ),
    "message": ActionRisk(
        RiskTier.HIGH, reversible=False,
        preconditions=["message content reviewed"],
        verification="user_confirm",
    ),
    "cron": ActionRisk(
        RiskTier.HIGH, reversible=False,
        preconditions=["schedule is intentional"],
        rollback="delete cron job",
    ),
    "sandbox": ActionRisk(
        RiskTier.HIGH, reversible=True,
        preconditions=["sandbox policy enforced"],
    ),
    "spawn": ActionRisk(
        RiskTier.HIGH, reversible=False,
        preconditions=["subagent scope is bounded"],
    ),
    # Destructive operations
    "delete_file": ActionRisk(
        RiskTier.CRITICAL, reversible=False,
        preconditions=["file is safe to delete", "backup exists"],
        verification="user_confirm",
    ),
    "git_push": ActionRisk(
        RiskTier.CRITICAL, reversible=False,
        preconditions=["changes reviewed", "tests pass"],
        verification="user_confirm",
    ),
    "git_force_push": ActionRisk(
        RiskTier.CRITICAL, reversible=False,
        preconditions=["explicit user authorization"],
        verification="user_confirm",
    ),
}

# Low-risk actions that never need approval
PREAPPROVED_LOW_RISK: set[str] = {
    "read_file", "search", "list_files", "web_search",
}


def classify_action_risk(tool_name: str) -> ActionRisk:
    """Get the risk profile for a tool. Unknown tools default to MEDIUM."""
    return ACTION_RISKS.get(tool_name, ActionRisk(RiskTier.MEDIUM))
