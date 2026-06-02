"""Fixed agent roles — planner, executor, verifier, researcher, referee.

Roles are FIXED, not dynamically assigned. Each role has a bounded
responsibility, an allowed tool set, and output constraints.

Roles communicate through the blackboard (event ledger), NOT through
free-form agent-to-agent conversation. This prevents uncontrolled chatter
and makes every communication traceable.
"""

from dataclasses import dataclass, field
from enum import Enum


class AgentRole(str, Enum):
    PLANNER = "planner"        # decompose task, propose action sequence
    EXECUTOR = "executor"      # execute proposed actions (read, edit, run)
    VERIFIER = "verifier"      # independently check executor results
    RESEARCHER = "researcher"  # search, read docs, gather context
    REFEREE = "referee"        # resolve conflicts, arbitrate disputes


@dataclass
class RoleDefinition:
    """What a role can do, what it produces, and what tools it uses."""
    role: AgentRole
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    output_type: str = ""  # "plan", "action", "verdict", "research", "ruling"
    requires_evidence: bool = False
    can_disagree: bool = False  # whether this role can challenge others

    def may_use(self, tool_name: str) -> bool:
        if not self.allowed_tools:
            return True  # no restrictions
        return tool_name in self.allowed_tools


ROLE_DEFINITIONS: dict[AgentRole, RoleDefinition] = {
    AgentRole.PLANNER: RoleDefinition(
        role=AgentRole.PLANNER,
        description="Decompose goals into action sequences. Propose, don't execute.",
        allowed_tools=["read_file", "search", "web_search", "list_files"],
        output_type="plan",
        can_disagree=False,
    ),
    AgentRole.EXECUTOR: RoleDefinition(
        role=AgentRole.EXECUTOR,
        description="Execute planned actions. Report results honestly.",
        allowed_tools=["read_file", "edit_file", "write_file", "shell",
                        "exec", "web_fetch", "notebook", "sandbox"],
        output_type="action",
        can_disagree=False,
    ),
    AgentRole.VERIFIER: RoleDefinition(
        role=AgentRole.VERIFIER,
        description="Independently verify executor results. Use independent evidence.",
        allowed_tools=["read_file", "shell", "search", "web_search",
                        "list_files"],
        output_type="verdict",
        requires_evidence=True,
        can_disagree=True,
    ),
    AgentRole.RESEARCHER: RoleDefinition(
        role=AgentRole.RESEARCHER,
        description="Gather context, search documentation, read relevant files.",
        allowed_tools=["read_file", "search", "web_search", "web_fetch",
                        "list_files"],
        output_type="research",
        can_disagree=False,
    ),
    AgentRole.REFEREE: RoleDefinition(
        role=AgentRole.REFEREE,
        description="Resolve conflicts between roles. Final authority on disagreements.",
        allowed_tools=["read_file", "search", "list_files"],
        output_type="ruling",
        requires_evidence=True,
        can_disagree=True,
    ),
}


class RoleRegistry:
    """Look up role definitions and enforce tool access control."""

    def __init__(self):
        self._active_roles: dict[AgentRole, str] = {}  # role -> instance_id

    def register(self, role: AgentRole, instance_id: str) -> None:
        if role in self._active_roles:
            raise ValueError(f"Role {role.value} already assigned to {self._active_roles[role]}")
        self._active_roles[role] = instance_id

    def unregister(self, role: AgentRole) -> None:
        self._active_roles.pop(role, None)

    def is_active(self, role: AgentRole) -> bool:
        return role in self._active_roles

    def get_active(self) -> dict[AgentRole, str]:
        return dict(self._active_roles)

    def check_tool_access(self, role: AgentRole, tool_name: str) -> bool:
        definition = ROLE_DEFINITIONS.get(role)
        if definition is None:
            return False
        return definition.may_use(tool_name)

    def required_roles_for_task(self, task_complexity: str = "medium") -> list[AgentRole]:
        """Minimum roles needed for a given task complexity."""
        if task_complexity == "simple":
            return [AgentRole.EXECUTOR]
        if task_complexity == "medium":
            return [AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.VERIFIER]
        # complex
        return [
            AgentRole.RESEARCHER,
            AgentRole.PLANNER,
            AgentRole.EXECUTOR,
            AgentRole.VERIFIER,
            AgentRole.REFEREE,
        ]
