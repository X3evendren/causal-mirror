"""Blackboard — structured, traceable inter-role communication.

All role-to-role communication goes through the blackboard. Nothing is
passed directly between agents. This ensures:
  - Every message is recorded in the event ledger
  - Conflicting claims can be traced to their source
  - The referee has a complete record to arbitrate from
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid

from causal_harness.agents.roles import AgentRole


class EntryType(str, Enum):
    PLAN = "plan"              # planner's proposed action sequence
    ACTION_RESULT = "action_result"  # executor's report
    VERDICT = "verdict"        # verifier's assessment
    RESEARCH = "research"      # researcher's findings
    RULING = "ruling"          # referee's decision
    QUESTION = "question"      # any role asking another for input
    OBJECTION = "objection"    # verifier/referee challenging a result


@dataclass
class BlackboardEntry:
    """A single item on the blackboard."""
    entry_id: str
    entry_type: EntryType
    role: AgentRole
    content: str
    references: list[str] = field(default_factory=list)  # entry_ids this responds to
    evidence: list[str] = field(default_factory=list)   # file paths, test names, URLs
    confidence: float = 0.5
    timestamp: str = ""

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class Blackboard:
    """Thread-safe shared workspace for multi-agent coordination.

    All entries are immutable once posted. Entries can reference each
    other to build a conversation graph that the referee can traverse.
    """

    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self._entries: list[BlackboardEntry] = []
        self._by_type: dict[EntryType, list[BlackboardEntry]] = {t: [] for t in EntryType}

    def post(self, entry: BlackboardEntry) -> str:
        """Post an entry. Returns entry_id."""
        self._entries.append(entry)
        self._by_type[entry.entry_type].append(entry)
        self._trim()
        return entry.entry_id

    def query(
        self,
        entry_type: EntryType | None = None,
        role: AgentRole | None = None,
        limit: int = 20,
    ) -> list[BlackboardEntry]:
        """Query entries by type and/or role, most recent first."""
        results = []
        for entry in reversed(self._entries):
            if entry_type and entry.entry_type != entry_type:
                continue
            if role and entry.role != role:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def get_thread(self, entry_id: str) -> list[BlackboardEntry]:
        """Get an entry and all entries that reference it (recursively)."""
        root = self._find(entry_id)
        if root is None:
            return []
        thread = [root]
        for entry in self._entries:
            if entry_id in entry.references:
                thread.append(entry)
                thread.extend(self.get_thread(entry.entry_id))
        return thread

    def latest_plan(self) -> BlackboardEntry | None:
        plans = self._by_type.get(EntryType.PLAN, [])
        return plans[-1] if plans else None

    def latest_verdict(self) -> BlackboardEntry | None:
        verdicts = self._by_type.get(EntryType.VERDICT, [])
        return verdicts[-1] if verdicts else None

    def objections_pending(self) -> list[BlackboardEntry]:
        """Find objections that haven't been ruled on yet."""
        objections = self._by_type.get(EntryType.OBJECTION, [])
        rulings = {r.references[0] for r in self._by_type.get(EntryType.RULING, [])
                   if r.references}
        return [obj for obj in objections if obj.entry_id not in rulings]

    def clear(self) -> None:
        self._entries.clear()
        for t in EntryType:
            self._by_type[t] = []

    def _find(self, entry_id: str) -> BlackboardEntry | None:
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def _trim(self) -> None:
        while len(self._entries) > self.max_entries:
            removed = self._entries.pop(0)
            self._by_type[removed.entry_type].remove(removed)
