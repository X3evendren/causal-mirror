"""Typed memory items — identity, preferences, project state, skills.

Every memory item links back to its source events via event_ids.
Stale or contradicted items are superseded, not blindly accumulated.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    IDENTITY = "identity"          # stable agent operating principles
    PREFERENCES = "preferences"    # user preferences and constraints
    PROJECT_STATE = "project_state"  # files, decisions, risks, milestones
    SKILLS = "skills"              # reusable procedures from successful episodes


@dataclass
class MemoryItem:
    """A single typed memory record with provenance tracking.

    Attributes:
        memory_id: Unique identifier.
        memory_type: Which store this belongs to.
        content: The knowledge payload (text, structured data, etc.).
        source_events: Event IDs that produced this memory.
        confidence: How certain this memory is (0–1).
        created_at: When this was first recorded.
        updated_at: When last modified.
        superseded_by: If this item was replaced, points to the replacement.
        last_accessed: When last retrieved (for decay/eviction).
        access_count: How many times this has been retrieved.
        tags: Searchable tags for retrieval.
    """
    memory_id: str
    memory_type: MemoryType
    content: str = ""
    source_events: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = ""
    updated_at: str = ""
    superseded_by: str | None = None
    last_accessed: str = ""
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def touch(self) -> None:
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        self.access_count += 1

    def supersede(self, replacement_id: str) -> None:
        self.superseded_by = replacement_id

    def is_active(self) -> bool:
        return self.superseded_by is None


def memory_item_to_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "memory_id": item.memory_id,
        "memory_type": item.memory_type.value,
        "content": item.content,
        "source_events": item.source_events,
        "confidence": item.confidence,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "superseded_by": item.superseded_by,
        "last_accessed": item.last_accessed,
        "access_count": item.access_count,
        "tags": item.tags,
    }


def memory_item_from_dict(d: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        memory_id=d["memory_id"],
        memory_type=MemoryType(d["memory_type"]),
        content=d.get("content", ""),
        source_events=d.get("source_events", []),
        confidence=d.get("confidence", 1.0),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
        superseded_by=d.get("superseded_by"),
        last_accessed=d.get("last_accessed", ""),
        access_count=d.get("access_count", 0),
        tags=d.get("tags", []),
    )
