"""Memory consolidation — distill event ledger data into typed memories.

Consolidation runs after episodes complete. It extracts:
  - identity: what behaviors led to success/failure
  - preferences: user feedback and constraints
  - project_state: files edited, decisions made, open risks
  - skills: reusable action patterns from successful episodes
"""

from datetime import datetime, timezone
import uuid

from causal_harness.events.schema import Event, EventType, EventOutcome
from causal_harness.memory.schema import MemoryType, MemoryItem
from causal_harness.memory.store import MemoryStore


def consolidate_episode(
    events: list[Event],
    store: MemoryStore,
    episode_goal: str = "",
) -> list[str]:
    """Extract typed memories from an episode's events.

    Returns list of created memory_ids.
    """
    if not events:
        return []

    episode_id = events[0].episode_id
    event_ids = [e.event_id for e in events]
    created: list[str] = []

    # Extract project state: files touched, tools used
    files = _extract_files(events)
    if files:
        item = MemoryItem(
            memory_id="",
            memory_type=MemoryType.PROJECT_STATE,
            content=f"Files modified in episode {episode_id}: {', '.join(files)}",
            source_events=event_ids,
            confidence=0.9,
            tags=["files", "episode", episode_id],
        )
        created.append(store.add(item))

    # Extract skills from successful tool patterns
    success = _episode_success(events)
    if success:
        pattern = _extract_pattern(events)
        if pattern:
            item = MemoryItem(
                memory_id="",
                memory_type=MemoryType.SKILLS,
                content=f"Successful pattern: {' → '.join(pattern)}",
                source_events=event_ids,
                confidence=0.7,
                tags=["skill", "success", episode_id],
            )
            created.append(store.add(item))

    # Extract preferences from user interactions
    approvals = _extract_approvals(events)
    for approval in approvals:
        item = MemoryItem(
            memory_id="",
            memory_type=MemoryType.PREFERENCES,
            content=f"User feedback: {approval}",
            source_events=event_ids,
            confidence=0.8,
            tags=["preference", "user"],
        )
        created.append(store.add(item))

    # Project state: risks and decisions
    errors = [e for e in events if e.event_type == EventType.ERROR_DETECTED]
    for err in errors:
        error_class = err.payload.get("error_class", "unknown")
        item = MemoryItem(
            memory_id="",
            memory_type=MemoryType.PROJECT_STATE,
            content=f"Open risk: {error_class} — {err.payload.get('message', '')[:200]}",
            source_events=[err.event_id],
            confidence=0.8,
            tags=["risk", error_class, episode_id],
        )
        created.append(store.add(item))

    return created


def update_identity_from_outcomes(
    store: MemoryStore,
    total_episodes: int,
    success_rate: float,
    common_errors: list[str],
) -> str | None:
    """Update identity with aggregate performance summary."""
    content = (
        f"Completed {total_episodes} episodes. "
        f"Success rate: {success_rate:.1%}. "
    )
    if common_errors:
        content += f"Common errors: {', '.join(common_errors[:5])}."

    # Check if an identity summary already exists
    existing = store.retrieve("Completed", memory_type=MemoryType.IDENTITY, limit=1)
    if existing:
        return store.update(
            existing[0].memory_id, MemoryType.IDENTITY,
            content=content,
            confidence=min(1.0, 0.5 + success_rate * 0.5),
        ).memory_id if existing[0] else None  # type: ignore

    item = MemoryItem(
        memory_id="",
        memory_type=MemoryType.IDENTITY,
        content=content,
        confidence=min(1.0, 0.5 + success_rate * 0.5),
        tags=["identity", "summary"],
    )
    return store.add(item)


# -- helpers --

def _extract_files(events: list[Event]) -> list[str]:
    files = set()
    for e in events:
        if e.event_type == EventType.TOOL_CALLED:
            target = e.payload.get("target", "") or e.payload.get("file_path", "")
            if target:
                files.add(target)
    return sorted(files)[:20]


def _episode_success(events: list[Event]) -> bool:
    for e in events:
        if e.event_type == EventType.EPISODE_COMPLETED:
            return e.outcome == EventOutcome.SUCCESS
    # Heuristic: no errors → likely success
    return not any(e.event_type == EventType.ERROR_DETECTED for e in events)


def _extract_pattern(events: list[Event]) -> list[str]:
    """Extract the tool call sequence as a skill pattern."""
    tools = []
    for e in events:
        if e.event_type == EventType.TOOL_CALLED:
            name = e.payload.get("tool", "")
            if name and name not in ("ask", "message"):
                tools.append(name)
    return tools[:10]  # cap at 10 steps


def _extract_approvals(events: list[Event]) -> list[str]:
    approvals = []
    for e in events:
        if e.event_type == EventType.USER_APPROVAL_REQUESTED:
            action = e.payload.get("action", "")
            granted = e.payload.get("granted", False)
            if granted:
                approvals.append(f"Approved: {action}")
            else:
                approvals.append(f"Denied: {action}")
    return approvals
