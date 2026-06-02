"""Event schema — typed event definitions for the event ledger.

Each event records observable agent behavior, not hidden chain-of-thought.
The schema is intentionally narrow: it captures what happened, not why.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class EventType(str, Enum):
    """Observable event types in the agent lifecycle."""
    TURN_STARTED = "turn_started"
    GOAL_LOADED = "goal_loaded"
    OBSERVATION_ADDED = "observation_added"
    ACTION_PROPOSED = "action_proposed"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    VERIFICATION_RUN = "verification_run"
    ERROR_DETECTED = "error_detected"
    REPAIR_ATTEMPTED = "repair_attempted"
    USER_APPROVAL_REQUESTED = "user_approval_requested"
    STATE_TRANSITION = "state_transition"
    TURN_COMPLETED = "turn_completed"
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"


class EventOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    PENDING = "pending"


EVENT_TYPES: set[EventType] = set(EventType)


@dataclass
class Event:
    """A single observable event in the agent lifecycle.

    Attributes:
        event_id: Unique event identifier.
        episode_id: Which episode this belongs to.
        turn_id: Which turn within the episode.
        timestamp: UTC timestamp of the event.
        actor: Who/what produced this event (agent, tool, user, system).
        event_type: The type of event.
        payload: Type-specific data (tool call args, error info, etc.).
        risk_before: Numeric risk estimate before the event (0.0–1.0).
        risk_after: Numeric risk estimate after the event (0.0–1.0).
        outcome: The outcome classification.
    """
    event_id: str
    episode_id: str
    turn_id: int
    timestamp: str
    actor: str
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    risk_before: float | None = None
    risk_after: float | None = None
    outcome: EventOutcome = EventOutcome.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "risk_before": self.risk_before,
            "risk_after": self.risk_after,
            "outcome": self.outcome.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        d["event_type"] = EventType(d["event_type"])
        d["outcome"] = EventOutcome(d.get("outcome", "unknown"))
        return cls(**d)


def new_event(
    event_type: EventType,
    episode_id: str,
    turn_id: int,
    actor: str = "agent",
    payload: dict[str, Any] | None = None,
    risk_before: float | None = None,
    risk_after: float | None = None,
    outcome: EventOutcome = EventOutcome.UNKNOWN,
) -> Event:
    """Factory for creating events with auto-generated id and timestamp."""
    return Event(
        event_id=uuid.uuid4().hex[:12],
        episode_id=episode_id,
        turn_id=turn_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor=actor,
        event_type=event_type,
        payload=payload or {},
        risk_before=risk_before,
        risk_after=risk_after,
        outcome=outcome,
    )
