"""Event ledger — the source of truth for observable agent behavior."""

from causal_harness.events.schema import (
    Event,
    EventType,
    EventOutcome,
    new_event,
    EVENT_TYPES,
)
from causal_harness.events.ledger import EventLedger
from causal_harness.events.episode import Episode, EpisodeManager
from causal_harness.events.adapters import (
    classify_error,
    TOOL_CATEGORY_MAP,
    tool_event,
    tool_result_event,
    error_event,
    verification_event,
    approval_event,
)

__all__ = [
    "Event",
    "EventType",
    "EventOutcome",
    "new_event",
    "EVENT_TYPES",
    "EventLedger",
    "Episode",
    "EpisodeManager",
    "classify_error",
    "TOOL_CATEGORY_MAP",
    "tool_event",
    "tool_result_event",
    "error_event",
    "verification_event",
    "approval_event",
]
