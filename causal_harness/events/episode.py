"""Episode management — lifecycle, boundaries, and state tracking.

An episode is a coherent unit of work: a single task, a user request,
or a benchmark problem. Episode boundaries prevent the CSSR from
treating artificial transitions between unrelated tasks as meaningful.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid

from causal_harness.events.schema import Event, EventType, EventOutcome, new_event


class EpisodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Episode:
    """A single episode — one task or interaction unit.

    Episodes are the unit of analysis for CSSR. Each episode produces
    a discrete symbol sequence with a BOUNDARY marker between episodes
    so that the behavior encoder does not create artificial cross-task
    transitions.
    """
    episode_id: str
    goal: str
    status: EpisodeStatus = EpisodeStatus.PENDING
    turn_count: int = 0
    started_at: str = ""
    completed_at: str = ""
    metadata: dict = field(default_factory=dict)

    def start(self) -> Event:
        self.status = EpisodeStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        return new_event(
            EventType.EPISODE_STARTED,
            episode_id=self.episode_id,
            turn_id=0,
            actor="system",
            payload={"goal": self.goal, **self.metadata},
            outcome=EventOutcome.PENDING,
        )

    def complete(self, outcome: EventOutcome = EventOutcome.SUCCESS) -> Event:
        self.status = (
            EpisodeStatus.COMPLETED if outcome == EventOutcome.SUCCESS
            else EpisodeStatus.FAILED
        )
        self.completed_at = datetime.now(timezone.utc).isoformat()
        return new_event(
            EventType.EPISODE_COMPLETED,
            episode_id=self.episode_id,
            turn_id=self.turn_count,
            actor="system",
            payload={
                "goal": self.goal,
                "turns": self.turn_count,
                "status": self.status.value,
            },
            outcome=outcome,
        )

    def next_turn(self) -> int:
        self.turn_count += 1
        return self.turn_count


class EpisodeManager:
    """Tracks active and completed episodes. Not thread-safe by itself;
    wrap in a lock if used from multiple threads.
    """

    def __init__(self):
        self._active: Episode | None = None
        self._history: dict[str, Episode] = {}

    @property
    def active(self) -> Episode | None:
        return self._active

    def begin_episode(
        self, goal: str, metadata: dict | None = None
    ) -> tuple[Episode, Event, Event | None]:
        """Start a new episode. If one is already active, it is auto-completed.

        Returns:
            (episode, start_event, auto_complete_event) where auto_complete_event
            is the EPISODE_COMPLETED event from the previously active episode,
            or None if no episode was auto-completed.
        """
        auto_complete_event: Event | None = None
        if self._active and self._active.status == EpisodeStatus.RUNNING:
            auto_complete_event = self._active.complete(EventOutcome.UNKNOWN)

        ep = Episode(
            episode_id=uuid.uuid4().hex[:12],
            goal=goal,
            metadata=metadata or {},
        )
        start_event = ep.start()
        self._active = ep
        self._history[ep.episode_id] = ep
        return ep, start_event, auto_complete_event

    def end_episode(self, outcome: EventOutcome = EventOutcome.SUCCESS) -> Event | None:
        if self._active is None:
            return None
        event = self._active.complete(outcome)
        self._history[self._active.episode_id] = self._active
        self._active = None
        return event

    def get(self, episode_id: str) -> Episode | None:
        if self._active and self._active.episode_id == episode_id:
            return self._active
        return self._history.get(episode_id)

    def all_episode_ids(self) -> list[str]:
        seen = set(self._history.keys())
        if self._active:
            seen.add(self._active.episode_id)
        return list(seen)
