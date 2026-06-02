"""Append-only JSONL event ledger.

Thread-safe, corruption-resistant event storage. Each line is a complete
JSON object — one event per line. No line is ever modified after being written.
"""

import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from causal_harness.events.schema import Event, EventType


class EventLedger:
    """Append-only JSONL ledger for agent events.

    Usage:
        ledger = EventLedger("logs/experiment_01")
        ledger.append(event)
        ledger.append(event2)
        for event in ledger.replay():
            ...
    """

    def __init__(self, ledger_dir: str | Path, max_file_mb: int = 100):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_file_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._file_index = self._find_latest_file_index()
        self._current_path = self._make_path(self._file_index)
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def append(self, event: Event) -> str:
        """Append one event to the ledger. Returns the event_id."""
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            self._rotate_if_needed()
            with open(self._current_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self._count += 1
        return event.event_id

    def replay(
        self,
        episode_id: str | None = None,
        event_type: EventType | None = None,
    ) -> Iterator[Event]:
        """Yield events from all ledger files, optionally filtered.

        Args:
            episode_id: If set, only yield events from this episode.
            event_type: If set, only yield events of this type.
        """
        import logging
        log = logging.getLogger(__name__)
        for path in self._sorted_ledger_paths():
            for line in _read_lines(path):
                try:
                    record = json.loads(line)
                    event = Event.from_dict(record)
                    if episode_id and event.episode_id != episode_id:
                        continue
                    if event_type and event.event_type != event_type:
                        continue
                    yield event
                except (json.JSONDecodeError, KeyError) as exc:
                    log.warning(
                        "Skipping corrupted record in %s: %s", path, exc
                    )
                    continue

    def replay_episode(self, episode_id: str) -> list[Event]:
        """Return all events for one episode in chronological order."""
        return list(self.replay(episode_id=episode_id))

    def count_events(self) -> int:
        """Count total events by scanning all files. Slow; prefer len() for runtime."""
        total = 0
        for path in self._sorted_ledger_paths():
            total += sum(1 for _ in _read_lines(path))
        return total

    # -- internal helpers --

    def _sorted_ledger_paths(self) -> list[Path]:
        return sorted(
            self.ledger_dir.glob("ledger_*.jsonl"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    def _make_path(self, index: int) -> Path:
        return self.ledger_dir / f"ledger_{index:04d}.jsonl"

    def _find_latest_file_index(self) -> int:
        paths = self._sorted_ledger_paths()
        if not paths:
            return 0
        return int(paths[-1].stem.split("_")[1])

    def _rotate_if_needed(self) -> None:
        if self._current_path.exists() and self._current_path.stat().st_size >= self._max_bytes:
            self._file_index += 1
            self._current_path = self._make_path(self._file_index)


def _read_lines(path: Path) -> Iterator[str]:
    """Yield non-empty lines from a file, stripping trailing whitespace."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped:
                yield stripped
