"""Loop detection — identify repeated state-action patterns.

When an agent gets stuck, it often repeats the same (state, action) pairs.
This module tracks recent behavior and raises an alarm when a loop is detected.
"""

from collections import deque


class LoopDetector:
    """Detects repeated state-action patterns in the recent event stream.

    Tracks the last N (state_id, action) pairs. If the same pair or the
    same action appears K times within the window, a loop is flagged.

    Usage:
        detector = LoopDetector(window=10, threshold=3)
        detector.observe("EDIT_FILE")
        detector.observe("EDIT_FILE", state_id=2)
        loop_detected = detector.is_looping()  # True if threshold exceeded
    """

    def __init__(self, window: int = 10, threshold: int = 3):
        self.window = window
        self.threshold = threshold
        self._actions: deque[str] = deque(maxlen=window)
        self._state_actions: deque[tuple[int | None, str]] = deque(maxlen=window)

    def observe(self, action: str, state_id: int | None = None) -> None:
        """Record an action (and optional state) in the observation window."""
        self._actions.append(action)
        self._state_actions.append((state_id, action))

    def is_looping(self) -> bool:
        """Check if any state-action pair or action appears >= threshold times.

        State-action loops are checked first (more specific signal);
        falls back to action-only if no state_ids are available.
        """
        if len(self._actions) < self.threshold:
            return False
        return self._check_looping(self._state_actions) or self._check_looping(
            [(None, a) for a in self._actions]
        )

    def loop_count(self) -> int:
        """Return the count of the most frequent pattern in the window."""
        if not self._actions:
            return 0
        items: list = list(self._state_actions)
        counts: dict[tuple, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return max(counts.values())

    def most_frequent(self) -> str:
        """Return the most frequent action in the window."""
        if not self._actions:
            return ""
        counts: dict[str, int] = {}
        for a in self._actions:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)  # type: ignore

    def recent_pattern(self) -> list[str]:
        """Return the last N actions as a list."""
        return list(self._actions)

    def reset(self) -> None:
        self._actions.clear()
        self._state_actions.clear()

    def _check_looping(self, items) -> bool:
        if len(items) < self.threshold:  # type: ignore
            return False
        counts: dict = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return max(counts.values()) >= self.threshold  # type: ignore
