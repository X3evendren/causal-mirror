"""Behavior Encoder V2 — converts event sequences into CSSR-ready symbol streams.

Replaces the old 16-symbol action×quality alphabet with:
  Layer A: 11 canonical action symbols + BOUNDARY markers between episodes
  Layer B: structured side-channel features for risk models (TODO)

Episode boundaries are CRITICAL: without them, CSSR treats the last
action of episode N and the first action of episode N+1 as a meaningful
state transition, creating phantom causal states.
"""

from collections.abc import Iterator

from causal_harness.events.schema import Event, EventType
from causal_harness.encoding.symbols import (
    ActionSymbol,
    BOUNDARY,
    TOOL_TO_SYMBOL,
)


class BehaviorEncoderV2:
    """Encodes event streams into discrete symbol sequences for CSSR.

    Usage:
        encoder = BehaviorEncoderV2()
        symbols = encoder.encode_all(ledger.replay())

        from cssr import CSSR, CSSRConfig
        cssr = CSSR(CSSRConfig()).fit(symbols)

    Note:
        This encoder is stateful: it tracks `_error_in_turn` across calls
        to `encode_event()` so that a TOOL_CALLED following an ERROR_DETECTED
        is correctly labelled as REPAIR. Events MUST be passed in chronological
        order within a turn. For replay scenarios, use `encode_episode()` or
        `encode_all()` which reset state at episode boundaries.
    """

    def __init__(self, insert_boundaries: bool = True):
        self.insert_boundaries = insert_boundaries
        self._error_in_turn: bool = False

    def encode_event(self, event: Event) -> str | None:
        """Convert one event into a Layer A symbol (or None if not encodable).

        Does NOT handle episode boundaries — use encode_all() or encode_episode()
        for multi-episode streams.

        IMPORTANT: This method mutates internal state (`_error_in_turn`).
        Events must be passed in chronological order within a single turn.
        """
        if event.event_type == EventType.TOOL_CALLED:
            return self._encode_tool_call(event)

        if event.event_type == EventType.TOOL_RESULT:
            return None

        if event.event_type == EventType.ERROR_DETECTED:
            self._error_in_turn = True
            return ActionSymbol.ERROR.value

        if event.event_type == EventType.REPAIR_ATTEMPTED:
            self._error_in_turn = False
            return ActionSymbol.REPAIR.value

        if event.event_type == EventType.VERIFICATION_RUN:
            return ActionSymbol.VERIFY.value

        if event.event_type == EventType.USER_APPROVAL_REQUESTED:
            return ActionSymbol.ESCALATE.value

        if event.event_type == EventType.ACTION_PROPOSED:
            return ActionSymbol.PLAN.value

        if event.event_type == EventType.EPISODE_COMPLETED:
            return ActionSymbol.COMPLETE.value

        if event.event_type == EventType.TURN_STARTED:
            self._error_in_turn = False
            return None

        if event.event_type in (EventType.EPISODE_STARTED, EventType.GOAL_LOADED,
                                 EventType.OBSERVATION_ADDED, EventType.STATE_TRANSITION,
                                 EventType.TURN_COMPLETED):
            return None

        return None

    def encode_episode(self, events: list[Event]) -> list[str]:
        """Encode one episode's events. Appends a trailing BOUNDARY."""
        symbols: list[str] = []
        self._error_in_turn = False
        for event in events:
            sym = self.encode_event(event)
            if sym is not None:
                symbols.append(sym)
        symbols.append(BOUNDARY)
        return symbols

    def encode_all(self, events: Iterator[Event]) -> list[str]:
        """Encode a full multi-episode stream with automatic BOUNDARY insertion."""
        symbols: list[str] = []
        last_ep: str | None = None

        for event in events:
            if self.insert_boundaries and last_ep is not None and event.episode_id != last_ep:
                symbols.append(BOUNDARY)
            last_ep = event.episode_id

            sym = self.encode_event(event)
            if sym is not None:
                symbols.append(sym)

        return symbols

    # -- internal --

    def _encode_tool_call(self, event: Event) -> str | None:
        tool_name = event.payload.get("tool", "")

        if self._error_in_turn:
            self._error_in_turn = False
            return ActionSymbol.REPAIR.value

        symbol = TOOL_TO_SYMBOL.get(tool_name, ActionSymbol.PLAN)
        return symbol.value
