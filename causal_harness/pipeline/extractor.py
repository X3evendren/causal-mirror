"""TrainingDataExtractor — read EventLedger episodes for RiskModel training.

Converts stored episodes into (symbol_sequences, outcomes) pairs suitable
for RiskModel.train(). Each episode is encoded via BehaviorEncoderV2 and
labeled 0 (success) or 1 (failure) based on its events.
"""

from causal_harness.events.schema import Event, EventType, EventOutcome
from causal_harness.events.ledger import EventLedger
from causal_harness.encoding.encoder import BehaviorEncoderV2
from causal_harness.encoding.symbols import BOUNDARY


class TrainingDataExtractor:
    """Extract (episode_symbols, outcome) pairs from an EventLedger.

    Usage:
        extractor = TrainingDataExtractor()
        episodes, outcomes = extractor.extract(ledger, encoder)
        risk_model.train(episodes, outcomes)
    """

    def extract(
        self,
        ledger: EventLedger,
        encoder: BehaviorEncoderV2 | None = None,
    ) -> tuple[list[list[str]], list[int]]:
        """Scan all ledger events and produce training data.

        Args:
            ledger: The event ledger to read from.
            encoder: BehaviorEncoderV2 instance. Created if None.

        Returns:
            (episodes, outcomes) where episodes[i] is a symbol list
            and outcomes[i] is 0 (success) or 1 (failure).
        """
        enc = encoder or BehaviorEncoderV2()

        # Group events by episode_id
        episodes_map: dict[str, list[Event]] = {}
        for event in ledger.replay():
            episodes_map.setdefault(event.episode_id, []).append(event)

        symbol_sequences: list[list[str]] = []
        outcomes: list[int] = []

        for ep_id, events in sorted(episodes_map.items()):
            # Sort events by (turn_id, timestamp) for correct chronology
            events.sort(key=lambda e: (e.turn_id, e.timestamp))

            # Encode to symbols
            symbols = enc.encode_episode(events)

            # Remove trailing BOUNDARY for training (not needed in sequences)
            clean = [s for s in symbols if s != BOUNDARY]

            if clean:  # skip empty episodes
                symbol_sequences.append(clean)
                outcomes.append(self._determine_episode_outcome(events))

        return symbol_sequences, outcomes

    @staticmethod
    def _determine_episode_outcome(events: list[Event]) -> int:
        """Determine episode outcome: 0 = success, 1 = failure.

        Priority order:
          1. Any ERROR_DETECTED event → failure (1)
          2. EPISODE_COMPLETED with FAILURE outcome → failure (1)
          3. EPISODE_COMPLETED with UNKNOWN → failure (conservative, 1)
          4. Any TOOL_RESULT with FAILURE → failure (1)
          5. Otherwise → success (0)
        """
        for event in events:
            if event.event_type == EventType.ERROR_DETECTED:
                return 1

        for event in events:
            if event.event_type == EventType.EPISODE_COMPLETED:
                if event.outcome in (EventOutcome.FAILURE, EventOutcome.UNKNOWN):
                    return 1
                return 0

        for event in events:
            if event.event_type == EventType.TOOL_RESULT:
                if event.outcome == EventOutcome.FAILURE:
                    return 1

        return 0
