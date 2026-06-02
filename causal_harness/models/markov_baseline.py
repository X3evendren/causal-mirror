"""Finite-order Markov risk model — simple frequency baseline.

Counts how often each (context, next_action) pattern leads to failure.
If CSSR can't beat this, it shouldn't be used for control decisions.
"""

from collections import defaultdict

from causal_harness.models.base import RiskModel, RiskPrediction


class MarkovRiskModel(RiskModel):
    """Order-N Markov model: P(failure | last N symbols + next action).

    This is intentionally simple. A more sophisticated model should
    beat this baseline before being trusted for runtime control.
    """

    def __init__(self, order: int = 2, min_count: int = 1):
        self.order = order
        self.min_count = min_count
        self._failure_counts: dict[tuple, int] = defaultdict(int)
        self._total_counts: dict[tuple, int] = defaultdict(int)
        self._global_failure_rate: float = 0.0
        self._trained: bool = False

    def train(self, episodes: list[list[str]], outcomes: list[int]) -> None:
        """Count (context, action) -> failure patterns across all episodes."""
        self._failure_counts.clear()
        self._total_counts.clear()
        total_failures = 0
        total_actions = 0

        for symbols, outcome in zip(episodes, outcomes):
            is_failure = 1 if outcome == 1 else 0
            total_failures += is_failure

            for i in range(len(symbols)):
                # Build context from up to `order` previous symbols
                context_start = max(0, i - self.order)
                context = tuple(symbols[context_start:i])
                action = symbols[i]
                key = (context, action)

                self._total_counts[key] += 1
                self._failure_counts[key] += is_failure
                total_actions += 1

        self._global_failure_rate = total_failures / max(1, len(episodes))
        self._trained = True

    def predict(self, context: list[str], next_action: str) -> RiskPrediction:
        if not self._trained:
            return RiskPrediction(p_failure=self._global_failure_rate or 0.5,
                                  confidence=0.1, state_id=-1)

        ctx = tuple(context[-self.order:]) if len(context) > self.order else tuple(context)
        # Try progressively shorter contexts if full context unseen
        for trim in range(len(ctx) + 1):
            key = (ctx[trim:], next_action)
            total = self._total_counts.get(key, 0)
            if total >= self.min_count:
                p_fail = self._failure_counts.get(key, 0) / total
                confidence = min(1.0, total / (total + 10))  # Laplace-ish confidence
                return RiskPrediction(
                    p_failure=float(p_fail),
                    confidence=float(confidence),
                    state_id=hash(key) % 10000,
                )

        # Fallback to global rate
        return RiskPrediction(
            p_failure=self._global_failure_rate,
            confidence=0.1,
            state_id=-1,
            reasoning="insufficient data — using global failure rate",
        )
