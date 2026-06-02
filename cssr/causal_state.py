"""CausalState data structure."""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CausalState:
    """A causal state: equivalence class of histories with identical future distributions.

    Attributes:
        state_id: Unique integer identifier.
        histories: List of suffix tuples belonging to this state.
        emission_counts: Aggregated next-symbol counts across all histories.
        emission_probs: Normalized emission probabilities.
        transitions: Dict mapping symbol -> next CausalState (populated during determinization).
    """
    state_id: int
    histories: list = field(default_factory=list)
    emission_counts: dict = field(default_factory=lambda: defaultdict(int))
    emission_probs: dict = field(default_factory=dict)
    transitions: dict = field(default_factory=dict)

    def add_history(self, suffix: tuple, next_counts: dict):
        """Add a history suffix and its next-symbol counts to this state."""
        self.histories.append(suffix)
        for sym, cnt in next_counts.items():
            self.emission_counts[sym] += cnt

    def normalize_emissions(self):
        """Compute emission probabilities from counts."""
        total = sum(self.emission_counts.values())
        if total > 0:
            self.emission_probs = {
                s: c / total for s, c in self.emission_counts.items()
            }
        else:
            self.emission_probs = {}

    def emission_total(self) -> int:
        """Total number of observations for this state."""
        return sum(self.emission_counts.values())

    def __repr__(self):
        return f"CausalState(id={self.state_id}, n_histories={len(self.histories)})"
