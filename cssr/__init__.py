"""
CSSR: Causal State Splitting Reconstruction.

Pure Python implementation of the CSSR algorithm (Shalizi & Klinkner, UAI 2004)
for discovering epsilon-machines from discrete symbol sequences.

Usage:
    from cssr import CSSR, CSSRConfig

    config = CSSRConfig(L_max=3, alpha=0.01)
    model = CSSR(config).fit(sequence)
    print(f"C_mu  = {model.metrics['statistical_complexity']:.4f}")
    print(f"h_mu  = {model.metrics['entropy_rate']:.4f}")
    print(f"E     = {model.metrics['excess_entropy']:.4f}")
    print(f"chi   = {model.metrics['predictive_information']:.4f}")
    print(model.to_dot())
"""

from cssr.config import CSSRConfig
from cssr.machine import EpsilonMachine
from cssr.suffix_trie import SuffixTrie
from cssr.causal_state import CausalState
from cssr.splitting import split_states
from cssr.determinization import determinize, remove_transient_states, _rebuild_transitions
from cssr.metrics import compute_all_metrics
from cssr.visualization import to_dot, complexity_entropy_point
from cssr.utils import estimate_L_max, detect_alphabet


class CSSR:
    """Main CSSR engine: fit an epsilon-machine to a discrete symbol sequence."""

    def __init__(self, config: CSSRConfig | None = None):
        self.config = config or CSSRConfig()
        self.alphabet: set | None = None
        self.suffix_trie: SuffixTrie | None = None
        self.state_map: dict | None = None
        self.machine: EpsilonMachine | None = None
        self.metrics: dict | None = None

    def fit(self, sequence) -> "CSSR":
        """
        Run the full CSSR pipeline on a discrete symbol sequence.

        Args:
            sequence: List of hashable symbols (ints or strings).

        Returns:
            self, with .machine and .metrics populated.
        """
        sequence = list(sequence)
        if len(sequence) == 0:
            raise ValueError("Sequence must have at least 1 element.")

        self.alphabet = detect_alphabet(sequence)
        L_max = self.config.L_max or estimate_L_max(len(sequence), len(self.alphabet))
        L_max = min(L_max, len(sequence) - 1) if len(sequence) > 1 else 1

        # Phase 1: Build suffix trie
        self.suffix_trie = SuffixTrie(self.alphabet)
        self.suffix_trie.build(sequence, L_max)

        # Phase 2: State splitting
        from cssr.splitting import split_states
        states, raw_state_map = split_states(
            self.suffix_trie, self.alphabet, L_max, self.config
        )

        # Phase 3: Determinization + transient removal
        from cssr.determinization import determinize, remove_transient_states
        from cssr.splitting import _aggregate_state_emissions
        states, self.state_map = determinize(states, raw_state_map, self.alphabet)
        # Recompute emission counts from suffix trie after determinization
        # (splitting creates new sub-states with inherited, incorrect emissions)
        _aggregate_state_emissions(states, self.suffix_trie)
        # Rebuild transitions from state_map — determinization can leave them
        # incomplete when states are split (new sub-states get empty transitions)
        _rebuild_transitions(states, self.state_map, self.alphabet)
        if self.config.remove_transient:
            states, self.state_map = remove_transient_states(
                states, self.state_map, self.alphabet
            )

        # Normalize emission probabilities for all states
        for state in states:
            state.normalize_emissions()

        # Phase 4: Assemble epsilon-machine + compute metrics
        self.machine = EpsilonMachine(
            states=states,
            start_state=states[0] if states else None,
            alphabet=self.alphabet,
            L_max=L_max,
        )
        self.metrics = compute_all_metrics(self.machine)

        return self

    def to_dot(self) -> str:
        """Return Graphviz DOT representation of the epsilon-machine."""
        if self.machine is None:
            raise RuntimeError("Call fit() before to_dot().")
        return to_dot(self.machine)

    def visualize(self, filename: str) -> None:
        """Render the epsilon-machine to a file using Graphviz."""
        if self.machine is None:
            raise RuntimeError("Call fit() before visualize().")
        dot_str = to_dot(self.machine)
        import graphviz
        g = graphviz.Source(dot_str)
        g.render(filename, cleanup=True)


__all__ = [
    "CSSR",
    "CSSRConfig",
    "EpsilonMachine",
    "SuffixTrie",
    "CausalState",
    "split_states",
    "determinize",
    "remove_transient_states",
    "compute_all_metrics",
    "to_dot",
    "complexity_entropy_point",
    "estimate_L_max",
    "detect_alphabet",
]
