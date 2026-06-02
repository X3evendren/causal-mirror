"""EpsilonMachine dataclass and metric computation.

Phase 4 output: the assembled epsilon-machine with all metrics.
"""

from dataclasses import dataclass, field
import numpy as np
from cssr.causal_state import CausalState


@dataclass
class EpsilonMachine:
    """An epsilon-machine: the minimal, unifilar, optimal predictor of a process.

    Attributes:
        states: List of recurrent causal states.
        start_state: The state containing the null history (if present).
        alphabet: Set of distinct symbols.
        L_max: Maximum history length used in reconstruction.
    """
    states: list[CausalState]
    start_state: CausalState | None
    alphabet: set
    L_max: int

    def __post_init__(self):
        """Validate that all states have non-empty emission distributions.

        Catches the case where determinization creates sub-states with empty
        emissions and the caller forgot to call _aggregate_state_emissions.

        Only enforces when alphabet has more than 1 symbol — single-symbol
        sequences naturally have empty/trivial emission_probs (H=0).
        """
        if len(self.alphabet) <= 1:
            return
        for s in self.states:
            if not s.emission_probs:
                raise ValueError(
                    f"State {s.state_id} has empty emission_probs. "
                    f"Did you forget to call _aggregate_state_emissions "
                    f"after determinization? "
                    f"Histories: {len(s.histories)}, "
                    f"emission_counts: {dict(s.emission_counts)}"
                )

    def build_transition_matrix(self) -> np.ndarray:
        """Build the state transition probability matrix P.

        P[i, j] = probability of transitioning from state i to state j.
        """
        n = len(self.states)
        P = np.zeros((n, n))

        for s in self.states:
            i = s.state_id
            total = sum(s.emission_probs.values())
            if total == 0:
                P[i, i] = 1.0  # Absorbing state
                continue
            for b, prob in s.emission_probs.items():
                target = s.transitions.get(b)
                if target is not None:
                    j = target if isinstance(target, int) else target.state_id
                    P[i, j] += prob
                else:
                    # Stay in same state if transition undefined
                    P[i, i] += prob

        # Renormalize rows
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P = P / row_sums

        return P

    def steady_state(self) -> np.ndarray:
        """Compute the steady-state distribution pi over states.

        Solves pi @ P = pi, sum(pi) = 1.
        """
        P = self.build_transition_matrix()
        n = P.shape[0]

        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # Solve (P^T - I) @ pi = 0, with last row replaced by sum(pi) = 1
        A = P.T - np.eye(n)
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        try:
            pi = np.linalg.solve(A, b)
            pi = np.clip(pi, 0, None)
            pi = pi / pi.sum()
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition
            eigvals, eigvecs = np.linalg.eig(P.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            pi = np.real(eigvecs[:, idx])
            pi = np.clip(pi, 0, None)
            pi = pi / pi.sum() if pi.sum() > 0 else np.ones(n) / n

        return pi

    def statistical_complexity(self, pi: np.ndarray | None = None) -> float:
        """C_mu = H[pi]: Shannon entropy of the steady-state distribution."""
        if pi is None:
            pi = self.steady_state()
        entropy = 0.0
        for p in pi:
            if p > 1e-15:
                entropy -= p * np.log2(p)
        return entropy

    def entropy_rate(self, pi: np.ndarray | None = None) -> float:
        """h_mu: Conditional entropy of next symbol given current state.

        h_mu = sum_i pi_i * H[emissions | s_i]
        """
        if pi is None:
            pi = self.steady_state()
        h_mu = 0.0
        for s in self.states:
            if pi[s.state_id] == 0:
                continue
            state_h = 0.0
            for prob in s.emission_probs.values():
                if prob > 1e-15:
                    state_h -= prob * np.log2(prob)
            h_mu += pi[s.state_id] * state_h
        return h_mu

    def excess_entropy(self, pi: np.ndarray | None = None) -> float:
        """E = I[S_t; S_{t+1}]: Mutual information between consecutive states.

        This is the one-step predictive information. For unifilar HMMs,
        this equals the excess entropy (mutual information between past and future).
        """
        if pi is None:
            pi = self.steady_state()
        P = self.build_transition_matrix()
        n = len(self.states)

        E = self.statistical_complexity(pi)
        for i in range(n):
            if pi[i] == 0:
                continue
            row_entropy = 0.0
            for j in range(n):
                if P[i, j] > 1e-15:
                    row_entropy -= P[i, j] * np.log2(P[i, j])
            E -= pi[i] * row_entropy

        return max(0.0, E)

    def predictive_information(self, pi: np.ndarray | None = None) -> float:
        """chi = H[X] - h_mu: How much knowing the state reduces symbol uncertainty."""
        if pi is None:
            pi = self.steady_state()

        # Marginal symbol entropy: H[X] = -sum_x P(x) log P(x)
        # where P(x) = sum_s pi_s * P(x | s)
        symbol_weights: dict[str, float] = {}
        for s in self.states:
            weight = pi[s.state_id]
            if weight <= 0:
                continue
            for sym, prob in s.emission_probs.items():
                symbol_weights[sym] = symbol_weights.get(sym, 0.0) + weight * prob

        total_prob = sum(symbol_weights.values())
        total_symbol_entropy = 0.0
        for cnt in symbol_weights.values():
            p = cnt / total_prob if total_prob > 0 else 0
            if p > 1e-15:
                total_symbol_entropy -= p * np.log2(p)

        h_mu = self.entropy_rate(pi)
        return max(0.0, total_symbol_entropy - h_mu)
