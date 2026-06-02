"""Phase 2: Causal State Splitting.

The core CSSR algorithm: starting from a single trivial state (containing only
the null suffix), iteratively split states by testing whether longer histories
have different future distributions than their parent state.

Algorithm from Shalizi & Klinkner (UAI 2004).
"""

from collections import defaultdict

from cssr.causal_state import CausalState
from cssr.config import CSSRConfig
from cssr.suffix_trie import SuffixTrie
from cssr.testing import compare_distributions, get_corrected_alpha


def _aggregate_state_emissions(states: list[CausalState], suffix_trie: SuffixTrie) -> None:
    """Recompute emission_counts for each state from its member histories.

    This must be called after any modification to state memberships.
    """
    for state in states:
        state.emission_counts = defaultdict(int)
        for hist in state.histories:
            next_counts = suffix_trie.get_next_counts(hist)
            for sym, cnt in next_counts.items():
                state.emission_counts[sym] += cnt


def split_states(
    suffix_trie: SuffixTrie,
    alphabet: set,
    L_max: int,
    config: CSSRConfig,
) -> tuple[list[CausalState], dict]:
    """Run the CSSR state-splitting loop.

    Starting from one state containing only the empty suffix, iteratively
    test longer suffixes against their parent state's distribution.
    Create new states when the data forces it.

    Args:
        suffix_trie: Built SuffixTrie with substring frequencies.
        alphabet: Set of distinct symbols in the sequence.
        L_max: Maximum history length.
        config: CSSR configuration.

    Returns:
        (states, state_map) where:
          states: list of CausalState objects
          state_map: dict mapping suffix tuple -> state_id
    """
    # Initialize: one state with the null suffix
    empty = ()
    state0 = CausalState(state_id=0)
    n_empty = suffix_trie.get_count(empty)
    if n_empty > 0:
        state0.emission_counts = dict(suffix_trie.get_next_counts(empty))
        state0.histories = [empty]
    else:
        state0.histories = [empty]

    states = [state0]
    state_map: dict[tuple, int] = {empty: 0}

    # Single-symbol alphabet: no splitting possible
    if len(alphabet) <= 1:
        state0.normalize_emissions()
        return states, state_map

    for L in range(1, L_max + 1):
        suffixes = suffix_trie.suffixes_of_length(L)
        if not suffixes:
            continue

        # Recompute state distributions from current assignments
        _aggregate_state_emissions(states, suffix_trie)

        # --- First pass: compute H0 p-values and handle untestable suffixes ---
        # Collect testable (suffix, p_value, suffix_counts, parent_id) tuples.
        candidates: list[tuple[tuple, float, dict, int]] = []

        for suffix in suffixes:
            # suffix = (s_{t-L}, ..., s_{t-1}) in oldest-first order
            # a = s_{t-L} (oldest symbol), x = (s_{t-L+1}, ..., s_{t-1}) (shorter suffix)
            a = suffix[0]
            x = suffix[1:]  # length L-1

            parent_id = state_map.get(x)
            if parent_id is None:
                # Parent suffix not observed; create new state conservatively
                new_state = CausalState(state_id=len(states))
                next_counts = suffix_trie.get_next_counts(suffix)
                new_state.add_history(suffix, next_counts)
                states.append(new_state)
                state_map[suffix] = new_state.state_id
                continue

            parent_state = states[parent_id]

            # Get suffix's own next-symbol distribution
            suffix_counts = suffix_trie.get_next_counts(suffix)
            total_suffix = sum(suffix_counts.values())

            if total_suffix < config.min_count:
                # Insufficient data: conservatively assign to parent state
                parent_state.histories.append(suffix)
                state_map[suffix] = parent_id
                for sym, cnt in suffix_counts.items():
                    parent_state.emission_counts[sym] += cnt
                continue

            # H0: P(future | a x) == P(future | state(x))
            parent_counts = dict(parent_state.emission_counts)
            _parent_total = sum(parent_counts.values())

            if _parent_total < config.min_count:
                parent_state.histories.append(suffix)
                state_map[suffix] = parent_id
                for sym, cnt in suffix_counts.items():
                    parent_state.emission_counts[sym] += cnt
                continue

            p_value = compare_distributions(
                suffix_counts, parent_counts, alphabet, config.test
            )
            candidates.append((suffix, p_value, suffix_counts, parent_id))

        if not candidates:
            continue

        # --- Apply multiple testing correction ---
        pvalues = [c[1] for c in candidates]
        n_tests = len(pvalues)

        if config.correction == "bh":
            # Benjamini-Hochberg: use corrected p-values from statsmodels
            # BH-corrected p < alpha  ⇔  reject H0
            corrected_pvalues = correct_pvalues(pvalues, config.alpha, "bh")
            reject = [cp < config.alpha for cp in corrected_pvalues]
        elif config.correction == "bonferroni":
            threshold = config.alpha / n_tests if n_tests > 0 else config.alpha
            reject = [p < threshold for p in pvalues]
        else:
            reject = [p < config.alpha for p in pvalues]

        # --- Second pass: assign suffixes based on corrected rejection ---
        for (suffix, _p_value, suffix_counts, parent_id), is_rejected in zip(candidates, reject):
            parent_state = states[parent_id]

            if not is_rejected:
                # H0 not rejected: suffix belongs in parent's state
                parent_state.histories.append(suffix)
                state_map[suffix] = parent_id
                for sym, cnt in suffix_counts.items():
                    parent_state.emission_counts[sym] += cnt
            else:
                # H0 rejected: test H1 — does it match any other existing state?
                # H1 tests use uncorrected alpha (testing against a specific
                # pre-existing state is a more targeted hypothesis)
                matched = False
                for other in states:
                    if other.state_id == parent_id:
                        continue
                    other_counts = dict(other.emission_counts)
                    _total_other = sum(other_counts.values())
                    if _total_other < config.min_count:
                        continue

                    p_val = compare_distributions(
                        suffix_counts, other_counts, alphabet, config.test
                    )
                    if p_val > config.alpha:
                        # Matches this existing state
                        other.histories.append(suffix)
                        state_map[suffix] = other.state_id
                        for sym, cnt in suffix_counts.items():
                            other.emission_counts[sym] += cnt
                        matched = True
                        break

                if not matched:
                    # Create a brand new state
                    new_state = CausalState(state_id=len(states))
                    new_state.histories.append(suffix)
                    for sym, cnt in suffix_counts.items():
                        new_state.emission_counts[sym] += cnt
                    states.append(new_state)
                    state_map[suffix] = new_state.state_id

    # Normalize all emission probabilities
    for state in states:
        state.normalize_emissions()

    return states, state_map
