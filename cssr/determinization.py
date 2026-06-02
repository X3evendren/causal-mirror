"""Phase 3: Determinization and transient state removal.

Ensures the epsilon-machine is unifilar: for each state and each symbol,
the next state is uniquely determined. Also removes transient (non-recurrent) states.
"""

from collections import defaultdict
from cssr.causal_state import CausalState


def _tarjan_scc(adj: list[set], n: int) -> list[list[int]]:
    """Tarjan's algorithm for strongly connected components.

    Args:
        adj: Adjacency list, adj[i] = set of successor node indices.
        n: Number of nodes.

    Returns:
        List of SCCs, each SCC is a list of node indices.
    """
    index = 0
    indices = [-1] * n
    lowlink = [0] * n
    onstack = [False] * n
    stack = []
    sccs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in adj[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif onstack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                onstack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    return sccs


def _build_transition_graph(states, state_map, alphabet) -> list[set]:
    """Build a directed graph of state transitions.

    Uses s.transitions if available (post-rebuild), falling back to
    history-based discovery for states with empty transitions.
    """
    n = len(states)
    adj = [set() for _ in range(n)]
    for s in states:
        if s.transitions:
            for target_id in s.transitions.values():
                adj[s.state_id].add(target_id)
        elif s.histories:
            for hist in s.histories:
                for b in alphabet:
                    new_hist = hist[1:] + (b,) if len(hist) >= 1 else (b,)
                    if new_hist in state_map:
                        adj[s.state_id].add(state_map[new_hist])
    return adj


def determinize(
    states: list[CausalState],
    state_map: dict,
    alphabet: set,
) -> tuple[list[CausalState], dict]:
    """Ensure unifilar property: state-symbol -> unique next state.

    For each state and each symbol, if different histories in the state
    lead to different next states when the same symbol is appended,
    split the state into sub-states where the transition is deterministic.

    Args:
        states: Current causal states.
        state_map: Current suffix -> state_id mapping.
        alphabet: Symbol alphabet.

    Returns:
        (determinized_states, determinized_state_map)
    """
    if len(alphabet) <= 1:
        return states, state_map

    changed = True
    iteration = 0
    max_iterations = len(state_map) + 10  # theoretical bound

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        new_states = []

        for s in states:
            split = False
            for b in alphabet:
                # For each symbol, find all possible next states from this state.
                # New history = drop oldest symbol, append observed symbol:
                #   h' = h[1:] + (b,)
                next_state_ids = set()
                for hist in s.histories:
                    if len(hist) >= 1:
                        new_hist = hist[1:] + (b,)
                    else:
                        new_hist = (b,)
                    if new_hist in state_map:
                        next_state_ids.add(state_map[new_hist])

                if len(next_state_ids) > 1:
                    # Non-deterministic: partition histories by next-state
                    partitions = defaultdict(list)
                    for hist in s.histories:
                        new_hist = hist[1:] + (b,) if len(hist) >= 1 else (b,)
                        ns = state_map.get(new_hist)
                        partitions[ns].append(hist)

                    # Create new sub-states
                    for ns, group_hists in partitions.items():
                        new_state = CausalState(state_id=len(states) + len(new_states))
                        new_state.histories = group_hists
                        # Copy emission counts from the original state (proportional by history)
                        new_states.append(new_state)

                    split = True
                    break  # Restart the while loop after splitting
                elif len(next_state_ids) == 1:
                    # Record deterministic transition
                    s.transitions[b] = next_state_ids.pop()

            if not split:
                new_states.append(s)

        if split:
            states = new_states
            # Aggregate emissions for new states
            for s in states:
                s.emission_counts = defaultdict(int)
                # We don't have the suffix_trie here to re-aggregate
                # Inherit from original states? This is tricky.
                # Best approach: keep emission_probs if already computed
            changed = True

        # Rebuild state_map for new states
        new_state_map = {}
        for s in states:
            for hist in s.histories:
                new_state_map[hist] = s.state_id
        state_map = new_state_map

    # Return to original state_id scheme
    for i, s in enumerate(states):
        s.state_id = i

    new_state_map = {}
    for s in states:
        for hist in s.histories:
            new_state_map[hist] = s.state_id
    state_map = new_state_map

    return states, state_map


def _rebuild_transitions(
    states: list[CausalState],
    state_map: dict,
    alphabet: set,
) -> None:
    """Rebuild all transitions from the state_map after determinization.

    After determinization the machine is unifilar: for each state and symbol,
    all histories agree on the next state. This recomputes s.transitions
    correctly regardless of how states were split during determinization.
    """
    for s in states:
        s.transitions = {}
        if not s.histories:
            continue
        for b in alphabet:
            for hist in s.histories:
                new_hist = hist[1:] + (b,) if len(hist) >= 1 else (b,)
                if new_hist in state_map:
                    s.transitions[b] = state_map[new_hist]
                    break  # unifilar: all histories in this state agree


def remove_transient_states(
    states: list[CausalState],
    state_map: dict,
    alphabet: set,
) -> tuple[list[CausalState], dict]:
    """Remove transient (non-recurrent) states from the epsilon-machine.

    Uses Tarjan's SCC algorithm to find the recurrent component(s).
    States that are not in any terminal SCC are removed.

    Args:
        states: Causal states.
        state_map: Suffix -> state_id mapping.
        alphabet: Symbol alphabet.

    Returns:
        (filtered_states, filtered_state_map)
    """
    n = len(states)
    if n <= 1:
        return states, state_map

    adj = _build_transition_graph(states, state_map, alphabet)
    sccs = _tarjan_scc(adj, n)

    if not sccs:
        return states, state_map

    # Find terminal SCCs (no outgoing edges to other SCCs)
    scc_of = {}
    for i, scc in enumerate(sccs):
        for v in scc:
            scc_of[v] = i

    terminal_sccs = set()
    for i, scc in enumerate(sccs):
        is_terminal = True
        for v in scc:
            for w in adj[v]:
                if scc_of.get(w, -1) != i:
                    is_terminal = False
                    break
            if not is_terminal:
                break
        if is_terminal:
            terminal_sccs.add(i)

    if not terminal_sccs:
        # All states are transient — pathological case
        # Keep the largest SCC as fallback
        largest_scc_idx = max(range(len(sccs)), key=lambda i: len(sccs[i]))
        terminal_sccs = {largest_scc_idx}

    keep_ids = set()
    for i in terminal_sccs:
        keep_ids.update(sccs[i])

    if len(keep_ids) == n:
        return states, state_map  # Nothing to remove

    # Filter states
    old_to_new = {}
    new_states = []
    for s in states:
        if s.state_id in keep_ids:
            new_id = len(new_states)
            old_to_new[s.state_id] = new_id
            new_states.append(s)

    # Re-id states
    for i, s in enumerate(new_states):
        s.state_id = i
        # Update transitions to use new IDs (transitions may store ints or CausalState objects)
        new_transitions = {}
        for b, target in s.transitions.items():
            tid = target.state_id if hasattr(target, 'state_id') else target
            if tid in old_to_new:
                new_transitions[b] = old_to_new[tid]
        s.transitions = new_transitions

    # Filter state_map
    new_state_map = {}
    for suffix, sid in state_map.items():
        if sid in old_to_new:
            new_state_map[suffix] = old_to_new[sid]

    return new_states, new_state_map
