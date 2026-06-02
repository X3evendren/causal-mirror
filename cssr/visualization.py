"""Phase 5: Visualization — DOT output and complexity-entropy diagram."""


def to_dot(machine) -> str:
    """Generate Graphviz DOT representation of the epsilon-machine.

    Args:
        machine: An EpsilonMachine instance.

    Returns:
        DOT format string.
    """
    lines = [
        "digraph EpsilonMachine {",
        "  rankdir=LR;",
        "  node [shape=circle, style=filled, fillcolor=lightyellow];",
        "  edge [fontsize=10];",
    ]

    # Node labels
    for s in machine.states:
        probs_items = sorted(s.emission_probs.items(), key=lambda x: str(x[0]))
        probs_str = "\\n".join(
            f"{b}: {p:.3f}" for b, p in probs_items
        )
        label = f"State {s.state_id}\\n{probs_str}"
        lines.append(
            f'  s{s.state_id} [label="{label}", tooltip="{len(s.histories)} histories"];'
        )

    # Edges
    for s in machine.states:
        for b, target in s.transitions.items():
            prob = s.emission_probs.get(b, 0)
            tid = target if isinstance(target, int) else target.state_id
            lines.append(
                f'  s{s.state_id} -> s{tid} [label="{b} ({prob:.3f})"];'
            )

    lines.append("}")
    return "\n".join(lines)


def complexity_entropy_point(machine) -> tuple[float, float]:
    """Return (h_mu, C_mu) coordinates for the complexity-entropy diagram.

    Args:
        machine: An EpsilonMachine instance.

    Returns:
        (entropy_rate, statistical_complexity)
    """
    pi = machine.steady_state()
    h = machine.entropy_rate(pi)
    C = machine.statistical_complexity(pi)
    return h, C


def complexity_entropy_label(machine) -> str:
    """Human-readable complexity-entropy summary."""
    h, C = complexity_entropy_point(machine)
    n = len(machine.states)
    pi = machine.steady_state()
    E = machine.excess_entropy(pi)
    return (
        f"C_mu = {C:.4f} bits, h_mu = {h:.4f} bits, "
        f"E = {E:.4f} bits, n_states = {n}"
    )
