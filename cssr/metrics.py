"""Phase 4: Metric computation from an epsilon-machine."""


def compute_all_metrics(machine) -> dict:
    """Compute all standard computational mechanics metrics.

    Args:
        machine: An EpsilonMachine instance.

    Returns:
        dict with keys: statistical_complexity, entropy_rate, excess_entropy,
        predictive_information, n_states, alphabet_size.
    """
    pi = machine.steady_state()

    return {
        "statistical_complexity": machine.statistical_complexity(pi),
        "entropy_rate": machine.entropy_rate(pi),
        "excess_entropy": machine.excess_entropy(pi),
        "predictive_information": machine.predictive_information(pi),
        "n_states": len(machine.states),
        "alphabet_size": len(machine.alphabet),
    }
