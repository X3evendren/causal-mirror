"""Integration tests: CSSR on known reference processes.

Validates the full pipeline against processes with known epsilon-machine structure.
"""

import pytest
from cssr import CSSR, CSSRConfig


def generate_iid(n: int, probs: dict = None) -> list:
    """Generate IID sequence. Single state, C_mu = 0."""
    import random
    random.seed(42)
    if probs is None:
        probs = {"H": 0.5, "T": 0.5}
    symbols = list(probs.keys())
    weights = list(probs.values())
    return random.choices(symbols, weights=weights, k=n)


def generate_even_process(n: int) -> list:
    """Generate Even Process: no consecutive 1s, only even runs of 0s between 1s.

    Causal states (from external observer):
      - Histories ending in "1": next is always "0" (deterministic)
      - Histories ending in "0": next is "0" or "1" with equal prob
    Stationary: pi(ends-in-0)=2/3, pi(ends-in-1)=1/3
    Theoretical: n_states=2, C_mu ≈ 0.918, h_mu ≈ 0.667
    """
    import random
    random.seed(42)
    result = []
    # Start in state A (even)
    even = True
    for _ in range(n):
        if even:
            result.append("0")
            if random.random() < 0.5:
                even = False
        else:
            result.append("1")
            even = True
    return result


def generate_golden_mean(n: int) -> list:
    """Generate Golden Mean Process: no consecutive 0s.

    Causal states (from external observer):
      - Histories ending in "1": next is "0" or "1" with equal prob
      - Histories ending in "0": next must be "1" (deterministic)
    Stationary: pi(ends-in-1)=2/3, pi(ends-in-0)=1/3
    Theoretical: n_states=2, C_mu ≈ 0.918, h_mu ≈ 0.667
    """
    import random
    random.seed(42)
    result = ["1"]
    for _ in range(n - 1):
        last = result[-1]
        if last == "0":
            result.append("1")
        else:
            result.append("1" if random.random() < 0.5 else "0")
    return result


def test_iid_single_state():
    """IID process should converge to a single causal state."""
    seq = generate_iid(5000)
    config = CSSRConfig(L_max=3, alpha=0.01, min_count=3)
    model = CSSR(config).fit(seq)
    n_states = model.metrics["n_states"]
    C_mu = model.metrics["statistical_complexity"]
    h_mu = model.metrics["entropy_rate"]
    assert n_states == 1, f"IID: expected 1 state, got {n_states}"
    assert C_mu < 0.05, f"IID: C_mu ≈ 0, got {C_mu:.4f}"
    assert h_mu > 0.95, f"IID: h_mu ≈ 1.0, got {h_mu:.4f}"


def test_even_process_two_states():
    """Even Process: 2 states, C_mu ≈ 0.92, h_mu ≈ 0.67."""
    seq = generate_even_process(5000)
    config = CSSRConfig(L_max=3, alpha=0.01, min_count=5)
    model = CSSR(config).fit(seq)
    n_states = model.metrics["n_states"]
    C_mu = model.metrics["statistical_complexity"]
    h_mu = model.metrics["entropy_rate"]
    assert n_states == 2, f"Even: expected 2 states, got {n_states}"
    assert 0.85 < C_mu < 0.99, f"Even: C_mu ≈ 0.92, got {C_mu:.4f}"
    assert 0.55 < h_mu < 0.80, f"Even: h_mu ≈ 0.67, got {h_mu:.4f}"


def test_golden_mean_structure():
    """Golden Mean Process: 2 states, C_mu ≈ 0.92, h_mu ≈ 0.67."""
    seq = generate_golden_mean(5000)
    config = CSSRConfig(L_max=3, alpha=0.01, min_count=5)
    model = CSSR(config).fit(seq)
    n_states = model.metrics["n_states"]
    C_mu = model.metrics["statistical_complexity"]
    h_mu = model.metrics["entropy_rate"]
    assert n_states == 2, f"GM: expected 2 states, got {n_states}"
    assert 0.85 < C_mu < 0.99, f"GM: C_mu ≈ 0.92, got {C_mu:.4f}"
    assert 0.55 < h_mu < 0.80, f"GM: h_mu ≈ 0.67, got {h_mu:.4f}"


def test_single_symbol_alphabet():
    """Single-symbol sequences: return one state immediately."""
    seq = ["0"] * 100
    config = CSSRConfig(L_max=5)
    model = CSSR(config).fit(seq)
    assert model.metrics["n_states"] == 1
    assert model.metrics["statistical_complexity"] == 0.0
    assert model.metrics["entropy_rate"] == 0.0


def test_short_sequence():
    """Very short sequences should not crash."""
    seq = ["A", "B", "A"]
    config = CSSRConfig(L_max=2, alpha=0.05, min_count=1)
    model = CSSR(config).fit(seq)
    assert model.metrics["n_states"] >= 1


def test_to_dot():
    """DOT output should be a valid string."""
    seq = ["A", "B", "A", "B", "A"]
    config = CSSRConfig(L_max=1, alpha=0.05, min_count=1)
    model = CSSR(config).fit(seq)
    dot = model.to_dot()
    assert dot.startswith("digraph")
    assert "EpsilonMachine" in dot


def generate_order2_markov(n: int) -> list:
    """Generate a strongly recurrent 2nd-order Markov process with 3 causal states.

    All four 2-symbol contexts have well-separated emission distributions:
      ("0","0") -> P(0)=0.8, P(1)=0.2
      ("0","1") -> P(0)=0.3, P(1)=0.7
      ("1","0") -> P(0)=0.6, P(1)=0.4
      ("1","1") -> P(0)=0.9, P(1)=0.1
    Every context can emit both symbols; no absorbing states.
    CSSR should discover >=2 states with C_mu > 0.
    """
    import random
    random.seed(42)
    result = ["0", "1"]
    for _ in range(n - 2):
        last2 = (result[-2], result[-1])
        if last2 == ("0", "0"):
            result.append("0" if random.random() < 0.8 else "1")
        elif last2 == ("0", "1"):
            result.append("0" if random.random() < 0.3 else "1")
        elif last2 == ("1", "0"):
            result.append("0" if random.random() < 0.6 else "1")
        else:  # ("1","1")
            result.append("0" if random.random() < 0.9 else "1")
    return result


def test_order2_markov():
    """2nd-order Markov process: CSSR should discover ≥2 causal states with C_mu > 0."""
    seq = generate_order2_markov(10000)
    config = CSSRConfig(L_max=4, alpha=0.01, min_count=5)
    model = CSSR(config).fit(seq)
    n_states = model.metrics["n_states"]
    C_mu = model.metrics["statistical_complexity"]
    h_mu = model.metrics["entropy_rate"]
    assert n_states >= 2, f"2nd-order Markov: expected >=2 states, got {n_states}"
    assert n_states <= 8, f"2nd-order Markov: expected <=8 states, got {n_states}"
    assert C_mu > 0, f"2nd-order Markov: C_mu should be > 0 for multi-state process"
    total_emit = sum(len(s.emission_probs) for s in model.machine.states)
    assert total_emit >= 2, f"2nd-order Markov: expected diverse emissions"


def test_metrics_keys():
    """All expected metric keys present."""
    seq = generate_iid(1000)
    config = CSSRConfig(L_max=2)
    model = CSSR(config).fit(seq)
    expected_keys = {
        "statistical_complexity", "entropy_rate", "excess_entropy",
        "predictive_information", "n_states", "alphabet_size"
    }
    assert expected_keys.issubset(set(model.metrics.keys()))


def test_multi_state_implies_nonzero_c_mu():
    """Regression: n_states > 1 must produce C_mu > 0.

    This guards against the determinization bug where sub-states lost
    transitions, causing a degenerate steady-state distribution with
    all probability mass on one state (C_mu = 0 despite n_states > 1).
    """
    seq = generate_even_process(5000)
    config = CSSRConfig(L_max=3, alpha=0.01, min_count=5)
    model = CSSR(config).fit(seq)
    n_states = model.metrics["n_states"]
    C_mu = model.metrics["statistical_complexity"]
    E = model.metrics["excess_entropy"]
    if n_states > 1:
        assert C_mu > 0, (
            f"n_states={n_states} but C_mu={C_mu}. "
            f"Determinization may be losing transitions."
        )
        assert E > 0, (
            f"n_states={n_states} but E={E}. "
            f"Transition structure may be degenerate."
        )


def test_steady_state_is_non_degenerate():
    """Steady-state distribution should not collapse to a single state
    when multiple recurrent states exist."""
    seq = generate_even_process(5000)
    config = CSSRConfig(L_max=3, alpha=0.01, min_count=5)
    model = CSSR(config).fit(seq)
    if model.metrics["n_states"] > 1:
        pi = model.machine.steady_state()
        nonzero = sum(1 for p in pi if p > 0.01)
        assert nonzero >= 2, (
            f"Steady state collapsed: pi={pi}, only {nonzero} states have mass > 0.01"
        )
