"""Utility functions for CSSR."""

import math


def estimate_L_max(N: int, k: int) -> int:
    """Estimate a reasonable maximum history length.

    Heuristic: L_max <= floor(log_k(N)), capped at 10.

    Args:
        N: Sequence length.
        k: Alphabet size.

    Returns:
        Estimated L_max (>= 1).
    """
    if k <= 1:
        return 1
    if N <= k:
        return 1
    L = int(math.log(N) / max(math.log(k), 1e-10))
    return max(1, min(L, 10))


def detect_alphabet(sequence) -> set:
    """Detect the alphabet (set of unique symbols) from a sequence.

    Args:
        sequence: Iterable of hashable symbols.

    Returns:
        Set of unique symbols.
    """
    alphabet = set(sequence)
    if len(alphabet) == 0:
        raise ValueError("Cannot detect alphabet from empty sequence.")
    return alphabet
