"""Statistical tests for CSSR state splitting.

Provides chi-squared and Kolmogorov-Smirnov two-sample tests for comparing
conditional probability distributions of discrete symbols, plus multiple
testing correction via Bonferroni or Benjamini-Hochberg.
"""

import math
from collections import defaultdict

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def chi2_test(counts_observed: dict, counts_reference: dict, alphabet: set) -> float:
    """Two-sample chi-squared test for discrete multinomial distributions.

    Tests H0: the observed distribution equals the reference distribution.
    Returns a p-value. Low p-values indicate the distributions differ.

    NOTE: This test is asymmetric — swapping `counts_observed` and
    `counts_reference` may produce different p-values. The first argument
    should be the smaller sample (the suffix being tested); the second
    should be the larger reference (the parent state).

    Args:
        counts_observed: Observed next-symbol counts (from the suffix being tested).
        counts_reference: Reference next-symbol counts (from the parent state).
        alphabet: Full set of possible symbols.

    Returns:
        p-value from the chi-squared distribution.
    """
    n_obs = sum(counts_observed.values())
    n_ref = sum(counts_reference.values())

    if n_obs == 0 or n_ref == 0:
        return 1.0  # Not enough data to reject

    chi2_stat = 0.0
    df = 0

    for sym in alphabet:
        observed = counts_observed.get(sym, 0)
        ref_prob = counts_reference.get(sym, 0) / n_ref if n_ref > 0 else 0
        expected = n_obs * ref_prob

        if expected > 1e-10:
            chi2_stat += (observed - expected) ** 2 / expected
            df += 1
        elif observed > 0:
            # Symbol seen in observed but never in reference
            return 0.0

    # effective degrees of freedom
    df = max(df - 1, 1)

    if chi2_stat > 1e10:
        return 0.0

    return 1.0 - stats.chi2.cdf(chi2_stat, df)


def ks_test(counts_observed: dict, counts_reference: dict, alphabet: set) -> float:
    """Two-sample Kolmogorov-Smirnov test for discrete distributions.

    Converts counts to empirical CDFs and compares. More conservative than chi-squared
    for discrete data, but provided for compatibility with original CSSR.

    Args:
        counts_observed: Observed next-symbol counts.
        counts_reference: Reference next-symbol counts.
        alphabet: Full set of possible symbols.

    Returns:
        p-value from the KS test.
    """
    n_obs = sum(counts_observed.values())
    n_ref = sum(counts_reference.values())

    if n_obs == 0 or n_ref == 0:
        return 1.0

    sorted_alphabet = sorted(alphabet, key=lambda s: (type(s).__name__, str(s)))

    obs_ecdf = []
    ref_ecdf = []
    obs_cum = 0.0
    ref_cum = 0.0

    for sym in sorted_alphabet:
        obs_cum += counts_observed.get(sym, 0) / n_obs
        ref_cum += counts_reference.get(sym, 0) / n_ref
        obs_ecdf.append(obs_cum)
        ref_ecdf.append(ref_cum)

    d_stat = max(abs(o - r) for o, r in zip(obs_ecdf, ref_ecdf))

    # Two-sample KS: effective n = n1*n2/(n1+n2)
    effective_n = n_obs * n_ref / (n_obs + n_ref)
    lambda_stat = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * d_stat

    # Kolmogorov distribution
    p_value = 0.0
    for k in range(1, 100):
        p_value += 2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lambda_stat * lambda_stat)
    p_value = min(1.0, max(0.0, p_value))

    return p_value


def compare_distributions(
    counts_a: dict, counts_b: dict, alphabet: set, method: str = "chi2"
) -> float:
    """Compare two discrete distributions via the specified test method.

    Args:
        counts_a: First distribution's counts.
        counts_b: Second distribution's counts.
        alphabet: Full set of symbols.
        method: "chi2" or "ks".

    Returns:
        p-value (low => distributions differ).
    """
    if method == "ks":
        return ks_test(counts_a, counts_b, alphabet)
    return chi2_test(counts_a, counts_b, alphabet)


def correct_pvalues(pvalues: list[float], alpha: float, method: str | None) -> list[float]:
    """Apply multiple testing correction to a list of p-values.

    Args:
        pvalues: Raw p-values from individual tests.
        alpha: Desired family-wise significance level.
        method: "bonferroni", "bh" (Benjamini-Hochberg), or None (no correction).

    Returns:
        For "bonferroni": a list of the same Bonferroni threshold repeated.
        For "bh": BH-corrected p-values — compare each against `alpha` directly
                 (reject H0 when corrected_p < alpha).
        For None: a list of the original alpha repeated.
        Length always matches len(pvalues).
    """
    n = len(pvalues)
    if n == 0 or method is None:
        return [alpha] * n

    if method == "bonferroni":
        return [alpha / n] * n

    if method in ("bh", "fdr_bh"):
        # Use statsmodels for Benjamini-Hochberg
        # multipletests returns: (rejections, corrected_pvalues, alpha_sidak, alpha_bonf)
        _, corrected, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
        return list(corrected)

    return [alpha] * n


def get_corrected_alpha(alpha: float, n_tests: int, method: str | None) -> float:
    """Get the corrected significance threshold for a batch of n_tests.

    Note: This only works for Bonferroni and no-correction. For BH/FDR,
    the threshold depends on p-value ranking and cannot be pre-computed
    to a single value. Use correct_pvalues() instead for BH.

    Args:
        alpha: Original significance level.
        n_tests: Number of tests in this batch.
        method: "bonferroni", or None.

    Returns:
        Corrected alpha value for each individual test.

    Raises:
        ValueError: If method is "bh" (BH cannot be reduced to a single threshold).
    """
    if n_tests == 0 or method is None:
        return alpha
    if method in ("bonferroni",):
        return alpha / n_tests if n_tests > 0 else alpha
    if method in ("bh", "fdr_bh"):
        raise ValueError(
            "BH/FDR correction cannot be reduced to a single threshold. "
            "Use correct_pvalues(pvalues, alpha, 'bh') instead: collect all "
            "p-values, get back BH-corrected p-values, and compare each "
            "against alpha."
        )
    return alpha
