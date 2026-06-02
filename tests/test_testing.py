"""Tests for statistical testing functions."""

import pytest
from cssr.testing import chi2_test, ks_test, compare_distributions, get_corrected_alpha


def test_chi2_identical_distributions():
    """Identical distributions should give high p-value (fail to reject H0)."""
    p = chi2_test({"A": 50, "B": 50}, {"A": 50, "B": 50}, {"A", "B"})
    assert p > 0.05


def test_chi2_very_different():
    """Very different distributions should give low p-value (reject H0)."""
    p = chi2_test({"A": 100, "B": 0}, {"A": 0, "B": 100}, {"A", "B"})
    assert p < 0.001


def test_chi2_similar():
    """Similar but not identical distributions."""
    p = chi2_test({"A": 55, "B": 45}, {"A": 45, "B": 55}, {"A", "B"})
    # Should not be extremely significant with small samples
    assert p > 0.001  # not extremely different


def test_chi2_empty():
    """Empty counts should return 1.0 (not enough evidence)."""
    p = chi2_test({}, {"A": 10}, {"A", "B"})
    assert p == 1.0


def test_chi2_new_symbol():
    """Symbol only in observed, never in reference."""
    p = chi2_test({"C": 10, "A": 0}, {"A": 10}, {"A", "C"})
    # C appears in observed but never in reference => very significant
    assert p < 0.01


def test_ks_test():
    p = ks_test({"A": 50, "B": 50}, {"A": 50, "B": 50}, {"A", "B"})
    assert p > 0.05


def test_dispatcher_chi2():
    p1 = compare_distributions({"A": 50, "B": 50}, {"A": 50, "B": 50}, {"A", "B"}, "chi2")
    assert p1 > 0.05


def test_dispatcher_ks():
    p2 = compare_distributions({"A": 50, "B": 50}, {"A": 50, "B": 50}, {"A", "B"}, "ks")
    assert p2 > 0.05


def test_bonferroni_correction():
    alpha = get_corrected_alpha(0.01, 10, "bonferroni")
    assert alpha == 0.001


def test_no_correction():
    alpha = get_corrected_alpha(0.01, 10, None)
    assert alpha == 0.01
