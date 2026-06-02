"""CSSR configuration."""

from dataclasses import dataclass, field


@dataclass
class CSSRConfig:
    """Configuration for the CSSR algorithm.

    Attributes:
        L_max: Maximum history length to consider. None triggers auto-estimate.
        alpha: Significance level for hypothesis tests (default: 0.001).
        test: Statistical test to use: "chi2" or "ks".
        correction: Multiple testing correction: "bonferroni", "bh", or None.
        min_count: Minimum suffix count required for statistical testing.
        remove_transient: Whether to remove transient (non-recurrent) states.
        epsilon: Numerical tolerance for zero-checking.
    """
    L_max: int | None = None
    alpha: float = 0.001
    test: str = "chi2"
    correction: str | None = "bonferroni"
    min_count: int = 5
    remove_transient: bool = True
    epsilon: float = 1e-10
