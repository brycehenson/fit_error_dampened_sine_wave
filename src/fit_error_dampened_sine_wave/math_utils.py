"""Utility math helpers for statistical estimation."""

import numpy as np


def sample_std_expected_std(emp_std: float, sample_count: int) -> float:
    """Estimate uncertainty of the sample-standard-deviation estimator.

    For IID normal data with true standard deviation ``sigma`` and sample size
    ``n = sample_count``, the sample standard deviation ``s`` satisfies:

    - ``E[s] = c4(n) * sigma``
    - ``Var(s) = sigma^2 * (1 - c4(n)^2)``

    where
    ``c4(n) = sqrt(2 / (n - 1)) * Gamma(n / 2) / Gamma((n - 1) / 2)``.

    For moderate/large ``n``, ``c4(n)^2 ~= 1 - 1 / (2 * (n - 1))``, giving
    ``Std(s) ~= sigma / sqrt(2 * (n - 1))``.
    Replacing unknown ``sigma`` by measured ``s`` yields the plug-in estimate
    used here:
    ``Std(s) ~= s / sqrt(2 * (n - 1))``.

    References:
    - Wikipedia: Unbiased estimation of standard deviation.
    - SAS QC documentation: c4 function.
    - CRAN rQCC documentation: unbiasing factor ``c4``.

    Args:
        emp_std: Measured sample standard deviation.
        sample_count: Number of samples used to compute `emp_std`.

    Returns:
        Estimated uncertainty (standard deviation) of the sample-standard-deviation
        estimator. Returns `np.nan` when inputs are invalid.

    Example:
        >>> sample_std_expected_std(emp_std=0.2, sample_count=30)
        0.02626128657194451
    """

    if sample_count < 2 or not np.isfinite(emp_std):
        return float(np.nan)

    return float(emp_std / np.sqrt(2.0 * (sample_count - 1)))
