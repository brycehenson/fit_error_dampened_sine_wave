"""Tests for statistical math utility helpers."""

import numpy as np
from src.math_utils import sample_std_expected_std


def test_sample_std_expected_std_matches_monte_carlo_spread() -> None:
    """Match plug-in prediction to Monte Carlo spread of sample std estimates."""

    np_rng = np.random.default_rng(42)
    sample_count = 30
    num_repeats = 4000
    true_sigma = 1.7

    std_estimates = np.empty(num_repeats, dtype=float)
    for repeat_i in range(num_repeats):
        draws = np_rng.normal(loc=0.0, scale=true_sigma, size=sample_count)
        std_estimates[repeat_i] = float(np.std(draws, ddof=1))

    measured_std_of_std = float(np.std(std_estimates, ddof=1))
    measured_emp_std = float(np.mean(std_estimates))
    predicted_std_of_std = sample_std_expected_std(
        emp_std=measured_emp_std,
        sample_count=sample_count,
    )

    np.testing.assert_allclose(
        measured_std_of_std,
        predicted_std_of_std,
        rtol=0.10,
        atol=0.0,
    )
