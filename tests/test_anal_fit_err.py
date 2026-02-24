"""Tests for analytic uncertainty estimates in damped sine-wave fitting."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import curve_fit

from fit_error_dampened_sine_wave.anal_fit_err import (
    analy_err_in_fit_cw_sine,
    analy_err_in_fit_damp_sine,
)
from fit_error_dampened_sine_wave.math_functions import damp_sine_wave


@dataclass(frozen=True, kw_only=True)
class _NoisyFitConfig:
    """Input parameters for repeated noisy nonlinear fits."""

    amp: float
    """Signal amplitude."""
    freq: float
    """Signal frequency in Hz."""
    phase: float
    """Signal phase in radians."""
    damp_rate: float
    """Exponential damping rate in inverse seconds."""
    samp_num: int
    """Number of uniformly-spaced samples."""
    samp_time: float
    """Total observation duration in seconds."""
    sigma_obs: float
    """Standard deviation of additive Gaussian observation noise."""


def _std_from_many_noisy_fits(
    config: _NoisyFitConfig,
    repeats: int = 120,
) -> npt.NDArray[np.float64]:
    """Estimate parameter spread from repeated nonlinear fits to noisy traces.
    See _NoisyFitConfig for input parameters."""

    np_rng = np.random.default_rng(42)
    t = np.linspace(0.0, config.samp_time, config.samp_num)
    y_clean = damp_sine_wave(
        t,
        frequency=config.freq,
        damping_rate=config.damp_rate,
        amplitude=config.amp,
        phase=config.phase,
    )

    def model(
        x: npt.NDArray[np.float64],
        fit_amp: float,
        fit_freq: float,
        fit_phase: float,
        fit_damp: float,
    ) -> npt.NDArray[np.float64]:
        return cast(
            npt.NDArray[np.float64],
            damp_sine_wave(
                x,
                frequency=fit_freq,
                damping_rate=fit_damp,
                amplitude=fit_amp,
                phase=fit_phase,
            ),
        )

    fit_results: list[npt.NDArray[np.float64]] = []
    initial_guess = np.array(
        [
            config.amp * 0.95,
            config.freq * 1.02,
            config.phase + 0.05,
            config.damp_rate * 1.05,
        ]
    )

    for _ in range(repeats):
        y_noisy = y_clean + np_rng.normal(0.0, config.sigma_obs, size=config.samp_num)
        fitted_output = curve_fit(
            model,
            t,
            y_noisy,
            p0=initial_guess,
            maxfev=20000,
            bounds=((0.0, 0.0, -2 * np.pi, 1e-8), (np.inf, np.inf, 2 * np.pi, np.inf)),
        )
        # curve_fit returns (popt, pcov):
        # - popt: best-fit parameters [amp, freq, phase, damp_rate]
        # - pcov: covariance matrix of those estimates
        # We collect only popt because this helper estimates spread from repeated fits.
        fit_results.append(cast(npt.NDArray[np.float64], fitted_output[0]))

    return cast(npt.NDArray[np.float64], np.std(np.array(fit_results), axis=0, ddof=1))


@pytest.mark.parametrize(
    "freq, cycles, damping_time_cycles",
    [
        (0.8, 4.0, 6.0),
        (1.0, 5.0, 8.0),
        (1.4, 7.5, 12.0),
    ],
)
def test_damped_fit_errors_match_spread_from_noisy_fits(
    freq: float,
    cycles: float,
    damping_time_cycles: float,
) -> None:
    """Compare analytic uncertainty against Monte Carlo spread of fitted parameters.

    Validity regime:
    - observation duration > 3 cycles
    - damping time > 5 cycles
    """

    amp = 1.3
    phase = 0.4
    sigma_obs = 0.02
    samp_num = 1500

    samp_time = cycles / freq
    damping_time = damping_time_cycles / freq
    damp_rate = 1.0 / damping_time

    analytic = analy_err_in_fit_damp_sine(
        amplitude=amp,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=sigma_obs,
        damp_rate=damp_rate,
    )

    fit_spread = _std_from_many_noisy_fits(
        config=_NoisyFitConfig(
            amp=amp,
            freq=freq,
            phase=phase,
            damp_rate=damp_rate,
            samp_num=samp_num,
            samp_time=samp_time,
            sigma_obs=sigma_obs,
        ),
    )

    analytic_std = np.array(
        [
            analytic.amplitude,
            analytic.frequency,
            analytic.phase,
            analytic.damping_rate,
        ]
    )

    np.testing.assert_allclose(analytic_std, fit_spread, rtol=0.35, atol=0.0)


@pytest.mark.parametrize("cycles", [3.25, 5.0, 9.0])
def test_damped_fit_errors_are_finite_and_positive_in_valid_range(
    cycles: float,
) -> None:
    """Check all returned uncertainties are finite and positive in valid regime."""

    freq = 1.0
    amp = 1.0
    samp_num = 1200
    sigma_obs = 0.03

    samp_time = cycles / freq
    damp_rate = freq / 7.0  # damping time = 7 cycles (> 5 cycles)

    estimate = analy_err_in_fit_damp_sine(
        amplitude=amp,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=sigma_obs,
        damp_rate=damp_rate,
    )

    values = np.array(
        [estimate.amplitude, estimate.frequency, estimate.phase, estimate.damping_rate]
    )
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)


def test_damped_fit_errors_scale_linearly_with_observation_noise() -> None:
    """Verify analytic uncertainties scale linearly with observation noise."""

    amp = 1.1
    freq = 1.0
    samp_num = 1500
    cycles = 6.0
    samp_time = cycles / freq
    damp_rate = freq / 10.0  # damping time = 10 cycles

    base_sigma = 0.01
    scale = 2.5

    est_base = analy_err_in_fit_damp_sine(
        amplitude=amp,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=base_sigma,
        damp_rate=damp_rate,
    )
    est_scaled = analy_err_in_fit_damp_sine(
        amplitude=amp,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=base_sigma * scale,
        damp_rate=damp_rate,
    )

    base_vals = np.array(
        [est_base.amplitude, est_base.frequency, est_base.phase, est_base.damping_rate]
    )
    scaled_vals = np.array(
        [
            est_scaled.amplitude,
            est_scaled.frequency,
            est_scaled.phase,
            est_scaled.damping_rate,
        ]
    )

    np.testing.assert_allclose(scaled_vals / base_vals, scale, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("damp_rate", [None, 0.0, 0])
def test_damped_fit_errors_fallback_to_undamped_when_damping_is_missing_or_zero(
    damp_rate: float | None,
) -> None:
    """Use undamped analytic uncertainties when damping rate is None or zero."""

    amplitude = 1.2
    samp_num = 1000
    samp_time = 5.0
    sigma_obs = 0.015

    damped_estimate = analy_err_in_fit_damp_sine(
        amplitude=amplitude,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=sigma_obs,
        damp_rate=damp_rate,
    )
    undamped_estimate = analy_err_in_fit_cw_sine(
        amplitude=amplitude,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=sigma_obs,
    )

    np.testing.assert_allclose(
        np.array(
            [
                damped_estimate.amplitude,
                damped_estimate.frequency,
                damped_estimate.phase,
            ]
        ),
        np.array(
            [
                undamped_estimate.amplitude,
                undamped_estimate.frequency,
                undamped_estimate.phase,
            ]
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert np.isnan(damped_estimate.damping_rate)


def test_damped_fit_errors_fallback_to_undamped_for_tiny_damping_ratio() -> None:
    """Use undamped uncertainties when samp_time / damp_time is tiny."""

    amplitude = 1.2
    samp_num = 1000
    samp_time = 5.0
    sigma_obs = 0.015
    damp_rate = 1e-8  # damp_rate * samp_time = 5e-8 << threshold

    damped_estimate = analy_err_in_fit_damp_sine(
        amplitude=amplitude,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=sigma_obs,
        damp_rate=damp_rate,
    )
    undamped_estimate = analy_err_in_fit_cw_sine(
        amplitude=amplitude,
        samp_num=samp_num,
        samp_time=samp_time,
        sigma_obs=sigma_obs,
    )

    np.testing.assert_allclose(
        np.array(
            [
                damped_estimate.amplitude,
                damped_estimate.frequency,
                damped_estimate.phase,
            ]
        ),
        np.array(
            [
                undamped_estimate.amplitude,
                undamped_estimate.frequency,
                undamped_estimate.phase,
            ]
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert np.isnan(damped_estimate.damping_rate)
