from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class FitSineUncertaintyEstimate:
    amplitude: float
    frequency: float
    phase: float


@dataclass
class FitDampSineUncertaintyEstimate(FitSineUncertaintyEstimate):
    damping_rate: float  # renamed because `lambda` is a reserved keyword


def analy_err_in_fit_cw_sine(
    amplitude: float,
    samp_num: int,
    samp_time: float,
    sigma_obs: float,
) -> FitSineUncertaintyEstimate:
    """Estimate parameter uncertainties for damped sine wave fit.

    from
    "A derivation of the errors for least squares fitting to time series data"
    Michael Houston Montgomery, D. Odonoghue

    bibtex

    @ARTICLE{MontgomeryOdonoghue1999,
    author = {{Montgomery}, M.~H. and {Odonoghue}, D.},
        title = "{A derivation of the errors for least squares fitting to time series data}",
    journal = {Delta Scuti Star Newsletter},
        year = 1999,
        month = jul,
    volume = 13,
        pages = {28},
    url = {http://adsabs.harvard.edu/abs/1999DSSN...13...28M},
    }


    Parameters
    ----------
    amp : float
        Amplitude of the sine wave.
    samp_num : int
        Number of samples.
    samp_time : float
        Total time duration of the sample.
    sigma_obs : float
        Standard deviation of the noise on observations.

    Returns
    -------
    FitSineUncertaintyEstimate
        Data structure containing uncertainty estimates.
    """

    sigma_amp = np.sqrt(2 / samp_num) * sigma_obs
    sigma_freq = np.sqrt(6 / samp_num) * (sigma_obs / (np.pi * samp_time * amplitude))
    sigma_phi = np.sqrt(2 / samp_num) * sigma_obs / amplitude

    return FitSineUncertaintyEstimate(
        amplitude=sigma_amp,
        frequency=sigma_freq,
        phase=sigma_phi,
    )


def analy_err_in_fit_damp_sine(
    amplitude: float,
    samp_num: int,
    samp_time: float,
    sigma_obs: float,
    damp_rate: Optional[float] = None,
) -> FitDampSineUncertaintyEstimate:
    """Estimate parameter uncertainties for damped sine wave fit.

    Valid when the damping time is long compare to the oscillation period.
    In practice when the damping time > ~ 3 oscillation periods, the formula should be reasonably accurate.

    Parameters
    ----------
    amplitude : float
        Amplitude of the sine wave.
    samp_num : int
        Number of samples.
    samp_time : float
        Total time duration of the sample.
    sigma_obs : float
        Standard deviation of the noise on observations.
    damp_rate : Optional[float]
        Damping rate (if None or negligible, undamped case is used).

    Returns
    -------
    FitUncertaintyEstimate
        Data structure containing uncertainty estimates.
    """
    # Treat damping as negligible when the observation duration is tiny
    # compared with the damping time: samp_time / damp_time << 1.
    damp_ratio_threshold = 1e-6
    if damp_rate is None or np.abs(damp_rate) * samp_time <= damp_ratio_threshold:
        undamped_result = analy_err_in_fit_cw_sine(
            amplitude=amplitude,
            samp_num=samp_num,
            samp_time=samp_time,
            sigma_obs=sigma_obs,
        )
        return FitDampSineUncertaintyEstimate(
            amplitude=undamped_result.amplitude,
            frequency=undamped_result.frequency,
            phase=undamped_result.phase,
            damping_rate=float("nan"),
        )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        var_a = (
            8
            * damp_rate
            * samp_time
            * sigma_obs**2
            * (
                damp_rate**2 * samp_time**2
                + damp_rate * samp_time
                - 1 / 2 * np.exp(2 * damp_rate * samp_time)
                + 1 / 2
            )
            / (
                samp_num
                * (
                    2 * damp_rate**2 * samp_time**2
                    - np.cosh(2 * damp_rate * samp_time)
                    + 1
                )
            )
        )

        sigma_amp = np.sqrt(var_a)
        # fudge factor
        # sigma_amp = sigma_amp * 1.5

        var_omega = (
            -8
            * damp_rate**3
            * samp_time
            * sigma_obs**2
            * (np.exp(2 * damp_rate * samp_time) - 1)
            / (
                amplitude**2
                * samp_num
                * (
                    2 * damp_rate**2 * samp_time**2
                    - np.cosh(2 * damp_rate * samp_time)
                    + 1
                )
            )
        )

        if np.isinf(var_omega):
            var_omega = np.nan
        sigma_freq = np.sqrt(var_omega) / (np.pi * 2)

        var_phi = (
            8
            * damp_rate
            * samp_time
            * sigma_obs**2
            * (
                damp_rate**2 * samp_time**2
                + damp_rate * samp_time
                - 1 / 2 * np.exp(2 * damp_rate * samp_time)
                + 1 / 2
            )
            / (
                amplitude**2
                * samp_num
                * (
                    2 * damp_rate**2 * samp_time**2
                    - np.cosh(2 * damp_rate * samp_time)
                    + 1
                )
            )
        )

        if np.isinf(var_phi):
            var_phi = np.nan
        sigma_phi = np.sqrt(var_phi)

        var_lambda = (
            -8
            * damp_rate**3
            * samp_time
            * sigma_obs**2
            * (np.exp(2 * damp_rate * samp_time) - 1)
            / (
                amplitude**2
                * samp_num
                * (
                    2 * damp_rate**2 * samp_time**2
                    - np.cosh(2 * damp_rate * samp_time)
                    + 1
                )
            )
        )

        if np.isinf(var_lambda):
            var_lambda = np.nan
        sigma_lambda = np.sqrt(var_lambda)

    return FitDampSineUncertaintyEstimate(
        amplitude=sigma_amp,
        frequency=sigma_freq,
        phase=sigma_phi,
        damping_rate=sigma_lambda,
    )
