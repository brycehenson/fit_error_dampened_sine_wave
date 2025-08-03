import math
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


def analy_err_in_fit_sine(
    amplitude: float,
    samp_num: int,
    samp_time: float,
    sigma_obs: float,
) -> FitSineUncertaintyEstimate:
    """Estimate parameter uncertainties for damped sine wave fit.

    from https://github.com/HeBECANU/Core_BEC_Analysis/blob/dev/lib/fits/analy_err_in_fit_sine.m

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
    sigma_phi = np.sqrt(2 / samp_num) * sigma_obs / amplitude * 2

    return FitSineUncertaintyEstimate(
        amplitude=sigma_amp,
        frequency=sigma_freq,
        phase=sigma_phi,
    )


def analy_err_in_fit_damp_sine(
    amp: float,
    samp_num: int,
    samp_time: float,
    sigma_obs: float,
    damp_rate: Optional[float] = None,
) -> FitDampSineUncertaintyEstimate:
    """Estimate parameter uncertainties for damped sine wave fit.


    from https://github.com/HeBECANU/Core_BEC_Analysis/blob/dev/lib/fits/analy_err_in_fit_sine.m

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
    damp_rate : Optional[float]
        Damping rate (if None or negligible, undamped case is used).

    Returns
    -------
    FitUncertaintyEstimate
        Data structure containing uncertainty estimates.
    """
    damp_threshold = 1e-9

    damp_time = 1 / damp_rate
    if damp_rate is not None and samp_time / damp_time > damp_threshold:

        coth_term = 1 / np.tanh(damp_rate * samp_time)
        sigma_amp = (
            sigma_obs
            * np.sqrt(2 / samp_num)
            * np.sqrt(samp_time * damp_rate * (1 + coth_term))
        )
        unc_undamped = np.sqrt(2 / samp_num) * sigma_obs
        sigma_amp = sigma_amp + unc_undamped
        # fudge factor
        # sigma_amp = sigma_amp * 1.5

        num_freq = samp_time * damp_rate**3 * (-1 + np.exp(2 * damp_rate * samp_time))
        den_freq = (
            -1 - 2 * (damp_rate * samp_time) ** 2 + np.cosh(2 * damp_rate * samp_time)
        )
        sigma_freq = (
            (sigma_obs)
            / (amp * np.pi * np.sqrt(samp_num))
            * np.sqrt(num_freq / den_freq)
        )
        # formula says 4
        sigma_freq = sigma_freq * np.sqrt(2)

        num_phi = (
            damp_rate
            * samp_time
            * (
                -1
                + np.exp(2 * damp_rate * samp_time)
                - 2 * damp_rate * samp_time * (1 + damp_rate * samp_time)
            )
        )
        den_phi = (
            -1 - 2 * (damp_rate * samp_time) ** 2 + np.cosh(2 * damp_rate * samp_time)
        )
        sigma_phi = np.sqrt(4 / samp_num) * sigma_obs / amp * np.sqrt(num_phi / den_phi)

        num_lam = (
            damp_rate**3
            * samp_time
            * np.exp(2 * damp_rate * samp_time)
            * (
                -2 * damp_rate * samp_time * (damp_rate * samp_time + 1)
                + np.exp(2 * damp_rate * samp_time)
                - 1
            )
        )
        den_lam = (
            (2 * damp_rate - 1) * np.exp(2 * damp_rate * samp_time)
            + 2 * damp_rate**2 * (samp_time - 2) * samp_time
            + 2 * damp_rate * samp_time
            - 2 * damp_rate
            + 1
        ) ** 2
        sigma_lambda = (
            2 * (sigma_obs / (amp * np.sqrt(samp_num))) * np.sqrt(num_lam / den_lam)
        )

    else:
        result = analy_err_in_fit_sine(
            amplitude=amp, samp_num=samp_num, samp_time=samp_time, sigma_obs=sigma_obs
        )
        sigma_amp = result.amplitude
        sigma_freq = result.frequency
        sigma_phi = result.phase
        sigma_lambda = float("nan")

    return FitDampSineUncertaintyEstimate(
        amplitude=sigma_amp,
        frequency=sigma_freq,
        phase=sigma_phi,
        damping_rate=sigma_lambda,
    )


def analy_err_in_fit_damp_sine_testing(
    amp: float,
    samp_num: int,
    samp_time: float,
    sigma_obs: float,
    damp_rate: Optional[float] = None,
) -> FitDampSineUncertaintyEstimate:
    """Estimate parameter uncertainties for damped sine wave fit.


    from https://github.com/HeBECANU/Core_BEC_Analysis/blob/dev/lib/fits/analy_err_in_fit_sine.m

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
    damp_rate : Optional[float]
        Damping rate (if None or negligible, undamped case is used).

    Returns
    -------
    FitUncertaintyEstimate
        Data structure containing uncertainty estimates.
    """

    var_a = (
        8
        * damp_rate
        * samp_time
        * sigma_obs**2
        * (
            2 * damp_rate**2 * samp_time**2
            + 2 * damp_rate * samp_time
            - np.exp(2 * damp_rate * samp_time)
            + 1
        )
        * np.exp(2 * damp_rate * samp_time)
        / (
            samp_num
            * (
                4 * damp_rate**2 * samp_time**2 * np.exp(2 * damp_rate * samp_time)
                - np.exp(4 * damp_rate * samp_time)
                + 2 * np.exp(2 * damp_rate * samp_time)
                - 1
            )
        )
    )

    sigma_amp = np.sqrt(var_a)
    # fudge factor
    # sigma_amp = sigma_amp * 1.5

    var_omega = (
        16
        * damp_rate**3
        * samp_time
        * sigma_obs**2
        * (1 - math.exp(2 * damp_rate * samp_time))
        * math.exp(2 * damp_rate * samp_time)
        / (
            amp**2
            * samp_num
            * (
                4 * damp_rate**2 * samp_time**2 * math.exp(2 * damp_rate * samp_time)
                - math.exp(4 * damp_rate * samp_time)
                + 2 * math.exp(2 * damp_rate * samp_time)
                - 1
            )
        )
    )

    sigma_freq = math.sqrt(var_omega) / (np.pi * 2)

    var_phi = (
        8
        * damp_rate
        * samp_time
        * sigma_obs**2
        * (
            2 * damp_rate**2 * samp_time**2
            + 2 * damp_rate * samp_time
            - math.exp(2 * damp_rate * samp_time)
            + 1
        )
        * math.exp(2 * damp_rate * samp_time)
        / (
            amp**2
            * samp_num
            * (
                4 * damp_rate**2 * samp_time**2 * math.exp(2 * damp_rate * samp_time)
                - math.exp(4 * damp_rate * samp_time)
                + 2 * math.exp(2 * damp_rate * samp_time)
                - 1
            )
        )
    )

    sigma_phi = math.sqrt(var_phi)

    var_lambda = (
        16
        * damp_rate**3
        * samp_time
        * sigma_obs**2
        * (1 - math.exp(2 * damp_rate * samp_time))
        * math.exp(2 * damp_rate * samp_time)
        / (
            amp**2
            * samp_num
            * (
                4 * damp_rate**2 * samp_time**2 * math.exp(2 * damp_rate * samp_time)
                - math.exp(4 * damp_rate * samp_time)
                + 2 * math.exp(2 * damp_rate * samp_time)
                - 1
            )
        )
    )

    sigma_lambda = math.sqrt(var_lambda)

    return FitDampSineUncertaintyEstimate(
        amplitude=sigma_amp,
        frequency=sigma_freq,
        phase=sigma_phi,
        damping_rate=sigma_lambda,
    )
