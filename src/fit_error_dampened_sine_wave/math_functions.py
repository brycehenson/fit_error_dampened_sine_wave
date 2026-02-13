"""Numeric helpers for damped sine models and frequency aliasing."""

import numpy as np
import numpy.typing as npt


def damp_sine_wave(
    time: float | npt.NDArray[np.float64],
    frequency: float,
    damping_rate: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damped sine wave value at a given time."""

    return (
        amplitude
        * np.exp(-damping_rate * time)
        * np.sin(2 * np.pi * frequency * time + phase)
    )


def sine_wave(
    time: float | npt.NDArray[np.float64],
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> float | npt.NDArray[np.float64]:
    """Calculate the damped sine wave value at a given time."""

    return amplitude * np.sin(2 * np.pi * frequency * time + phase)


def apparent_frequency(f: float, f_s: float) -> float:
    """Compute alias frequency from true frequency and sample rate.

    from
    "Trap Frequency Measurement with a Pulsed Atom Laser"
    B. M. Henson,∗ K. F. Thomas, Z. Mehdi, T. G. Burnett, J. A. Ross,
    S. S. Hodgman, and A. G. Truscot
    https://arxiv.org/pdf/2201.10021

    This follows the piecewise formula given in the figure:
    f_a = -f + N*f_s/2 for even N
    f_a =  f - (N-1)*f_s/2 for odd N

    Both axes are assumed to be normalized by the true frequency f, so f=1 in most uses.

    Args:
        f: True frequency of the signal (must be positive and nonzero)
        f_s: Sampling frequency

    Returns:
        Apparent frequency `f_a`, normalized by `f`
    """
    assert f > 0, "True frequency f must be positive and nonzero"

    nyquist_zone_index = int(np.floor(2 * f / f_s)) + 1

    if nyquist_zone_index % 2 == 0:
        f_a: float = -f + nyquist_zone_index * f_s / 2
    else:
        f_a = f - (nyquist_zone_index - 1) * f_s / 2

    return f_a
