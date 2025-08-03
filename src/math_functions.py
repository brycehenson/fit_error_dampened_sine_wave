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
