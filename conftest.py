"""Shared pytest fixtures for notebook-focused test behavior."""

import os
from collections.abc import Generator
from typing import Any

import numpy as np
import plotly.io as pio
import pytest


def _numpy_rng_states_equal(previous: Any, current: Any) -> bool:
    """Compare NumPy RNG states from legacy tuple or dict-based APIs.

    Args:
        previous: RNG state captured before a test.
        current: RNG state captured after a test.

    Returns:
        True when states are equivalent, False otherwise.

    Example:
        >>> _numpy_rng_states_equal(("MT19937", [1, 2], 0, 0, 0.0), ("MT19937", [1, 2], 0, 0, 0.0))
        True
    """

    if isinstance(previous, tuple) and isinstance(current, tuple):
        if len(previous) != len(current) or len(previous) < 5:
            return False
        return (
            previous[0] == current[0]
            and previous[2] == current[2]
            and previous[3] == current[3]
            and previous[4] == current[4]
            and np.array_equal(np.asarray(previous[1]), np.asarray(current[1]))
        )

    if isinstance(previous, dict) and isinstance(current, dict):
        if previous.keys() != current.keys():
            return False
        matches = True
        for key in previous:
            previous_value = previous[key]
            current_value = current[key]
            if isinstance(previous_value, np.ndarray) or isinstance(
                current_value, np.ndarray
            ):
                matches = np.array_equal(
                    np.asarray(previous_value), np.asarray(current_value)
                )
            else:
                matches = previous_value == current_value
            if not matches:
                break
        return matches

    return previous == current


@pytest.fixture(autouse=True)
def seed_and_lock_numpy_rng(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Lock NumPy RNG state for tests marked with ``notebook``.

    Args:
        request: Pytest fixture request object.

    Example:
        This fixture runs automatically for tests marked with ``notebook``.
    """

    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    np.random.seed(0)
    initial_state: Any = np.random.get_state()
    yield
    final_state: Any = np.random.get_state()
    if not _numpy_rng_states_equal(initial_state, final_state):
        raise AssertionError("NumPy RNG state changed; random was called.")


@pytest.fixture(autouse=True)
def set_plotly_renderer(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Force Plotly to use non-interactive rendering for notebook tests.

    Args:
        request: Pytest fixture request object.

    Example:
        This fixture runs automatically for tests marked with ``notebook``.
    """

    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    os.environ.setdefault("PLOTLY_RENDERER", "json")
    pio.renderers.default = "json"
    pio.renderers.render_on_display = False
    yield
