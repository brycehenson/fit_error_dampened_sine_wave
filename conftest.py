import pytest


@pytest.fixture(autouse=True)
def seed_and_lock_numpy_rng(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("notebook") is None:
        yield
        return

    try:
        import numpy as np
    except ModuleNotFoundError:
        yield
        return

    np.random.seed(0)
    state = np.random.get_state()
    yield
    new_state = np.random.get_state()

    same_state = (
        state[0] == new_state[0]
        and state[2:] == new_state[2:]
        and np.array_equal(state[1], new_state[1])
    )
    if not same_state:
        raise AssertionError("NumPy RNG state changed; random was called.")
