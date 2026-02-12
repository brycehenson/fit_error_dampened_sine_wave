# Repository Agent Instructions

## Testing conventions
- Add a docstring to every `test_*` function.
- Any test that uses pseudorandom data must use a seeded NumPy generator:
  - `np_rng = np.random.default_rng(42)`
- Rationale: fixed seeds prevent intermittent CI failures from random noise.
