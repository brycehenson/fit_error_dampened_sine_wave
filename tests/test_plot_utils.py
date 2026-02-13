"""Tests for confidence interval plotting utilities."""

import numpy as np
import pytest
from plotly.subplots import make_subplots
from src.plot_utils import make_ci_polygon, plot_curve_with_ci


def test_make_ci_polygon_with_constant_ci_returns_closed_polygon() -> None:
    """Build a closed CI polygon from scalar half-width input."""

    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 0.5, 1.0], dtype=float)

    x_poly, y_poly = make_ci_polygon(x=x, y=y, ci=0.2)

    np.testing.assert_allclose(x_poly, np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(y_poly, np.array([-0.2, 0.3, 0.8, 1.2, 0.7, 0.2, -0.2]))


def test_make_ci_polygon_with_vector_ci_uses_pointwise_width() -> None:
    """Use each CI sample to construct lower and upper polygon boundaries."""

    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    ci = np.array([0.1, 0.2, 0.3], dtype=float)

    _, y_poly = make_ci_polygon(x=x, y=y, ci=ci)

    np.testing.assert_allclose(y_poly, np.array([0.9, 1.8, 2.7, 3.3, 2.2, 1.1, 0.9]))


def test_make_ci_polygon_drops_rows_containing_nan_values() -> None:
    """Remove rows with NaN in x, y, or ci before building the polygon."""

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y = np.array([0.0, np.nan, 0.9, 0.1], dtype=float)
    ci = np.array([0.1, 0.1, 0.2, 0.2], dtype=float)

    x_poly, _ = make_ci_polygon(x=x, y=y, ci=ci)

    np.testing.assert_allclose(x_poly, np.array([0.0, 2.0, 3.0, 3.0, 2.0, 0.0, 0.0]))


def test_make_ci_polygon_requires_at_least_two_valid_samples() -> None:
    """Raise when NaN filtering leaves fewer than two samples."""

    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([np.nan, 1.0], dtype=float)

    with pytest.raises(ValueError, match="At least two valid samples"):
        make_ci_polygon(x=x, y=y, ci=0.1)


def test_plot_curve_with_ci_adds_fill_and_line_traces() -> None:
    """Create a new figure with a filled CI region and central curve line."""

    x = np.linspace(0.0, 2.0, 50)
    y = np.sin(x)

    fig = plot_curve_with_ci(x=x, y=y, ci=0.1)

    assert len(fig.data) == 2
    assert fig.data[0].fill == "toself"
    assert fig.data[0].showlegend is False
    assert fig.data[1].mode == "lines"


def test_plot_curve_with_ci_supports_subplot_add_trace_parameters() -> None:
    """Forward subplot row/col arguments to Plotly add_trace."""

    x = np.linspace(0.0, 1.0, 10)
    y = x**2
    fig = make_subplots(rows=1, cols=2)

    plot_curve_with_ci(
        x=x,
        y=y,
        ci=0.1,
        fig=fig,
        add_trace_parameters={"row": 1, "col": 2},
    )

    assert len(fig.data) == 2
    assert fig.data[0].xaxis == "x2"
    assert fig.data[1].xaxis == "x2"
