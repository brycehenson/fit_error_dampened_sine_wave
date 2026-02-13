"""Plotly helpers for plotting curves with symmetric confidence interval bands."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go


def _validate_and_clean_ci_inputs(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ci: float | npt.NDArray[np.float64],
    *,
    sort_by_x: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate curve inputs and remove rows containing NaN values.

    Args:
        x: Sample x coordinates with shape `(n,)`.
        y: Central curve values with shape `(n,)`.
        ci: Symmetric confidence interval half-width, either scalar or shape `(n,)`.
        sort_by_x: Whether to sort cleaned samples by ascending x coordinate.

    Returns:
        Cleaned arrays `(x_clean, y_clean, ci_clean)`, each with shape `(m,)`.

    Raises:
        ValueError: If inputs are not 1D, shapes differ, fewer than 2 valid rows
            remain after NaN removal, or CI contains negative values.

    Example:
        >>> x = np.array([0.0, 1.0, 2.0])
        >>> y = np.array([1.0, np.nan, 3.0])
        >>> x_clean, y_clean, ci_clean = _validate_and_clean_ci_inputs(
        ...     x=x,
        ...     y=y,
        ...     ci=0.2,
        ...     sort_by_x=False,
        ... )
        >>> x_clean
        array([0., 2.])
    """

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have identical shapes.")

    if np.isscalar(ci):
        ci_array = np.full_like(x_array, fill_value=float(ci), dtype=float)
    else:
        ci_array = np.asarray(ci, dtype=float)
        if ci_array.ndim != 1:
            raise ValueError("ci must be a scalar or one-dimensional array.")
        if ci_array.shape != x_array.shape:
            raise ValueError("When ci is an array, it must match x and y shape.")

    valid_mask = ~(np.isnan(x_array) | np.isnan(y_array) | np.isnan(ci_array))
    x_clean = x_array[valid_mask]
    y_clean = y_array[valid_mask]
    ci_clean = ci_array[valid_mask]

    if x_clean.size < 2:
        raise ValueError("At least two valid samples are required after NaN removal.")
    if np.any(ci_clean < 0.0):
        raise ValueError("ci half-width values must be non-negative.")

    if sort_by_x:
        sort_idx = np.argsort(x_clean)
        x_clean = x_clean[sort_idx]
        y_clean = y_clean[sort_idx]
        ci_clean = ci_clean[sort_idx]

    return x_clean, y_clean, ci_clean


def make_ci_polygon(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ci: float | npt.NDArray[np.float64],
    *,
    sort_by_x: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build a closed polygon describing a symmetric confidence interval band.

    The polygon starts at the lower boundary and moves forward in x, then returns
    along the upper boundary in reverse, and repeats the first point to close.

    Args:
        x: Sample x coordinates with shape `(n,)`.
        y: Central curve values with shape `(n,)`.
        ci: Symmetric half-width of the confidence interval. Can be a scalar or
            an array with shape `(n,)`.
        sort_by_x: If true, sort valid samples by x after NaN removal.

    Returns:
        Tuple `(x_poly, y_poly)` where both arrays have shape `(2m + 1,)`, with
        `m` equal to the number of valid samples after filtering.

    Raises:
        ValueError: If input validation fails or fewer than two valid samples
            remain after NaN removal.

    Example:
        >>> x = np.linspace(0.0, 10.0, 200)
        >>> y = np.sin(x)
        >>> x_poly, y_poly = make_ci_polygon(x, y, 0.1)
    """

    x_clean, y_clean, ci_clean = _validate_and_clean_ci_inputs(
        x=x,
        y=y,
        ci=ci,
        sort_by_x=sort_by_x,
    )
    y_low = y_clean - ci_clean
    y_high = y_clean + ci_clean

    x_poly = np.concatenate((x_clean, x_clean[::-1], x_clean[:1]))
    y_poly = np.concatenate((y_low, y_high[::-1], y_low[:1]))
    return x_poly, y_poly


def plot_curve_with_ci(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ci: float | npt.NDArray[np.float64],
    *,
    fig: go.Figure | go.FigureWidget | None = None,
    add_trace_parameters: Mapping[str, Any] | None = None,
    curve_name: str = "Curve",
    ci_name: str = "CI",
    curve_line: Mapping[str, Any] | None = None,
    ci_fillcolor: str = "rgba(31, 119, 180, 0.20)",
    ci_showlegend: bool = False,
    sort_by_x: bool = False,
) -> go.Figure | go.FigureWidget:
    """Plot a curve and its symmetric confidence interval band in a Plotly figure.

    Args:
        x: Sample x coordinates with shape `(n,)`.
        y: Central curve values with shape `(n,)`.
        ci: Symmetric half-width of the confidence interval. Can be a scalar or
            an array with shape `(n,)`.
        fig: Existing figure to draw into. If omitted, a new Figure is created.
        add_trace_parameters: Optional keyword arguments forwarded to `fig.add_trace`,
            e.g. `{"row": 1, "col": 2}` for subplot placement.
        curve_name: Legend label for the central curve.
        ci_name: Legend label for the CI polygon.
        curve_line: Optional Plotly line style mapping for the curve trace.
        ci_fillcolor: Fill color of the confidence polygon.
        ci_showlegend: Whether the CI trace should appear in the legend.
        sort_by_x: If true, sort valid samples by x after NaN removal.

    Returns:
        Plotly figure containing the CI polygon trace and the curve trace.

    Raises:
        ValueError: If input validation fails.

    Example:
        >>> x = np.linspace(0.0, 6.0, 200)
        >>> y = np.exp(-0.2 * x) * np.cos(2.0 * x)
        >>> fig = plot_curve_with_ci(x, y, 0.05)
    """

    # pylint: disable=too-many-arguments,too-many-locals
    target_fig: go.Figure | go.FigureWidget = go.Figure() if fig is None else fig
    trace_kwargs = dict(add_trace_parameters or {})

    x_clean, y_clean, _ = _validate_and_clean_ci_inputs(
        x=x,
        y=y,
        ci=ci,
        sort_by_x=sort_by_x,
    )
    x_poly, y_poly = make_ci_polygon(x=x, y=y, ci=ci, sort_by_x=sort_by_x)

    ci_trace = go.Scatter(
        x=x_poly,
        y=y_poly,
        mode="lines",
        fill="toself",
        fillcolor=ci_fillcolor,
        line={"width": 0.0, "color": "rgba(0, 0, 0, 0)"},
        name=ci_name,
        showlegend=ci_showlegend,
        hoverinfo="skip",
    )
    target_fig.add_trace(ci_trace, **trace_kwargs)

    curve_trace = go.Scatter(
        x=x_clean,
        y=y_clean,
        mode="lines",
        line=dict(curve_line or {}),
        name=curve_name,
        showlegend=True,
    )
    target_fig.add_trace(curve_trace, **trace_kwargs)

    return target_fig
