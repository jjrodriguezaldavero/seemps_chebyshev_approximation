import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

line = lambda x, a, b: a + b * x

SUBPLOT_HEIGHT = 5
ASPECT_RATIO = 4 / 3

# STYLES
# fmt: off
STYLE_FIT_1 = {"linestyle": (0, (7, 5)), "marker": None, "color": "k", "alpha": 0.8, "zorder": 3}
STYLE_FIT_2 = {"linestyle": (0, (2, 5)), "marker": None, "color": "k", "alpha": 0.8, "zorder": 3}
# fmt: on


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    subplot_height: float = SUBPLOT_HEIGHT,
    aspect_ratio: float = ASPECT_RATIO,
    **kwargs,
):
    fig_height = nrows * subplot_height
    fig_width = ncols * (subplot_height * aspect_ratio)
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), **kwargs)
    return fig, axs


def set_plot_parameters(
    fig_size=(SUBPLOT_HEIGHT * ASPECT_RATIO, SUBPLOT_HEIGHT),
    font_size=30,
    title_size=30,
    label_size=30,
    major_tick_size=12,
    minor_tick_size=6,
    legend_size=30,
    marker_size=10,
    line_width=3.5,
) -> None:
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["axes.titlesize"] = title_size
    plt.rcParams["xtick.major.size"] = major_tick_size
    plt.rcParams["ytick.major.size"] = major_tick_size
    plt.rcParams["xtick.minor.size"] = minor_tick_size
    plt.rcParams["ytick.minor.size"] = minor_tick_size
    plt.rcParams["legend.fontsize"] = legend_size
    plt.rcParams["lines.markersize"] = marker_size
    plt.rcParams["lines.linewidth"] = line_width
    plt.rcParams["text.usetex"] = True
    plt.rc("font", family="serif")
    plt.rc("text", usetex=True)
    # Grid
    plt.rcParams["grid.color"] = "grey"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.4


def fit_power_law(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    style: dict = STYLE_FIT_1,
    label: str = "x",
    fit_range: slice = slice(None),
    plot_range: Optional[slice] = None,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    x_fit = x[fit_range]
    y_fit = y[fit_range]
    (a, b), _ = curve_fit(line, np.log(x_fit), np.log(y_fit))
    a, b = np.exp(a), b
    if plot_range is None:
        plot_range = fit_range
    x_plot = x[plot_range]
    y_plot = a * x_plot**b
    label = rf"$\mathcal{{O}}({label}^{{{b:.2f}}})$"
    ax.plot(x_plot, y_plot, label=label, **style)


def fit_exponential(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    style: dict = STYLE_FIT_1,
    base: float = np.e,
    label: str = "x",
    fit_range: slice = slice(None),
    plot_range: Optional[slice] = None,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    x_fit = x[fit_range]
    y_fit = y[fit_range]
    if base == np.e:
        (a, b), _ = curve_fit(line, x_fit, np.log(y_fit))
        a, b = np.exp(a), b
        label = rf"$\mathcal{{O}}(e^{{{round(b, 2)}{label}}})$"
    else:
        (a, b), _ = curve_fit(line, x_fit, np.emath.logn(base, y_fit))
        a, b = base**a, b
        label = rf"$\mathcal{{O}}({base}^{{{round(b, 2)}{label}}})$"
    if plot_range is None:
        plot_range = fit_range
    x_plot = x[plot_range]
    y_plot = a * base ** (b * x_plot)
    ax.plot(x_plot, y_plot, label=label, **style)


def integer_ticks(x_range, step, start_idx=None, start_value=None, stop_value=None):
    """Computes the x-ticks in a given format for plots"""
    if start_idx is not None and (start_idx < 0 or start_idx >= len(x_range)):
        raise ValueError("start_idx is out of the range of x_range")
    if start_value is not None:
        tick_start = start_value
    elif start_idx is not None:
        tick_start = int(x_range[start_idx])
    else:
        tick_start = int(min(x_range))
    tick_end = max(x_range) + 1 if stop_value is None else stop_value
    ticks = list(range(tick_start, tick_end + 1, step))
    return ticks


def scientific_format(number):
    """Computes the scientific formatting of a number."""
    sci_notation = "{:.2e}".format(number)
    mantissa, exponent = sci_notation.split("e")
    exponent = exponent.replace("+", "").lstrip("0")
    if exponent == "":
        exponent = "0"
    return f"{mantissa} ⋅ 10^{{{exponent}}}"


def custom_errorbar(
    ax: Axes, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, alpha_fill=0.25, **kwargs
):
    ax.errorbar(x, y, yerr, elinewidth=2, capsize=4, capthick=2, **kwargs)
    # Shaded errorbar
    # line = ax.errorbar(x, y, yerr, elinewidth=2, capsize=4, capthick=2, **kwargs)
    # ax.fill_between(x, y - yerr, y + yerr, color=line[0].get_color(), alpha=alpha_fill)
