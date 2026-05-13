# snia_rise/_utils/_plt.py
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..model.lightcurve import SNLightCurveLib

def set_plot_style():
    """
    Set a consistent plot style for matplotlib plots.
    """
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Ariel",
            "font.size": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )


set_plot_style()


def show_and_save(f: Callable) -> Callable:
    """
    A decorator that adds common plotting options (show and save) to any plotting function.
    By default, the plot is not shown or saved.

    Parameters
    ----------
    f : Callable
        The plotting function to be decorated

    Returns
    -------
    Callable
        The wrapped function with additional show and save parameters
    """
    from functools import wraps

    @wraps(f)
    def wrapper(*args, show: bool = False, save: Optional[str] = None, **kwargs):
        # Call the original plotting function
        result = f(*args, **kwargs)

        # Handle saving if a path is provided
        if save:
            plt.savefig(save)

        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()

        return result

    return wrapper


def plot_box_spec(
    wave: ArrayLike, flux: ArrayLike, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """
    Plot a box spectrum.

    Parameters
    ----------
    wave : array-like
        The wavelength array
    flux : array-like
        The flux array
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """

    from ._data_binning import get_box_spec

    wave_plot, flux_plot = get_box_spec(wave, flux)
    try:
        ax = plt.gca() if ax is None else ax
    except ValueError:
        fig, ax = plt.subplots()

    ax.plot(wave_plot, flux_plot, **kwargs)

    return ax


def show_kde_posterior(
    lib: "SNLightCurveLib",
    param: str,
    ax: plt.Axes | list[plt.Axes],
    idx_filt: int = None,
    x_range=None,
    **kwargs,
):
    import numpy as np
    import seaborn as sns

    sample = lib.post_sample.copy()

    if idx_filt is None:
        param_1d = sample[param].values.flatten()
    else:
        param_1d = sample[param].values[..., idx_filt].flatten()

    if x_range is not None:
        param_1d = param_1d[(param_1d >= x_range[0]) & (param_1d <= x_range[1])]

    if len(param_1d) == 0:
        print(f"Warning: No data for parameter '{param}' in range {x_range}")
        return

    bw_adjust = kwargs.pop("bw_adjust", 1)
    params = dict(fill=True, alpha=0.25, lw=2)
    params.update(**kwargs)

    sns.kdeplot(x=param_1d, ax=ax, bw_adjust=bw_adjust, **params)
    ax.set_xlim(x_range)
    ax.set_ylabel("")


def show_param_comparison(
    mock_libs,
    colors,
    labels,
    param,
    x_range_mean=None,
    x_range_sigma=None,
    idx_filt=None,
    truths=None,
    ax=None,
    sublibs=None,
):
    """Generic two-panel KDE comparison for a parameter's mean and sigma."""

    n_lib = len(mock_libs)

    if ax is None:
        fig, ax = plt.subplots(
            1, 2, figsize=(12, 3), constrained_layout=True, sharey="col", sharex="col"
        )

    if truths is not None:
        for j in range(len(ax)):
            ax[j].axvline(
                truths[j], color="0.5", linestyle=":", lw=5, alpha=0.5, zorder=-1
            )

    param_mean = f"mean_{param}"
    param_sigma = f"sigma_{param}"

    for k in range(n_lib):
        mock_lib = mock_libs[k]
        color = colors[k]
        label = labels[k]
        show_kde_posterior(
            mock_lib,
            param=param_mean,
            idx_filt=idx_filt,
            ax=ax[0],
            x_range=x_range_mean,
            color=color,
            label=label,
        )
        show_kde_posterior(
            mock_lib,
            param=param_sigma,
            idx_filt=idx_filt,
            ax=ax[1],
            x_range=x_range_sigma,
            color=color,
            label=label,
        )
        for _ax in ax[1:]:
            _ax.set_ylabel("")
        for _ax in ax:
            _ax.set_yticks([])

    if sublibs is not None:
        for k in range(n_lib):
            if sublibs[k] is not None:
                show_kde_posterior(
                    sublibs[k],
                    param=param_mean,
                    idx_filt=idx_filt,
                    ax=ax[0],
                    x_range=x_range_mean,
                    color=colors[k],
                    linestyle="--",
                    alpha=0.1,
                )
                show_kde_posterior(
                    sublibs[k],
                    param=param_sigma,
                    idx_filt=idx_filt,
                    ax=ax[1],
                    x_range=x_range_sigma,
                    color=colors[k],
                    linestyle="--",
                    alpha=0.1,
                )

    return ax


def show_weighted_mean_comparison(
    mock_libs,
    colors,
    labels,
    param,
    x_range_mean=None,
    x_range_weighted=None,
    idx_filt=None,
    truth=None,
    ax=None,
    sublibs=None,
    param_weighted=None,
):
    """Two-panel KDE: naive population mean (left) vs weighted bootstrap mean (right).

    Parameters
    ----------
    mock_libs : list
        List of SNLightCurveLib objects.
    colors : list
        Colors for each library.
    labels : list
        Labels for each library.
    param : str
        Base parameter name.  The left panel shows ``mean_{param}``.
        The right panel shows ``weighted_{param}`` (or *param_weighted*
        if provided), falling back to ``mean_{param}`` when the weighted
        variable is absent from a library's post_sample.
    x_range_mean, x_range_weighted : tuple or None
        x-axis limits for each panel.
    idx_filt : int or None
        Filter index for per-filter parameters.  Passed through to
        :func:`show_kde_posterior`.
    truth : float or None
        Vertical reference line.
    ax : array of Axes or None
        Pre-existing axes (1×2 layout expected).
    sublibs : list or None
        Sublibraries for faint overlay (same length as mock_libs).
    param_weighted : str or None
        Explicit name for the weighted parameter.  If None,
        ``weighted_{param}`` is used.
    """
    n_lib = len(mock_libs)

    if ax is None:
        fig, ax = plt.subplots(
            1, 2, figsize=(12, 3), constrained_layout=True, sharey="col", sharex="col"
        )

    if truth is not None:
        for j in range(len(ax)):
            ax[j].axvline(
                truth, color="0.5", linestyle=":", lw=5, alpha=0.5, zorder=-1
            )

    param_naive = f"mean_{param}"
    param_w = param_weighted if param_weighted is not None else f"weighted_{param}"

    for k in range(n_lib):
        mock_lib = mock_libs[k]
        color = colors[k]
        label = labels[k]

        show_kde_posterior(
            mock_lib,
            param=param_naive,
            idx_filt=idx_filt,
            ax=ax[0],
            x_range=x_range_mean,
            color=color,
            label=label,
        )

        p_right = param_w if param_w in mock_lib.post_sample else param_naive
        show_kde_posterior(
            mock_lib,
            param=p_right,
            idx_filt=idx_filt,
            ax=ax[1],
            x_range=x_range_weighted,
            color=color,
            label=label,
        )

        for _ax in ax[1:]:
            _ax.set_ylabel("")
        for _ax in ax:
            _ax.set_yticks([])

    if sublibs is not None:
        for k in range(n_lib):
            if sublibs[k] is not None:
                show_kde_posterior(
                    sublibs[k],
                    param=param_naive,
                    idx_filt=idx_filt,
                    ax=ax[0],
                    x_range=x_range_mean,
                    color=colors[k],
                    linestyle="--",
                    alpha=0.1,
                )
                p_right = (
                    param_w
                    if param_w in sublibs[k].post_sample
                    else param_naive
                )
                show_kde_posterior(
                    sublibs[k],
                    param=p_right,
                    idx_filt=idx_filt,
                    ax=ax[1],
                    x_range=x_range_weighted,
                    color=colors[k],
                    linestyle="--",
                    alpha=0.1,
                )

    return ax


def show_corr_comparison(mock_libs, colors, labels, truths=None, ax=None, sublibs=None):
    if ax is None:  # Create subplots if no axes are provided
        fix, ax = plt.subplots(
            1, 4, figsize=(14, 4), constrained_layout=True, sharey="col", sharex="col"
        )
    n_lib = len(mock_libs)

    if truths is not None:
        for j in range(len(ax)):
            ax[j].axvline(
                truths[j], color="0.5", linestyle=":", lw=5, alpha=0.5, zorder=-1
            )

    params = [
        "corr_t_rise_alpha_flt2",
        "corr_t_rise_log_Aprime_flt2",
        "corr_alpha_log_Aprime_flt2",
        "corr_alpha_flt1_flt2",
    ]
    for k in range(n_lib):
        mock_lib = mock_libs[k]
        color = colors[k]
        label = labels[k]
        for i in range(len(ax)):
            param = params[i]
            show_kde_posterior(
                mock_lib,
                param=param,
                ax=ax[i],
                x_range=(-1, 1),
                color=color,
                label=label,
            )
        for _ax in ax[1:]:
            _ax.set_ylabel("")
        for _ax in ax:
            _ax.set_yticks([])

    if sublibs is not None:
        for k in range(n_lib):
            if sublibs[k] is not None:
                for i in range(len(ax)):
                    show_kde_posterior(
                        sublibs[k],
                        param=params[i],
                        ax=ax[i],
                        x_range=(-1, 1),
                        color=colors[k],
                        linestyle="--",
                        alpha=0.1,
                    )

    try:
        ax[0].set_xlabel(r"$\rho(t_\mathrm{rise}, \alpha_r)$")
        ax[1].set_xlabel(r"$\rho(t_\mathrm{rise}, \ln A_r)$")
        ax[2].set_xlabel(r"$\rho(\alpha_r, \ln A_r)$")
        ax[3].set_xlabel(r"$\rho(\alpha_g, \alpha_r)$")
    except IndexError:
        pass

    return ax


def show_t_rise_corr_comparison(mock_libs, colors, labels, truths=None, ax=None):
    if ax is None:  # Create subplots if no axes are provided
        fix, ax = plt.subplots(
            1, 4, figsize=(14, 3), constrained_layout=True, sharey="col", sharex="col"
        )
    n_lib = len(mock_libs)

    if truths is not None:
        for j in range(len(ax)):
            # if truths[j] is not None:
            #     for k in range(n_lib):
            #         _ax = ax[k, j]
            ax[j].axvline(
                truths[j], color="0.5", linestyle=":", lw=5, alpha=0.5, zorder=-1
            )

    params = [
        "corr_t_rise_alpha_flt2",
        "corr_t_rise_alpha_flt1-flt2",
        "corr_alpha_flt1_flt2",
        "corr_t_rise_log_Aprime_flt2",
    ]
    for k in range(n_lib):
        mock_lib = mock_libs[k]
        color = colors[k]
        label = labels[k]
        for i in range(len(ax)):
            param = params[i]
            show_kde_posterior(
                mock_lib,
                param=param,
                ax=ax[i],
                x_range=(-1, 1),
                color=color,
                label=label,
            )
        for _ax in ax[1:]:
            _ax.set_ylabel("")
        for _ax in ax:
            _ax.set_yticks([])

    try:
        ax[0].set_xlabel(r"$\rho(t_\mathrm{rise}, \alpha_r)$")
        ax[1].set_xlabel(r"$\rho(t_\mathrm{rise}, \alpha_g - \alpha_r)$")
        ax[2].set_xlabel(r"$\rho(\alpha_g, \alpha_r)$")
        ax[3].set_xlabel(r"$\rho(t_\mathrm{rise}, \ln A_r)$")
    except IndexError:
        pass

    return ax
