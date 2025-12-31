# snia_rise/_utils/_plt.py
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from pandas._libs.tslibs.offsets import BDay

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
    range=None,
    show_prior=False,
    **kwargs,
):
    import numpy as np
    import seaborn as sns

    prior_sample = lib.prior_sample.copy()
    post_sample = lib.post_sample.copy()

    for sample in [prior_sample, post_sample]:
        if sample is None:
            return
        sample["mean_alpha_flt1"] = sample["mean_alpha_0"][..., 0]
        sample["mean_alpha_flt2"] = sample["mean_alpha_0"][..., 1]
        sample["sigma_alpha_flt1"] = sample["sigma_alpha_0"][..., 0]
        sample["sigma_alpha_flt2"] = sample["sigma_alpha_0"][..., 1]

    # Flatten the xarray DataArray to 1D array for seaborn
    param_post = post_sample[param].values.flatten()
    param_prior = prior_sample[param].values.flatten()

    if range is not None:
        param_post = param_post[(param_post >= range[0]) & (param_post <= range[1])]
        param_prior = param_prior[(param_prior >= range[0]) & (param_prior <= range[1])]

    bw_adjust = kwargs.pop("bw_adjust", None)
    if bw_adjust is None:
        bw_adjust = (np.percentile(param_post, 95) - np.percentile(param_post, 5)) * 2
    params = dict(fill=True, alpha=0.25, lw=2, **kwargs)
    params_prior = dict(
        alpha=0.25, lw=2, linestyle="--", color=kwargs.get("color", "0.5")
    )

    sns.kdeplot(x=param_prior, ax=ax, bw_adjust=bw_adjust, **params_prior)
    # sns.histplot(
    #     x=param_prior,
    #     ax=ax,
    #     stat="density",
    #     bins=30,
    #     element="step",
    #     **params_prior,
    # )

    sns.kdeplot(x=param_post, ax=ax, bw_adjust=bw_adjust, **params)
    ax.set_xlim(range)
