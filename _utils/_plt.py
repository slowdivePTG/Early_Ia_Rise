# hostsub_gp/_plt.py
import matplotlib.pyplot as plt

from typing import Callable, Optional
from numpy.typing import ArrayLike

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
