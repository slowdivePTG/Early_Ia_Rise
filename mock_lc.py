from matplotlib.pylab import ranf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from matplotlib.ticker import MultipleLocator
from requests import post
import seaborn as sns

import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist, infer

# %matplotlib notebook

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

from fit_early_lc import f_t, Ia_lc


def generate_flux_err(flux, ZP=0, method="broken_power_law"):
    """
    Estimate the flux error based on the broken power law relation between magnitude and S/N
    """
    mag = -2.5 * jnp.log10(flux) + ZP
    if method == "broken_power_law":

        def log_broken_powerlaw(x, x0=18.2123, y0=-8.9946, k1=-0.3044, k2=0):
            """
            Broken power law function - mag v.s. S/N (default values adopted from the fit to the r-band data in Yao et al. 2019)
            """
            log_err = 0.5 * jnp.log10((jnp.power(10, (y0 + k1 * (x - x0)) * 2) + jnp.power(10, (y0 + k2 * (x - x0)) * 2)))
            log_sky_noise = y0
            return np.where(np.isfinite(x) & ~np.isnan(x), log_err, log_sky_noise)
                
        return 10 ** (log_broken_powerlaw(mag) + 0.4 * ZP)
    else:
        raise ValueError("Method not recognized")


class Ia_lc_lib:

    def __init__(self, cadence, n_lc, tfl_true, C_true, A_true, alpha_true, mag_peak: float = 17.5):
        t_sample = jnp.arange(-100, 0, step=cadence)
        n_sample = len(t_sample)

        lc_lib = np.empty(shape=n_lc, dtype=object)

        for k in range(n_lc):
            np.random.seed(k + 114514)
            t_jitter = np.random.randn(n_sample) * 0.05
            t_jitter -= t_jitter[
                np.argmin(np.abs(t_sample - tfl_true))
            ]  # ensure jitter=0 when t=t_fl -> SN observed right after the exposion
            t_shift = cadence / n_lc * k  # delay the first detection time
            t_mock = t_sample + t_jitter + t_shift
            flux_true = f_t(t=t_mock, tfl=tfl_true, C=C_true, A=A_true, alpha=alpha_true)

            mag_40 = mag_peak + 2.5 * np.log10(0.4)  # 40% of peak
            idx_early = t_mock < -10  # assuming 40% of peak is achieved at phase=-10 days
            flux_40 = f_t(t=-10, tfl=tfl_true, C=0, A=A_true, alpha=alpha_true)
            ZP_mock = 2.5 * np.log10(flux_40) + mag_40

            t_mock_early = t_mock[idx_early]
            flux_true_early = flux_true[idx_early]
            flux_err_mock_early = generate_flux_err(flux_true_early, ZP=ZP_mock, method="broken_power_law")
            flux_mock_early = flux_true_early + np.random.randn(idx_early.sum()) * flux_err_mock_early

            mock = Ia_lc(lc_early=dict(phase=t_mock_early, flux=flux_mock_early, flux_err=flux_err_mock_early))
            lc_lib[k] = mock

        self.lc_lib = lc_lib
        self.n_lc = n_lc
        self.var_true = dict(t_fl=tfl_true, C=C_true, A=A_true, alpha=alpha_true)
        self.var_name = dict(t_fl=r"$t_\mathrm{fl}$", C=r"$C$", A=r"$A$", alpha=r"$\alpha$")

    def sampling(self, prior_params: dict = None, prior_pred_samples: int = 4000, hierarchical: bool = False):
        if not hierarchical:
            lc_lib_inf = np.empty(shape=len(self.lc_lib), dtype=object)
            for k, mock in enumerate(self.lc_lib):
                mock.sampling(prior_params=prior_params, prior_pred_samples=prior_pred_samples)
                lc_lib_inf[k] = mock.inf_data
            self.lc_lib_inf = lc_lib_inf
        else:
            pass

    def plot_prior_posterior(self, var_name: list = ["alpha", "t_fl"], var_range: dict = None):
        if var_range is None:
            var_range = dict(alpha=(0, 4), t_fl=(-30, -10), C=(-1, 1), A=(0, 10))
        _, ax = plt.subplots(
            2, len(var_name), sharex="col", sharey="row", constrained_layout=True, figsize=(3 * len(var_name), 6)
        )
        cmap = plt.get_cmap("coolwarm")  # Choose a colormap

        for i, var in enumerate(var_name):
            prior, posterior = [], []
            for k in range(self.n_lc):
                prior.append(self.lc_lib_inf[k].prior[var].data.ravel())
                posterior.append(self.lc_lib_inf[k].posterior[var].data.ravel())
                ax[1, i].hist(
                    posterior[k],
                    histtype="step",
                    bins=50,
                    color=cmap(k / self.n_lc),  # Use color mapping based on k
                    range=var_range[var],
                )
            ax[0, i].hist(
                np.array(prior).ravel(),
                histtype="step",
                bins=50,
                color="k",
                lw=3,
                weights=np.ones_like(np.array(prior).ravel()) / len(prior),
                range=var_range[var],
            )
            ax[1, i].hist(
                np.array(posterior).ravel(), histtype="step", bins=25 * self.n_lc, color="k", lw=2, range=var_range[var]
            )

            ax[1, i].axvline(self.var_true[var], color="crimson", ls="--")
            ax[1, i].axvline(np.median(np.array(posterior).ravel()), color="k", ls="--")
            ax[0, i].set_title(f"Prior: {self.var_name[var]}")
            ax[1, i].set_title(f"Posterior: {self.var_name[var]}")
            ax[1, i].set_xlabel(self.var_name[var])
        ax[0, 0].set_yticks([])
        ax[1, 0].set_yticks([])
        ax[0, 0].set_ylabel("Normalized count")
        ax[1, 0].set_ylabel("Normalized count")

        return ax
