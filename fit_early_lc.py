from math import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from matplotlib.ticker import MultipleLocator
import seaborn as sns

import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist, infer
import arviz as az

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


def f_t(t, tfl, C, A, alpha, eps: float = 1e-12):
    """
    Calculate the flux with a power-law rise model.

    Parameters:
    -----------
    t : float or array-like
        Time value.
    tfl : float or array-like
        Time of the first light.
    C : float or array-like
        Baseline flux.
    A : float or array-like
        Proportionality factor.
    alpha : float or array-like
        Rising power-law index.
    eps : float, optional
        Small value to avoid numerical issues (default: 1e-12).

    Returns:
    --------
    float
        The calculated value of f(t).
    """
    eps = 1e-20  # avoid numerical issues when t - tfl is small and alpha < 1
    f = jnp.where(t < tfl, 0, A * jnp.power(jnp.maximum(t - tfl, eps), alpha)) + C
    return f


####################################################################################################
############ Probabilistic model for the individual/hierarchical light curve modeling ##############
####################################################################################################


def single_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    n_fcqfid: int = 1,
    n_filt: int = 1,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_params: dict = {},
) -> None:
    """
    Function to model the single light curve of a supernova
    in multiple fields, CCDs, quadrants, as well as filters.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : array-like
        Time values (phase) of the light curve.
        Phase = (t_obs - t_max) / (1 + z)
    flux : array-like
        Flux values of the light curve.
    flux_err : array-like
        Flux error values of the light curve.
    n_fcqfid, idx_fcqfid : int, array-like
        Number of unique fcqf IDs and indices used to index the fcqf IDs
        for each measurement.
    n_filt, idx_filt : int, array-like
        Number of unique filters and their indices.
    prior_params : dict, optional
        Dictionary containing the prior information for the model.
        The dictionary should contain the following keys:
            - prior_type : str
                Type of prior to use for the model.
                Options: "Miller", "Jeffreys", "Maximum_Entropy"
            - mean_alpha : float, optional
                Mean value of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - std_alpha : float, optional
                Standard deviation of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".

    Returns
    -------
    None
    """

    prior_type = prior_params.get("prior_type", "Miller")

    # t_fl : Time of the first light
    tfl = numpyro.sample("t_fl", dist.Uniform(-100, 0))

    # Parameters specific to each fcqf ID (n_fcqfid)
    # C : Baseline flux
    # beta : Uncertainty scale factor
    C = numpyro.sample(
        "C",
        dist.Uniform(-100, 100),
        sample_shape=(n_fcqfid,),
    )
    log_beta = numpyro.sample(
        "log_beta",
        dist.Uniform(-1, 1),
        sample_shape=(n_fcqfid,),
    )
    beta = numpyro.deterministic("beta", 10**log_beta)

    # Parameters specific to each filter (n_filt)
    # alpha : Rising power-law index
    # A : Proportionality factor
    if prior_type == "Miller":  # priors adopted in Miller+2020
        # Aprime = A * 10^alpha
        alpha = numpyro.sample("alpha", dist.Exponential(jnp.log(10)), sample_shape=(n_filt,))
        log_Aprime = numpyro.sample(
            "log_Aprime",
            dist.Uniform(-5, 5),
            sample_shape=(n_filt,),
        )
        numpyro.deterministic("Aprime", jnp.power(10, log_Aprime))
        A = numpyro.deterministic("A", jnp.power(10, log_Aprime - alpha))
    else:
        if prior_type == "Jeffreys":  # Jeffreys prior
            log_alpha = numpyro.sample("log_alpha", dist.Uniform(-1, 1), sample_shape=(n_filt,))
            alpha = numpyro.deterministic("alpha", jnp.power(10, log_alpha))

        elif prior_type == "Maximum_Entropy":  # Maximum entropy prior
            mean_alpha = prior_params.get("mean_alpha", 2)
            std_alpha = prior_params.get("std_alpha", None)

            if std_alpha is None:  # alpha > 0, known E --> Exponential
                lambda_ = 1 / mean_alpha
                alpha = numpyro.sample("alpha", dist.Exponential(lambda_), sample_shape=(n_filt,))

            else:  # alpha > 0, known E and Var --> Gamma
                kappa_ = mean_alpha**2 / std_alpha**2
                lambda_ = mean_alpha / std_alpha**2
                alpha = numpyro.sample("alpha", dist.Gamma(kappa_, lambda_), sample_shape=(n_filt,))
        else:
            raise ValueError("Invalid prior type. Options: 'Miller', 'Jeffreys', 'Maximum_Entropy'")

        log_A = numpyro.sample(
            "log_A",
            dist.Uniform(-5, 5),
            sample_shape=(n_filt,),
        )
        A = numpyro.deterministic("A", jnp.power(10, log_A))
        Aprime = numpyro.deterministic("Aprime", A * jnp.power(10, alpha))

    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(t, tfl, C[idx_fcqfid], A[idx_filt], alpha[idx_filt]),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def hierarchical_model(t, flux_err, n_fcqfid, idx_fcqfid, n_filt, idx_filt, flux=None):
    """
    Function to model the single light curve of a supernova
    in multiple fields, CCDs, quadrants, as well as filters.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : array-like
        Time values (phase) of the light curve.
        Phase = (t_obs - t_max) / (1 + z)
    flux : array-like
        Flux values of the light curve.
    flux_err : array-like
        Flux error values of the light curve.
    n_fcqfid, idx_fcqfid : int
        Number of unique fcqf IDs and indices used to index the fcqf IDs
        for each measurement.
    n_filt, idx_filt : int
        Number of unique filters and their indices.

    Returns
    -------
    None
    """

    pass

    # # t_fl : Time of the first light
    # tfl = numpyro.sample("t_fl", dist.Uniform(-50, 0))

    # # Parameters specific to each fcqf ID (n_fcqfid)
    # # C : Baseline flux
    # C = numpyro.sample(
    #     "C",
    #     dist.Normal(0, 1e2),
    #     sample_shape=(n_fcqfid,),
    # )
    # # Uncertainty scale factor
    # log_beta = numpyro.sample(
    #     "log_beta",
    #     dist.Uniform(jnp.zeros(n_fcqfid), 1 * jnp.ones(n_fcqfid)),
    # )
    # beta = numpyro.deterministic("beta", 10**log_beta)

    # # Parameters specific to each filter (n_filt)
    # # alpha : Rising power-law index
    # alpha = numpyro.sample("alpha", dist.Uniform(jnp.zeros(n_filt), 5 * jnp.ones(n_filt)))
    # # Aprime : Proportionality factor
    # log_Aprime = numpyro.sample(
    #     "log_A_prime",
    #     dist.Uniform(0 * jnp.ones(n_filt), 5 * jnp.ones(n_filt)),
    # )
    # A =


####################################################################################################
################# Class to organize the light curve data and perform MCMC sampling #################
####################################################################################################


class Ia_lc:

    def __init__(self, lc_early: dict = {}, lc_peak: dict = None, ZTFID: str = None) -> None:
        self.ID = ZTFID

        # observations between 40% and 100% of max flux
        self.lc_early = lc_early

        # observations between -100 days and peak
        self.lc_peak = lc_peak

    def sampling(
        self,
        num_samples: int = 2000,
        num_warmup: int = 2000,
        num_chains: int = 2,
        random_seed: int = 11,
        prior_pred_samples: int = 500,
        prior_params: dict = {},
    ):
        """
        Perform MCMC sampling using NUTS algorithm.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw from the posterior distribution (default: 2000).
        num_warmup : int, optional
            Number of warmup samples to discard (default: 2000).
        num_chains : int, optional
            Number of chains to run (default: 2).
        random_seed : int, optional
            Random seed for reproducibility (default: 11).
        prior_pred_samples : int, optional
            Number of samples to draw from the prior predictive distribution (default: 500).
        prior_params : dict, optional
            Dictionary containing the prior information for the model.

        Returns
        -------
        None
        """

        fcqfid = self.lc_early["fcqfid"]
        filt = self.lc_early["filt"]
        Phase = self.lc_early["Phase"]
        flux = self.lc_early["flux"]
        flux_err = self.lc_early["flux_err"]

        # calculate number of unique fcqfid and filter and their indices
        uni_fcqfid = sorted(list(set(fcqfid)))
        n_fcqfid = len(uni_fcqfid)
        idx_fcqfid = np.array([jnp.where(uni_fcqfid == fcqf)[0][0] for fcqf in fcqfid])

        uni_filt = sorted(list(set(filt)))
        n_filt = len(uni_filt)
        idx_filt = np.array([jnp.where(uni_filt == f)[0][0] for f in filt])

        self.sampler = infer.MCMC(
            infer.NUTS(single_model),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
        )
        running_params = {
            "t": Phase,
            "flux": flux,
            "flux_err": flux_err,
            "n_fcqfid": n_fcqfid,
            "n_filt": n_filt,
            "idx_fcqfid": idx_fcqfid,
            "idx_filt": idx_filt,
        }
        self.sampler.run(jax.random.PRNGKey(random_seed), **running_params, prior_params=prior_params)

        # prior and posterior predictive checks
        prior_pred = infer.Predictive(single_model, num_samples=prior_pred_samples)(
            jax.random.PRNGKey(1919810), **running_params, prior_params=prior_params
        )
        post_pred = infer.Predictive(single_model, self.sampler.get_samples())(
            jax.random.PRNGKey(114514), **running_params, prior_params=prior_params
        )
        # convert to arviz InferenceData
        self.inf_data = az.from_numpyro(self.sampler, prior=prior_pred, posterior_predictive=post_pred)

    def plot_lc(self, save: bool = False, filename: str = None, offset: float = 30, post_pred_samples: int = 25):
        """
        Plot the light curve and the inferred model.

        Parameters
        ----------
        save : bool, optional
            Save the figure if True (default: False).
        filename : str, optional, default=self.ID
            Filename to save the figure.
        offset : float, optional, default=30
            Offset to separate g & r light curves in the plot.

        Returns
        -------
        None
        """

        _, ax = plt.subplots(figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True)

        post_sample = self.sampler.get_samples()

        idx_post_check = np.random.choice(len(post_sample["t_fl"]), post_pred_samples)

        for k, fcqf in enumerate(np.sort(np.unique(self.lc_early["fcqfid"]))):
            C_ = np.median(post_sample["C"][:, k])
            beta_ = np.median(post_sample["beta"][:, k])
            is_g = np.all(self.lc_early["filt"][self.lc_early["fcqfid"] == fcqf] == 1)
            color = "tab:green" if is_g else "tab:red"
            ax.errorbar(
                self.lc_early["Phase"][self.lc_early["fcqfid"] == fcqf],
                self.lc_early["flux"][self.lc_early["fcqfid"] == fcqf] - C_ + (is_g - 0.5) * offset,
                yerr=self.lc_early["flux_err"][self.lc_early["fcqfid"] == fcqf] * beta_,
                color="w",
                markeredgecolor=color,
                ecolor=color,
                fmt="o",
                zorder=10,
            )
            if self.lc_peak is not None:
                ax.errorbar(
                    self.lc_peak["Phase"][self.lc_peak["fcqfid"] == fcqf],
                    self.lc_peak["flux"][self.lc_peak["fcqfid"] == fcqf] - C_ + (is_g - 0.5) * offset,
                    yerr=self.lc_peak["flux_err"][self.lc_peak["fcqfid"] == fcqf] * beta_,
                    color="w",
                    markeredgecolor=color,
                    ecolor=color,
                    fmt="o",
                    alpha=0.25,
                    zorder=10,
                )

            ax.set_xlim(-31, -4)
            ax.set_ylim(-offset * 1.5, 100)

            t_pred = jnp.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
            for i in idx_post_check:
                # ax.plot(
                #     Phase[fcqfid == fcqf],
                #     post_pred["flux"][i, fcqfid == fcqf] - C_ + (is_g - 0.5) * offset,
                #     ".",
                #     color="0.5",
                #     alpha=0.1,
                # )

                A = post_sample["A"][i, fcqf % 10 - 1]
                alpha = post_sample["alpha"][i, fcqf % 10 - 1]
                ax.plot(
                    t_pred,
                    f_t(t_pred, post_sample["t_fl"][i], 0, A, alpha) + (is_g - 0.5) * offset,
                    color="0.2",
                    lw=0.1,
                    zorder=-1,
                )

        for i in idx_post_check:
            ax.axvline(post_sample["t_fl"][i], color="0.2", lw=0.1)

        ax.set_xlabel(r"$t - T_{B, \mathrm{max}}\ [\mathrm{restframe\ d}]$")
        ax.set_ylabel(r"$f + \mathrm{offset}$")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_title(self.ID)

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + ".pdf")

    def plot_corner(self, save: bool = False, filename: str = None, var_names: list = ["t_fl", "alpha", "Aprime"]):
        import corner

        corner.corner(
            self.inf_data,
            var_names=var_names,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.5, 0.95],
            title_quantiles=[0.05, 0.5, 0.95],
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf")


class ZTF_SN_Ia(Ia_lc):

    def __init__(self, tab_info, tab_lc, ZTFID: str) -> None:
        """
        Initialize the class instance.

        Parameters
        ----------
        tab_info : astropy.table.Table
            DataFrame containing information about the object.
        tab_lc : astropy.table.Table
            DataFrame containing light curve data.
        ZTFID : str
            ZTF ID of the object.

        Returns
        -------
        None
        """

        info = tab_info[tab_info["name"] == ZTFID]
        dat = tab_lc[tab_lc["ZTF"] == ZTFID]

        # Phase = dat["Phase"].value.astype("<f4")
        ZP = dat["ZP"].value.astype("<f4")
        flux = dat["Flux"].value.astype("<f4") / (10 ** (0.4 * ZP))
        flux_err = dat["e_Flux"].value.astype("<f4") / (10 ** (0.4 * ZP))

        self.t0_g = info["t0_g_adopted"].value[0]
        self.t0_B = info["t0_B_salt2"].value[0]
        self.t0_g_unc = info["t0_g_adopted_unc"].value[0]
        self.t0_B_unc = info["t0_salt2_unc"].value[0]

        z = info["z_adopt"].value[0]

        Phase = (dat["JD"].value - self.t0_B) / (1 + z)

        fcqfid = dat["fcqfid"].value
        filt = fcqfid % 10

        from spec_tool.data_binning import data_binning

        t_g, f_g, _ = data_binning(np.array([Phase, flux, flux_err]).T[filt == 1], 0.5).T
        t_r, f_r, _ = data_binning(np.array([Phase, flux, flux_err]).T[filt == 2], 0.5).T

        # max flux from data
        # flux_g_max = np.max(f_g)
        # flux_r_max = np.max(f_r)

        # max flux from SALT2 fit
        flux_g_max = info["fratio_gmax_2adam"].value[0]
        flux_r_max = info["fratio_rmax_2adam"].value[0]

        t_g_40 = t_g[np.where((f_g < 0.4 * flux_g_max) & (t_g < 0))[0][-1]] + 0.25
        t_r_40 = t_r[np.where((f_r < 0.4 * flux_r_max) & (t_r < 0))[0][-1]] + 0.25

        # normalization
        flux[filt == 1] /= flux_g_max / 100
        flux_err[filt == 1] /= flux_g_max / 100
        flux[filt == 2] /= flux_r_max / 100
        flux_err[filt == 2] /= flux_r_max / 100

        # filter out observations < 40% of max flux
        idx_rise = (Phase < 0) & (Phase > -100)
        idx_g = (filt == 1) & (Phase < t_g_40)
        idx_r = (filt == 2) & (Phase < t_r_40)
        idx = idx_rise & (idx_g | idx_r)

        # observations between 40% and 100% of max flux
        lc_early = {
            "Phase": Phase[idx],
            "flux": flux[idx],
            "flux_err": flux_err[idx],
            "fcqfid": fcqfid[idx],
            "filt": filt[idx],
        }

        # observations between -100 days and peak
        lc_peak = {
            "Phase": Phase[idx_rise],
            "flux": flux[idx_rise],
            "flux_err": flux_err[idx_rise],
            "fcqfid": fcqfid[idx_rise],
            "filt": filt[idx_rise],
        }

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ZTFID=ZTFID)
