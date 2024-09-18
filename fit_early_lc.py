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

from sklearn.preprocessing import LabelEncoder

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


def f_t(t, tfl, C, A, alpha, eps: float = 1e-10):
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
    eps : float, optional, default = 1e-10
        Small value to avoid numerical issues when t - tfl is small and alpha < 1

    Returns:
    --------
    float
        The calculated value of f(t).
    """
    f = jnp.where(t < tfl, 0, A * jnp.power(jnp.maximum(t - tfl, eps), alpha)) + C
    return f


####################################################################################################
############ Probabilistic model for the individual/hierarchical light curve modeling ##############
####################################################################################################


def single_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_params: dict = {},
) -> None:
    """
    Bayesian model of the early rise for a single supernova
    in multiple fields, CCDs, quadrants, as well as filters.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : array-like
        Time values (phase) of the light curve.
        phase = (t_obs - t_max) / (1 + z)
    flux : array-like
        Flux values of the light curve.
    flux_err : array-like
        Flux error values of the light curve.
    idx_fcqfid : int, array-like
        Indices used to index the fcqf IDs for each measurement.
    idx_filt : int, array-like
        Indices of unique filters.
    prior_params : dict, optional
        Dictionary containing the prior information for the model.
        The dictionary should contain the following keys:
            - prior_type : str
                Type of prior to use for the model.
                Options: "Miller", "Jeffreys", "Maximum_Entropy", "Flat"
            - mean_alpha : float, optional
                Mean value of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - std_alpha : float, optional
                Standard deviation of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - min_alpha : float, optional, default = 0
                Minimum value of the prior distribution for alpha.

    Returns
    -------
    None
    """

    # single filter, single fcqfid
    if idx_fcqfid is None:
        idx_fcqfid = jnp.ones_like(t, dtype=int)
    if idx_filt is None:
        idx_filt = jnp.zeros_like(t, dtype=int)

    prior_type = prior_params.get("prior_type", "Miller")

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))

    with numpyro.plate("n_fcqfid", n_fcqfid):
        # Parameters specific to each fcqf ID (n_fcqfid)
        # C : Baseline flux
        # beta : Uncertainty scale factor
        C = numpyro.sample("C", dist.Uniform(-50, 50))
        log_beta = numpyro.sample("log_beta", dist.Uniform(-0.3, 0.3))  # ~50%
        beta = numpyro.deterministic("beta", jnp.power(10, log_beta))

    with numpyro.plate("n_filt", n_filt):

        # Parameters specific to each filter (n_filt)
        # alpha : Rising power-law index
        # A : Proportionality factor

        min_alpha = prior_params.get("min_alpha", 0)
        assert min_alpha >= 0, "Minimum value of alpha must be non-negative"
        if prior_type == "Miller":  # priors adopted in Miller+2020
            # Aprime = A * 10**alpha
            prior_alpha_Miller = dist.Exponential(jnp.log(10))
            prior_alpha_Miller.support = dist.constraints.interval(min_alpha, 10)
            alpha = numpyro.sample("alpha", prior_alpha_Miller)
            log_Aprime = numpyro.sample("log_Aprime", dist.Uniform(-5, 5))
            A = numpyro.deterministic("A", jnp.power(10, log_Aprime - alpha))
            numpyro.deterministic("Aprime", jnp.power(10, log_Aprime))
        else:
            if prior_type == "Jeffreys":  # Jeffreys prior
                log_alpha = numpyro.sample("log_alpha-", dist.Uniform(-3, 1))
                alpha = numpyro.deterministic("alpha", jnp.power(10, log_alpha) + min_alpha)

            elif prior_type in ["Flat", "Uniform"]:
                alpha = numpyro.sample("alpha", dist.Uniform(min_alpha, 10))

            elif prior_type == "Maximum_Entropy":  # Maximum entropy prior
                mean_alpha = prior_params.get("mean_alpha", 2)
                std_alpha = prior_params.get("std_alpha", None)

                if std_alpha is None:  # alpha > min_alpha, known E --> Exponential
                    lambda_ = 1 / (mean_alpha - min_alpha)
                    alpha_ = numpyro.sample("alpha-", dist.Exponential(lambda_))
                else:  # alpha > 1, known E and Var --> Gamma
                    kappa_ = (mean_alpha - min_alpha) ** 2 / std_alpha**2
                    lambda_ = (mean_alpha - min_alpha) / std_alpha**2
                    alpha_ = numpyro.sample("alpha-", dist.Gamma(kappa_, lambda_))
                alpha = numpyro.deterministic("alpha", alpha_ + min_alpha)
            else:
                raise ValueError(
                    "Invalid prior type. Options: 'Miller', 'Jeffreys', 'Maximum_Entropy', 'Flat' (or 'Uniform')"
                )

            log_A = numpyro.sample("log_A", dist.Uniform(-5, 5))
            A = numpyro.deterministic("A", jnp.power(10, log_A))
            numpyro.deterministic("Aprime", A * jnp.power(10, alpha))

    # t_fl : Time of the first light
    tfl = numpyro.sample("t_fl", dist.Uniform(-100, 0))

    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(t, tfl, C[idx_fcqfid], A[idx_filt], alpha[idx_filt]),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )

def unpooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_params: dict = {},
):
    """

    Returns
    -------
    None
    """

    pass

def pooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    idx_filt_grz: list = None,
    prior_params: dict = {},
):
    """
    Bayesian model of the early rise for a library of supernovae
    in multiple fields, CCDs, quadrants, as well as filters.

    In this pooled model, it is assumed that the rising power-law index alpha is the same
    in each filter among different objects.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : list
        A list of time value (phase) array of each light curve.
        phase = (t_obs - t_max) / (1 + z)
    flux : list
        A list of flux array of each light curve.
    flux_err : list
        A list of flux error array of each light curve.
    idx_obj : list
        Indices used to index the objects.
    idx_fcqfid : list
        Indices used to index the fcqf IDs for each measurement.
        Same icqf IDs on different objects are labeled as different.
    idx_filt : list
        Indices of unique filters.
        Same filters on different objects are labeled as different.
    idx_filt_grz : list
        Indices of unique filters for g, r, and z bands.
    prior_params : dict, optional
        Dictionary containing the prior information for the model.
        The dictionary should contain the following keys:
            - prior_type : str
                Type of prior to use for the model.
                Options: "Miller", "Jeffreys", "Maximum_Entropy", "Flat"
            - mean_alpha : float, optional
                Mean value of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - std_alpha : float, optional
                Standard deviation of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - min_alpha : float, optional, default = 0
                Minimum value of the prior distribution for alpha.

    Returns
    -------
    None
    """

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_filt_grz = len(np.unique(idx_filt_grz))
    n_obj = len(np.unique(idx_obj))

    with numpyro.plate("fcqfid", n_fcqfid):
        # Parameters specific to each fcqf ID for each object (n_fcqfid)
        # C : Baseline flux
        C = numpyro.sample("C", dist.Uniform(-50, 50))

        # beta : Uncertainty scale factor
        log_beta = numpyro.sample(
            "log_beta",
            dist.Uniform(-0.3, 0.3),  # ~50%
        )
        beta = numpyro.deterministic(f"beta", jnp.power(10, log_beta))

    prior_type = prior_params.get("prior_type", "Maximum_Entropy")
    min_alpha = prior_params.get("min_alpha", 0)
    with numpyro.plate("filt_grz", n_filt_grz):
        # Parameters specific to each filter for g, r, z bands (n_filt_grz)
        # alpha : Rising power-law index
        if prior_type == "Jeffreys":  # Jeffreys prior
            log_alpha = numpyro.sample("log_alpha-", dist.Uniform(-3, 1))
            alpha = numpyro.deterministic("alpha", jnp.power(10, log_alpha) + min_alpha)

        elif prior_type in ["Flat", "Uniform"]:
            alpha = numpyro.sample("alpha", dist.Uniform(min_alpha, 10))

        elif prior_type == "Maximum_Entropy":  # Maximum entropy prior
            mean_alpha = prior_params.get("mean_alpha", 2)
            std_alpha = prior_params.get("std_alpha", None)

            if std_alpha is None:  # alpha > min_alpha, known E --> Exponential
                lambda_ = 1 / (mean_alpha - min_alpha)
                alpha_ = numpyro.sample("alpha-", dist.Exponential(lambda_))
            else:  # alpha > 1, known E and Var --> Gamma
                kappa_ = (mean_alpha - min_alpha) ** 2 / std_alpha**2
                lambda_ = (mean_alpha - min_alpha) / std_alpha**2
                alpha_ = numpyro.sample("alpha-", dist.Gamma(kappa_, lambda_))
            alpha = numpyro.deterministic("alpha", alpha_ + min_alpha)
        else:
            raise ValueError(
                "Invalid prior type. Options: 'Jeffreys', 'Maximum_Entropy', 'Flat' (or 'Uniform')"
            )

    with numpyro.plate("filt", n_filt):
        # Parameters specific to each filter (n_filt)
        # A : Proportionality factor
        log_A = numpyro.sample(f"log_A", dist.Uniform(-5, 5))
        A = numpyro.deterministic("A", jnp.power(10, log_A))

    numpyro.deterministic(f"Aprime", A[idx_filt] * jnp.power(10, alpha[idx_filt_grz]))

    with numpyro.plate("obj", n_obj):
        # Parameters specific to each object (n_obj)
        # t_fl : Time of the first light
        tfl = numpyro.sample(f"t_fl", dist.Uniform(-100, 0))

    with numpyro.plate(f"data", len(t)):
        numpyro.sample(
            f"flux",
            dist.Normal(
                f_t(t, tfl[idx_obj], C[idx_fcqfid], A[idx_filt], alpha[idx_filt]),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def hierarchical_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_params: dict = {},
):
    """
    Hierarchical Bayesian model of the early rise for a library of supernovae
    in multiple fields, CCDs, quadrants, as well as filters.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : list
        A list of time value (phase) array of each light curve.
        phase = (t_obs - t_max) / (1 + z)
    flux : list
        A list of flux array of each light curve.
    flux_err : list
        A list of flux error array of each light curve.
    idx_obj : list
        Indices used to index the objects.
    idx_fcqfid : list
        Indices used to index the fcqf IDs for each measurement.
        Same icqf IDs on different objects are labeled as different.
    idx_filt : list
        Indices of unique filters.
        Same filters on different objects are labeled as different.
    prior_params : dict, optional
        Dictionary containing the prior information for the model.
        The dictionary should contain the following keys:
            - prior_type : str
                Type of prior to use for the model.
                Options: "Maximum_Entropy", "Gaussian"/"Gauss"/"Normal"
            - min_alpha : float, optional, default = 1
                Minimum value of the prior distribution for alpha.

    Returns
    -------
    None
    """

    # hyperpriors
    # alpha : Rising power-law index
    prior_type = prior_params.get("prior_type", "Maximum_Entropy")

    # maximum entropy prior --> alpha > 1
    min_alpha = prior_params.get("min_alpha", 0)
    assert min_alpha >= 0, "Minimum value of alpha must be non-negative"
    mean_alpha = numpyro.sample("mean_alpha", dist.Uniform(min_alpha, 10))

    ## maximum entropy prior --> positive
    # log_std_alpha = numpyro.sample("log_std_alpha", dist.Uniform(-3, 1))
    # std_alpha = numpyro.deterministic("std_alpha", jnp.power(10, log_std_alpha))
    std_alpha = numpyro.sample("std_alpha", dist.HalfNormal(0.1))

    mean_tfl = numpyro.sample("mean_t_fl", dist.Uniform(-100, 0))
    # log_std_tfl = numpyro.sample("log_std_t_fl", dist.Uniform(-3, 1))
    # std_tfl = numpyro.deterministic("std_t_fl", jnp.power(10, log_std_tfl))
    std_tfl = numpyro.sample("std_t_fl", dist.HalfNormal(1))

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    with numpyro.plate("fcqfid", n_fcqfid):
        # Parameters specific to each fcqf ID for each object (n_fcqfid)
        # C : Baseline flux
        C = numpyro.sample("C", dist.Uniform(-50, 50))

        # beta : Uncertainty scale factor
        log_beta = numpyro.sample(
            "log_beta",
            dist.Uniform(-0.3, 0.3),  # ~50%
        )
        beta = numpyro.deterministic(f"beta", jnp.power(10, log_beta))

    with numpyro.plate("filt", n_filt):
        # Parameters specific to each filter (n_filt)
        # alpha : Rising power-law index
        if prior_type in ["Gaussian", "Gauss", "Normal"]:  # Gaussian hyperpriors
            prior_alpha = dist.Normal(mean_alpha, std_alpha)
            prior_alpha.support = dist.constraints.interval(min_alpha, 10)
            alpha = numpyro.sample(f"alpha", prior_alpha)
        elif prior_type == "Maximum_Entropy":  # Maximum entropy (Gamma) hyperpriors
            concentration_alpha = (mean_alpha - min_alpha) ** 2 / std_alpha**2
            rate_alpha = (mean_alpha - min_alpha) / std_alpha**2
            alpha_ = numpyro.sample(f"alpha-", dist.Gamma(concentration_alpha, rate_alpha))
            alpha = numpyro.deterministic(f"alpha", alpha_ + min_alpha)
        else:
            raise ValueError("Invalid hyperprior type. Options: 'Maximum_Entropy' and 'Gaussian'/'Gauss'/'Normal'")

        # A : Proportionality factor
        log_A = numpyro.sample(f"log_A", dist.Uniform(-5, 5))
        A = numpyro.deterministic("A", jnp.power(10, log_A))
        numpyro.deterministic(f"Aprime", A * jnp.power(10, alpha))

    with numpyro.plate("obj", n_obj):
        # Parameters specific to each object (n_obj)
        # t_fl : Time of the first light
        # tfl = numpyro.sample(f"t_fl", dist.Uniform(-100, 0))
        tfl = numpyro.sample(f"t_fl", dist.Normal(mean_tfl, std_tfl))

    with numpyro.plate(f"data", len(t)):
        numpyro.sample(
            f"flux",
            dist.Normal(
                f_t(t, tfl[idx_obj], C[idx_fcqfid], A[idx_filt], alpha[idx_filt]),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


####################################################################################################
################# Class to organize the light curve data and perform MCMC sampling #################
####################################################################################################


def init_lc_package(lc):
    """
    Initialize a light curve package.

    Parameters
    ----------
    lc : dict
        A dictionary containing the light curve data.

    Returns
    -------
    dict
        A dictionary containing the initialized light curve package.

    Raises
    ------
    AssertionError
        If 'phase', 'flux', or 'flux_err' columns are missing in the lc dictionary.
    AssertionError
        If the lengths of 'phase', 'flux', and 'flux_err' columns do not match.
    """

    if lc is None:
        return None
    assert "phase" in lc.keys(), "'phase' column required"
    assert "flux" in lc.keys(), "'flux' column required"
    assert "flux_err" in lc.keys(), "'flux_err' column required"
    phase = lc["phase"]
    flux = lc["flux"]
    flux_err = lc["flux_err"]
    assert len(phase) == len(flux) == len(flux_err), "Lengths of the data columns do not match"

    fcqfid = lc.get("fcqfid", np.ones_like(phase, dtype=int))
    filt = lc.get("filt", np.zeros_like(phase, dtype=int))

    return dict(phase=phase, flux=flux, flux_err=flux_err, fcqfid=fcqfid, filt=filt)


class Ia_lc:

    def __init__(self, lc_early: dict = {}, lc_peak: dict = None, ztfid: str = None) -> None:
        self.ID = ztfid

        # observations between 40% and 100% of max flux
        self.lc_early = init_lc_package(lc_early)

        fcqfid = self.lc_early["fcqfid"]
        filt = self.lc_early["filt"]

        # calculate number of unique fcqfid and filter and their indices
        fcqfid_encoder = LabelEncoder()
        self.idx_fcqfid = fcqfid_encoder.fit_transform(fcqfid)

        filt_encoder = LabelEncoder()
        self.idx_filt = filt_encoder.fit_transform(filt)

        # observations between -100 days and peak
        self.lc_peak = init_lc_package(lc_peak)

    def sampling(
        self,
        num_samples: int = 3000,
        num_warmup: int = 1000,
        num_chains: int = 2,
        random_seed: int = 11,
        prior_pred_samples: int = 500,
        prior_params: dict = {},
        nuts_params: dict = {},
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
        nuts_params : dict, optional
            Dictionary containing the parameters for infer.NUTS.

        Returns
        -------
        None
        """

        self.sampler = infer.MCMC(
            infer.NUTS(single_model, **nuts_params),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
        )
        running_params = {
            "t": self.lc_early["phase"],
            "flux": self.lc_early["flux"],
            "flux_err": self.lc_early["flux_err"],
            "idx_fcqfid": self.idx_fcqfid,
            "idx_filt": self.idx_filt,
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
                self.lc_early["phase"][self.lc_early["fcqfid"] == fcqf],
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
                    self.lc_peak["phase"][self.lc_peak["fcqfid"] == fcqf],
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
                #     phase[fcqfid == fcqf],
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

    def plot_corner(self, save: bool = False, filename: str = None, var_name: list = ["t_fl", "alpha", "Aprime"]):
        import corner

        corner.corner(
            self.inf_data.posterior[var_name],
            # var_name=var_name,
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

    def __init__(self, tab_info, tab_lc, ztfid: str) -> None:
        """
        Initialize the class instance.

        Parameters
        ----------
        tab_info : astropy.table.Table
            DataFrame containing information about the object.
        tab_lc : astropy.table.Table
            DataFrame containing light curve data.
        ztfid : str
            ZTF ID of the object.

        Returns
        -------
        None
        """

        info = tab_info[tab_info["name"] == ztfid]
        dat = tab_lc[tab_lc["ZTF"] == ztfid]

        # phase = dat["phase"].value.astype("<f4")
        ZP = dat["ZP"].value.astype("<f4")
        flux = dat["Flux"].value.astype("<f4") / (10 ** (0.4 * ZP))
        flux_err = dat["e_Flux"].value.astype("<f4") / (10 ** (0.4 * ZP))

        self.t0_g = info["t0_g_adopted"].value[0]
        self.t0_B = info["t0_B_salt2"].value[0]
        self.t0_g_unc = info["t0_g_adopted_unc"].value[0]
        self.t0_B_unc = info["t0_salt2_unc"].value[0]

        z = info["z_adopt"].value[0]

        phase = (dat["JD"].value - self.t0_B) / (1 + z)

        fcqfid = dat["fcqfid"].value
        filt = fcqfid % 10

        from spec_tool.data_binning import data_binning

        t_g, f_g, _ = data_binning(np.array([phase, flux, flux_err]).T[filt == 1], 0.5).T
        t_r, f_r, _ = data_binning(np.array([phase, flux, flux_err]).T[filt == 2], 0.5).T

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
        idx_rise = (phase < 0) & (phase > -100)
        idx_g = (filt == 1) & (phase < t_g_40)
        idx_r = (filt == 2) & (phase < t_r_40)
        idx = idx_rise & (idx_g | idx_r)

        # observations between 40% and 100% of max flux
        lc_early = {
            "phase": phase[idx],
            "flux": flux[idx],
            "flux_err": flux_err[idx],
            "fcqfid": fcqfid[idx],
            "filt": filt[idx],
        }

        # observations between -100 days and peak
        lc_peak = {
            "phase": phase[idx_rise],
            "flux": flux[idx_rise],
            "flux_err": flux_err[idx_rise],
            "fcqfid": fcqfid[idx_rise],
            "filt": filt[idx_rise],
        }

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid)


class Ia_lc_library:

    def __init__(self, lc_early_lib: list = None, lc_peak_lib: list = None, ztfid_lib: list = None) -> None:

        self.lc_library = []
        if (lc_peak_lib is not None) and (ztfid_lib is not None):
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(Ia_lc(lc_early=lc_early, lc_peak=lc_peak_lib[k], ztfid=ztfid_lib[k]))
        elif lc_peak_lib is not None:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(Ia_lc(lc_early=lc_early, lc_peak=lc_peak_lib[k]))
        elif ztfid_lib is not None:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(Ia_lc(lc_early=lc_early, ztfid=ztfid_lib[k]))
        else:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(Ia_lc(lc_early=lc_early))

        self.phase, self.flux, self.flux_err = [], [], []
        self.idx_filt, self.idx_fcqfid = np.array([], dtype=int), np.array([], dtype=int)
        self.idx_obj = np.array([], dtype=int)

        for k in range(len(lc_early_lib)):
            # concatenate the indices
            self.idx_filt = np.append(self.idx_filt, self.lc_library[k].idx_filt + len(np.unique(self.idx_filt)))
            self.idx_fcqfid = np.append(
                self.idx_fcqfid, self.lc_library[k].idx_fcqfid + len(np.unique(self.idx_fcqfid))
            )
            self.idx_obj = np.append(self.idx_obj, np.ones_like(self.lc_library[k].idx_filt) * k)
            # concatenate the light curve data
            self.phase = np.append(self.phase, self.lc_library[k].lc_early["phase"])
            self.flux = np.append(self.flux, self.lc_library[k].lc_early["flux"])
            self.flux_err = np.append(self.flux_err, self.lc_library[k].lc_early["flux_err"])

        print("Number of objects:", len(lc_early_lib))
        print("Number of unique filters:", len(np.unique(self.idx_filt)))
        print("Number of unique fcqfid:", len(np.unique(self.idx_fcqfid)))
        print("Light curves compiled...")

    def sampling(
        self,
        num_samples: int = 1000,
        num_warmup: int = 3000,
        num_chains: int = 2,
        random_seed: int = 11,
        prior_pred_samples: int = 500,
        prior_params: dict = {},
        nuts_params: dict = {},
    ):
        """
        Perform MCMC sampling using NUTS algorithm.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw from the posterior distribution (default: 1000).
        num_warmup : int, optional
            Number of warmup samples to discard (default: 3000).
        num_chains : int, optional
            Number of chains to run (default: 2).
        random_seed : int, optional
            Random seed for reproducibility (default: 11).
        prior_pred_samples : int, optional
            Number of samples to draw from the prior predictive distribution (default: 500).
        prior_params : dict, optional
            Dictionary containing the prior information for the model.
        nuts_params : dict, optional
            Dictionary containing the parameters for infer.NUTS.

        Returns
        -------
        None
        """

        self.sampler = infer.MCMC(
            infer.NUTS(hierarchical_model, **nuts_params),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        running_params = {
            "t": self.phase,
            "flux": self.flux,
            "flux_err": self.flux_err,
            "idx_obj": self.idx_obj,
            "idx_fcqfid": self.idx_fcqfid,
            "idx_filt": self.idx_filt,
        }
        self.sampler.run(
            jax.random.PRNGKey(random_seed),
            **running_params,
            prior_params=prior_params,
        )

        # prior and posterior predictive checks
        prior_pred = infer.Predictive(hierarchical_model, num_samples=prior_pred_samples)(
            jax.random.PRNGKey(1919810), **running_params, prior_params=prior_params
        )
        post_pred = infer.Predictive(hierarchical_model, self.sampler.get_samples())(
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

        # _, ax = plt.subplots(figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True)

        # post_sample = self.sampler.get_samples()

        # idx_post_check = np.random.choice(len(post_sample["t_fl"]), post_pred_samples)

        # for k, fcqf in enumerate(np.sort(np.unique(self.lc_early["fcqfid"]))):
        #     C_ = np.median(post_sample["C"][:, k])
        #     beta_ = np.median(post_sample["beta"][:, k])
        #     is_g = np.all(self.lc_early["filt"][self.lc_early["fcqfid"] == fcqf] == 1)
        #     color = "tab:green" if is_g else "tab:red"
        #     ax.errorbar(
        #         self.lc_early["phase"][self.lc_early["fcqfid"] == fcqf],
        #         self.lc_early["flux"][self.lc_early["fcqfid"] == fcqf] - C_ + (is_g - 0.5) * offset,
        #         yerr=self.lc_early["flux_err"][self.lc_early["fcqfid"] == fcqf] * beta_,
        #         color="w",
        #         markeredgecolor=color,
        #         ecolor=color,
        #         fmt="o",
        #         zorder=10,
        #     )
        #     if self.lc_peak is not None:
        #         ax.errorbar(
        #             self.lc_peak["phase"][self.lc_peak["fcqfid"] == fcqf],
        #             self.lc_peak["flux"][self.lc_peak["fcqfid"] == fcqf] - C_ + (is_g - 0.5) * offset,
        #             yerr=self.lc_peak["flux_err"][self.lc_peak["fcqfid"] == fcqf] * beta_,
        #             color="w",
        #             markeredgecolor=color,
        #             ecolor=color,
        #             fmt="o",
        #             alpha=0.25,
        #             zorder=10,
        #         )

        #     ax.set_xlim(-31, -4)
        #     ax.set_ylim(-offset * 1.5, 100)

        #     t_pred = jnp.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
        #     for i in idx_post_check:
        #         # ax.plot(
        #         #     phase[fcqfid == fcqf],
        #         #     post_pred["flux"][i, fcqfid == fcqf] - C_ + (is_g - 0.5) * offset,
        #         #     ".",
        #         #     color="0.5",
        #         #     alpha=0.1,
        #         # )

        #         A = post_sample["A"][i, fcqf % 10 - 1]
        #         alpha = post_sample["alpha"][i, fcqf % 10 - 1]
        #         ax.plot(
        #             t_pred,
        #             f_t(t_pred, post_sample["t_fl"][i], 0, A, alpha) + (is_g - 0.5) * offset,
        #             color="0.2",
        #             lw=0.1,
        #             zorder=-1,
        #         )

        # for i in idx_post_check:
        #     ax.axvline(post_sample["t_fl"][i], color="0.2", lw=0.1)

        # ax.set_xlabel(r"$t - T_{B, \mathrm{max}}\ [\mathrm{restframe\ d}]$")
        # ax.set_ylabel(r"$f + \mathrm{offset}$")
        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(25))
        # ax.yaxis.set_minor_locator(MultipleLocator(5))
        # ax.set_title(self.ID)

        # if save:
        #     if filename is None:
        #         filename = self.ID
        #     plt.savefig(filename + ".pdf")
        pass

    def plot_corner(self, save: bool = False, filename: str = None, var_name: list = ["mean_alpha", "std_alpha"]):

        import corner

        corner.corner(
            self.inf_data.posterior[var_name],
            # var_name=var_name,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.5, 0.95],
            title_quantiles=[0.05, 0.5, 0.95],
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf")
