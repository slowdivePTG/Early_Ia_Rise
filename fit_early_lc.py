import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import warnings

import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist, infer
import arviz as az
import corner
import xarray as xr

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


def f_t(t, t_fl, C, A, alpha, eps: float = 1e-10):
    """
    Calculate the flux with a power-law rise model.

    Parameters:
    -----------
    t : float or array-like
        Time value.
    t_fl : float or array-like
        Time of the first light.
    C : float or array-like
        Baseline flux.
    A : float or array-like
        Proportionality factor.
    alpha : float or array-like
        Rising power-law index.
    eps : float, optional, default = 1e-10
        Small value to avoid numerical issues when t - t_fl is small and alpha < 1

    Returns:
    --------
    float
        The calculated value of f(t).
    """
    f = jnp.where(t < t_fl, 0, A * jnp.power(jnp.maximum(t - t_fl, eps), alpha)) + C
    return f


####################################################################################################
####################### Probabilistic models for SNe Ia light curve modeling #######################
####################################################################################################


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
    Bayesian model of the early rise for a library of supernovae
    in multiple fields, CCDs, quadrants, as well as filters.

    In this unpooled model, different rising power-law indices are assumed
    among different objects.

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
                Options: "Miller", "Jeffreys", "Maximum_Entropy", "Flat"
            - mean_alpha_0 : float, optional
                Mean value of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - std_alpha_0 : float, optional
                Standard deviation of the prior distribution for alpha.
                Required if prior_type == "Maximum_Entropy".
            - min_alpha : float, optional, default = 0
                Minimum value of the prior distribution for alpha.

    Returns
    -------
    None
    """

    prior_type = prior_params.get("prior_type", "Miller")

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    with numpyro.plate("n_fcqfid", n_fcqfid):
        # Parameters specific to each fcqf ID (n_fcqfid)
        # C : Baseline flux
        # beta : Uncertainty scale factor
        C = numpyro.sample("C", dist.Uniform(-50, 50))
        beta = numpyro.sample("beta", dist.LogUniform(0.7, 1.3))  # ~30%

    with numpyro.plate("n_filt", n_filt):
        # Parameters specific to each filter for g, r, z bands (n_filt_gr)
        # alpha : Rising power-law index
        # A : Proportionality factor
        min_alpha = prior_params.get("min_alpha", 0)
        assert min_alpha >= 0, "Minimum value of alpha must be non-negative"
        if prior_type == "Miller":  # priors adopted in Miller+2020
            # Aprime = A * 10**alpha
            prior_alpha_Miller = dist.Exponential(jnp.log(10))
            prior_alpha_Miller.support = dist.constraints.interval(min_alpha, 10)
            alpha = numpyro.sample("alpha", prior_alpha_Miller)
        else:
            if prior_type == "Jeffreys":  # Jeffreys prior
                alpha = numpyro.sample(
                    "alpha",
                    dist.LogUniform(max(min_alpha, 1e-2), 10),
                )

            elif prior_type in ["Flat", "Uniform"]:
                alpha = numpyro.sample("alpha", dist.Uniform(min_alpha, 10))

            elif prior_type == "Maximum_Entropy":  # Maximum entropy prior
                mean_alpha_0 = prior_params.get("mean_alpha_0", 2)
                std_alpha_0 = prior_params.get("std_alpha_0", None)

                if std_alpha_0 is None:  # alpha > min_alpha, known E --> Exponential
                    rate_alpha_ = 1 / (mean_alpha_0 - min_alpha)
                    alpha_ = numpyro.sample("alpha-", dist.Exponential(rate_alpha_))
                else:  # alpha > alpha_min, known E and Var --> Gamma
                    concentration_alpha_ = (mean_alpha_0 - min_alpha) ** 2 / std_alpha_0**2
                    rate_alpha_ = (mean_alpha_0 - min_alpha) / std_alpha_0**2
                    alpha_ = numpyro.sample("alpha-", dist.Gamma(concentration_alpha_, rate_alpha_))
                alpha = numpyro.deterministic("alpha", alpha_ + min_alpha)
            else:
                raise ValueError(
                    "Invalid prior type. Options: 'Miller', 'Jeffreys', 'Maximum_Entropy', 'Flat' (or 'Uniform')"
                )

        Aprime = numpyro.sample("Aprime", dist.LogUniform(1e-5, 1e5))
        A = numpyro.deterministic("A", Aprime / jnp.power(10, alpha))

    with numpyro.plate("obj", n_obj):
        # Parameters specific to each object (n_obj)
        # t_fl : Time of the first light
        t_fl = numpyro.sample(f"t_fl", dist.Uniform(-40, 0))

    with numpyro.plate(f"data", len(t)):
        numpyro.sample(
            f"flux",
            dist.Normal(
                f_t(t, t_fl[idx_obj], C[idx_fcqfid], A[idx_filt], alpha[idx_filt]),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def pooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    idx_filt_gr: list = None,
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
    idx_filt_gr : list
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
    n_filt_gr = len(np.unique(idx_filt_gr))

    with numpyro.plate("fcqfid", n_fcqfid):
        # Parameters specific to each fcqf ID for each object (n_fcqfid)
        # C : Baseline flux
        # beta : Uncertainty scale factor
        C = numpyro.sample("C", dist.Uniform(-50, 50))
        beta = numpyro.sample("beta", dist.LogUniform(0.7, 1.3))  # ~30%

    prior_type = prior_params.get("prior_type", "Maximum_Entropy")
    min_alpha = prior_params.get("min_alpha", 0)
    with numpyro.plate("filt_gr", n_filt_gr):
        # Parameters specific to each filter for g, r, z bands (n_filt_gr)
        # alpha : Rising power-law index
        if prior_type == "Jeffreys":  # Jeffreys prior
            alpha = numpyro.sample(
                "alpha",
                dist.LogUniform(max(min_alpha, 1e-2), 10),
            )

        elif prior_type in ["Flat", "Uniform"]:
            alpha = numpyro.sample("alpha", dist.Uniform(min_alpha, 10))

        elif prior_type == "Maximum_Entropy":  # Maximum entropy prior
            mean_alpha_0 = prior_params.get("mean_alpha_0", 2)
            std_alpha_0 = prior_params.get("std_alpha_0", None)

            if std_alpha_0 is None:  # alpha > min_alpha, known E --> Exponential
                rate_alpha_ = 1 / (mean_alpha_0 - min_alpha)
                alpha_ = numpyro.sample("alpha-", dist.Exponential(rate_alpha_))
            else:  # alpha > 1, known E and Var --> Gamma
                concentration_alpha_ = (mean_alpha_0 - min_alpha) ** 2 / std_alpha_0**2
                rate_alpha_ = (mean_alpha_0 - min_alpha) / std_alpha_0**2
                alpha_ = numpyro.sample("alpha-", dist.Gamma(concentration_alpha_, rate_alpha_))
            alpha = numpyro.deterministic("alpha", alpha_ + min_alpha)
        else:
            raise ValueError("Invalid prior type. Options: 'Jeffreys', 'Maximum_Entropy', 'Flat' (or 'Uniform')")

    with numpyro.plate("filt", n_filt):
        # Parameters specific to each filter (n_filt)
        # A : Proportionality factor
        Aprime = numpyro.sample("Aprime", dist.LogUniform(1e-5, 1e5))

    A = numpyro.deterministic(f"A", Aprime[idx_filt] / jnp.power(10, alpha[idx_filt_gr]))

    # t_fl : Time of the first light
    t_fl = numpyro.sample(f"t_fl", dist.Uniform(-40, 0))

    with numpyro.plate(f"data", len(t)):
        numpyro.sample(
            f"flux",
            dist.Normal(
                f_t(t, t_fl, C[idx_fcqfid], A[idx_filt], alpha[idx_filt_gr]),
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
    std_alpha = numpyro.sample("std_alpha", dist.HalfNormal(0.1))

    mean_t_fl = numpyro.sample("mean_t_fl", dist.Uniform(-40, 0))
    std_t_fl = numpyro.sample("std_t_fl", dist.HalfNormal(1))

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    with numpyro.plate("fcqfid", n_fcqfid):
        # Parameters specific to each fcqf ID for each object (n_fcqfid)
        # C : Baseline flux
        C = numpyro.sample("C", dist.Uniform(-50, 50))

        # beta : Uncertainty scale factor
        # beta = numpyro.sample("beta", dist.LogUniform(0.8, 1.2))  # ~20%
        beta = numpyro.sample("beta", dist.LogNormal(0, 0.1))

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
        A = numpyro.sample("A", dist.LogUniform(1e-5, 1e5))
        numpyro.deterministic(f"Aprime", A * jnp.power(10, alpha))

    with numpyro.plate("obj", n_obj):
        # Parameters specific to each object (n_obj)
        # t_fl : Time of the first light
        # t_fl = numpyro.sample(f"t_fl", dist.Uniform(-40, 0))
        t_fl = numpyro.sample(f"t_fl", dist.Normal(mean_t_fl, std_t_fl))

    with numpyro.plate(f"data", len(t)):
        numpyro.sample(
            f"flux",
            dist.Normal(
                f_t(t, t_fl[idx_obj], C[idx_fcqfid], A[idx_filt], alpha[idx_filt]),
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
    """
    Class to organize light curves of individual SNe
    """

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

        self.post_sample = None

    def sampling(
        self,
        num_samples: int = 1000,
        num_warmup: int = 5000,
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

        kernel = unpooled_model
        init_strategy = nuts_params.pop("init_strategy", infer.init_to_median())
        self.sampler = infer.MCMC(
            infer.NUTS(kernel, init_strategy=init_strategy, **nuts_params),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        running_params = {
            "t": self.lc_early["phase"],
            "flux": self.lc_early["flux"],
            "flux_err": self.lc_early["flux_err"],
            "idx_obj": np.zeros_like(self.idx_fcqfid, dtype=int),  # only one object
            "idx_fcqfid": self.idx_fcqfid,
            "idx_filt": self.idx_filt,
        }
        self.sampler.run(
            jax.random.PRNGKey(random_seed),
            **running_params,
            prior_params=prior_params,
        )

        # prior and posterior predictive checks
        prior_pred = infer.Predictive(kernel, num_samples=prior_pred_samples)(
            jax.random.PRNGKey(1919810 + random_seed), **running_params, prior_params=prior_params
        )
        post_pred = infer.Predictive(kernel, self.sampler.get_samples())(
            jax.random.PRNGKey(114514 + random_seed), **running_params, prior_params=prior_params
        )
        # convert to arviz InferenceData
        self.inf_data = az.from_numpyro(self.sampler, prior=prior_pred, posterior_predictive=post_pred)

        # store the posterior samples
        self.post_sample = self.inf_data.posterior

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

        post_sample = self.post_sample
        if post_sample is None:
            C_ = np.zeros_like(self.lc_early["flux"])
            beta_ = np.ones_like(self.lc_early["flux"])

            warnings.warn("No posterior samples available.")
        else:
            C_ = np.median(post_sample["C"][:, :, self.idx_fcqfid], axis=(0, 1))
            beta_ = np.median(post_sample["beta"][:, :, self.idx_fcqfid], axis=(0, 1))
            t_fl = np.ravel(post_sample["t_fl"])

            idx_post_check = np.random.choice(len(t_fl), post_pred_samples)
            for i in idx_post_check:
                ax.axvline(t_fl[i], color="0.2", lw=0.1)

        colors = np.array(["tab:green", "tab:red", "tab:orange"])
        color = colors[self.idx_filt[self.idx_fcqfid]]
        n_color = len(np.unique(self.idx_filt))

        for k, flt in enumerate(np.sort(np.unique(self.idx_filt))):
            for j, fcqfid in enumerate(np.unique(self.idx_fcqfid)):
                idx = (self.idx_filt == flt) & (self.idx_fcqfid == fcqfid)
                if idx.sum() == 0:
                    continue
                ax.errorbar(
                    self.lc_early["phase"][idx],
                    self.lc_early["flux"][idx] - C_[idx] + (flt - 0.5 * (n_color - 1)) * offset,
                    yerr=self.lc_early["flux_err"][idx] * beta_[idx],
                    color="w",
                    markeredgecolor=colors[k],
                    ecolor=colors[k],
                    fmt="o",
                    zorder=10,
                )
                if self.lc_peak is not None:
                    ax.errorbar(
                        self.lc_early["phase"][idx],
                        self.lc_early["flux"][idx] - C_[idx] + (flt - 0.5 * (n_color - 1)) * offset,
                        yerr=self.lc_early["flux_err"][idx] * beta_[idx],
                        color="w",
                        markeredgecolor=color,
                        ecolor=color,
                        fmt="o",
                        alpha=0.25,
                        zorder=10,
                    )

            ax.set_xlim(-31, -4)
            ax.set_ylim(-offset * 1.5, 100)

            if post_sample is not None:
                A_ = np.ravel(post_sample["A"][:, :, flt])
                alpha_ = np.ravel(post_sample["alpha"][:, :, flt])
                t_pred = jnp.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
                for i in idx_post_check:
                    ax.plot(
                        t_pred,
                        f_t(t_pred, t_fl[i], 0, A_[i], alpha_[i]) + (flt - 0.5 * (n_color - 1)) * offset,
                        color="0.2",
                        lw=0.1,
                        zorder=-1,
                    )

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

    def plot_corner(self, save: bool = False, filename: str = None, **kwargs):
        """
        Plot the corner plot of the posterior samples.

        Parameters
        ----------
        save : bool, optional
            Save the figure if True (default: False).
        filename : str, optional, default=self.ID
            Filename to save the figure.

        Returns
        -------
        None
        """

        corner.corner(
            self.post_sample,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.5, 0.95],
            title_quantiles=[0.05, 0.5, 0.95],
            **kwargs,
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf", bbox_inches="tight")


class Ia_lc_library:
    """
    Class to organize a library of light curves of SNe Ia
    """

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
        self.idx_filt, self.idx_filt_gr = np.array([], dtype=int), np.array([], dtype=int)
        self.idx_fcqfid = np.array([], dtype=int)
        self.idx_obj = np.array([], dtype=int)

        for k in range(len(lc_early_lib)):
            # concatenate the indices
            self.idx_filt = np.append(self.idx_filt, self.lc_library[k].idx_filt + len(np.unique(self.idx_filt)))
            self.idx_filt_gr = np.append(self.idx_filt_gr, self.lc_library[k].idx_filt)
            self.idx_fcqfid = np.append(
                self.idx_fcqfid, self.lc_library[k].idx_fcqfid + len(np.unique(self.idx_fcqfid))
            )
            self.idx_obj = np.append(self.idx_obj, np.ones_like(self.lc_library[k].idx_filt) * k)
            # concatenate the light curve data
            self.phase = np.append(self.phase, self.lc_library[k].lc_early["phase"])
            self.flux = np.append(self.flux, self.lc_library[k].lc_early["flux"])
            self.flux_err = np.append(self.flux_err, self.lc_library[k].lc_early["flux_err"])

        n_obj = len(np.unique(self.idx_obj))
        n_fcqfid = len(np.unique(self.idx_fcqfid))
        n_filt = len(np.unique(self.idx_filt))
        n_filt_gr = len(np.unique(self.idx_filt_gr))
        assert n_obj == self.idx_obj.max() + 1, "Indexing error: idx_obj"
        assert n_fcqfid == self.idx_fcqfid.max() + 1, "Indexing error: idx_fcqfid"
        assert n_filt == self.idx_filt.max() + 1, "Indexing error: idx_filt"
        assert n_filt_gr == self.idx_filt_gr.max() + 1, "Indexing error: idx_filt_gr"
        print("Number of objects:", n_obj)
        print("Number of unique fcqfid:", len(np.unique(self.idx_fcqfid)))
        print("Number of unique filters:", n_filt)
        print("Number of gr filters:", n_filt_gr)
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
        model_structure: str = "hierarchical",
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
        model_structure : str, optional
            Type of model to use for the MCMC sampling (default: "hierarchical").
            Options: "pooled", "unpooled", "hierarchical"

        Returns
        -------
        None
        """

        if model_structure == "pooled":
            kernel = pooled_model
        elif model_structure == "unpooled":
            kernel = unpooled_model
        elif model_structure == "hierarchical":
            kernel = hierarchical_model
        else:
            raise ValueError("Invalid model structure. Options: 'pooled', 'unpooled', 'hierarchical'")
        init_strategy = nuts_params.pop("init_strategy", infer.init_to_median())
        self.sampler = infer.MCMC(
            infer.NUTS(kernel, init_strategy=init_strategy, **nuts_params),
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
        if model_structure == "pooled":
            running_params["idx_filt_gr"] = self.idx_filt_gr
        self.sampler.run(
            jax.random.PRNGKey(random_seed),
            **running_params,
            prior_params=prior_params,
        )

        # prior and posterior predictive checks
        prior_pred = infer.Predictive(kernel, num_samples=prior_pred_samples)(
            jax.random.PRNGKey(1919810 + random_seed), **running_params, prior_params=prior_params
        )
        post_pred = infer.Predictive(kernel, self.sampler.get_samples())(
            jax.random.PRNGKey(114514 + random_seed), **running_params, prior_params=prior_params
        )
        # convert to arviz InferenceData
        self.inf_data = az.from_numpyro(self.sampler, prior=prior_pred, posterior_predictive=post_pred)

        # store the posterior samples
        self.post_sample = self.inf_data.posterior

        for k, lc in enumerate(self.lc_library):
            fcqfid_in_obj = np.unique(self.idx_fcqfid[self.idx_obj == k])
            filt_in_obj = np.unique(self.idx_filt[self.idx_obj == k])

            lc.post_sample = {}  # self.post_sample[["C", "beta", "A", "alpha", "t_fl"]]
            lc.post_sample["C"] = self.post_sample["C"][:, :, fcqfid_in_obj]
            lc.post_sample["beta"] = self.post_sample["beta"][:, :, fcqfid_in_obj]
            lc.post_sample["A"] = self.post_sample["A"][:, :, filt_in_obj]
            if model_structure != "pooled":
                lc.post_sample["alpha"] = self.post_sample["alpha"][:, :, filt_in_obj]
                lc.post_sample["t_fl"] = self.post_sample["t_fl"][:, :, k]
            else:
                lc.post_sample["alpha"] = self.post_sample["alpha"]
                lc.post_sample["t_fl"] = self.post_sample["t_fl"]
            lc.post_sample = xr.Dataset(lc.post_sample)

    def plot_corner(self, save: bool = False, filename: str = None, var_name: list = ["mean_alpha", "std_alpha"]):
        corner.corner(
            self.inf_data.posterior[var_name],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.5, 0.95],
            title_quantiles=[0.05, 0.5, 0.95],
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf", bbox_inches="tight")
