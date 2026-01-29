import warnings

import arviz as az
import corner
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpyro import infer
from numpyro.infer.initialization import init_to_median
from sklearn.preprocessing import LabelEncoder

from .._utils import plt
from ..constants import T_PIVOT
from .model_structure import (
    f_t,
    hierarchical_model,
    pooled_model,
    unpooled_model,
)

####################################################################################################
################# Class to organize the light curve data and perform MCMC sampling #################
####################################################################################################


class SNLightCurve(object):
    """
    Class to organize light curves of individual SNe
    """

    def __init__(
        self,
        lc_early: dict = {},
        lc_peak: dict = None,
        t0_err: float = None,
        ztfid: str = None,
    ) -> None:
        try:
            self.ID = ztfid
            self.t0_err = t0_err

            # observations between 40% and 100% of max flux
            self.lc_early = self._init_lc_package(lc_early)
            self.n_obs = len(self.lc_early["phase"])

            fcqfid = self.lc_early["fcqfid"]
            filt = self.lc_early["filt"]

            # calculate number of unique fcqfid and filter and their indices
            fcqfid_encoder = LabelEncoder()
            self.idx_fcqfid = fcqfid_encoder.fit_transform(fcqfid)

            filt_encoder = LabelEncoder()
            self.idx_filt = filt_encoder.fit_transform(filt)

            # observations between -100 days and peak
            self.lc_peak = self._init_lc_package(lc_peak, fcqfids=np.unique(fcqfid))

            if self.lc_peak is not None:
                fcqfid_peak = self.lc_peak["fcqfid"]
                filt_peak = self.lc_peak["filt"]
                self.idx_fcqfid_peak = fcqfid_encoder.transform(fcqfid_peak)
                self.idx_filt_peak = filt_encoder.transform(filt_peak)

            self.post_sample = None
        except:
            print(
                f"Error initializing SNLightCurve for {ztfid}. Please check the input data."
            )
            raise

    @staticmethod
    def _init_lc_package(lc, fcqfids: dict = None):
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
        assert len(phase) == len(flux) == len(flux_err), (
            "Lengths of the data columns do not match"
        )
        if "beta" in lc.keys():
            beta = lc["beta"]
            assert len(beta) == len(phase), "Lengths of the data columns do not match"
        else:
            beta = np.ones_like(phase)

        fcqfid = lc.get("fcqfid", np.ones_like(phase, dtype=int))
        filt = lc.get("filt", np.zeros_like(phase, dtype=int))

        if fcqfids is not None:
            # filter the fcqfid and filt based on the provided fcqfids
            mask = np.isin(fcqfid, fcqfids)
            phase = phase[mask]
            flux = flux[mask]
            flux_err = flux_err[mask]
            fcqfid = fcqfid[mask]
            filt = filt[mask]
            beta = beta[mask]

        return dict(
            phase=phase,
            flux=flux,
            flux_err=flux_err,
            fcqfid=fcqfid,
            filt=filt,
            beta=beta,
        )

    def sampling(
        self,
        num_samples: int = 1000,
        num_warmup: int = 3000,
        num_chains: int = 2,
        thinning: int = 1,
        random_seed: int = 11,
        prior_pred_samples: int = 500,
        prior_config: dict = {},
        nuts_params: dict = {},
    ):
        """
        Perform MCMC sampling using NUTS algorithm.
        """
        kernel = unpooled_model

        # Fix t0_err for unpooled model
        if self.t0_err is None:
            t0_err_model = np.array([0.0])
        else:
            t0_err_model = np.array([float(self.t0_err)])

        # Prepare inputs as JAX arrays
        running_params = {
            "t": jnp.array(self.lc_early["phase"]),
            "flux": jnp.array(self.lc_early["flux"]),
            "flux_err": jnp.array(self.lc_early["flux_err"]),
            "beta": jnp.array(self.lc_early["beta"]),
            "t0_err": jnp.array(t0_err_model),
            "idx_obj": jnp.zeros_like(self.idx_fcqfid, dtype=int),
            "idx_fcqfid": jnp.array(self.idx_fcqfid),
            "idx_filt": jnp.array(self.idx_filt),
        }

        # --- DEFINE DIMENSIONS AND COORDINATES ---
        # This ensures ArviZ names the dimensions correctly (obj, filt, fcqfid)
        # instead of generic names like dim_0, dim_1.
        dims = {
            "t_rise": ["obj"],
            "t_fl": ["obj"],
            "alpha_0": ["obj", "filt"],
            "Aprime": ["obj", "filt"],
            "log_Aprime": ["obj", "filt"],
            "alpha_1": ["obj", "filt"],
            "C": ["fcqfid"],
            "beta": ["fcqfid"],
            "t_thresh": ["obj"],
        }

        # Local coordinates for this specific object run
        coords = {
            "obj": [0],  # Single object, index 0
            "filt": np.arange(len(np.unique(self.idx_filt))),
            "fcqfid": np.arange(len(np.unique(self.idx_fcqfid))),
        }

        # Standard MCMC
        self.sampler = infer.MCMC(
            infer.NUTS(
                kernel,
                init_strategy=init_to_median_with_alpha0(alpha_0_init=2.0),
                # dense_mass=True,
                target_accept_prob=0.95,
                **nuts_params,
            ),
            num_warmup=num_warmup,
            num_samples=num_samples * thinning,
            num_chains=num_chains,
            thinning=thinning,
            progress_bar=True,
            chain_method="vectorized",  # Forced
        )

        self.sampler.run(
            jax.random.PRNGKey(random_seed),
            **running_params,
            prior_config=prior_config,
        )

        prior_pred = infer.Predictive(kernel, num_samples=prior_pred_samples)(
            jax.random.PRNGKey(1919810 + random_seed),
            **running_params,
            prior_config=prior_config,
        )
        post_pred = infer.Predictive(kernel, self.sampler.get_samples())(
            jax.random.PRNGKey(114514 + random_seed),
            **running_params,
            prior_config=prior_config,
        )

        # PASS DIMS/COORDS TO ARVIZ
        self.inf_data = az.from_numpyro(
            self.sampler,
            prior=prior_pred,
            posterior_predictive=post_pred,
            dims=dims,
            coords=coords,
        )
        self.post_sample = self.inf_data.posterior

    def plot_lc(
        self,
        save: bool = False,
        filename: str = None,
        offset: float = 30,
        post_pred_samples: int = 25,
    ):
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

        colors = np.array(["tab:green", "tab:red", "tab:cyan", "tab:orange"])
        n_color = len(np.unique(self.idx_filt))

        _, ax = plt.subplots(
            figsize=(8, 2.25 * max(n_color, 2)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        post_sample = self.post_sample
        if post_sample is None:
            base_ = np.zeros_like(self.lc_early["flux"])
            beta_ = np.ones_like(self.lc_early["flux"])

            warnings.warn("No posterior samples available.")
        else:
            base_ = np.median(post_sample["C"][:, :, self.idx_fcqfid], axis=(0, 1))
            # beta_ = np.median(post_sample["beta"][:, :, self.idx_fcqfid], axis=(0, 1))
            beta_ = self.lc_early["beta"]
            t_fl = np.ravel(post_sample["t_fl"])

            idx_post_check = np.random.choice(len(t_fl), post_pred_samples)
            for i in idx_post_check:
                ax.axvline(t_fl[i], color="0.2", lw=0.1)

        for k, flt in enumerate(np.sort(np.unique(self.idx_filt))):
            for j, fcqfid in enumerate(np.unique(self.idx_fcqfid)):
                idx = (self.idx_filt == flt) & (self.idx_fcqfid == fcqfid)
                if idx.sum() == 0:
                    continue
                ax.errorbar(
                    self.lc_early["phase"][idx],
                    self.lc_early["flux"][idx]
                    - base_[idx]
                    - (flt - 0.5 * (n_color - 1)) * offset,
                    yerr=self.lc_early["flux_err"][idx] * beta_[idx],
                    color="w",
                    markeredgecolor=colors[k],
                    ecolor=colors[k],
                    fmt="o",
                    zorder=10,
                )
                if self.lc_peak is not None:
                    idx_peak = (self.idx_filt_peak == flt) & (
                        self.idx_fcqfid_peak == fcqfid
                    )
                    ax.errorbar(
                        self.lc_peak["phase"][idx_peak],
                        self.lc_peak["flux"][idx_peak]
                        - base_[idx][0]
                        - (flt - 0.5 * (n_color - 1)) * offset,
                        yerr=self.lc_peak["flux_err"][idx_peak] * beta_[idx][0],
                        color="w",
                        markeredgecolor=colors[k],
                        ecolor=colors[k],
                        fmt="o",
                        alpha=0.25,
                        zorder=10,
                    )

            ax.set_xlim(-31, +1)
            ax.set_ylim(-offset * (max(n_color, 2) - 1), 100 + offset)

            if post_sample is not None:
                amp_prime_ = np.ravel(post_sample["Aprime"][..., flt])
                alpha_0_ = np.ravel(post_sample["alpha_0"][..., flt])
                if "alpha_1" in post_sample.keys():
                    alpha_1_ = np.ravel(post_sample["alpha_1"][..., flt])
                else:
                    alpha_1_ = np.zeros_like(alpha_0_)
                t_pred = jnp.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
                for i in idx_post_check:
                    ax.plot(
                        t_pred,
                        f_t(
                            t_pred,
                            t_fl[i],
                            0,
                            amp_prime_[i],
                            alpha_0_[i],
                            alpha_1_[i],
                            t_pivot=T_PIVOT,
                        )
                        - (flt - 0.5 * (n_color - 1)) * offset,
                        color="0.2",
                        lw=0.1,
                        zorder=-1,
                    )

        ax.set_xlabel(r"$t - T_{B, \mathrm{max}}\ [\mathrm{restframe\ d}]$")
        ax.set_ylabel(r"$f + \mathrm{offset}$")
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(25))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
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

        var_names = kwargs.pop("var_names", ["t_rise", "Aprime", "alpha_0"])
        if "alpha_1" in self.post_sample.keys() and "alpha_1" not in var_names:
            var_names.append("alpha_1")

        corner.corner(
            self.post_sample,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.16, 0.5, 0.84],
            title_quantiles=[0.16, 0.5, 0.84],
            **kwargs,
            var_names=var_names,
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf", bbox_inches="tight")


class SNLightCurveLib(object):
    """
    Class to organize a library of light curves of SNe Ia
    """

    def __init__(
        self,
        lc_early_lib: list = None,
        lc_peak_lib: list = None,
        ztfid_lib: list = None,
        t0_err: list = None,
        sampling_model: str = "hierarchical",
    ) -> None:
        self.lc_library: list[SNLightCurve] = []
        self.ztfid_lib: list = ztfid_lib if ztfid_lib is not None else []
        self.model_structure = sampling_model

        self.phase, self.flux, self.flux_err, self.beta = [], [], [], []
        self.idx_filt = np.array([], dtype=int)
        self.idx_fcqfid = np.array([], dtype=int)
        self.idx_obj = np.array([], dtype=int)

        if lc_early_lib is None:
            return

        self.t0_err = np.array(t0_err, dtype=float) if t0_err is not None else None
        if t0_err is None:
            t0_err = [None] * len(lc_early_lib)

        if (lc_peak_lib is not None) and (ztfid_lib is not None):
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(
                        lc_early=lc_early,
                        lc_peak=lc_peak_lib[k],
                        ztfid=ztfid_lib[k],
                        t0_err=t0_err[k],
                    )
                )
        elif lc_peak_lib is not None:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(
                        lc_early=lc_early, lc_peak=lc_peak_lib[k], t0_err=t0_err[k]
                    )
                )
        elif ztfid_lib is not None:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(
                        lc_early=lc_early, ztfid=ztfid_lib[k], t0_err=t0_err[k]
                    )
                )
        else:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(lc_early=lc_early), t0_err=t0_err[k]
                )

        # Identify n_filt: number of unique filters across all objects
        n_filt = max([len(np.unique(lc.idx_filt)) for lc in self.lc_library])

        for k, lc in enumerate(self.lc_library):
            if lc.n_obs == 0:
                print(f"Warning: Light curve {lc.ID} has no observations. Skipping...")
            # concatenate the indices
            self.idx_filt = np.append(self.idx_filt, lc.idx_filt)
            self.idx_fcqfid = np.append(
                self.idx_fcqfid, lc.idx_fcqfid + len(np.unique(self.idx_fcqfid))
            )
            self.idx_obj = np.append(self.idx_obj, np.full(lc.n_obs, k))

            # concatenate the light curve data
            self.phase = np.append(self.phase, lc.lc_early["phase"])
            self.flux = np.append(self.flux, lc.lc_early["flux"])
            self.flux_err = np.append(self.flux_err, lc.lc_early["flux_err"])
            self.beta = np.append(self.beta, lc.lc_early["beta"])

        n_obj = len(np.unique(self.idx_obj))
        n_fcqfid = len(np.unique(self.idx_fcqfid))
        assert n_obj == self.idx_obj.max() + 1, "Indexing error: idx_obj"
        assert n_fcqfid == self.idx_fcqfid.max() + 1, "Indexing error: idx_fcqfid"
        assert n_filt == self.idx_filt.max() + 1, "Indexing error: idx_filt"
        print("Number of objects:", n_obj)
        print("Number of unique fcqfid:", len(np.unique(self.idx_fcqfid)))
        print("Number of filters:", n_filt)
        print("Number of total observations:", len(self.phase))
        print("Light curves compiled...")

        self.inf_data = None
        self.post_sample: xr.DataArray = None
        self.prior_sample: xr.DataArray = None

    @staticmethod
    def decode_sample(
        sample: xr.DataArray, f_thresh=0.025, t_pivot=T_PIVOT
    ) -> xr.DataArray:
        """
        Decode the posterior samples of each light curve from the packed hierarchical model.

        Parameters
        ----------
        sample : xr.DataArray
            The prior/posterior samples to decode.

        Returns
        -------
        None
        """

        # Decode Corr, Variance matrices if present
        if ("Corr" in sample) and ("Sigma" in sample):
            n_filt = sample.sizes.get("filt", 0)
            if n_filt > 0:
                for i in range(n_filt):
                    sample[f"corr_t_rise_alpha_flt{i + 1}"] = sample["Corr"][
                        ..., i + 1, 0
                    ]
                    sample[f"corr_t_rise_log_Aprime_flt{i + 1}"] = sample["Corr"][
                        ..., i + n_filt + 1, 0
                    ]
                    sample[f"corr_alpha_log_Aprime_flt{i + 1}"] = sample["Corr"][
                        ..., i + 1, i + n_filt + 1
                    ]
                    for j in range(i + 1, n_filt):
                        sample[f"corr_alpha_flt{i + 1}_flt{j + 1}"] = sample["Corr"][
                            ..., i + 1, j + 1
                        ]
                        sample[f"corr_t_rise_alpha_flt{i + 1}-flt{j + 1}"] = (
                            sample["Sigma"][..., i + 1, 0]
                            - sample["Sigma"][..., j + 1, 0]
                        ) / (
                            sample["Sigma"][..., 0, 0]
                            * (
                                sample["Sigma"][..., i + 1, i + 1]
                                + sample["Sigma"][..., j + 1, j + 1]
                                - 2 * sample["Sigma"][..., i + 1, j + 1]
                            )
                        ) ** 0.5

            for matrix in ["Corr", "Sigma"]:
                sample.drop_vars(matrix)

        # Post-calculate population-level parameters if not sampled directly
        if "mean_t_rise" not in sample:
            sample["mean_t_rise"] = sample["t_rise"].mean(dim="obj")
            sample["sigma_t_rise"] = sample["t_rise"].std(dim="obj", ddof=1)

        # Post-calculate differences between filters for mean_alpha_0 (color evolution)
        if "obj" in sample["alpha_0"].dims:
            if "mean_alpha_0" not in sample:
                sample["mean_alpha_0"] = sample["alpha_0"].mean(dim="obj")
                sample["sigma_alpha_0"] = sample["alpha_0"].std(dim="obj", ddof=1)
            if "mean_log_Aprime" not in sample:
                sample["mean_log_Aprime"] = sample["log_Aprime"].mean(dim="obj")
                sample["sigma_log_Aprime"] = sample["log_Aprime"].std(dim="obj", ddof=1)

            n_filt = sample.sizes["filt"]
            for j in range(n_filt):
                if f"corr_t_rise_alpha_flt{j + 1}" not in sample:
                    sample[f"corr_t_rise_alpha_flt{j + 1}"] = xr.corr(
                        sample["t_rise"],
                        sample["alpha_0"][..., j],
                        dim="obj",
                    )
                if f"corr_t_rise_log_Aprime_flt{j + 1}" not in sample:
                    sample[f"corr_t_rise_log_Aprime_flt{j + 1}"] = xr.corr(
                        sample["t_rise"],
                        sample["log_Aprime"][..., j],
                        dim="obj",
                    )
                if f"corr_alpha_log_Aprime_flt{j + 1}" not in sample:
                    sample[f"corr_alpha_log_Aprime_flt{j + 1}"] = xr.corr(
                        sample["alpha_0"][..., j],
                        sample["log_Aprime"][..., j],
                        dim="obj",
                    )
                for k in range(j + 1, n_filt):
                    sample[f"mean_alpha_flt{j + 1}-flt{k + 1}"] = (
                        sample["mean_alpha_0"][..., j] - sample["mean_alpha_0"][..., k]
                    )
                    sample[f"sigma_alpha_flt{j + 1}-flt{k + 1}"] = (
                        sample["sigma_alpha_0"][..., j] ** 2
                        + sample["sigma_alpha_0"][..., k] ** 2
                        - 2
                        * sample["sigma_alpha_0"][..., j]
                        * sample["sigma_alpha_0"][..., k]
                        * sample[f"corr_alpha_flt{j + 1}_flt{k + 1}"]
                    )
                    if f"corr_t_rise_alpha_flt{j + 1}-flt{k + 1}" not in sample:
                        sample[f"corr_t_rise_alpha_flt{j + 1}-flt{k + 1}"] = xr.corr(
                            sample["t_rise"],
                            sample["alpha_0"][..., j] - sample["alpha_0"][..., k],
                            dim="obj",
                        )
                    if f"corr_t_rise_alpha_flt{j + 1}_flt{k + 1}" not in sample:
                        sample[f"corr_alpha_flt{j + 1}_flt{k + 1}"] = xr.corr(
                            sample["alpha_0"][..., j],
                            sample["alpha_0"][..., k],
                            dim="obj",
                        )

        # Post-calculate t_thresh, xi_thresh and the related correlations
        exponent = sample["alpha_0"]  # only for power-law model right now
        log_t_thresh = (
            np.log10(f_thresh * 100) - np.log10(sample["Aprime"])
        ) / exponent
        sample["t_thresh"] = 10**log_t_thresh * t_pivot
        sample["xi"] = sample["t_thresh"] / sample["t_rise"]
        n_filt = sample.sizes["filt"]
        for j in range(n_filt):
            sample[f"corr_t_rise_xi_flt{j + 1}"] = xr.corr(
                sample["t_rise"],
                sample["xi"][..., j],
                dim="obj",
            )

        return sample

    def decode_prior_sample(self):
        """
        Decode the prior samples of each light curve from the packed hierarchical model.

        Returns
        -------
        None
        """
        if self.prior_sample is None:
            print("Inference data not yet available.")
            return

        prior_sample = self.decode_sample(self.prior_sample)
        self.prior_sample = self.drop_nuisances(prior_sample)

    def decode_post_sample(self, model_structure: str = "hierarchical"):
        """
        Decode the posterior samples of each light curve from the packed hierarchical model.

        Parameters
        ----------
        model_structure : str, optional
            Type of model used for the MCMC sampling (default: "hierarchical").
            Options:  "pooled", "unpooled", "hierarchical"
            Note: All hierarchical variants ("mvn", "independent", "tfl_only")
                use the same data structure

        Returns
        -------
        None
        """
        if self.post_sample is None:
            print("Inference data not yet available.")
            return

        self.post_sample = self.decode_sample(self.post_sample)

        # Decode the posterior samples for each light curve
        for k, lc in enumerate(self.lc_library):
            lc.post_sample = {}

            # Parameters common to all unique fcqfid in this object
            fcqfid_in_obj = np.unique(self.idx_fcqfid[self.idx_obj == k])
            lc.post_sample["C"] = self.post_sample["C"][..., fcqfid_in_obj]
            # lc.post_sample["beta"] = self.post_sample["beta"][..., fcqfid_in_obj]

            # A: (n_chains, n_samples, n_obj, n_filt)
            lc.post_sample["Aprime"] = self.post_sample["Aprime"][..., k, :]
            # lc.post_sample["A"] = self.post_sample["A"][..., k, :]

            # t_fl: (n_chains, n_samples, n_obj)
            lc.post_sample["t_rise"] = self.post_sample["t_rise"][..., k]
            lc.post_sample["t_fl"] = self.post_sample["t_fl"][..., k]

            if model_structure == "pooled":  # Pooled:  all objects share alpha_0
                # alpha_0: (n_chains, n_samples, n_filt)
                lc.post_sample["alpha_0"] = self.post_sample["alpha_0"]
                # alpha_1: (n_chains, n_samples, n_filt)
                if "alpha_1" in self.post_sample.keys():
                    lc.post_sample["alpha_1"] = self.post_sample["alpha_1"]

            else:  # Unpooled or Hierarchical (all variants:  mvn, independent, tfl_only)
                # alpha_0: (n_chains, n_samples, n_obj, n_filt)
                lc.post_sample["alpha_0"] = self.post_sample["alpha_0"][..., k, :]
                # alpha_1: (n_chains, n_samples, n_obj, n_filt)
                if "alpha_1" in self.post_sample.keys():
                    lc.post_sample["alpha_1"] = self.post_sample["alpha_1"][..., k, :]

            lc.post_sample = xr.Dataset(lc.post_sample)

    def append(self, lc_lib: "SNLightCurveLib"):
        """
        Append another SNLightCurveLib to the current one.

        Parameters
        ----------
        lc_lib : SNLightCurveLib
            The SNLightCurveLib to append.

        Returns
        -------
        None
        """
        if not isinstance(lc_lib, SNLightCurveLib):
            raise TypeError("lc_lib must be an instance of SNLightCurveLib")

        # concatenate the indices and light curve data
        n_obj_current = len(np.unique(self.idx_obj))
        if n_obj_current == 0:
            # Copy directly if the current library is empty
            self.idx_filt = lc_lib.idx_filt
            self.idx_fcqfid = lc_lib.idx_fcqfid
            self.idx_obj = lc_lib.idx_obj
            self.phase = lc_lib.phase
            self.flux = lc_lib.flux
            self.flux_err = lc_lib.flux_err
            self.beta = lc_lib.beta
            self.t0_err = lc_lib.t0_err

        else:
            for k, lc in enumerate(lc_lib.lc_library):
                self.idx_filt = np.append(self.idx_filt, lc.idx_filt)

                self.idx_fcqfid = np.append(
                    self.idx_fcqfid,
                    lc.idx_fcqfid + len(np.unique(self.idx_fcqfid)),
                )
                self.idx_obj = np.append(
                    self.idx_obj, np.full_like(lc.idx_filt, len(self.lc_library) + k)
                )
                self.phase = np.append(self.phase, lc.lc_early["phase"])
                self.flux = np.append(self.flux, lc.lc_early["flux"])
                self.flux_err = np.append(self.flux_err, lc.lc_early["flux_err"])
                self.beta = np.append(self.beta, lc.lc_early["beta"])
                assert (self.t0_err is None) == (lc.t0_err is None), (
                    "t0_err presence mismatch"
                )
                if self.t0_err is None:
                    self.t0_err = None
                else:
                    self.t0_err = np.append(self.t0_err, lc.t0_err)

            n_obj = len(np.unique(self.idx_obj))
            n_fcqfid = len(np.unique(self.idx_fcqfid))
            n_filt = len(np.unique(self.idx_filt))

            assert n_obj == self.idx_obj.max() + 1, "Indexing error: idx_obj"
            assert n_fcqfid == self.idx_fcqfid.max() + 1, "Indexing error: idx_fcqfid"
            assert n_filt == self.idx_filt.max() + 1, "Indexing error: idx_filt"
            print("Number of objects:", n_obj)
            print("Number of unique fcqfid:", len(np.unique(self.idx_fcqfid)))
            print("Number of filters:", n_filt)
            print("Number of total observations:", len(self.phase))
            print("Light curves appended...")

        self.lc_library.extend(lc_lib.lc_library)
        self.ztfid_lib.extend(lc_lib.ztfid_lib)

    def _sampling_unpooled(
        self,
        num_samples: int,
        num_warmup: int,
        num_chains: int,
        thinning: int,
        random_seed: int,
        prior_config: dict,
        nuts_params: dict,
    ):
        """Perform sampling for unpooled model using parallel processing."""
        from joblib import Parallel, delayed

        run_params = {
            "num_samples": num_samples,
            "num_warmup": num_warmup,
            "num_chains": num_chains,
            "thinning": thinning,
            "random_seed": random_seed,
            "prior_config": prior_config,
        }

        print(f"Parallelizing fits across {len(self.lc_library)} objects...")

        def _run_fit(lc, seed_offset):
            local_params = run_params.copy()
            local_params["random_seed"] += seed_offset
            lc.sampling(**local_params, nuts_params=nuts_params)

            # Clear heavy objects for clean pickling
            if hasattr(lc, "sampler"):
                lc.sampler = None
            if hasattr(lc, "inf_data"):
                lc.inf_data = None

            return lc

        num_devices = jax.local_device_count()
        print(f"Using {num_devices} devices for parallel processing.")
        results = Parallel(n_jobs=num_devices)(
            delayed(_run_fit)(lc, i) for i, lc in enumerate(self.lc_library)
        )

        self.lc_library = results
        self.aggregate_samples()

    def _run_warmup(
        self,
        kernel,
        running_params: dict,
        prior_config: dict,
        num_warmup: int,
        num_chains: int,
        chain_method: str,
        nuts_params: dict,
        rng_key,
        model_structure: str,
        debug_save: bool,
        debug_dir: str | None,
        debug_basename: str | None,
    ):
        """Run warmup stages: SA warmup and/or warmup without t0_err."""

        if "hierarchical" in model_structure:
            print("\nSimulated Annealing warmup...")
            running_params_sa = running_params.copy()
            running_params_sa["t0_err"] = jnp.zeros_like(running_params["t"])
            num_sa = int(num_warmup * 0.1)

            rng_key, sa_key = jax.random.split(rng_key)

            sampler_sa = infer.MCMC(
                infer.SA(
                    kernel,
                    init_strategy=init_to_median_with_alpha0(alpha_0_init=2.0),
                    dense_mass=False,
                ),
                num_warmup=0,
                num_samples=num_sa,
                num_chains=1,
                chain_method="sequential",
            )
            sampler_sa.run(sa_key, **running_params_sa, prior_config=prior_config)

            # Use median of late SA window for initialization
            samples_sa = sampler_sa.get_samples()
            late_frac = (0.2, 0.1)
            start = int(jnp.floor((1.0 - late_frac[0]) * num_sa))
            start = int(jnp.clip(start, 0, max(num_sa - 1, 0)))
            end = int(jnp.floor((1.0 - late_frac[1]) * num_sa))
            end = int(jnp.clip(end, 0, max(num_sa - 1, 0)))

            init_values_warmup = {}
            for k, v in samples_sa.items():
                v_win = v[start:end]
                init_values_warmup[k] = jnp.median(v_win, axis=0)

            for k, v in init_values_warmup.items():
                if not jnp.all(jnp.isfinite(v)):
                    raise ValueError(f"Non-finite value found in {k}")

            init_strategy_warmup = infer.init_to_value(values=init_values_warmup)

            del sampler_sa, samples_sa

        else:
            num_sa = 0
            init_strategy_warmup = init_to_median_with_alpha0(alpha_0_init=2.0)

        if self.t0_err is not None:
            print("\nWarmup without t0_err...")
            running_params_no_t0_err = running_params.copy()
            running_params_no_t0_err["t0_err"] = jnp.zeros_like(running_params["t"])
            num_no_t0_err = int(num_warmup * 0.25)

            rng_key, no_t0_err_key = jax.random.split(rng_key)

            if chain_method == "vectorized":
                print("Doubling the number of chains for exploration")
                num_chains_warmup = num_chains * 2
            else:
                num_chains_warmup = num_chains

            sampler_no_t0_err = infer.MCMC(
                infer.NUTS(
                    kernel,
                    init_strategy=init_strategy_warmup,
                    target_accept_prob=0.95,
                    **nuts_params,
                ),
                num_warmup=num_no_t0_err,
                num_samples=max(int(num_no_t0_err * 0.1), 1),
                num_chains=num_chains_warmup,
                chain_method=chain_method,
            )
            sampler_no_t0_err.run(
                no_t0_err_key,
                **running_params_no_t0_err,
                prior_config=prior_config,
                extra_fields=("num_steps", "energy", "accept_prob"),
            )

            samples_no_t0_err = sampler_no_t0_err.get_samples()
            init_values_no_t0_err = {
                k: jnp.median(v, axis=0) for k, v in samples_no_t0_err.items()
            }
            init_strategy_main = infer.init_to_value(values=init_values_no_t0_err)

            # Optional: save only warmup posterior (xarray) via ArviZ
            if debug_save:
                from pathlib import Path

                import arviz as az

                save_dir = (
                    Path(debug_dir)
                    if debug_dir
                    else Path(__file__).resolve().parent / "debug"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                post_warmup = az.from_numpyro(sampler_no_t0_err).posterior.astype(
                    "float16"
                )
                post_warmup.to_netcdf(
                    save_dir / f"{debug_basename}_warmup_no_t0_err.nc"
                )

                # Save extra fields (if available)
                extra = sampler_no_t0_err.get_extra_fields()
                if extra:
                    np.savez_compressed(
                        save_dir / f"{debug_basename}_extra_fields.npz",
                        **{k: np.array(v) for k, v in extra.items()},
                    )

            del sampler_no_t0_err, samples_no_t0_err
        else:
            num_no_t0_err = 0
            init_strategy_main = init_strategy_warmup

        return init_strategy_main, num_no_t0_err, rng_key

    def _run_main_sampling(
        self,
        kernel,
        running_params: dict,
        prior_config: dict,
        num_samples: int,
        num_warmup: int,
        num_chains: int,
        thinning: int,
        chain_method: str,
        nuts_params: dict,
        init_strategy_main,
        effective_warmup: int,
        model_structure: str,
        rng_key,
    ):
        """Run the main NUTS sampling."""
        print("\nMain sampling...")

        if model_structure == "hierarchical_mvn":
            dense_mass_site = [
                (
                    "mean_t_rise",
                    "mean_alpha_0",
                    "mean_log_Aprime",
                    "sigma_t_rise",
                    "sigma_alpha_0",
                    "sigma_log_Aprime",
                    "chol_corr",
                )
            ]
        else:
            dense_mass_site = []

        rng_key, main_key = jax.random.split(rng_key)

        self.sampler = infer.MCMC(
            infer.NUTS(
                kernel,
                init_strategy=init_strategy_main,
                dense_mass=dense_mass_site,
                target_accept_prob=0.95,
                **nuts_params,
            ),
            num_warmup=effective_warmup,
            num_samples=num_samples * thinning,
            thinning=thinning,
            num_chains=num_chains,
            chain_method=chain_method,
        )
        self.sampler.run(
            main_key,
            **running_params,
            prior_config=prior_config,
            extra_fields=("num_steps", "energy", "accept_prob"),
        )

        return rng_key

    def _check_sampler_health(self, nuts_params: dict):
        """Check sampler diagnostics and report health status."""
        extra = self.sampler.get_extra_fields()
        num_steps = extra["num_steps"]
        accept_probs = extra["accept_prob"]
        energies = extra["energy"]

        print("\n--- Sampler Health Check ---")

        # Flag A: Max Tree Depth (Truncation)
        max_steps_limit = 2 ** nuts_params.get("max_tree_depth", 10)
        pct_at_limit = np.mean(num_steps >= max_steps_limit) * 100
        print(f"   Maximum tree depth reached: {np.log2(np.max(num_steps)):.0f}")
        if pct_at_limit > 5:
            print(
                f"⚠️  WARNING: {pct_at_limit:.1f}% of samples hit max_tree_depth ({max_steps_limit} steps)."
            )
            print(
                "   The sampler is being truncated. Increase max_tree_depth or reparameterize."
            )

        # Flag B: E-BFMI (Energy Flow)
        energy_diff = np.diff(energies)
        ebfmi = np.var(energy_diff) / np.var(energies)
        if ebfmi < 0.3:
            print(f"⚠️  WARNING: Low E-BFMI ({ebfmi:.2f}).")
            print(
                "   The sampler is struggling to explore the tails. Check for high correlations."
            )

        # Flag C: Acceptance Rate
        avg_accept = np.mean(accept_probs)
        if avg_accept < 0.5:
            print(f"⚠️  WARNING: Low average acceptance probability ({avg_accept:.2f}).")
            print("   The sampler is 'stuck' or the step size is too large.")

        # Flag D: R_hat (Convergence)
        rhat_data = az.rhat(self.post_sample)
        # Compute numeric max across all variables, ignoring NaNs
        max_rhat = np.nanmax(
            [np.nanmax(rhat_data[var_name].values) for var_name in rhat_data.data_vars]
        )
        if max_rhat > 1.01:
            # Find variables with high R_hat
            bad_vars = []
            for var_name in rhat_data.data_vars:
                var_rhat = rhat_data[var_name]
                if var_rhat.ndim == 0:  # scalar variable
                    if float(var_rhat) > 1.01:
                        bad_vars.append(f"{var_name} ({float(var_rhat):.4f})")
                else:  # array variable
                    bad_indices = np.where(var_rhat.values > 1.01)
                    if len(bad_indices[0]) > 0:
                        max_val = float(var_rhat.values[bad_indices].max())
                        bad_vars.append(f"{var_name} ({max_val:.4f})")

            print(f"⚠️  WARNING: High R_hat detected (max = {max_rhat:.4f}).")
            print(f"   Variables with R_hat > 1.01: {', '.join(bad_vars)}")
            print(
                "   Chains have not converged. Consider increasing num_warmup or num_samples."
            )

        if ebfmi >= 0.3 and pct_at_limit <= 5:
            print("✅ All NUTS diagnostics look healthy.")

    def sampling(
        self,
        num_samples: int = 1000,
        num_warmup: int = 3000,
        num_chains: int = 2,
        thinning: int = 1,
        random_seed: int = 11,
        sample_prior: bool = False,
        prior_config: dict = {},
        nuts_params: dict = {},
        debug_save: bool = False,
        debug_dir: str = None,
        debug_basename: str = None,
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
        thinning : int, optional
            Thinning factor for MCMC samples (default: 1).
        random_seed : int, optional
            Random seed for reproducibility (default: 11).
        sample_prior : bool, optional
            Whether to sample from the prior distribution only (default: False).
        prior_pred_samples : int, optional
            Number of samples to draw from the prior predictive distribution (default: 500).
        prior_config : dict, optional
            Dictionary containing the prior information for the model.
        nuts_params : dict, optional
            Dictionary containing the parameters for infer.NUTS.

        Returns
        -------
        None

        Notes
        -----
        If `debug_save` is True, warmup sampler properties and posterior samples are saved under `debug_dir` (defaults to a subdirectory named "debug" next to this file if not provided).
        """
        from .._utils import (
            extract_coords_dims_from_model,
            get_recommended_chain_method,
        )

        model_structure = self.model_structure

        assert model_structure in [
            "pooled",
            "unpooled",
            "hierarchical",
            "hierarchical_trise",
            "hierarchical_mvn",
        ], f"Invalid model structure: {model_structure}"

        if model_structure == "unpooled":
            print("Using unpooled model for sampling...")
            kernel = unpooled_model
        elif model_structure == "pooled":
            print("Using pooled model for sampling...")
            kernel = pooled_model
        elif model_structure == "hierarchical":
            print("Using hierarchical model for sampling...")
            prior_config["correlation_structure"] = "independent"
            kernel = hierarchical_model
        elif model_structure == "hierarchical_trise":
            print("Using hierarchical t_rise model for sampling...")
            prior_config["correlation_structure"] = "trise_only"
            kernel = hierarchical_model
        elif model_structure == "hierarchical_mvn":
            print("Using hierarchical mvn model for sampling...")
            prior_config["correlation_structure"] = "mvn"
            kernel = hierarchical_model
        else:
            raise ValueError(
                "Invalid model structure. Options: 'pooled', 'unpooled', 'hierarchical' (as well as '_mvn' and '_trise')"
            )

        running_params = {
            "t": self.phase,
            "flux": self.flux,
            "flux_err": self.flux_err,
            "beta": self.beta,
            "t0_err": self.t0_err
            if self.t0_err is not None
            else jnp.zeros_like(self.phase),
            "idx_obj": self.idx_obj,
            "idx_fcqfid": self.idx_fcqfid,
            "idx_filt": self.idx_filt,
        }

        rng_key = jax.random.PRNGKey(random_seed)

        platform = jax.default_backend()
        chain_method = get_recommended_chain_method(platform, num_chains)

        if self.prior_sample is None:
            print("Sampling from prior...")
            prior_pred = infer.Predictive(kernel, num_samples=num_samples * num_chains)(
                rng_key,
                **running_params,
                prior_config=prior_config,
            )

            coords, dims = extract_coords_dims_from_model(
                kernel,
                model_kwargs={**running_params, "prior_config": prior_config},
                num_samples=1,
            )

            self.prior_sample = az.from_numpyro(
                prior=prior_pred, coords=coords, dims=dims
            ).prior.astype("float32")

        if sample_prior:
            self.decode_prior_sample()
            return

        if model_structure == "unpooled":
            self._sampling_unpooled(
                num_samples,
                num_warmup,
                num_chains,
                thinning,
                random_seed,
                prior_config,
                nuts_params,
            )
            return

        # Run warmup stages
        init_strategy_main, num_no_t0_err, rng_key = self._run_warmup(
            kernel,
            running_params,
            prior_config,
            num_warmup,
            num_chains,
            chain_method,
            nuts_params,
            rng_key,
            model_structure,
            debug_save,
            debug_dir,
            debug_basename,
        )

        # Run main sampling
        effective_warmup = num_warmup - num_no_t0_err

        rng_key = self._run_main_sampling(
            kernel,
            running_params,
            prior_config,
            num_samples,
            num_warmup,
            num_chains,
            thinning,
            chain_method,
            nuts_params,
            init_strategy_main,
            effective_warmup,
            model_structure,
            rng_key,
        )

        inf_data = az.from_numpyro(self.sampler)
        self.post_sample = self.drop_nuisances(inf_data.posterior.astype("float32"))

        # Check sampler health
        self._check_sampler_health(nuts_params)

    @staticmethod
    def drop_nuisances(sample: xr.DataArray | None):
        NUISANCES = ["alpha-", "theta", "chol_corr"]
        if sample is not None:
            vars_to_remove = [
                var
                for var in sample
                if var in NUISANCES or "raw" in var or var.startswith("_")
            ]
            sample = sample.drop_vars(vars_to_remove)
        return sample

    def drop_bad_chains(
        self,
        n_thresh: int = 1,
        sigma_thresh: float = 3.0,
        iterative: bool = True,
        max_iters: int = 5,
        var_thresh: float = 1e-8,
        stuck_percentile: float = 0.5,
    ):
        """
        Identify and drop bad chains iteratively using:
        (i) Z-score thresholding of parameter means; and
        (ii) detection of chains with extremely low within-chain variance (stuck chains); and
        (iii) a Gaussian z-test over per-chain dispersion across chains to reject variance outliers.

        Parameters
        ----------
        n_thresh : int, optional
            Minimum number of parameters that must flag a chain as bad for it to be dropped (default: 1).
        sigma_thresh : float, optional
            Z-score threshold for flagging bad chains (default: 3.0).
        iterative : bool, optional
            If True, repeat detection and removal until no new bad chains are found or max_iters is reached (default: True).
        max_iters : int, optional
            Maximum number of iterations when iterative=True (default: 5).
        var_thresh : float, optional
            Absolute variance threshold below which a chain is considered stuck (default: 1e-8).
        stuck_percentile : float, optional
            Additionally mark chains stuck if their median std across parameters is below this percentile (0-100) of all chains (default: 0.5).
        """
        import numpy as np
        from astropy.stats import mad_std

        # Candidate parameters to evaluate
        params = [
            k
            for k in self.post_sample.data_vars.keys()
            if "mean" in k or "sigma" in k or "corr" in k
        ]

        def find_bad_chains(ds):
            bad_chains_accum = []

            # (A) Mean-shift detection via robust Z-scores
            for param in params:
                dims_to_reduce = [d for d in ds[param].dims if d != "chain"]
                means = ds[param].median(dim=dims_to_reduce).values  # per-chain means
                robust_scale = mad_std(means)
                # Guard against zero scale
                if robust_scale == 0 or not np.isfinite(robust_scale):
                    # If no dispersion, treat large deviations from global mean cautiously
                    z_scores = np.zeros_like(means)
                else:
                    z_scores = (means - np.mean(means)) / robust_scale

                bad_chains = np.where(np.abs(z_scores) > sigma_thresh)[0]
                if len(bad_chains) > 0:
                    print(f"⚠️  WARNING: found {len(bad_chains)} bad chains for {param}")
                    print(f"    Suspected bad chain indices: {bad_chains}")

                bad_chains_accum = np.append(bad_chains_accum, bad_chains)

            # (B) Stuck-chain detection via extremely low within-chain variance
            # Compute per-chain std across draws for each param, then aggregate by median across params
            per_param_std = []
            for param in params:
                # std over draws per chain -> (chain,)
                dims_to_reduce = [d for d in ds[param].dims if d != "chain"]
                stds = (
                    ds[param].std(dim=dims_to_reduce).values
                )  # per-chain std across all non-chain axes
                per_param_std.append(stds)
            if len(per_param_std) > 0:
                per_param_std = np.vstack(per_param_std)  # (n_params, n_chains)
                median_chain_std = np.median(per_param_std, axis=0)  # (n_chains,)

                # Absolute threshold
                stuck_abs = np.where(median_chain_std < np.sqrt(var_thresh))[0]

                # Percentile-based threshold (guard against degenerate distribution)
                finite_stds = median_chain_std[np.isfinite(median_chain_std)]
                if finite_stds.size > 0:
                    perc_val = np.percentile(finite_stds, stuck_percentile)
                    stuck_pct = np.where(median_chain_std <= perc_val)[0]
                else:
                    stuck_pct = np.array([], dtype=int)

                stuck_chains = np.unique(np.concatenate([stuck_abs, stuck_pct]))
                if len(stuck_chains) > 0:
                    print(
                        f"⚠️  WARNING: found {len(stuck_chains)} stuck chains (low variance)"
                    )
                    print(f"    Suspected stuck chain indices: {stuck_chains}")

                bad_chains_accum = np.append(bad_chains_accum, stuck_chains)

            # (C) Gaussian z-test on per-chain dispersion across chains
            if len(per_param_std) > 0:
                # Use median std per chain across parameters as a dispersion metric
                dispersion = median_chain_std
                disp_scale = np.std(dispersion)
                if disp_scale == 0 or not np.isfinite(disp_scale):
                    z_var = np.zeros_like(dispersion)
                else:
                    z_var = (dispersion - np.mean(dispersion)) / disp_scale
                var_outliers = np.where(np.abs(z_var) > sigma_thresh)[0]
                if len(var_outliers) > 0:
                    print(
                        f"⚠️  WARNING: found {len(var_outliers)} variance outlier chains (z-test)"
                    )
                    print(
                        f"    Suspected variance outlier chain indices: {var_outliers}"
                    )
                bad_chains_accum = np.append(bad_chains_accum, var_outliers)

            # Consolidate by requiring a chain to be flagged more than n_thresh times
            if len(bad_chains_accum) == 0:
                return np.array([], dtype=int)

            counts = np.bincount(bad_chains_accum.astype(int))
            flagged = np.where(counts > n_thresh)[0]
            return flagged.astype(int)

        it = 0
        total_removed = []
        while True:
            it += 1
            ds = self.post_sample
            flagged = find_bad_chains(ds)

            # Stop if nothing to remove
            if flagged.size == 0:
                if it == 1:
                    print("    No bad or stuck chains detected.")
                else:
                    print("    No additional bad or stuck chains detected.")
                break

            # Drop flagged chains
            before = list(ds.chain.values)
            to_keep = [c for c in before if c not in flagged]
            self.post_sample = ds.sel(chain=to_keep)
            total_removed.extend(flagged.tolist())
            print(
                f"    Iteration {it}: deleted {len(flagged)} bad/stuck chains -> remaining {len(to_keep)} chains"
            )

            # Stop if not iterative or reached max iterations
            if not iterative or it >= max_iters:
                break

        if len(total_removed) > 0:
            unique_removed = np.unique(np.array(total_removed, dtype=int))
            print(f"    Total deleted {len(unique_removed)} chains: {unique_removed}")

    def drop_chains(self, chain_ids):
        """
        Drop specified chains from the dataset.
        """
        self.post_sample = self.post_sample.drop_sel(chain=chain_ids)

    def aggregate_samples(self):
        """
        Aggregate the prior and posterior samples from individual light curves into a single
        xarray.Dataset, creating a data structure comparable to the hierarchical model output.
        This method is intended for use with unpooled models where each light curve
        is sampled independently.
        """
        # Identify nuisance parameters which are handled differently from physics parameters.
        # Nuisance parameters are stacked along 'fcqfid', while physics parameters are stacked along 'obj'.
        nuisance_vars = ["C", "beta"]
        # Variables to exclude from aggregation
        exclude_vars = ["flux"]

        def _aggregate_samples(sample_attr: str):
            """Helper function to aggregate either 'post_sample' or 'prior_sample'."""
            # Filter for light curves that have the specified sample attribute
            valid_lcs = [
                lc
                for lc in self.lc_library
                if hasattr(lc, sample_attr) and getattr(lc, sample_attr) is not None
            ]

            if not valid_lcs:
                print(f"No {sample_attr} found to aggregate.")
                return None

            # Get variable names from first valid light curve
            all_vars = list(getattr(valid_lcs[0], sample_attr).data_vars)
            # Filter out nuisance and excluded variables
            physics_vars = [
                v for v in all_vars if v not in nuisance_vars and v not in exclude_vars
            ]

            physics_datasets = []
            nuisance_datasets = []

            # Process each light curve's sample
            for i, lc in enumerate(valid_lcs):
                sample = getattr(lc, sample_attr)

                # Extract physics parameters
                ds_phys = sample[physics_vars]

                # If obj dimension exists with size > 1, select only the first element (unpooled should have size 1)
                if "obj" in ds_phys.dims and ds_phys.sizes["obj"] > 1:
                    # This shouldn't happen for unpooled, but handle it just in case
                    ds_phys = ds_phys.isel(obj=0)

                # If obj dimension exists with size 1, squeeze it out
                if "obj" in ds_phys.dims and ds_phys.sizes["obj"] == 1:
                    ds_phys = ds_phys.squeeze("obj", drop=True)

                # Now expand with a fresh obj dimension
                ds_phys = ds_phys.expand_dims(dim={"obj": 1}).assign_coords(obj=[i])

                physics_datasets.append(ds_phys)

                # Extract nuisance parameters if they exist
                lc_nuisance_vars = [
                    v for v in nuisance_vars if v in sample and v not in exclude_vars
                ]
                if lc_nuisance_vars:
                    nuisance_datasets.append(sample[lc_nuisance_vars])

            # Concatenate all physics datasets along the 'obj' dimension
            aggregated_physics = xr.concat(
                physics_datasets,
                dim="obj",
                coords="all",
                compat="override",
                join="outer",
            )

            # Transpose each variable to put 'obj' in the correct position
            transposed_vars = {}
            for var in aggregated_physics.data_vars:
                dims = list(aggregated_physics[var].dims)
                if "obj" in dims:
                    # Remove 'obj' from current position
                    dims.remove("obj")

                    # If variable has 'filt' dimension, obj should be second-to-last
                    # Otherwise, obj should be last
                    if "filt" in dims:
                        # Insert 'obj' before 'filt' (second-to-last)
                        filt_idx = dims.index("filt")
                        dims.insert(filt_idx, "obj")
                    else:
                        # Append 'obj' as last dimension
                        dims.append("obj")

                    transposed_vars[var] = aggregated_physics[var].transpose(*dims)
                else:
                    transposed_vars[var] = aggregated_physics[var]

            aggregated_physics = xr.Dataset(transposed_vars)

            # If nuisance parameters were found, concatenate them and merge with physics parameters
            if nuisance_datasets:
                aggregated_nuisance = xr.concat(
                    nuisance_datasets,
                    dim="fcqfid",
                    coords="all",
                    compat="override",
                    join="outer",
                )
                result = xr.merge([aggregated_physics, aggregated_nuisance])
            else:
                result = aggregated_physics

            print(
                f"{sample_attr} aggregation complete. Final Dimensions: {result.sizes}"
            )
            return result

        # Aggregate posteriors
        aggregated_post = _aggregate_samples("post_sample")
        if aggregated_post is not None:
            self.post_sample = self.decode_sample(aggregated_post)

        # Aggregate priors
        aggregated_prior = _aggregate_samples("prior_sample")
        if aggregated_prior is not None:
            self.prior_sample = self.decode_sample(aggregated_prior)

    def plot_corner(
        self,
        save: bool = False,
        filename: str = None,
        var_names: list = ["mean_alpha_0", "sigma_alpha_0"],
        **kwargs,
    ):
        corner.corner(
            self.post_sample[var_names],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.16, 0.5, 0.84],
            title_quantiles=[0.16, 0.5, 0.84],
            **kwargs,
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf", bbox_inches="tight")

    def plot_trace(self):
        """
        Arviz plot_trace wrapper.
        """

        az.plot_trace(
            self.post_sample,
            sorted(
                [var for var in self.post_sample if "mean" in var or "sigma" in var]
            ),
        )

    def show_summary(self):
        """
        Arviz summary wrapper.
        """

        return az.summary(
            self.post_sample,
            sorted(
                [
                    var
                    for var in self.post_sample
                    if (
                        ("mean" in var or "sigma" in var or "corr" in var)
                        and "chain" in self.post_sample[var].dims
                        and "draw" in self.post_sample[var].dims
                    )
                ]
            ),
            stat_focus="median",
            hdi_prob=0.95,
        )

    def compare_true_vs_fitted_params(self, band: str = "g"):
        """
        Compare true vs fitted parameters for a specific band.

        Parameters
        ----------
        lib : SNLightCurveLib
            Light curve library with posterior samples and true parameters.
        band : str
            Band to plot for alpha_0. Must be "g" or "r".
        """
        import statsmodels.api as sm

        assert band in ["g", "r"], "band must be 'g' or 'r'"
        assert hasattr(self, "params_true"), "true parameters not found"

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 6),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 0.35]},
        )
        ax = [axs[0, 0], axs[0, 1]]
        ax_res = [axs[1, 0], axs[1, 1]]

        for k, varname in enumerate(["alpha_0", "t_rise"]):
            # Extract true parameters
            if varname == "alpha_0":
                true_vals = np.asarray(self.params_true["alpha_0"])
            else:
                true_vals = np.asarray(self.params_true["t_rise"])

            # Posterior medians and uncertainties
            if varname == "alpha_0":
                band_idx = 0 if band == "g" else 1

                # Handle different model structures
                if self.model_structure == "pooled":
                    # Pooled:  alpha_0 is shared across all objects
                    # Shape: (n_chains, n_samples, n_filt)
                    post_alpha = self.post_sample[varname][:, :, band_idx]

                    # Replicate for all objects
                    fitted_vals = np.full(len(true_vals), np.median(post_alpha))
                    percentiles = np.percentile(post_alpha, (16, 84))
                    fitted_vals_down = np.full(
                        len(true_vals), np.abs(percentiles[0] - fitted_vals[0])
                    )
                    fitted_vals_up = np.full(
                        len(true_vals), percentiles[1] - fitted_vals[0]
                    )

                else:  # hierarchical (mvn, independent)
                    # Hierarchical:  alpha_0 per object, per filter
                    # Shape: (n_chains, n_samples, n_obj, n_filt)
                    # Select filters for this band
                    post_alpha = self.post_sample[varname][..., band_idx]

                    # Now post_alpha_avg has shape (n_chains, n_samples, n_obj)
                    fitted_vals = np.median(post_alpha, axis=(0, 1))
                    percentiles = np.percentile(post_alpha, (16, 84), axis=(0, 1))
                    fitted_vals_down = np.abs(percentiles[0] - fitted_vals)
                    fitted_vals_up = percentiles[1] - fitted_vals

            else:  # t_rise
                # Shape: (n_chains, n_samples, n_obj)
                fitted_vals = np.median(self.post_sample[varname], axis=(0, 1))
                percentiles = np.percentile(
                    self.post_sample[varname], (16, 84), axis=(0, 1)
                )
                fitted_vals_down = np.abs(percentiles[0] - fitted_vals)
                fitted_vals_up = percentiles[1] - fitted_vals

            fitted_vals_std = 0.5 * (fitted_vals_up + fitted_vals_down)

            # For plotting ranges
            min_val = np.round(np.min(true_vals), 1) - 0.1
            max_val = np.round(np.max(true_vals), 1) + 0.1

            # Scatter with error bars
            ax[k].errorbar(
                true_vals,
                fitted_vals,
                yerr=[fitted_vals_down, fitted_vals_up],
                alpha=0.5,
                color="#4F6D7A",
                fmt="o",
                linewidth=0.75,
            )

            # 1: 1 line
            ax[k].plot(
                [min_val, max_val],
                [min_val, max_val],
                color="k",
                linestyle="--",
                zorder=0,
            )

            # -------- Weighted linear regression (WLS) --------
            weights = 1.0 / fitted_vals_std**2

            X = sm.add_constant(true_vals)  # add intercept
            wls_model = sm.WLS(fitted_vals, X, weights=weights)
            results = wls_model.fit()

            slope = results.params[1]
            intercept = results.params[0]

            reg_x = np.linspace(min_val, max_val, 200)
            reg_y = intercept + slope * reg_x

            ax[k].plot(
                reg_x,
                reg_y,
                color="#4F6D7A",
                linestyle="--",
                zorder=1,
            )

            print(f"WLS fit for {varname} ({band}-band, {self.model_structure}):")
            print(f"  Slope: {slope:.3f}, Intercept: {intercept:.3f}")

            # -------- Residuals / pulls --------
            pulls = (fitted_vals - true_vals) / fitted_vals_std

            ax_res[k].scatter(
                true_vals,
                pulls,
                alpha=0.5,
                color="#4F6D7A",
            )
            ax_res[k].axhline(0, color="k", linestyle="--", zorder=0)
            ax_res[k].fill_between(
                [min_val, max_val], 1, -1, color="gray", alpha=0.2, zorder=-1
            )

            # Axis formatting
            ax[k].set_xticks([])
            ax_res[k].set_ylabel(r"$\mathrm{Pull}$")
            ax_res[k].set_ylim(-3.5, 3.5)
            ax[k].set_xlim(min_val, max_val)
            ax[k].set_ylim(
                min_val - 0.1 * (max_val - min_val),
                max_val + 0.1 * (max_val - min_val),
            )
            ax_res[k].set_xlim(min_val, max_val)

        # Global labels
        ax[0].set_ylabel(rf"$\widehat{{\alpha}}_{band}$")
        ax[1].set_ylabel(r"$\widehat{t}_\mathrm{fl}\ [\mathrm{days}]$")
        ax_res[0].set_xlabel(rf"$\alpha_{band}$")
        ax_res[1].set_xlabel(r"$t_\mathrm{fl}\ [\mathrm{days}]$")

        return fig, axs


def init_to_median_with_alpha0(site=None, alpha_0_init=2.0):
    """
    Custom initialization strategy that uses median for most parameters
    but sets alpha_0 (and related alpha parameters) to a specific value.

    Parameters
    ----------
    site : dict, optional
        Site dictionary from NumPyro
    alpha_0_init : float, optional
        Initial value for alpha_0 parameter (default: 2.0)

    Returns
    -------
    init_fn : callable or array
        Initialization function compatible with NumPyro NUTS
    """
    from functools import partial

    if site is None:
        return partial(init_to_median_with_alpha0, alpha_0_init=alpha_0_init)

    # Override alpha_0 related parameters
    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and site["name"] in ["alpha_0", "mean_alpha_0"]
    ):
        # Get the shape from the site
        sample_shape = site["kwargs"].get("sample_shape", ())
        param_shape = site["fn"].shape()
        full_shape = sample_shape + param_shape

        # Return constant value with the correct shape
        return jnp.full(full_shape, alpha_0_init)

    # For all other parameters, use median initialization
    return init_to_median(site)
