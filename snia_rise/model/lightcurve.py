import numpy as np
import warnings

import jax
import jax.numpy as jnp
import arviz as az
import corner
import xarray as xr
from numpyro import infer
from sklearn.preprocessing import LabelEncoder

from .._utils import plt

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
        ztfid: str = None,
    ) -> None:
        try:
            self.ID = ztfid

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
                f"Error initializing SNLightCurve for {ztfid}.Please check the input data."
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

        return dict(phase=phase, flux=flux, flux_err=flux_err, fcqfid=fcqfid, filt=filt)

    def sampling(
        self,
        num_samples: int = 1000,
        num_warmup: int = 5000,
        num_chains: int = 2,
        random_seed: int = 11,
        prior_pred_samples: int = 500,
        prior_config: dict = {},
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
        prior_config : dict, optional
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
            infer.NUTS(
                kernel,
                init_strategy=init_strategy,
                target_accept_prob=0.95,
                **nuts_params,
            ),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
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
            prior_config=prior_config,
        )

        # prior and posterior predictive checks
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
        # convert to arviz InferenceData
        self.inf_data = az.from_numpyro(
            self.sampler, prior=prior_pred, posterior_predictive=post_pred
        )

        # store the posterior samples
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
            beta_ = np.median(post_sample["beta"][:, :, self.idx_fcqfid], axis=(0, 1))
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
                amp_ = np.ravel(post_sample["A"][:, :, flt])
                alpha_0_ = np.ravel(post_sample["alpha_0"][:, :, flt])
                if "alpha_1" in post_sample.keys():
                    alpha_1_ = np.ravel(post_sample["alpha_1"][:, :, flt])
                else:
                    alpha_1_ = np.zeros_like(alpha_0_)
                t_pred = jnp.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
                for i in idx_post_check:
                    ax.plot(
                        t_pred,
                        f_t(t_pred, t_fl[i], 0, amp_[i], alpha_0_[i], alpha_1_[i])
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

        params_names = kwargs.pop("params_names", ["t_fl", "Aprime", "alpha_0"])
        if "alpha_1" in self.post_sample.keys() and "alpha_1" not in params_names:
            params_names.append("alpha_1")

        corner.corner(
            self.post_sample,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.16, 0.5, 0.84],
            title_quantiles=[0.16, 0.5, 0.84],
            **kwargs,
            var_names=params_names,
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
        post_sample: xr.Dataset = None,
        sampling_model: str = "hierarchical",
    ) -> None:
        self.lc_library: list[SNLightCurve] = []
        self.ztfid_lib: list = ztfid_lib if ztfid_lib is not None else []

        self.phase, self.flux, self.flux_err = [], [], []
        self.idx_filt = np.array([], dtype=int)
        self.idx_fcqfid = np.array([], dtype=int)
        self.idx_obj = np.array([], dtype=int)

        if lc_early_lib is None:
            return

        if (lc_peak_lib is not None) and (ztfid_lib is not None):
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(
                        lc_early=lc_early, lc_peak=lc_peak_lib[k], ztfid=ztfid_lib[k]
                    )
                )
        elif lc_peak_lib is not None:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(lc_early=lc_early, lc_peak=lc_peak_lib[k])
                )
        elif ztfid_lib is not None:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(
                    SNLightCurve(lc_early=lc_early, ztfid=ztfid_lib[k])
                )
        else:
            for k, lc_early in enumerate(lc_early_lib):
                self.lc_library.append(SNLightCurve(lc_early=lc_early))

        # Identify n_filt: number of unique filters across all objects
        n_filt = max([len(np.unique(lc.idx_filt)) for lc in self.lc_library])

        for k, lc in enumerate(self.lc_library):
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

        n_obj = len(np.unique(self.idx_obj))
        n_fcqfid = len(np.unique(self.idx_fcqfid))
        assert n_obj == self.idx_obj.max() + 1, "Indexing error: idx_obj"
        assert n_fcqfid == self.idx_fcqfid.max() + 1, "Indexing error: idx_fcqfid"
        assert n_filt == self.idx_filt.max() + 1, "Indexing error: idx_filt_gr"
        print("Number of objects:", n_obj)
        print("Number of unique fcqfid:", len(np.unique(self.idx_fcqfid)))
        print("Number of filters:", n_filt)
        print("Light curves compiled...")

        self.inf_data = None
        self.post_sample = post_sample
        self.decode_post_sample(model_structure=sampling_model)

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

        # Decode the posterior samples for each light curve
        for k, lc in enumerate(self.lc_library):
            lc.post_sample = {}

            # Parameters common to all unique fcqfid in this object
            fcqfid_in_obj = np.unique(self.idx_fcqfid[self.idx_obj == k])
            lc.post_sample["C"] = self.post_sample["C"][..., fcqfid_in_obj]
            lc.post_sample["beta"] = self.post_sample["beta"][..., fcqfid_in_obj]

            # A: (n_chains, n_samples, n_obj, n_filt)
            lc.post_sample["A"] = self.post_sample["A"][..., k, :]
            lc.post_sample["Aprime"] = self.post_sample["A"][..., k, :]

            # t_fl: (n_chains, n_samples, n_obj)
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
        raise NotImplementedError(
            "Appending light curve libraries is not supported yet."
        )
        # if not isinstance(lc_lib, SNLightCurveLib):
        #     raise TypeError("lc_lib must be an instance of SNLightCurveLib")

        # # concatenate the indices and light curve data
        # n_obj_current = len(np.unique(self.idx_obj))
        # n_filt_gr_current = (
        #     len(np.unique(self.idx_filt_gr)) if len(self.idx_filt_gr) > 0 else 0
        # )

        # for k, lc in enumerate(lc_lib.lc_library):
        #     # NEW: idx_filt_gr stays as the filter group (0 for g, 1 for r, etc.)
        #     self.idx_filt_gr = np.append(self.idx_filt_gr, lc.idx_filt)

        #     # If this is the first object being appended, determine n_filt_gr from it
        #     if n_filt_gr_current == 0:
        #         n_filt_gr_current = len(np.unique(lc.idx_filt))

        #     # NEW: idx_filt now maps to fixed structure: obj_idx * n_filt_gr + filt_gr_idx
        #     obj_idx = n_obj_current + k
        #     idx_filt_for_obj = obj_idx * n_filt_gr_current + lc.idx_filt
        #     self.idx_filt = np.append(self.idx_filt, idx_filt_for_obj)

        #     self.idx_fcqfid = np.append(
        #         self.idx_fcqfid,
        #         lc.idx_fcqfid + len(np.unique(self.idx_fcqfid)),
        #     )
        #     self.idx_obj = np.append(
        #         self.idx_obj, np.ones_like(lc.idx_filt) * (len(self.lc_library) + k)
        #     )
        #     self.phase = np.append(self.phase, lc.lc_early["phase"])
        #     self.flux = np.append(self.flux, lc.lc_early["flux"])
        #     self.flux_err = np.append(self.flux_err, lc.lc_early["flux_err"])

        # self.lc_library.extend(lc_lib.lc_library)
        # self.ztfid_lib.extend(lc_lib.ztfid_lib)

        # n_obj = len(np.unique(self.idx_obj))
        # n_fcqfid = len(np.unique(self.idx_fcqfid))
        # n_filt_gr = len(np.unique(self.idx_filt_gr))
        # # NEW: n_filt is now fixed based on structure, not on observed data
        # n_filt = n_obj * n_filt_gr

        # assert n_obj == self.idx_obj.max() + 1, "Indexing error: idx_obj"
        # assert n_fcqfid == self.idx_fcqfid.max() + 1, "Indexing error: idx_fcqfid"
        # # NEW: idx_filt may have gaps (missing filters) so max+1 check removed
        # # assert n_filt == self.idx_filt.max() + 1, "Indexing error: idx_filt"
        # assert n_filt_gr == self.idx_filt_gr.max() + 1, "Indexing error: idx_filt_gr"
        # print("Number of objects:", n_obj)
        # print("Number of unique fcqfid:", len(np.unique(self.idx_fcqfid)))
        # print("Number of filters (n_obj * n_filt_gr):", n_filt)
        # print("Number of gr filters:", n_filt_gr)
        # print("Light curves appended...")

    def sampling(
        self,
        num_samples: int = 1000,
        num_warmup: int = 3000,
        num_chains: int = 2,
        random_seed: int = 11,
        prior_pred_samples: int = 500,
        prior_config: dict = {},
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
        prior_config : dict, optional
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
            print("Using pooled model for sampling...")
            kernel = pooled_model
        elif model_structure == "unpooled":
            print("Using unpooled model for sampling...")
            kernel = unpooled_model
        elif model_structure == "hierarchical":
            print("Using hierarchical model for sampling...")
            prior_config["correlation_structure"] = "independent"
            kernel = hierarchical_model
        elif model_structure == "hierarchical_tfl":
            print("Using hierarchical t_fl model for sampling...")
            prior_config["correlation_structure"] = "tfl_only"
            kernel = hierarchical_model
        elif model_structure == "hierarchical_mvn":
            print("Using hierarchical mvn model for sampling...")
            prior_config["correlation_structure"] = "mvn"
            kernel = hierarchical_model
        else:
            raise ValueError(
                "Invalid model structure.Options: 'pooled', 'unpooled', 'hierarchical' (as well as '_mvn' and '_tfl')"
            )
        init_strategy = nuts_params.pop(
            "init_strategy",
            infer.init_to_median(),
        )
        self.sampler = infer.MCMC(
            infer.NUTS(
                kernel,
                init_strategy=init_strategy,
                **nuts_params,
            ),
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
            prior_config=prior_config,
        )

        # prior and posterior predictive checks
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
        # convert to arviz InferenceData
        self.inf_data = az.from_numpyro(
            self.sampler, prior=prior_pred, posterior_predictive=post_pred
        )

        # store the posterior samples
        self.post_sample = self.inf_data.posterior
        self.decode_post_sample(model_structure=model_structure)

    def plot_corner(
        self,
        save: bool = False,
        filename: str = None,
        var_name: list = ["mean_alpha_0", "sigma_alpha_0"],
        **kwargs,
    ):
        corner.corner(
            self.post_sample[var_name],
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
