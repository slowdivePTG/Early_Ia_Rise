import shutil
import numpy as np
import jax.numpy as jnp
import xarray as xr

from ..model.fit_rise import f_t, SNLightCurveLib
from .._utils import plt


class RedbackLightCurveLib(SNLightCurveLib):
    """
    A mock light curve library using Redback to simulate ZTF light curves.
    """

    import pandas as pd

    def __init__(
        self,
        params_true: dict,
        lc_early_lib: list[pd.DataFrame],
        lc_peak_lib: list[pd.DataFrame],
        post_sample: xr.Dataset = None,
        sampling_model: str = "hierarchical",
    ) -> None:
        super().__init__(
            lc_early_lib=lc_early_lib,
            lc_peak_lib=lc_peak_lib,
            post_sample=post_sample,
            sampling_model=sampling_model,
        )

        self.params_true = params_true
        self.params_names = dict(
            t_fl=r"$t_\mathrm{fl}$",
            base=r"$C$",
            amp=r"$A$",
            alpha=r"$\alpha$",
            mean_alpha=r"$\mu_\alpha$",
            std_alpha=r"$\sigma_\alpha$",
            mean_t_fl=r"$\mu_{t_\mathrm{fl}}$",
            std_t_fl=r"$\sigma_{t_\mathrm{fl}}$",
        )

    @classmethod
    def simulate_mock_light_curve(
        cls,
        n_lc: int = 10,
        params_mean: dict = None,
        params_std: dict = None,
        early_threshold: float = 0.4,
        model: str = "curved_power_law",
    ) -> list[pd.DataFrame]:
        """
        Simulate light curves using Redback.
        """
        import os
        import pandas as pd
        from pathlib import Path
        from redback.simulate_transients import SimulateOpticalTransient
        from .._utils._plt import set_plot_style

        from .sed import power_law_rise_flat_sed

        T0_MJD_TRANSIENT = 59050.0
        PEAK_LUMINOSITY = 2e28  # erg/s/Hz

        # Sample the population parameters using numpy.random
        num_tot = n_lc * 10  # oversample to account for non-detections

        np.random.seed(n_lc * 114514)

        if params_mean is None:
            params_mean = {}
        if params_std is None:
            params_std = {}

        # True hyper-parameters for the power-law rise model
        params_true = dict(
            mean_alpha=params_mean.get("alpha", 2.0),
            std_alpha=params_std.get("alpha", 0.3),
            mean_t_fl=params_mean.get("t_fl", -18.0),
            std_t_fl=params_std.get("t_fl", 1.5),
        )

        # Parameters for simulating the population
        params_sim = dict(
            base=np.random.normal(
                params_mean.get("base", 0.0), params_std.get("base", 0.1), num_tot
            ),
            t_peak=np.random.normal(
                -params_true["mean_t_fl"], params_true["std_t_fl"], num_tot
            ),
            alpha_0=np.random.normal(
                params_true["mean_alpha"], params_true["std_alpha"], num_tot
            ),
        )

        # Compute alpha_1 based on other parameters
        params_sim["alpha_1"] = -1 / (
            params_sim["t_peak"] * (1 + np.log(params_sim["t_peak"]))
        )

        # Prior on intrinsic peak luminosity (erg/s/Hz) - fixed value
        params_sim["peak_luminosity"] = np.full(num_tot, PEAK_LUMINOSITY)

        # dist_lum ~ PowerLaw(alpha=2, min=10, max=250)
        # For power law: f(x) ∝ x^(alpha_lum), we use inverse transform sampling
        # CDF^(-1)(u) = (min^(1+alpha_lum) + u*(max^(1+alpha_lum) - min^(1+alpha_lum)))^(1/(1+alpha_lum))
        alpha_lum = 2
        min_dist, max_dist = 10, 250
        u = np.random.uniform(0, 1, num_tot)
        params_sim["dist_lum"] = (
            min_dist ** (1 + alpha_lum)
            + u * (max_dist ** (1 + alpha_lum) - min_dist ** (1 + alpha_lum))
        ) ** (1 / (1 + alpha_lum))

        # Add the required t0_mjd_transient parameter
        # This sets when each transient begins (in MJD)
        params_sim["t0_mjd_transient"] = np.full(num_tot, T0_MJD_TRANSIENT)
        params_sim["ra"] = np.random.uniform(240, 360, num_tot)
        params_sim["dec"] = np.random.uniform(-30, 90, num_tot)

        # Simulate each transient individually
        lc_early_lib = np.empty(shape=n_lc, dtype=object)
        lc_peak_lib = np.empty(shape=n_lc, dtype=object)
        params_valid_det = []

        data_dir = Path(f"./data/mock/{model}_frac{int(early_threshold * 100)}")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        for i in range(num_tot):
            # Extract parameters for this single transient
            single_params = {key: val[i] for key, val in params_sim.items()}

            single_params["force_power_law"] = model == "power_law"

            if len(params_valid_det) >= n_lc:
                break

            # Simulate single transient
            sim = SimulateOpticalTransient.simulate_transient_in_ztf(
                model=power_law_rise_flat_sed,
                survey="ztf",
                parameters=single_params,
                end_transient_time=100,
                snr_threshold=3.0,
                add_source_noise=True,
                source_noise=0.02,
                redback_compatible_model=True,
                model_kwargs={},
                obs_buffer=100,
                seed=42,
            )

            obs = sim.observations
            # only need g and r bands
            obs = obs[(obs["band"] == "ztfg") | (obs["band"] == "ztfr")].reset_index(
                drop=True
            )
            idx_snr = (obs["flux(erg/cm2/s)"] / obs["flux_error"]) > 10
            obs["phase"] = (
                obs["time"]
                - single_params["t0_mjd_transient"]
                - single_params["t_peak"]
            )
            idx_rise = (obs["phase"] < -10) & (obs["phase"] > -20)
            idx_peak = (obs["phase"] >= -10) & (obs["phase"] < 10)
            idx_baseline = obs["phase"] < -20

            if (
                np.sum(idx_snr & idx_rise) < 2
                or np.sum(idx_snr & idx_peak) < 2
                or np.sum(idx_baseline) < 10
            ):
                continue

            # Normalize fluxes
            obs["flux_norm"] = np.full_like(obs["flux(erg/cm2/s)"], np.nan)
            obs["flux_norm_error"] = np.full_like(obs["flux_error"], np.nan)

            for band in ["ztfg", "ztfr"]:
                idx_band = obs["band"] == band
                flux_norm_factor = (
                    obs.loc[idx_snr & idx_band, "flux(erg/cm2/s)"].max() / 100
                )
                obs["flux_norm"][idx_band] = (
                    obs["flux(erg/cm2/s)"][idx_band] / flux_norm_factor
                )
                obs["flux_norm_error"][idx_band] = (
                    obs["flux_error"][idx_band] / flux_norm_factor
                )

            # Remove the filter with no detections
            obs = obs[np.isfinite(obs["flux_norm"])].reset_index(drop=True)

            idx_obs = len(params_valid_det)
            print(
                f"Simulating transient {idx_obs + 1}/{n_lc} ({i + 1}/{num_tot} attempts)..."
            )
            print(f"  → {len(sim.inference_observations)} detections")

            # Generate early light curve up to early_threshold of peak flux
            phase = obs["phase"].values
            flux_mock = obs["flux_norm"].values + single_params["base"]
            flux_err_mock = obs["flux_norm_error"].values

            filt = np.where(obs["band"] == "ztfg", 1, 2).astype(np.int32)
            fcqfid = filt.astype(np.int32)

            idx_early = (phase < 0) & (obs["flux_norm"] <= early_threshold * 100)
            idx_peak = phase < 0

            lc_early = pd.DataFrame(
                dict(
                    phase=phase[idx_early],
                    flux=flux_mock[idx_early],
                    flux_err=flux_err_mock[idx_early],
                    fcqfid=fcqfid[idx_early],
                    filt=filt[idx_early],
                )
            )
            lc_peak = pd.DataFrame(
                dict(
                    phase=phase[idx_peak],
                    flux=flux_mock[idx_peak],
                    flux_err=flux_err_mock[idx_peak],
                    fcqfid=fcqfid[idx_peak],
                    filt=filt[idx_peak],
                )
            )

            lc_early.reset_index(drop=True, inplace=True)
            lc_peak.reset_index(drop=True, inplace=True)

            lc_early.to_csv(
                data_dir / f"lc_early_{str(idx_obs).zfill(4)}.csv",
                index=False,
            )
            lc_peak.to_csv(
                data_dir / f"lc_peak_{str(idx_obs).zfill(4)}.csv",
                index=False,
            )

            lc_early_lib[idx_obs] = lc_early
            lc_peak_lib[idx_obs] = lc_peak

            params_valid_det.append(single_params)

        params_valid_det = pd.DataFrame(params_valid_det).reset_index(drop=True)

        params_true["alpha_0"] = params_valid_det["alpha_0"].values
        params_true["alpha_1"] = params_valid_det["alpha_1"].values
        params_true["t_fl"] = -params_valid_det["t_peak"].values

        params_valid_det.to_csv(
            data_dir / f"simulated_lc_params.csv",
            index=False,
        )

        # Reset the plot style after Redback's modification
        set_plot_style()

        return cls(params_true, lc_early_lib=lc_early_lib, lc_peak_lib=lc_peak_lib)


class MockLightCurveLib(SNLightCurveLib):
    """
    A mock light curve library for testing the inference pipeline.
    The light curves are generated using the curved power-law rise model with added noise.
    """

    def __init__(
        self,
        cadence: float = 1,
        n_lc: int = 10,
        params_true: dict = dict(t_fl=-20.0, base=0.0, amp_prime=50, alpha=2.0),
        params_mean: dict = dict(t_fl=-20.0, base=0.0, amp_prime=50, alpha=2.0),
        params_std: dict = dict(t_fl=1.0, base=0.1, amp_prime=5, alpha=0.1),
        fix_values: bool = True,
        mag_peak: float = 18,
        realistic_mag: bool = False,
    ) -> None:
        import warnings

        warnings.warn(
            "MockLightCurveLib is deprecated and may be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
        t_sample = jnp.arange(-100, 0, step=cadence)
        n_sample = len(t_sample)

        lc_early_lib = np.empty(shape=n_lc, dtype=object)

        np.random.seed(n_lc * int(cadence) + 114514)

        if fix_values:
            t_fl = params_true.get("t_fl", -20.0) * jnp.ones(n_lc)
            base = params_true.get("C", 0.0) * jnp.ones(n_lc)
            amp_prime = params_true.get("Aprime", 50) * jnp.ones(n_lc)
            alpha = params_true.get("alpha", 2.0) * jnp.ones(n_lc)
            self.params_true = params_true
        else:
            t_fl = np.random.randn(n_lc) * (
                params_std.get("t_fl", 1.0)
            ) + params_mean.get("t_fl", -20.0)
            base = np.random.randn(n_lc) * (params_std.get("C", 0.1)) + params_mean.get(
                "C", 0.0
            )
            amp_prime = np.random.randn(n_lc) * (
                params_std.get("Aprime", 0.1)
            ) + params_mean.get("Aprime", 50)
            alpha = np.random.randn(n_lc) * (
                params_std.get("alpha", 0.1)
            ) + params_mean.get("alpha", 2.0)
            self.params_true = dict(
                t_fl=t_fl, base=base, amp_prime=amp_prime, alpha=alpha
            )
            self.params_true["mean_alpha"] = params_mean.get("alpha", 2.0)
            self.params_true["std_alpha"] = params_std.get("alpha", 0.1)
            self.params_true["mean_t_fl"] = params_mean.get("t_fl", -20.0)
            self.params_true["std_t_fl"] = params_std.get("t_fl", 1.0)

        amp = amp_prime / jnp.power(10, alpha)

        if realistic_mag:
            # n(mag) ~ 10^(0.6*mag)
            # dmag = mag_peak - mag_peak_0 ~ Exponential(0.6)
            dmag = np.random.exponential(0.6, n_lc)
        else:
            # mag_peak = constant
            dmag = np.zeros(n_lc)

        self.mag_peak = mag_peak - dmag

        for k in range(n_lc):
            t_jitter = np.random.randn(n_sample) * 0.05  # 0.05 days = 1.2 hours
            t_jitter -= t_jitter[
                np.argmin(np.abs(t_sample - t_fl[k]))
            ]  # ensure jitter=0 when t=t_fl -> SN observed right after the exposion
            t_shift = cadence / n_lc * k  # delay the first detection time
            self.t_first_det = t_shift
            t_mock = t_sample + t_jitter + t_shift
            flux_true = f_t(
                t=t_mock, t_fl=t_fl[k], base=base[k], amp=amp[k], alpha=alpha[k]
            )

            mag_40 = self.mag_peak[k] - 2.5 * np.log10(0.4)  # 40% of peak
            t_40 = (40 / amp[k]) ** (1 / alpha[k]) + t_fl[
                k
            ]  # 40% of maximum flux is achieved at t_40
            idx_early = t_mock <= t_40
            zp_mock = 2.5 * np.log10(40) + mag_40

            t_mock_early = t_mock[idx_early]
            flux_true_early = flux_true[idx_early]
            flux_err_mock_early = self._generate_flux_err(
                flux_true_early, zp=zp_mock, method="broken_power_law"
            )
            flux_mock_early = (
                flux_true_early + np.random.randn(idx_early.sum()) * flux_err_mock_early
            )

            lc_early_lib[k] = dict(
                phase=t_mock_early, flux=flux_mock_early, flux_err=flux_err_mock_early
            )

        super().__init__(lc_early_lib=lc_early_lib)
        self.n_lc = n_lc
        self.params_names = dict(
            t_fl=r"$t_\mathrm{fl}$",
            base=r"$C$",
            amp=r"$A$",
            alpha=r"$\alpha$",
            mean_alpha=r"$\mu_\alpha$",
            std_alpha=r"$\sigma_\alpha$",
            mean_t_fl=r"$\mu_{t_\mathrm{fl}}$",
            std_t_fl=r"$\sigma_{t_\mathrm{fl}}$",
        )

        self.inf_data = None

    @staticmethod
    def _generate_flux_err(flux, zp=0, method="broken_power_law"):
        """
        Estimate the flux error based on the broken power law relation between magnitude and S/N
        """
        mag = -2.5 * jnp.log10(flux) + zp
        if method == "broken_power_law":

            def log_broken_powerlaw(x, x0=18.2123, y0=-8.9946, k1=-0.3044, k2=0):
                """
                Broken power law function - mag v.s. S/N (default values adopted from the fit to the r-band data in Yao et al. 2019)
                """
                log_err = 0.5 * jnp.log10(
                    (
                        jnp.power(10, (y0 + k1 * (x - x0)) * 2)
                        + jnp.power(10, (y0 + k2 * (x - x0)) * 2)
                    )
                )
                log_sky_noise = y0
                return np.where(np.isfinite(x) & ~np.isnan(x), log_err, log_sky_noise)

            return 10 ** (log_broken_powerlaw(mag) + 0.4 * zp)
        else:
            raise ValueError("Method not recognized")

    def plot_prior_posterior(
        self, params_names: list = ["alpha", "t_fl"], params_range: dict = None
    ):
        if params_range is None:
            params_range = dict(
                alpha=(0, 4), t_fl=(-30, -10), base=(-1, 1), amp=(0, 10)
            )
        _, ax = plt.subplots(
            2,
            len(params_names),
            sharex="col",
            sharey="row",
            constrained_layout=True,
            figsize=(3 * len(params_names), 6),
        )
        cmap = plt.get_cmap("coolwarm")  # Choose a colormap

        for i, var in enumerate(params_names):
            # overall
            if self.inf_data is None:
                prior, posterior = [], []
                for lc in self.lc_library:
                    prior = np.append(prior, lc.inf_data.prior[var].data)
                    posterior = np.append(posterior, lc.inf_data.posterior[var].data)
                prior = np.array(prior).ravel()
                posterior = np.array(posterior).ravel()
            else:
                prior = self.inf_data.prior[var].data.ravel()
                posterior = self.inf_data.posterior[var].data.ravel()
            ax[0, i].hist(
                prior,
                histtype="step",
                bins=50,
                color="k",
                lw=3,
                weights=np.ones_like(np.array(prior).ravel()) / len(prior),
                range=params_range[var],
            )
            ax[1, i].hist(
                posterior,
                histtype="step",
                bins=25 * self.n_lc,
                color="k",
                lw=2,
                range=params_range[var],
            )

            # posterior for each light curve
            for k in range(self.n_lc):
                if self.inf_data is None:
                    posterior_k = (
                        self.lc_library[k].inf_data.posterior[var].data.ravel()
                    )
                else:
                    posterior_k = self.post_sample[var].data[:, :, k].ravel()
                ax[1, i].hist(
                    posterior_k,  # only valid for single filter (as assumed in the mock data)
                    histtype="step",
                    bins=50,
                    color=cmap(k / self.n_lc),
                    range=params_range[var],
                    zorder=-1,
                )
            try:
                ax[1, i].axvline(self.params_true[var], color="crimson", ls="--")
            except ValueError:
                ax[1, i].axvline(
                    self.params_true["mean_" + var], color="crimson", ls="--"
                )
            ax[1, i].axvline(np.median(np.array(posterior).ravel()), color="k", ls="--")
            ax[0, i].set_title(f"Prior: {self.params_names[var]}")
            ax[1, i].set_title(f"Posterior: {self.params_names[var]}")
            ax[1, i].set_xlabel(self.params_names[var])
        ax[0, 0].set_yticks([])
        ax[1, 0].set_yticks([])
        ax[0, 0].set_ylabel("Normalized count")
        ax[1, 0].set_ylabel("Normalized count")

        return ax

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
        import corner

        params_names = kwargs.pop("params_names", ["alpha", "t_fl"])

        corner.corner(
            self.post_sample,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.5, 0.95],
            title_quantiles=[0.05, 0.5, 0.95],
            var_names=params_names,
            # truths=[self.params_true[var] for var in params_names],
            **kwargs,
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf", bbox_inches="tight")
