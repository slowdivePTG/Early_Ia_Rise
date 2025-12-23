import shutil
import numpy as np
import jax.numpy as jnp
import xarray as xr

from ..model.lightcurve import f_t, SNLightCurveLib
from .._utils import plt


class RedbackLightCurveLib(SNLightCurveLib):
    """
    A mock light curve library using Redback to simulate ZTF light curves.
    """

    import pandas as pd

    def __init__(
        self,
        n_lc: int = None,
        early_threshold: float = 0.4,
        model: str = None,
        true_model: str = "power_law",
        sampling_model: str = "hierarchical",
        prior_type: str = "uniform",
    ) -> None:
        import os
        import glob
        import pandas as pd

        from pathlib import Path

        file_dir = Path(f"./data/mock/{true_model}")

        peak_files = sorted(glob.glob(str(Path(file_dir) / "lc_peak*.csv")))
        params_file = file_dir / f"simulated_lc_params.csv"

        post_sample_dir = Path(file_dir) / f"{model}_frac{int(early_threshold * 100)}"
        if sampling_model in ["hierarchical_tfl", "unpooled", "pooled"]:
            sampling_model_str = f"{sampling_model}_{prior_type.lower()}"
        else:
            sampling_model_str = sampling_model
        post_sample_full_file = (
            post_sample_dir / f"post_sample_{sampling_model_str}_{n_lc}.nc"
        )

        if n_lc is None:
            n_lc = len(peak_files)
        else:
            if len(peak_files) < n_lc:
                raise ValueError(
                    f"Insufficient light curve files in {file_dir}: found {len(peak_files)} simulated light curves, but {n_lc} are required."
                )
            peak_files = peak_files[:n_lc]

        lc_early_lib = []
        lc_peak_lib = []

        for pf in peak_files:
            lc_peak = pd.read_csv(pf)
            lc_peak_lib.append(
                dict(
                    phase=lc_peak["phase"].values,
                    flux=lc_peak["flux"].values,
                    flux_err=lc_peak["flux_err"].values,
                    fcqfid=lc_peak["fcqfid"].values.astype(np.int32),
                    filt=lc_peak["filt"].values.astype(np.int32),
                )
            )
            idx_early = (
                lc_peak["phase"]
                <= lc_peak["phase"].values[
                    lc_peak["flux"].values < early_threshold * 100
                ][-1]
            )
            lc_early_lib.append({key: item[idx_early] for key, item in lc_peak.items()})

        if not os.path.exists(post_sample_full_file):
            post_sample = None
        else:
            print("Loading existing .nc file...")
            post_sample = xr.load_dataset(post_sample_full_file)

        if os.path.exists(params_file):
            params_true = pd.read_csv(params_file)[:n_lc].to_dict(orient="list")
        else:
            print("No true parameters file found.")
            params_true = None

        super().__init__(
            lc_early_lib=lc_early_lib,
            lc_peak_lib=lc_peak_lib,
            post_sample=post_sample,
            sampling_model=sampling_model,
        )

        self.params_true = params_true
        self.params_names = dict(
            t_rise=r"$t_\mathrm{rise}$",
            base=r"$C$",
            amp=r"$A$",
            alpha_0=r"$\alpha$",
            mean_alpha_0=r"$\mu_\alpha$",
            sigma_alpha_0=r"$\sigma_\alpha$",
            mean_t_rise=r"$\mu_{t_\mathrm{rise}}$",
            sigma_t_rise=r"$\sigma_{t_\mathrm{rise}}$",
        )

    @classmethod
    def simulate_mock_light_curve(
        cls,
        n_lc: int = 10,
        params_mean: dict = None,
        params_sigma: dict = None,
        model: str = "curved_power_law",
        min_dist_lum: float = 10,
        max_dist_lum: float = 250,
    ) -> list[pd.DataFrame]:
        """
        Simulate light curves using Redback.
        """
        import os
        import pandas as pd

        from astropy.cosmology import FlatLambdaCDM
        from pathlib import Path
        from redback.simulate_transients import SimulateOpticalTransient
        from .._utils._plt import set_plot_style

        from .sed import power_law_rise_flat_sed, snf_2011fe_sed

        import logging

        logging.getLogger("redback").setLevel(logging.WARNING)

        T0_MJD_TRANSIENT = 59050.0
        PEAK_LUMINOSITY = 2e28  # intrinsic peak luminosity (erg/s/Hz)

        # Sample the population parameters using numpy.random
        num_tot = n_lc * 10  # oversample to account for non-detections

        np.random.seed(n_lc * 114514)

        if params_mean is None:
            params_mean = {}
        if params_sigma is None:
            params_sigma = {}

        params_sim = dict(
            dist_lum=np.random.uniform(10, 250, num_tot),
        )
        params_sim["redshift"] = (
            FlatLambdaCDM(H0=70, Om0=0.3).redshift(params_sim["dist_lum"] * u.Mpc).z
        )

        if model in ["power_law", "curved_power_law"]:
            # True hyper-parameters for the power-law rise model
            params_true = dict(
                mean_alpha=params_mean.get("alpha", 2.0),
                sigma_alpha=params_sigma.get("alpha", 0.3),
                mean_t_rise=params_mean.get("t_rise", 18.0),
                sigma_t_rise=params_sigma.get("t_rise", 1.5),
            )

            # Parameters for simulating the population
            params_sim = dict(
                base=np.random.normal(
                    params_mean.get("base", 0.0), params_sigma.get("base", 0.1), num_tot
                ),
                t_rise=np.random.normal(
                    params_true["mean_t_rise"], params_true["sigma_t_rise"], num_tot
                ),
                alpha_0=np.random.normal(
                    params_true["mean_alpha"], params_true["sigma_alpha"], num_tot
                ),
                peak_luminosity=np.full(num_tot, PEAK_LUMINOSITY),
            )

            # Compute alpha_1 based on other parameters
            params_sim["alpha_1"] = 1 / (
                params_sim["t_rise"] * (1 + np.log(params_sim["t_rise"]))
            )

        elif model == "snf_2011fe":
            params_true = None

        else:
            raise ValueError(f"Model {model} not recognized.")

        # dist_lum ~ PowerLaw(alpha=2)
        # For power law: f(x) ∝ x^(alpha_lum), we use inverse transform sampling
        # CDF^(-1)(u) = (min^(1+alpha_lum) + u*(max^(1+alpha_lum) - min^(1+alpha_lum)))^(1/(1+alpha_lum))
        if min_dist_lum < max_dist_lum:
            alpha_lum = 2
            u = np.random.uniform(0, 1, num_tot)
            params_sim["dist_lum"] = (
                min_dist_lum ** (1 + alpha_lum)
                + u
                * (max_dist_lum ** (1 + alpha_lum) - min_dist_lum ** (1 + alpha_lum))
            ) ** (1 / (1 + alpha_lum))
        elif min_dist_lum == max_dist_lum:
            print("Using fixed distance luminosity for all transients.")
            params_sim["dist_lum"] = np.full(num_tot, min_dist_lum)

        # Add the required t0_mjd_transient parameter
        # This sets when each transient begins (in MJD)
        params_sim["t0_mjd_transient"] = np.full(num_tot, T0_MJD_TRANSIENT)
        params_sim["ra"] = np.random.uniform(0, 360, num_tot)
        params_sim["dec"] = np.random.uniform(-30, 90, num_tot)

        # Simulate each transient individually
        lc_peak_lib = np.empty(shape=n_lc, dtype=object)
        params_valid_det = []

        data_dir = Path(f"./data/mock/{model}")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        for i in range(num_tot):
            # Extract parameters for this single transient
            single_params = {key: val[i] for key, val in params_sim.items()}

            single_params["force_power_law"] = model == "power_law"

            if len(params_valid_det) >= n_lc:
                break

            if model in ["power_law", "curved_power_law"]:
                sed_model = power_law_rise_flat_sed
            elif model == "snf_2011fe":
                sed_model = snf_2011fe_sed

            # Simulate single transient
            sim = SimulateOpticalTransient.simulate_transient_in_ztf(
                model=sed_model,
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
            idx_g = obs["band"] == "ztfg"
            idx_r = obs["band"] == "ztfr"
            obs = obs[idx_g | idx_r].reset_index(drop=True)
            idx_snr = obs["flux(erg/cm2/s)"] / obs["flux_error"] > 5
            obs["phase"] = (
                obs["time"]
                - single_params["t0_mjd_transient"]
                - single_params["t_rise"]
            ) / (1 + single_params["redshift"])
            idx_early = obs["phase"] < -10
            idx_rise = (obs["phase"] >= -10) & (obs["phase"] < 0)
            idx_fall = (obs["phase"] >= 0) & (obs["phase"] < 10)
            idx_baseline = obs["phase"] < -25

            if (
                # >= 2 high-SNR points in either g or r band during early phase
                (
                    (np.sum(idx_snr & idx_early & idx_g) < 2)
                    or (np.sum(idx_snr & idx_early & idx_r) < 2)
                )
                or
                # >= 2 high-SNR points during rise and fall in at least one band
                (
                    np.sum(idx_snr & idx_rise & idx_g) < 2
                    or np.sum(idx_snr & idx_fall & idx_g) < 2
                )
                and (
                    np.sum(idx_snr & idx_rise & idx_r) < 2
                    or np.sum(idx_snr & idx_fall & idx_r) < 2
                )
                # >= 5 baseline points in both bands
                or (np.sum(idx_baseline & idx_g) < 5)
                or (np.sum(idx_baseline & idx_r) < 5)
            ):
                continue

            idx_obs = len(params_valid_det)
            print(
                f"Simulating transient {idx_obs + 1}/{n_lc} ({i + 1}/{num_tot} attempts)..."
            )
            print(f"  → {len(sim.inference_observations)} detections")

            # Generate early light curve up to early_threshold of peak flux
            phase = obs["phase"].values
            flux_mock = obs["flux(erg/cm2/s)"].values
            flux_err_mock = obs["flux_error"].values

            # Normalize flux to 100 at peak for both g and r bands
            for band in ["ztfg", "ztfr"]:
                idx_band = obs["band"] == band
                peak_flux_band = cls.get_rise_flux(
                    flt=band, dist_lum=single_params["dist_lum"]
                )
                flux_mock[idx_band] = flux_mock[idx_band] / peak_flux_band * 100
                flux_err_mock[idx_band] = flux_err_mock[idx_band] / peak_flux_band * 100

            flux_mock += single_params["base"]

            filt = np.where(obs["band"] == "ztfg", 1, 2).astype(np.int32)
            fcqfid = (
                filt.astype(np.int32) + i * 10
            )  # unique fcqfid for each object and filter

            idx_peak = phase < 0

            lc_peak = pd.DataFrame(
                dict(
                    phase=phase[idx_peak],
                    flux=flux_mock[idx_peak],
                    flux_err=flux_err_mock[idx_peak],
                    fcqfid=fcqfid[idx_peak],
                    filt=filt[idx_peak],
                )
            )

            lc_peak.reset_index(drop=True, inplace=True)

            lc_peak.to_csv(
                data_dir / f"lc_peak_{str(idx_obs).zfill(4)}.csv",
                index=False,
            )

            lc_peak_lib[idx_obs] = lc_peak

            params_valid_det.append(single_params)

        if model in ["power_law", "curved_power_law"]:
            params_valid_det = pd.DataFrame(params_valid_det).reset_index(drop=True)

            params_true["alpha_0"] = params_valid_det["alpha_0"].values
            params_true["alpha_1"] = params_valid_det["alpha_1"].values
            params_true["t_rise"] = params_valid_det["t_rise"].values

            params_valid_det.to_csv(
                data_dir / f"simulated_lc_params.csv",
                index=False,
            )
        elif model == "snf_2011fe":
            params_valid_det = pd.DataFrame(params_valid_det).reset_index(drop=True)

            params_valid_det.to_csv(
                data_dir / f"simulated_lc_params.csv",
                index=False,
            )

        # Reset the plot style after Redback's modification
        set_plot_style()

    @staticmethod
    def get_rise_flux(flt: str, dist_lum: float, peak_luminosity: float = 2e28):
        """
        Calculate the peak flux (in erg/cm^2/s) given the distance luminosity (in Mpc) within ZTF filters.
        """
        # Definition in redback/tables/filters.csv
        # bands, wavelength [Hz], wavelength [Angstrom], color, reference_flux, sncosmo_name, label, effective_width [Hz]
        # ztfg,  6.27200e+14,     4783.50000,            black, 5.78500e-06,    ztfg,         ZTF/g, 1.65636e+14
        # ztfr,  4.67500e+14,     6417.10000,            black, 3.79600e-06,    ztfr,         ZTF/r, 1.08076e+14
        # ztfi,  3.81300e+14,     7867.41000,            black, 2.34000e-06,    ztfi,         ZTF/i, 5.86690e+13

        MPC_TO_CM = 3.086e24

        dist_lum_cm = dist_lum * MPC_TO_CM

        peak_flux_cgs = peak_luminosity / (4 * np.pi * dist_lum_cm**2)  # erg/cm^2/s/Hz

        if flt == "ztfg":
            eff_width = 1.65636e14  # Hz
        elif flt == "ztfr":
            eff_width = 1.08076e14  # Hz
        elif flt == "ztfi":
            eff_width = 5.86690e13  # Hz
        else:
            raise ValueError(f"Filter {flt} not recognized.")

        peak_flux = peak_flux_cgs * eff_width  # erg/cm^2/s
        return peak_flux


class MockLightCurveLib(SNLightCurveLib):
    """
    A mock light curve library for testing the inference pipeline.
    The light curves are generated using the curved power-law rise model with added noise.
    """

    raise DeprecationWarning(
        "MockLightCurveLib is deprecated and may be removed in future versions."
    )

    def __init__(
        self,
        cadence: float = 1,
        n_lc: int = 10,
        params_true: dict = dict(t_rise=-20.0, base=0.0, amp_prime=50, alpha=2.0),
        params_mean: dict = dict(t_fl=-20.0, base=0.0, amp_prime=50, alpha=2.0),
        params_sigma: dict = dict(t_fl=1.0, base=0.1, amp_prime=5, alpha=0.1),
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
                params_sigma.get("t_fl", 1.0)
            ) + params_mean.get("t_fl", -20.0)
            base = np.random.randn(n_lc) * (
                params_sigma.get("C", 0.1)
            ) + params_mean.get("C", 0.0)
            amp_prime = np.random.randn(n_lc) * (
                params_sigma.get("Aprime", 0.1)
            ) + params_mean.get("Aprime", 50)
            alpha = np.random.randn(n_lc) * (
                params_sigma.get("alpha", 0.1)
            ) + params_mean.get("alpha", 2.0)
            self.params_true = dict(
                t_fl=t_fl, base=base, amp_prime=amp_prime, alpha=alpha
            )
            self.params_true["mean_alpha"] = params_mean.get("alpha", 2.0)
            self.params_true["sigma_alpha"] = params_sigma.get("alpha", 0.1)
            self.params_true["mean_t_fl"] = params_mean.get("t_fl", -20.0)
            self.params_true["sigma_t_fl"] = params_sigma.get("t_fl", 1.0)

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
            sigma_alpha=r"$\sigma_\alpha$",
            mean_t_fl=r"$\mu_{t_\mathrm{fl}}$",
            sigma_t_fl=r"$\sigma_{t_\mathrm{fl}}$",
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
