import os
import glob
import shutil
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from ..model.lightcurve import SNLightCurveLib


class RedbackLightCurveLib(SNLightCurveLib):
    """
    A mock light curve library using Redback to simulate ZTF light curves.
    """

    def __init__(
        self,
        n_lc: int = None,
        early_threshold: float = 0.4,
        model: str = None,
        true_model: str = "power_law",
        sampling_model: str = "hierarchical",
        prior_type: str = "uniform",
    ) -> None:
        file_dir = Path(f"./data/mock/{true_model}")

        peak_files = sorted(glob.glob(str(Path(file_dir) / "lc_peak*.csv")))
        params_file = file_dir / f"simulated_lc_params.csv"

        post_sample_dir = Path(file_dir) / f"{model}_frac{int(early_threshold * 100)}"
        if sampling_model in ["hierarchical_trise", "unpooled", "pooled"]:
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
        import astropy.units as u
        from astropy.cosmology import FlatLambdaCDM, z_at_value
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
            base=np.random.normal(
                params_mean.get("base", 0.0), params_sigma.get("base", 0.1), num_tot
            )
        )

        if model in ["power_law", "curved_power_law"]:
            # True hyper-parameters for the power-law rise model
            params_true = dict(
                mean_alpha=params_mean.get("alpha", 2.0),
                sigma_alpha=params_sigma.get("alpha", 0.3),
                mean_t_rise=params_mean.get("t_rise", 18.0),
                sigma_t_rise=params_sigma.get("t_rise", 1.5),
            )

            params_sim["t_rise"] = np.random.normal(
                params_true["mean_t_rise"], params_true["sigma_t_rise"], num_tot
            )
            params_sim["alpha_0"] = np.random.normal(
                params_true["mean_alpha"], params_true["sigma_alpha"], num_tot
            )
            params_sim["peak_luminosity"] = np.full(num_tot, PEAK_LUMINOSITY)

            # Compute alpha_1 based on other parameters
            params_sim["alpha_1"] = -1 / (
                params_sim["t_rise"] * (1 + np.log(params_sim["t_rise"]))
            )

        elif model == "snf_2011fe":
            params_sim["t_rise"] = np.full(num_tot, 20.0)

        else:
            raise ValueError(f"Model {model} not recognized.")

        # dist_lum ~ PowerLaw(alpha=2)
        # For power law: f(x) ∝ x^(alpha_lum), we use inverse transform sampling
        # CDF^(-1)(u) = (min^(1+alpha_lum) + u*(max^(1+alpha_lum) - min^(1+alpha_lum)))^(1/(1+alpha_lum))
        if min_dist_lum < max_dist_lum:
            alpha_lum = 2
            mu = np.random.uniform(0, 1, num_tot)
            params_sim["dist_lum"] = (
                min_dist_lum ** (1 + alpha_lum)
                + mu
                * (max_dist_lum ** (1 + alpha_lum) - min_dist_lum ** (1 + alpha_lum))
            ) ** (1 / (1 + alpha_lum))
        elif min_dist_lum == max_dist_lum:
            print("Using fixed distance luminosity for all transients.")
            params_sim["dist_lum"] = np.full(num_tot, min_dist_lum)

        # Compute redshift from distance luminosity
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        params_sim["redshift"] = np.array(
            [
                z_at_value(cosmo.luminosity_distance, dl * u.Mpc)
                for dl in params_sim["dist_lum"]
            ]
        )

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
            obs["phase"] = (obs["time"] - single_params["t0_mjd_transient"]) / (
                1 + single_params["redshift"]
            ) - single_params["t_rise"]

            idx_early = obs["phase"] < -10
            idx_rise = (obs["phase"] >= -10) & (obs["phase"] < 0)
            idx_fall = (obs["phase"] >= 0) & (obs["phase"] < 10)
            idx_baseline = (obs["phase"] < -25) & (obs["phase"] > -100)

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
                # >= 3 baseline points in both bands
                or (np.sum(idx_baseline & idx_g) < 3)
                or (np.sum(idx_baseline & idx_r) < 3)
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
                    flt=band,
                    dist_lum=single_params["dist_lum"],
                    redshift=single_params["redshift"],
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

            # plt.figure(figsize=(8, 3))
            # for band, color in zip(["ztfg", "ztfr"], ["tab:green", "tab:red"]):
            #     idx_band = obs["band"] == band
            #     plt.errorbar(
            #         lc_peak["phase"][lc_peak["filt"] == (1 if band == "ztfg" else 2)],
            #         lc_peak["flux"][lc_peak["filt"] == (1 if band == "ztfg" else 2)],
            #         yerr=lc_peak["flux_err"][
            #             lc_peak["filt"] == (1 if band == "ztfg" else 2)
            #         ],
            #         fmt="o",
            #         color=color,
            #         label=band,
            #     )
            # plt.xlim(-30, 5)
            # plt.axvline(0, color="k", ls=":")
            # plt.axvline(-single_params["t_rise"], color="k", ls=":")
            # plt.xlabel("Phase (days)")
            # plt.show()

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
    def get_rise_flux(
        flt: str, dist_lum: float, redshift: float, peak_luminosity: float = 2e28
    ):
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

        peak_flux_cgs = (
            peak_luminosity / (4 * np.pi * dist_lum_cm**2) / (1 + redshift)
        )  # erg/cm^2/s/Hz

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
