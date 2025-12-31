import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .._utils._plt import plt
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
        params_file = file_dir / "simulated_lc_params.csv"

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
            sampling_model=sampling_model,
        )

        self.params_true = params_true

        self.post_sample = post_sample
        self.decode_post_sample()

        if self.post_sample is not None:
            self.sampling(
                sample_prior=True,
                prior_config=dict(rise_model=model, prior_type=prior_type),
            )
            self.decode_prior_sample()

    @classmethod
    def simulate_mock_light_curve(
        cls,
        n_lc: int = 10,
        params_mean: dict = None,
        params_sigma: dict = None,
        model: str = "curved_power_law",
        min_dist_lum: float = 10,
        max_dist_lum: float = 270,
        z_fixed: float = None,
    ) -> list[pd.DataFrame]:
        """
        Simulate light curves using Redback.
        """
        import logging
        import os
        from pathlib import Path

        import pandas as pd
        from scipy.optimize import brentq

        from .._utils._plt import set_plot_style
        from .sed import broken_power_law_rise_flat_sed, power_law_rise_flat_sed

        logging.getLogger("redback").setLevel(logging.WARNING)

        # For the power-law rise models
        T0_MJD_TRANSIENT = 59050.0
        PEAK_LUMINOSITY = 2e28  # intrinsic peak luminosity (erg/s/Hz)

        # Sample the population parameters using numpy.random
        num_tot = n_lc * 20  # oversample to account for non-detections

        np.random.seed(n_lc * 114514 + 1919810)

        if params_mean is None:
            params_mean = {}
        if params_sigma is None:
            params_sigma = {}

        params_sim = dict(
            base=np.random.normal(
                params_mean.get("base", 0.0), params_sigma.get("base", 0.1), num_tot
            )
        )

        if "power_law" in model:
            # Add the required t0_mjd_transient parameter
            # This sets when each transient begins (in MJD)
            params_sim["t0_mjd_transient"] = np.full(num_tot, T0_MJD_TRANSIENT)
            params_sim["ra"] = np.random.uniform(0, 360, num_tot)
            params_sim["dec"] = np.random.uniform(-20, 70, num_tot)

            # True hyper-parameters for the power-law rise model
            params_true = dict(
                mean_alpha=params_mean.get("alpha", 2.0),
                sigma_alpha=params_sigma.get("alpha", 0.3),
                mean_t_rise=params_mean.get("t_rise", 18.5),
                sigma_t_rise=params_sigma.get("t_rise", 1.5),
            )

            params_sim["t_rise"] = np.random.normal(
                params_true["mean_t_rise"], params_true["sigma_t_rise"], num_tot
            )
            params_sim["alpha_0"] = np.random.normal(
                params_true["mean_alpha"], params_true["sigma_alpha"], num_tot
            )
            params_sim["peak_luminosity"] = np.full(num_tot, PEAK_LUMINOSITY)

            if model in ["power_law", "curved_power_law"]:
                # Compute alpha_1 based on other parameters
                params_sim["alpha_1"] = -1 / (
                    params_sim["t_rise"] * (1 + np.log(params_sim["t_rise"]))
                )

            elif model == "broken_power_law":
                # Compuate alpha_1 based on other parameters
                params_sim["t_b"] = (
                    np.random.uniform(-1, 1, num_tot) + params_sim["t_rise"]
                )
                params_sim["s"] = np.random.uniform(0.5, 1.5, num_tot)

                alpha_v_1 = params_sim["alpha_0"] / 2 - 1

                # Define the function we want to solve: func(alpha_v_2) = 0
                def _peak_equation(av2, av1, s, target_ratio):
                    # Avoid division by zero or invalid powers if the solver wanders
                    if av2 == av1 or (1 + av2) == 0:
                        return 1e9

                    # Calculate the theoretical ratio from the slopes and smoothness
                    # Using the equation provided
                    base = -(1 + av1) / (1 + av2)

                    if base <= 0:
                        return 1e9  # Invalid mathematical domain for fractional power

                    exponent = 1 / (s * (av1 - av2))
                    val = np.power(base, exponent)

                    return val - target_ratio

                # Solve for alpha_v_2 for each simulation
                alpha_v_2 = np.zeros(num_tot)

                # Target ratio is t_rise / t_b
                target_ratios = params_sim["t_rise"] / params_sim["t_b"]

                for i in range(num_tot):
                    av1 = alpha_v_1[i]
                    s_val = params_sim["s"][i]
                    ratio = target_ratios[i]

                    # Constraints: (1+av1)/(1+av2) < 0
                    # alpha_v_1 ~ 0 (-0.5 to 0.5).
                    # => alpha_v_2 < -1
                    try:
                        # Looking for a solution somewhat far from alpha_v_1 to avoid singularity
                        sol = brentq(
                            _peak_equation,
                            av1 - 5,
                            -1.01,
                            args=(av1, s_val, ratio),
                        )
                    except ValueError:
                        # Fallback or wider search if root is not bracketed in standard decay range
                        av2s = np.linspace(av1 - 5, -1.01, 1000)
                        func_vals = [
                            _peak_equation(av2, av1, s_val, ratio) for av2 in av2s
                        ]

                        plt.plot(av2s, func_vals)
                        plt.axhline(0, color="k", ls=":")
                        plt.title(f"Failed to bracket root for index {i}")
                        plt.xlabel("alpha_v_2")
                        plt.ylabel("Function Value")
                        plt.show()
                        raise RuntimeError(
                            f"Failed to find root for alpha_v_2 at index {i}: av1={av1}, s={s_val}, ratio={ratio}"
                        )

                    alpha_v_2[i] = sol

                params_sim["alpha_1"] = alpha_v_1 - alpha_v_2

        elif model == "snf_2011fe":
            params_sim["t_rise"] = np.full(num_tot, 0.0)

        else:
            raise ValueError(f"Model {model} not recognized.")

        params_sim["dist_lum"], params_sim["redshift"] = cls.get_dist_lum_redshift(
            n=num_tot,
            min_dist_lum=min_dist_lum,
            max_dist_lum=max_dist_lum,
            z_fixed=z_fixed,
        )

        # Simulate each transient individually
        lc_peak_lib = np.empty(shape=n_lc, dtype=object)
        params_valid_det = []

        model_dir = (
            model if z_fixed is None else f"{model}_z_{z_fixed:.2f}".replace(".", "_")
        )

        data_dir = Path("./data/mock") / model_dir
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        for i in range(num_tot):
            # Extract parameters for this single transient
            single_params = {key: val[i] for key, val in params_sim.items()}

            idx_obs = len(params_valid_det)

            if len(params_valid_det) >= n_lc:
                break

            if "power_law" in model:
                single_params["force_power_law"] = model == "power_law"

                if model in ["power_law", "curved_power_law"]:
                    sed_model = power_law_rise_flat_sed
                elif model == "broken_power_law":
                    sed_model = broken_power_law_rise_flat_sed

                lc_peak = cls._simulate_single_light_curve_redback(
                    sed_model=sed_model, params=single_params
                )

                if lc_peak is None:
                    continue

                print(
                    f"Simulating transient {idx_obs + 1}/{n_lc} ({i + 1} attempts)..."
                )
                print(f"  → {len(lc_peak['phase'])} detections before peak")

            elif "2011fe" in model:
                sed_model = "snf-2011fe"

                lc_peak = cls._simulate_single_light_curve_2011fe(params=single_params)

            # plt.figure(figsize=(8, 3))
            # for band, color in zip([1, 2], ["tab:green", "tab:red"]):
            #     idx_band = lc_peak["filt"] == band
            #     plt.errorbar(
            #         lc_peak["phase"][idx_band],
            #         lc_peak["flux"][idx_band],
            #         yerr=lc_peak["flux_err"][idx_band],
            #         fmt="o",
            #         color=color,
            #         label=band,
            #     )
            # plt.xlim(-30, 5)
            # plt.axvline(0, color="k", ls=":")
            # plt.axvline(-single_params["t_rise"], color="k", ls=":")
            # plt.xlabel("Phase (days)")
            # plt.show()

            lc_peak.to_csv(
                data_dir / f"lc_peak_{str(idx_obs).zfill(4)}.csv",
                index=False,
            )

            lc_peak_lib[idx_obs] = lc_peak

            params_valid_det.append(single_params)

        params_valid_det = pd.DataFrame(params_valid_det).reset_index(drop=True)

        params_valid_det.to_csv(
            data_dir / "simulated_lc_params.csv",
            index=False,
        )

        # Reset the plot style after Redback's modification
        set_plot_style()

    @staticmethod
    def get_dist_lum_redshift(
        n: int,
        min_dist_lum: float = 10,
        max_dist_lum: float = 270,
        z_fixed: float = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get distance luminosity (in Mpc) and redshift arrays.
        """
        import astropy.units as u
        from astropy.cosmology import FlatLambdaCDM, z_at_value

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        # dist_lum ~ PowerLaw(alpha=2)
        # For power law: f(x) ∝ x^(alpha_lum), we use inverse transform sampling
        # CDF^(-1)(u) = (min^(1+alpha_lum) + u*(max^(1+alpha_lum) - min^(1+alpha_lum)))^(1/(1+alpha_lum))
        if z_fixed is not None:
            z_fixed = max(z_fixed, 1e-3)
            print(f"Using fixed redshift z={z_fixed:.3f} for all transients.")
            dl_fixed = cosmo.luminosity_distance(z_fixed).value  # in Mpc
            dist_lum = np.full(n, dl_fixed)
            z = np.full(n, z_fixed)
        else:
            if min_dist_lum < max_dist_lum:
                alpha_lum = 2
                mu = np.random.uniform(0, 1, n)
                dist_lum = (
                    min_dist_lum ** (1 + alpha_lum)
                    + mu
                    * (
                        max_dist_lum ** (1 + alpha_lum)
                        - min_dist_lum ** (1 + alpha_lum)
                    )
                ) ** (1 / (1 + alpha_lum))
            elif min_dist_lum == max_dist_lum:
                print("Using fixed distance luminosity for all transients.")
                dist_lum = np.full(n, min_dist_lum)

            # Compute redshift from distance luminosity
            z = np.array(
                [z_at_value(cosmo.luminosity_distance, dl * u.Mpc) for dl in dist_lum]
            )

        return dist_lum, z

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

    @classmethod
    def _simulate_single_light_curve_redback(
        cls, sed_model, params: dict
    ) -> pd.DataFrame | None:
        """
        Simulate a single light curve using Redback.
        """
        from redback.simulate_transients import SimulateOpticalTransient

        # Simulate single transient
        sim = SimulateOpticalTransient.simulate_transient_in_ztf(
            model=sed_model,
            survey="ztf",
            parameters=params,
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
        obs["phase"] = (obs["time"] - params["t0_mjd_transient"]) / (
            1 + params["redshift"]
        ) - params["t_rise"]

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
            return None

        # Generate early light curve up to early_threshold of peak flux
        phase = obs["phase"].values
        flux_mock = obs["flux(erg/cm2/s)"].values
        flux_err_mock = obs["flux_error"].values

        # Normalize flux to 100 at peak for both g and r bands
        for band in ["ztfg", "ztfr"]:
            peak_flux_band = cls.get_rise_flux(
                flt=band,
                dist_lum=params["dist_lum"],
                redshift=params["redshift"],
            )

            idx_band = obs["band"] == band
            flux_mock[idx_band] = flux_mock[idx_band] / peak_flux_band * 100
            flux_err_mock[idx_band] = flux_err_mock[idx_band] / peak_flux_band * 100

        flux_mock += params["base"]

        filt = np.where(obs["band"] == "ztfg", 1, 2).astype(np.int32)
        fcqfid = filt.astype(np.int32)

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

        return lc_peak

    @classmethod
    def _simulate_single_light_curve_2011fe(cls, params: dict) -> pd.DataFrame | None:
        """
        Simulate a single light curve using Pereira et al. (2013) SNIFS spectrophotometric time series.
        """

        import glob
        import os

        import redback
        import sncosmo
        from astropy.cosmology import FlatLambdaCDM

        Z_11FE = 0.000804
        DIST_LUM_11FE = 7  # Mpc

        ztf_path = os.path.join(
            os.path.dirname(redback.__file__), "tables", "ztf.tar.gz"
        )
        filter_path = os.path.join(
            os.path.dirname(redback.__file__), "tables", "filters.csv"
        )

        # Load ZTF pointings and filter settings
        ztf_pointings = pd.read_csv(ztf_path, compression="gzip")
        ztf_filters = pd.read_csv(filter_path)
        ztf_filters = ztf_filters[ztf_filters["sncosmo_name"].isin(["ztfg", "ztfr"])]

        def get_flux_err_sky(filt: str, num: int) -> np.ndarray:
            """
            Get sky flux error for ZTF filter.
            """
            depth = ztf_pointings.loc[
                ztf_pointings["filter"] == filt, "fiveSigmaDepth"
            ].values
            skymaglim = np.random.choice(depth, size=num)
            flux_ref = ztf_filters[ztf_filters["sncosmo_name"] == filt][
                "reference_flux"
            ].values[0]
            skyfluxlim = flux_ref * 10 ** (-0.4 * skymaglim)
            flux_err_sky = skyfluxlim / 5.0

            return flux_err_sky

        # Load SNIFS spectra from Pereira et al. (2013)
        spec_files = sorted(glob.glob("./data/Pereira_2013/*.dat"))

        source_lst = []
        phase_det = []

        for spec_file in spec_files:
            with open(spec_file, "r") as f:
                header = f.readlines()
            for line in header:
                if "TMAX" in line:
                    phase = float(line.split("=")[1].strip().split()[0])
                    break
            if phase > 1.0:
                break
            spec = np.loadtxt(spec_file)
            source = [
                spec[:, 0] / (1 + Z_11FE),
                spec[:, 1] * (1 + Z_11FE),
                spec[:, 2] ** 0.5,
            ]
            source_lst.append(source)
            phase_det.append(phase)

        source_lst = np.array(source_lst)
        phase_det = np.array(phase_det)

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        # Set up synthetic light curves before t_fl (non-detection phase)
        z = params["redshift"]
        dist_lum = cosmo.luminosity_distance(z).value  # in Mpc

        # Get fluxes and errors for each filter
        phase = []
        flux, flux_err = [], []
        for filt in ["ztfg", "ztfr"]:
            # Non-detections
            phase_non_det = np.arange(-50.5, -16.5, 2)
            flux_err_sky_non_det = get_flux_err_sky(filt, len(phase_non_det))

            flux_err_non_det = flux_err_sky_non_det
            flux_obs_non_det = np.random.randn(len(phase_non_det)) * flux_err_non_det

            # Detections
            flux_err_sky_det = get_flux_err_sky(filt, len(phase_det))
            flux_ref = ztf_filters[ztf_filters["sncosmo_name"] == filt][
                "reference_flux"
            ].values[0]
            mag_src = []
            for source in source_lst:
                src = sncosmo.Spectrum(
                    wave=source[0] * (1 + z),
                    flux=source[1] / (1 + z) * (DIST_LUM_11FE / dist_lum) ** 2,
                    fluxerr=source[2] / (1 + z) * (DIST_LUM_11FE / dist_lum) ** 2,
                )
                mag_src.append(src.bandmag(filt, "ab"))
            mag_src = np.array(mag_src)
            # True fluxes
            flux_src = flux_ref * 10 ** (-0.4 * mag_src)
            # Photometric errors
            flux_err_phot = flux_src * 0.02  # 2% photometric error
            flux_err_det = (flux_err_sky_det**2 + flux_err_phot**2) ** 0.5
            flux_det = flux_src + np.random.randn(len(phase_det)) * flux_err_det

            phase.append(np.concatenate([phase_non_det, phase_det]))
            flux.append(
                np.concatenate([flux_obs_non_det, flux_det]) / flux_src[-1] * 100
            )
            flux_err.append(
                np.concatenate([flux_err_non_det, flux_err_det]) / flux_src[-1] * 100
            )

        filt = np.array([1] * len(phase[0]) + [2] * len(phase[1])).astype(np.int32)
        fcqfid = filt.astype(np.int32)

        lc_peak = pd.DataFrame(
            dict(
                phase=np.concatenate(phase),
                flux=np.concatenate(flux) + params["base"],
                flux_err=np.concatenate(flux_err),
                fcqfid=fcqfid,
                filt=filt,
            )
        )

        lc_peak.reset_index(drop=True, inplace=True)

        return lc_peak
