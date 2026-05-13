import glob
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .._utils import plt
from ..constants import EPS
from ..model.lightcurve import SNLightCurveLib


def _count_nights(phase, mask, bin_width=0.5):
    """Count distinct observing nights within a phase mask.

    Groups consecutive observations within `bin_width` phase days of the
    earliest ungrouped point, matching the sliding-window algorithm used
    by ``data_binning`` for ZTF data.
    """
    p = np.sort(phase[mask])
    if len(p) == 0:
        return 0
    n, i = 0, 0
    while i < len(p):
        n += 1
        limit = p[i] + bin_width
        i += 1
        while i < len(p) and p[i] < limit:
            i += 1
    return n


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
        true_param_dependence: str | None = None,
        early_coverage: bool = False,
        baseline_coverage: bool = False,
        pop_prior_config: str | None = None,
    ) -> None:
        if (true_param_dependence is not None) and ("power_law" in true_model):
            file_dir = Path(f"./data/mock/{true_model}_{true_param_dependence}")
        else:
            file_dir = Path(f"./data/mock/{true_model}")

        peak_files = sorted(glob.glob(str(Path(file_dir) / "lc_peak*.csv")))
        params_file = file_dir / "simulated_lc_params.csv"

        # Filter by coverage flags if requested
        keep_idx = None
        if (early_coverage or baseline_coverage) and os.path.exists(params_file):
            params_meta = pd.read_csv(params_file)
            mask = np.ones(len(params_meta), dtype=bool)
            if early_coverage and "early_coverage" in params_meta.columns:
                mask &= params_meta["early_coverage"].values == 1
            if baseline_coverage and "baseline_coverage" in params_meta.columns:
                mask &= params_meta["baseline_coverage"].values == 1
            keep_idx = np.where(mask)[0]
            peak_files = [peak_files[k] for k in keep_idx if k < len(peak_files)]
            print(
                f"Filtered to {len(peak_files)} light curves "
                f"(early_coverage={early_coverage}, baseline_coverage={baseline_coverage})"
            )

        post_sample_dir = Path(file_dir) / f"{model}_frac{int(early_threshold * 100)}"
        if pop_prior_config:
            sampling_model_str = f"{sampling_model}_pop_prior_{pop_prior_config}"
        elif sampling_model in ["unpooled", "pooled"]:
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
                    f"Insufficient light curve files in {file_dir}: found "
                    f"{len(peak_files)} simulated light curves, but {n_lc} are required."
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
            idx = lc_peak["flux"].values >= 5 * lc_peak["flux_err"].values
            idx_early = lc_peak["phase"] < 0
            if np.sum(idx) > 0:
                phase = lc_peak["phase"].values[idx]
                flux = lc_peak["flux"].values[idx]
                early_cut = flux < early_threshold * 100
                if np.sum(early_cut) > 0:
                    idx_early = lc_peak["phase"] < phase[early_cut][-1] + 0.5
                else:
                    idx_early = lc_peak["phase"] < phase[0]

            lc_early_lib.append({key: item[idx_early] for key, item in lc_peak.items()})

        if not os.path.exists(post_sample_full_file):
            post_sample = None
        else:
            print("Loading existing .nc file...")
            post_sample = xr.load_dataset(post_sample_full_file)

        if os.path.exists(params_file):
            params_df = pd.read_csv(params_file)
            if keep_idx is not None:
                params_df = params_df.iloc[keep_idx]
            params_true = params_df[:n_lc].to_dict(orient="list")
        else:
            print("No true parameters file found.")
            params_true = None

        super().__init__(
            lc_early_lib=lc_early_lib,
            lc_peak_lib=lc_peak_lib,
            sampling_model=sampling_model,
        )

        self.params_true = params_true

        # If SNe with no observations were removed from the library,
        # slice params_true to match the surviving objects.
        if self.params_true is not None and hasattr(self, "_obs_valid_idx"):
            vi = self._obs_valid_idx
            if len(vi) < len(next(iter(self.params_true.values()))):
                self.params_true = {
                    k: [v[i] for i in vi] for k, v in self.params_true.items()
                }

        self.post_sample = post_sample
        self.pop_prior = pop_prior_config
        if self.post_sample is not None and "pop_prior" in self.post_sample.attrs:
            val = self.post_sample.attrs["pop_prior"]
            if val == "True":
                self.pop_prior = "pop_prior"
            elif val in ("False", ""):
                self.pop_prior = None
            else:
                self.pop_prior = val
        self.decode_post_sample()

    @classmethod
    def simulate_mock_light_curve(
        cls,
        n_lc: int = 10,
        param_dependence: str = "independent",
        model: str = "curved_power_law",
        min_dist_lum: float = 10,
        max_dist_lum: float = 270,
        z_fixed: float = None,
        template_model_id: str | None = None,
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
        from .parametric_models import (
            broken_power_law_rise_flat_sed,
            curved_power_law_rise_flat_sed,
            power_law_plus_gaussian_bump_flat_sed,
            power_law_rise_flat_sed,
        )
        from .template_models import build_photometry_engine

        logging.getLogger("redback").setLevel(logging.WARNING)

        # For the power-law rise models
        PEAK_LUMINOSITY = 2e28  # intrinsic peak luminosity (erg/s/Hz)
        from ..constants import T_PIVOT

        # Sample the population parameters using numpy.random
        num_tot = n_lc * 10  # oversample to account for non-detections

        np.random.seed(num_tot + np.sum([ord(c) for c in model]))

        # Hyperparameters are fixed below; param_dependence controls independence vs correlation

        params_sim = dict(base=np.random.normal(0.0, 0.1, num_tot))

        if "power_law" in model:
            # Add the required t0_mjd_transient parameter
            # Each transient gets a random MJD to average over ZTF cadence
            params_sim["t0_mjd_transient"] = np.random.uniform(58200, 59150, num_tot)
            params_sim["ra"] = np.random.uniform(0, 360, num_tot)
            params_sim["dec"] = np.random.uniform(-10, 70, num_tot)

            # True hyper-parameters for the power-law rise model
            params_true = dict(
                mean_alpha=2.0,
                sigma_alpha=0.3,
                mean_t_rise=18.5,
                sigma_t_rise=1.5,
            )

            # Support independent or correlated sampling for (t_rise, alpha_0, log_Aprime)
            # Correlated sampling is enabled when sampling_mode == "correlated"
            # Fixed correlation matrix (order: [t_rise, alpha_0, log_Aprime]):
            # corr(t_rise, alpha_0) = +0.3, corr(t_rise, log_Aprime) = -0.3, corr(alpha_0, log_Aprime) = 0.0
            corr_matrix = (
                None
                if param_dependence == "independent"
                else np.array(
                    [
                        [1.0, 0.3, -0.3],
                        [0.3, 1.0, 0.0],
                        [-0.3, 0.0, 1.0],
                    ]
                )
            )

            if corr_matrix is None:
                # Original independent sampling (default)
                params_sim["t_rise"] = np.random.normal(
                    params_true["mean_t_rise"], params_true["sigma_t_rise"], num_tot
                )
                params_sim["alpha_0"] = np.clip(
                    np.random.normal(
                        params_true["mean_alpha"], params_true["sigma_alpha"], num_tot
                    ),
                    1.0 + EPS,
                    5.0,
                )
                params_sim["peak_luminosity"] = np.full(num_tot, PEAK_LUMINOSITY)

                if model == "power_law":
                    params_true["mean_log_Aprime"] = np.log(40)
                    params_true["sigma_log_Aprime"] = 0.2
                    params_sim["log_Aprime"] = np.random.normal(
                        params_true["mean_log_Aprime"],
                        params_true["sigma_log_Aprime"],
                        num_tot,
                    )
                else:
                    # For curved/broken models, set a placeholder for Aprime to keep downstream compatibility
                    params_sim["log_Aprime"] = np.full(num_tot, np.nan)
            else:
                # Correlated sampling via multivariate normal
                # Build mean vector and covariance matrix from provided means/sigmas and corr_matrix
                # Validate corr_matrix dimensions (2x2 or 3x3)
                corr_matrix = np.asarray(corr_matrix, dtype=float)
                if corr_matrix.shape not in [(2, 2), (3, 3)]:
                    raise ValueError(
                        "corr_matrix must be 2x2 ([t_rise, alpha_0]) or 3x3 ([t_rise, alpha_0, log_Aprime])."
                    )

                # Means
                mean_t = params_true["mean_t_rise"]
                mean_alpha = params_true["mean_alpha"]
                mean_log_Aprime = np.log(40)

                # Sigmas
                sig_t = params_true["sigma_t_rise"]
                sig_alpha = params_true["sigma_alpha"]
                sig_log_Aprime = 0.2

                if corr_matrix.shape == (2, 2):
                    mean_vec = np.array([mean_t, mean_alpha])
                    scale_vec = np.array([sig_t, sig_alpha])
                else:
                    mean_vec = np.array([mean_t, mean_alpha, mean_log_Aprime])
                    scale_vec = np.array([sig_t, sig_alpha, sig_log_Aprime])

                # Convert correlation to covariance: Sigma = D * R * D, where D = diag(sigmas)
                sigmas = np.diag(scale_vec)
                cov = sigmas @ corr_matrix @ sigmas

                # Sample
                samples = np.random.multivariate_normal(
                    mean=mean_vec, cov=cov, size=num_tot
                )

                # Assign sampled values
                params_sim["t_rise"] = samples[:, 0]
                params_sim["alpha_0"] = samples[:, 1]
                params_sim["peak_luminosity"] = np.full(num_tot, PEAK_LUMINOSITY)

                if corr_matrix.shape == (3, 3):
                    log_Aprime_vals = samples[:, 2]
                else:
                    log_Aprime_vals = np.random.normal(
                        mean_log_Aprime, sig_log_Aprime, num_tot
                    )

                # Aprime only used for pure power-law model; set NaN otherwise for compatibility
                if model == "power_law":
                    params_sim["log_Aprime"] = log_Aprime_vals
                else:
                    params_sim["log_Aprime"] = np.full(num_tot, np.nan)

            if model == "curved_power_law":
                # Compute alpha_1 based on other parameters
                params_sim["alpha_1"] = -1 / (
                    (params_sim["t_rise"] / T_PIVOT)
                    * (1 + np.log(params_sim["t_rise"] / T_PIVOT))
                )

            elif model == "broken_power_law":
                params_sim["s"] = np.random.uniform(0.5, 1.5, num_tot)
                ratio = np.random.uniform(1.0, 1.5, num_tot)
                params_sim["t_b"] = params_sim["t_rise"] / ratio

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

                # Target ratio is t_rise / t_b (already = ratio drawn above)
                target_ratios = params_sim["t_rise"] / params_sim["t_b"]

                for i in range(num_tot):
                    av1_i = alpha_v_1[i]
                    s_i = params_sim["s"][i]
                    ratio_i = target_ratios[i]
                    sol = brentq(
                        _peak_equation,
                        av1_i - 5,
                        -1.01,
                        args=(av1_i, s_i, ratio_i),
                    )
                    alpha_v_2[i] = sol

                params_sim["alpha_1"] = alpha_v_1 - alpha_v_2

            elif model == "power_law_bump":
                # Fix baseline rise parameters to isolate bump-induced fitting systematics
                params_sim["t_rise"] = np.full(num_tot, 18.5)
                params_sim["alpha_0"] = np.full(num_tot, 2.0)
                params_sim["log_Aprime"] = np.full(num_tot, np.log(40.0))

                # Gaussian bump amplitude in normalized flux units (peak=100)
                params_sim["amp"] = np.random.uniform(2.0, 5.0, num_tot)

                # Sample bump width via broad FWHM prior, then derive sigma and center
                params_sim["t_fwhm"] = np.random.uniform(1.0, 7.0, num_tot)
                params_sim["t_sigma"] = params_sim["t_fwhm"] / (
                    2.0 * np.sqrt(2.0 * np.log(2.0))
                )
                params_sim["t_cen"] = 2.0 * params_sim["t_sigma"]

        elif (
            "turtls" in model.lower()
            or "shen" in model.lower()
            or "observation" in model.lower()
        ):
            num_tot = n_lc  # no oversampling — all template transients accepted
            # Resolve template model ID
            if template_model_id is None:
                raise ValueError(
                    "template_model_id must be provided for template-based models."
                )
            for template in ["turtls", "shen2021", "observation"]:
                if template in model.lower():
                    template_model_id = f"{template}:{template_model_id}"
                    break

            # Load photometry engine once
            photometry_engine = build_photometry_engine(template_model_id)

            # Store metadata
            params_sim["template_model_id"] = [template_model_id] * num_tot
            params_sim["template_family"] = [photometry_engine.model.family] * num_tot

            # For Shen, pre-draw angles and compute angle-dependent peak properties
            if photometry_engine.model.family in ("shen2021", "shen"):
                n_angles = photometry_engine.get_num_viewing_angles()
                params_sim["n_shen_angles"] = np.full(num_tot, n_angles)

                # Assign viewing angles in balanced order, then shuffle
                angle_idxs = np.tile(np.arange(n_angles), num_tot // n_angles + 1)[:num_tot]
                np.random.shuffle(angle_idxs)
                params_sim["shen_angle_idx"] = angle_idxs

                # Precompute peak for each angle (since n_angles is small, e.g. ~14)
                # This avoids re-integrating SEDs num_tot times.
                t_peaks_by_angle = np.zeros(n_angles)
                m_peaks_by_angle = np.zeros(n_angles)

                print(f"Pre-computing peaks for {n_angles} viewing angles...")
                for i in range(n_angles):
                    tp, mp = photometry_engine.get_peak(
                        band="bessellb", z=0.0, angle_idx=i
                    )
                    t_peaks_by_angle[i] = tp
                    m_peaks_by_angle[i] = mp

                # Assign per-transient properties based on drawn angle
                params_sim["t_rise"] = t_peaks_by_angle[angle_idxs]
                params_sim["B_peak"] = m_peaks_by_angle[angle_idxs]

            else:
                # TURTLS or other band-template models (single implementation)
                t_peak_rest, M_peak_rest = photometry_engine.get_peak(
                    band="bessellb", z=0.0
                )
                params_sim["t_rise"] = np.full(num_tot, t_peak_rest)
                params_sim["B_peak"] = np.full(num_tot, M_peak_rest)

                params_sim["n_shen_angles"] = np.full(num_tot, -1)
                params_sim["shen_angle_idx"] = np.full(num_tot, -1)

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
        if ("power_law" in model) and ("bump" not in model):
            model_dir = f"{model_dir}_{param_dependence}"
        elif (
            "turtls" in model.lower()
            or "shen" in model.lower()
            or "observation" in model.lower()
        ):
            sanitized_id = template_model_id.replace(":", "_")
            model_dir = sanitized_id

            if z_fixed is not None:
                model_dir = f"{model_dir}_z_{z_fixed:.2f}".replace(".", "_")

        data_dir = Path("./data/mock") / model_dir
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        for i in range(num_tot):
            # Extract parameters for this single transient
            single_params = {key: val[i] for key, val in params_sim.items()}

            if "power_law" in model and (
                np.abs(single_params["alpha_0"] - params_true["mean_alpha"])
                / params_true["sigma_alpha"]
                > 3
            ):
                continue

            idx_obs = len(params_valid_det)

            if len(params_valid_det) >= n_lc:
                break

            early_cov = None
            n_obs_early_val = None
            base_cov = None

            if "power_law" in model:
                if model == "power_law":
                    sed_model = power_law_rise_flat_sed
                elif model == "power_law_bump":
                    sed_model = power_law_plus_gaussian_bump_flat_sed
                elif model == "curved_power_law":
                    sed_model = curved_power_law_rise_flat_sed
                elif model == "broken_power_law":
                    sed_model = broken_power_law_rise_flat_sed

                result = cls._simulate_single_light_curve_redback(
                    sed_model=sed_model, params=single_params
                )

                if result is None:
                    continue

                lc_peak, early_cov, n_obs_early_val, base_cov = result

                print(
                    f"Simulating transient {idx_obs + 1}/{n_lc} ({i + 1} attempts)..."
                )
                print(f"  → {len(lc_peak['phase'])} detections before peak")

            elif (
                "turtls" in model.lower()
                or "shen" in model.lower()
                or "observation" in model.lower()
            ):
                angle_idx = None
                if single_params.get("template_family") in ("shen2021", "shen"):
                    angle_idx = int(single_params["shen_angle_idx"])

                lc_peak = cls._simulate_single_light_curve_unified_template(
                    params=single_params,
                    photometry_engine=photometry_engine,
                    angle_idx=angle_idx,
                )

                if lc_peak is None:
                    continue

                early_cov = True
                base_cov = True

            # Reset the plot style after Redback's modification
            set_plot_style()

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

            single_params["early_coverage"] = int(early_cov)
            single_params["n_obs_early"] = (
                n_obs_early_val if n_obs_early_val is not None else -1
            )
            single_params["baseline_coverage"] = (
                int(base_cov) if base_cov is not None else -1
            )
            params_valid_det.append(single_params)

            if ((idx_obs + 1) % 50 == 0) and "power_law" in model:
                params_valid_det_df = pd.DataFrame(params_valid_det).reset_index(
                    drop=True
                )
                print(f"{'=' * 40}")
                print(
                    f"Current t_rise = {np.mean(params_valid_det_df['t_rise']):.2f} +/- {np.std(params_valid_det_df['t_rise']):.2f}"
                )
                print(
                    f"Current alpha = {np.mean(params_valid_det_df['alpha_0']):.2f} +/- {np.std(params_valid_det_df['alpha_0']):.2f}"
                )
                if model == "power_law":
                    print(
                        f"Current log_Aprime = {np.mean(params_valid_det_df['log_Aprime']):.2f} +/- {np.std(params_valid_det_df['log_Aprime']):.2f}"
                    )
                print(
                    f"Current rho(t_rise, alpha) = {np.corrcoef(params_valid_det_df['t_rise'], params_valid_det_df['alpha_0'])[0, 1]:.2f}"
                )
                if model == "power_law":
                    print(
                        f"Current rho(t_rise, log_Aprime) = {np.corrcoef(params_valid_det_df['t_rise'], params_valid_det_df['log_Aprime'])[0, 1]:.2f}"
                    )
                print(f"{'=' * 40}")

        params_valid_det = pd.DataFrame(params_valid_det).reset_index(drop=True)

        params_valid_det.to_csv(
            data_dir / "simulated_lc_params.csv",
            index=False,
        )

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
    ) -> tuple[pd.DataFrame, bool, int, bool] | None:
        """
        Simulate a single light curve using Redback.

        Returns
        -------
        (lc_peak, early_coverage, n_obs_early, baseline_coverage) | None
            lc_peak: DataFrame with pre-peak photometry.
            early_coverage: True if the transient passes the full
                early-time coverage criteria (baseline + early).
            n_obs_early: minimum of distinct high-SNR early-phase nights
                across g and r bands.
            baseline_coverage: True if the transient passes the baseline
                coverage criteria (>=10 nights in both bands at -100 < phase < -25).
            None: if the loose save gate fails (pre/post-peak nights
                or both bands not observed).
        """
        from redback.simulate_transients import SimulateOpticalTransient

        if "Aprime" in params:
            params["amp_prime"] = params["Aprime"]
        if "log_Aprime" in params:
            params["amp_prime"] = np.exp(params["log_Aprime"])
        else:
            raise ValueError("Either Aprime or log_Aprime must be provided")

        # Simulate single transient
        sim = SimulateOpticalTransient.simulate_transient_in_ztf(
            model=sed_model,
            survey="ztf",
            parameters=params,
            end_transient_time=100,
            snr_threshold=3.0,
            add_source_noise=True,
            source_noise=0.02**2,  # a bug in redback < 1.12.1 - has to be noise**2
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
        obs["phase"] = (obs["time"] - params["t0_mjd_transient"]) / (
            1 + params["redshift"]
        ) - params["t_rise"]

        phase = obs["phase"].values
        idx_snr = obs["flux(erg/cm2/s)"] / obs["flux_error"] > 5

        # Loose save gate: at least 2 distinct nights per filter before & after peak
        idx_g = obs["band"] == "ztfg"
        idx_r = obs["band"] == "ztfr"
        idx_prepeak = phase < 0
        idx_postpeak = phase >= 0
        if (
            _count_nights(phase, idx_prepeak & idx_snr) < 2
            or _count_nights(phase, idx_postpeak & idx_snr) < 2
            or np.sum(idx_g & idx_snr) == 0
            or np.sum(idx_r & idx_snr) == 0
        ):
            return None

        # Full early-coverage flag (the strict detection-quality criteria)
        idx_early = (phase < -10) & (phase > -25)
        idx_baseline = (phase < -25) & (phase > -100)

        n_obs_early = min(
            _count_nights(phase, idx_snr & idx_early & idx_g),
            _count_nights(phase, idx_snr & idx_early & idx_r),
        )

        baseline_coverage = (
            _count_nights(phase, idx_baseline & idx_g) >= 10
            and _count_nights(phase, idx_baseline & idx_r) >= 10
        )

        early_coverage = n_obs_early >= 2

        # Generate early light curve up to early_threshold of peak flux
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

        return lc_peak, early_coverage, n_obs_early, baseline_coverage

    @classmethod
    @staticmethod
    def _find_peak_poly4(time, magnitude):
        from scipy.optimize import curve_fit, fsolve

        def poly4(t, a, b, c, d, e):
            return a * t**4 + b * t**3 + c * t**2 + d * t + e

        idx_min = np.argmin(magnitude)
        t_peak_guess = time[idx_min]

        mask = np.abs(time - t_peak_guess) < 10
        time_fit = time[mask]
        mag_fit = magnitude[mask]

        phase = time_fit - t_peak_guess
        popt, _ = curve_fit(poly4, phase, mag_fit, p0=[1, -1, 0, 0, np.min(mag_fit)])

        # Create polynomial and its derivative
        poly_deriv_coeffs = [4 * popt[0], 3 * popt[1], 2 * popt[2], popt[3]]
        poly_deriv = np.poly1d(poly_deriv_coeffs)

        phase_peak = fsolve(poly_deriv, [0.0])[0]  # Time of peak in phase
        mag_peak = poly4(phase_peak, *popt)  # Magnitude at the peak
        return t_peak_guess + phase_peak, mag_peak

    @classmethod
    def _simulate_single_light_curve_unified_template(
        cls,
        params: dict,
        photometry_engine: Any,
        angle_idx: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Simulate a single light curve using the unified template photometry engine.
        Handles both TURTLS (band templates) and Shen+2021 (SED models).
        """
        import os

        import redback
        from astropy.cosmology import FlatLambdaCDM

        # Shared ZTF setup
        ztf_path = os.path.join(
            os.path.dirname(redback.__file__), "tables", "ztf.tar.gz"
        )
        filter_path = os.path.join(
            os.path.dirname(redback.__file__), "tables", "filters.csv"
        )
        ztf_pointings = pd.read_csv(ztf_path, compression="gzip")
        ztf_filters = pd.read_csv(filter_path)
        ztf_filters = ztf_filters[ztf_filters["sncosmo_name"].isin(["ztfg", "ztfr"])]

        def get_flux_err_sky(filt: str, num: int) -> np.ndarray:
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

        z = params["redshift"]
        # dist_lum is in Mpc, engine usually needs cosmo for accurate distance modulus if not provided
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        # 1. Determine Peak for this transient (at this z and angle)
        if photometry_engine.model.family == "observation":
            t_peak_obs = 0.0
        else:
            t_peak_obs, m_peak_obs = photometry_engine.get_peak(
                band="bessellb", z=0.0, cosmo=cosmo, angle_idx=angle_idx, method="poly4"
            )

        # 2. Define Observer Time Grid from native REST-FRAME model epochs
        # Use native model times directly (no synthetic observer-frame cadence grid)
        time_rest_native = np.asarray(photometry_engine.model.time_rest, dtype=float)

        if photometry_engine.model.family == "shen2021":
            random_idx = np.random.binomial(1, 0.5)
            time_rest_native = time_rest_native[random_idx::2]
            # Exclude the first 2 rest-frame days from template sampling
            time_rest_native = time_rest_native[time_rest_native >= 2.0]

        if time_rest_native.size == 0:
            return None

        # Explicit observer-frame grid constructed from native rest-frame epochs
        t_obs_grid = time_rest_native * (1 + z)

        # Calculate phases relative to peak (rest-frame)
        phase_obs_grid = t_obs_grid / (1 + z) - t_peak_obs

        # 3. Compute Observer Magnitudes
        filters = ["ztfg", "ztfr"]
        mags_dict = photometry_engine.get_obs_mag(
            filters=filters, t_obs=t_obs_grid, z=z, cosmo=cosmo, angle_idx=angle_idx
        )

        phase = []
        flux = []
        flux_err = []

        for filt in filters:
            # Get magnitudes for this filter
            mag_obs = mags_dict[filt]

            # Identify valid magnitudes (not nan/inf)
            valid_mask = np.isfinite(mag_obs)
            if not np.any(valid_mask):
                continue

            flux_ref = ztf_filters[ztf_filters["sncosmo_name"] == filt][
                "reference_flux"
            ].values[0]

            flux_model_grid = np.zeros_like(mag_obs)
            flux_model_grid[valid_mask] = flux_ref * 10 ** (-0.4 * mag_obs[valid_mask])

            # Define peak flux for this band for normalization to 100
            flux_peak_band = np.max(flux_model_grid) * 1.01
            if flux_peak_band <= 0:
                flux_peak_band = 1e-30

            # Simulate Non-Detections (Pre-explosion)
            # Pre-explosion corresponds to times before the first native sampled epoch
            t_obs_fl = t_obs_grid.min() - t_peak_obs

            if photometry_engine.model.family == "observation":
                phase_non_det = np.arange(-50.0, -18.0, 2.0) + 2.0
            else:
                phase_non_det = np.arange(t_obs_fl - 30, t_obs_fl - 2.0, 2) / (1 + z)

            phase_non_det += np.random.normal(0, 0.05 / (1 + z), len(phase_non_det))
            flux_err_sky_non_det = get_flux_err_sky(filt, len(phase_non_det))
            flux_obs_non_det = (
                np.random.randn(len(phase_non_det)) * flux_err_sky_non_det
            )

            # Simulate Detections (Model duration)
            phase_det = phase_obs_grid
            flux_src_det = flux_model_grid

            # Errors
            flux_err_sky_det = get_flux_err_sky(filt, len(flux_src_det))
            flux_err_phot = flux_src_det * 0.02
            flux_err_det = (flux_err_sky_det**2 + flux_err_phot**2) ** 0.5

            flux_obs_det = (
                flux_src_det + np.random.randn(len(flux_src_det)) * flux_err_det
            )

            # Store concatenated results, normalized
            phase.append(np.concatenate([phase_non_det, phase_det]))
            flux.append(
                np.concatenate([flux_obs_non_det, flux_obs_det]) / flux_peak_band * 100
            )
            flux_err.append(
                np.concatenate([flux_err_sky_non_det, flux_err_det])
                / flux_peak_band
                * 100
            )

        if not phase:
            return None

        filt_idx = []
        for i, p in enumerate(phase):
            filt_idx.append(np.full(len(p), i + 1, dtype=np.int32))

        lc_peak = pd.DataFrame(
            dict(
                phase=np.concatenate(phase),
                flux=np.concatenate(flux) + params.get("base", 0.0),
                flux_err=np.concatenate(flux_err),
                fcqfid=np.concatenate(filt_idx),
                filt=np.concatenate(filt_idx),
            )
        )

        # Only keep phase < 0 for the output format standard
        idx_peak = lc_peak["phase"] < 0
        lc_peak = lc_peak[idx_peak].reset_index(drop=True)

        return lc_peak
