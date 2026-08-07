import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from astropy.table import Table

from ._utils import data_binning
from .model.lightcurve import SNLightCurve, SNLightCurveLib


class ZTFDataProcessor:
    """Helper class to handle common ZTF data processing operations."""

    EXTERNAL_GR_FILTER_IDS = {
        "sdssg": 1,
        "swope2g": 1,
        "ps1::g": 1,
        "sdssr": 2,
        "swope2r": 2,
        "ps1::r": 2,
    }

    EXTERNAL_GR_PEAK_ALIASES = {
        "sdssg_AB",
        "sdssr_AB",
        "g_PS1",
        "r_PS1",
        "g_CSP2_AB",
        "r_CSP2_AB",
    }

    EXTERNAL_FCQFID_BASE = -900000

    @staticmethod
    def process_flux_normalization(
        flux, flux_err, filt, flux_g_max, flux_r_max, filtids=[1, 2]
    ):
        """Normalize flux data by maximum flux values."""
        flux = np.asarray(flux, dtype=np.float32)
        flux_err = np.asarray(flux_err, dtype=np.float32)

        flux[filt == filtids[0]] /= flux_g_max / 100
        flux_err[filt == filtids[0]] /= flux_g_max / 100
        flux[filt == filtids[1]] /= flux_r_max / 100
        flux_err[filt == filtids[1]] /= flux_r_max / 100

        return flux, flux_err

    @staticmethod
    def calculate_early_times(
        phase, flux, flux_err, filt, early_threshold=0.4, flux_max=None, filtid=1
    ):
        """Calculate 40% flux times in a filter."""
        try:
            data = np.array([phase, flux, flux_err]).T[filt == filtid]
            data = data[np.argsort(data[:, 0])]
            t, f, _ = data_binning(data, 0.5).T
        except ValueError:
            # Not enough data points
            return -np.inf, flux_max

        if flux_max is None:
            flux_max = np.max(f[np.abs(t) < 5])

        # Below flux threshold
        below_threshold = (
            (f <= early_threshold * flux_max)
            if early_threshold < 1
            else np.ones_like(
                f, dtype=bool
            )  # For early_threshold >= 1, consider all points
        )
        below_threshold[1:] &= below_threshold[:-1]  # ensure monotonicity
        idx_early = below_threshold & (t < 0)

        if np.sum(idx_early) == 0:
            t_early = -np.inf
        else:
            t_early = t[idx_early][-1] + 0.25

        return t_early, flux_max

    @staticmethod
    def create_light_curve_data(
        phase,
        flux,
        flux_err,
        fcqfid,
        filt,
        t_g_early,
        t_r_early,
        filtids=[1, 2],
        beta=None,
    ):
        """Create early and peak light curve dictionaries."""
        if beta is None:
            beta = np.ones_like(phase)

        # Filter out observations < 40% of max flux
        idx_i = filt == 3
        idx_rise = (phase < 0) & (phase > -100) & ~idx_i
        idx_g = (filt == filtids[0]) & (phase < t_g_early)
        idx_r = (filt == filtids[1]) & (phase < t_r_early)
        idx = idx_rise & (idx_g | idx_r)

        # Discard the fcqfID with few baseline points
        ## Already done in the preprocessing
        # idx_base = phase < -25
        # idx_fcqfid_few_base = []
        # for fcqfid_val in np.unique(fcqfid):
        # if np.sum((fcqfid == fcqfid_val) & idx_base) < 5:
        # idx_fcqfid_few_base.append(fcqfid_val)
        # print(f"Discarding fcqfid {fcqfid_val} with few baseline points")
        # idx &= ~np.isin(fcqfid, idx_fcqfid_few_base)

        lc_early = {
            "phase": phase[idx],
            "flux": flux[idx],
            "flux_err": flux_err[idx],
            "fcqfid": fcqfid[idx],
            "filt": filt[idx],
            "beta": beta[idx],
        }

        lc_peak = {
            "phase": phase[idx_rise],
            "flux": flux[idx_rise],
            "flux_err": flux_err[idx_rise],
            "fcqfid": fcqfid[idx_rise],
            "filt": filt[idx_rise],
            "beta": beta[idx_rise],
        }

        return lc_early, lc_peak

    @staticmethod
    def sort_light_curve(lc):
        """Return a light-curve dictionary sorted by phase."""

        order = np.argsort(np.asarray(lc["phase"], dtype=np.float64))
        return {key: np.asarray(value)[order] for key, value in lc.items()}

    @staticmethod
    def load_external_gr_photometry(
        filename,
        t0,
        z,
        peak_flux_by_filter,
        noise_floor=0.05,
    ):
        """Load external g/r photometry normalized by native passband peaks."""

        columns = ["phase", "flux", "flux_err", "fcqfid", "filt"]
        filename = Path(filename)
        if not filename.exists():
            return pd.DataFrame(columns=columns), {}

        external = pd.read_csv(filename)
        external = external[
            external["filter"].isin(ZTFDataProcessor.EXTERNAL_GR_FILTER_IDS)
            & external["bayesn_filter"].isin(ZTFDataProcessor.EXTERNAL_GR_PEAK_ALIASES)
            & np.isfinite(external["mjd"])
            & np.isfinite(external["flux"])
            & np.isfinite(external["fluxerr"])
            & np.isfinite(external["zp"])
            & external["source"].notna()
            & (external["fluxerr"] > 0)
        ].copy()
        if external.empty:
            return pd.DataFrame(columns=columns), {}

        missing_peaks = sorted(set(external["bayesn_filter"]) - set(peak_flux_by_filter))
        if missing_peaks:
            raise ValueError(
                "Missing peak-flux columns for external filters "
                f"{missing_peaks}; regenerate the calibration CSV."
            )

        external["filt"] = external["filter"].map(
            ZTFDataProcessor.EXTERNAL_GR_FILTER_IDS
        ).astype(np.int64)

        sources = sorted(external["source"].dropna().unique())
        source_index = {source: idx + 1 for idx, source in enumerate(sources)}
        groups = external[["source", "bayesn_filter"]].drop_duplicates()
        fcqfid_map = {
            (row.source, row.bayesn_filter): ZTFDataProcessor.EXTERNAL_FCQFID_BASE
            - source_index[row.source] * 10
            - idx
            for idx, row in enumerate(groups.itertuples(index=False), start=1)
        }
        external["fcqfid"] = [
            fcqfid_map[(row.source, row.bayesn_filter)]
            for row in external[["source", "bayesn_filter"]].itertuples(index=False)
        ]

        peak_flux = np.array(
            [peak_flux_by_filter[str(alias)] for alias in external["bayesn_filter"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(peak_flux)) or np.any(peak_flux <= 0):
            bad = sorted(set(external.loc[(~np.isfinite(peak_flux)) | (peak_flux <= 0), "bayesn_filter"]))
            raise ValueError(f"Missing finite positive external peak fluxes for filters {bad}.")

        scale = 10 ** (-0.4 * (external["zp"].to_numpy(dtype=np.float64) - 30.0))
        flux = external["flux"].to_numpy(dtype=np.float64) * scale
        flux_err = external["fluxerr"].to_numpy(dtype=np.float64) * scale
        floor = float(noise_floor) * np.abs(flux)
        flux_err = np.sqrt(flux_err**2 + floor**2)
        flux = flux / (peak_flux / 100.0)
        flux_err = flux_err / (peak_flux / 100.0)

        loaded = pd.DataFrame(
            {
                "phase": (external["mjd"].to_numpy(dtype=np.float64) - t0) / (1 + z),
                "flux": flux,
                "flux_err": flux_err,
                "fcqfid": external["fcqfid"].to_numpy(dtype=np.int64),
                "filt": external["filt"].to_numpy(dtype=np.int64),
            }
        )
        counts = external.groupby(["source", "filter", "bayesn_filter"]).size().to_dict()
        return loaded, counts


class ZTFIaEarlyLate(SNLightCurve):
    """
    Liu et al. in prep.
    ZTF SNe Ia with early light curves and nebular spectra
    """

    late_dir: str = "./data/ztf_snia_early_late/"
    phot_path: str = "ztf_early_Ia_phot.csv"
    lc_path: str = "light_curve_fps_ztf/*fnu.csv"
    external_lc_path: str = "light_curve_external/"

    def __init__(
        self,
        ztfid: str,
        early_threshold: float = 0.4,
        include_external_gr: bool = False,
        external_noise_floor: float = 0.05,
    ) -> None:
        phot_data = Table.read(self.late_dir + self.phot_path)
        lc_list = sorted(glob.glob(self.late_dir + self.lc_path))
        ztfid_list = [lc.split("/")[-1].split("_")[0] for lc in lc_list]
        # ztfid_list = meta_data["objid"].data
        if ztfid not in ztfid_list:
            raise ValueError(f"ZTF ID {ztfid} not found in early light curves.")
        try:
            tab_lc = pd.read_csv(
                lc_list[ztfid_list.index(ztfid)], sep=r"\s+", comment="#"
            )
        except:
            print(ztfid_list, ztfid)
            raise
        tab_lc = tab_lc.rename(
            columns={key: key.replace(",", "") for key in tab_lc.columns}
        )

        # Prepare filter id
        tab_lc["filter_id"] = np.select(
            [
                tab_lc["filter"] == "ZTF_g",
                tab_lc["filter"] == "ZTF_r",
                tab_lc["filter"] == "ZTF_i",
            ],
            [1, 2, 3],
            default=0,  # Or some other default value if needed
        )

        # Prepare fcqf id
        tab_lc["fcqfid"] = (
            tab_lc["field"].astype(np.int64) * 10000
            + tab_lc["ccdid"].astype(np.int64) * 100
            + tab_lc["qid"].astype(np.int64) * 10
            + tab_lc["filter_id"].astype(np.int64)
        )

        # Prepare flux and flux error
        tab_lc["zp"] = np.ones(len(tab_lc), dtype=np.float32) * 30.0
        delta_zp = tab_lc["zpdiff"] - tab_lc["zp"]
        tab_lc["flux"] = tab_lc["forcediffimflux"].astype(np.float32) * (
            10 ** (-0.4 * delta_zp)
        )
        tab_lc["flux_err"] = tab_lc["forcediffimfluxunc"].astype(np.float32) * (
            10 ** (-0.4 * delta_zp)
        )

        phot_row = phot_data[phot_data["ztfid"] == ztfid]
        if len(phot_row) != 1:
            raise ValueError(f"No unique photometric calibration row found for {ztfid}.")

        t0 = phot_row["t0"].data[0]
        t0_err = phot_row["t0_err"].data[0]
        z = phot_row["z"].data[0]
        flux_zp = phot_row["flux_zp"].data[0]
        flux_scale_to_zp30 = 10 ** (-0.4 * (flux_zp - 30.0))
        flux_g_max = phot_row["ztfg_flux_max"].data[0] * flux_scale_to_zp30
        flux_r_max = phot_row["ztfr_flux_max"].data[0] * flux_scale_to_zp30
        external_peak_flux = {}
        missing_external_peak_cols = []
        for alias in ZTFDataProcessor.EXTERNAL_GR_PEAK_ALIASES:
            col = f"{alias}_flux_max"
            if col not in phot_data.colnames:
                missing_external_peak_cols.append(col)
            else:
                external_peak_flux[alias] = phot_row[col].data[0] * flux_scale_to_zp30
        if include_external_gr and missing_external_peak_cols:
            raise ValueError(
                f"Missing external peak-flux columns in photometric calibration row for {ztfid}: "
                + ", ".join(missing_external_peak_cols)
            )
        required = [t0, t0_err, z, flux_zp, flux_g_max, flux_r_max]
        if not np.all(np.isfinite(required)) or flux_g_max <= 0 or flux_r_max <= 0:
            raise ValueError(f"Missing finite positive photometric calibration values for {ztfid}.")

        # Quality mask
        # https://github.com/BrightTransientSurvey/ztf_forced_phot/tree/main/explanation#-flags-bitmask
        mask = np.isfinite(tab_lc["flux"]) & (tab_lc["flux_err"] > 0)
        # tab_lc["infobitssci"] <= 33554432 #TODO: Figure out why some fluxes are NaN

        # Baseline outlier rejection (same as DR2, Liu et al. 2026)
        phase_full = (tab_lc["jd"].to_numpy() - 2400000.5 - t0) / (1 + z)
        idx_baseline = (phase_full < -30) & (phase_full > -100) & mask
        if idx_baseline.sum() > 0:
            bad_baseline = (
                np.abs(
                    tab_lc["flux"].to_numpy(dtype=np.float32)
                    / tab_lc["flux_err"].to_numpy(dtype=np.float32)
                )
                > 10
            ) & idx_baseline

            for _fcqfid in np.unique(tab_lc["fcqfid"]):
                idx_fcqfid = (tab_lc["fcqfid"] == _fcqfid).to_numpy()
                idx_baseline_fcqfid = idx_baseline & idx_fcqfid
                if idx_baseline_fcqfid.sum() > 0:
                    n_bad = (bad_baseline & idx_fcqfid).sum()
                    n_total = idx_baseline_fcqfid.sum()
                    if n_bad > 0.5 * n_total:
                        mask &= ~idx_fcqfid
                        print(
                            f"{ztfid}: {n_bad} bad points ({n_bad / n_total * 100:.1f}%) in {_fcqfid}: mask the field"
                        )
                    else:
                        mask &= ~(bad_baseline & idx_fcqfid)

        data = tab_lc[mask]
        flux = data["flux"]
        flux_err = data["flux_err"]
        phase = (data["jd"] - 2400000.5 - t0) / (1 + z)

        fcqfid = data["fcqfid"]
        filt = data["filter_id"]

        flux, flux_err = ZTFDataProcessor.process_flux_normalization(
            flux, flux_err, filt, flux_g_max, flux_r_max
        )

        if include_external_gr:
            external_filename = (
                self.late_dir
                + self.external_lc_path
                + f"{ztfid}_external.csv"
            )
            external_lc, external_counts = ZTFDataProcessor.load_external_gr_photometry(
                external_filename,
                t0,
                z,
                external_peak_flux,
                noise_floor=external_noise_floor,
            )
            if not external_lc.empty:
                phase = np.concatenate(
                    (np.asarray(phase, dtype=np.float64), external_lc["phase"].to_numpy())
                )
                flux = np.concatenate(
                    (np.asarray(flux, dtype=np.float64), external_lc["flux"].to_numpy())
                )
                flux_err = np.concatenate(
                    (
                        np.asarray(flux_err, dtype=np.float64),
                        external_lc["flux_err"].to_numpy(),
                    )
                )
                fcqfid = np.concatenate(
                    (
                        np.asarray(fcqfid, dtype=np.int64),
                        external_lc["fcqfid"].to_numpy(dtype=np.int64),
                    )
                )
                filt = np.concatenate(
                    (
                        np.asarray(filt, dtype=np.int64),
                        external_lc["filt"].to_numpy(dtype=np.int64),
                    )
                )
                counts_msg = ", ".join(
                    f"{source} {band} {alias}={count}"
                    for (source, band, alias), count in external_counts.items()
                )
                print(
                    f"{ztfid}: added external g/r photometry with "
                    f"{external_noise_floor:.1%} noise floor ({counts_msg})"
                )

        phase_order = np.argsort(np.asarray(phase, dtype=np.float64))
        phase = np.asarray(phase)[phase_order]
        flux = np.asarray(flux)[phase_order]
        flux_err = np.asarray(flux_err)[phase_order]
        fcqfid = np.asarray(fcqfid)[phase_order]
        filt = np.asarray(filt)[phase_order]

        # Calculate X% times and max flux from data
        t_g_early, _ = ZTFDataProcessor.calculate_early_times(
            phase,
            flux,
            flux_err,
            filt,
            early_threshold=early_threshold,
            flux_max=100.0,
            filtid=1,  # g filter
        )
        t_r_early, _ = ZTFDataProcessor.calculate_early_times(
            phase,
            flux,
            flux_err,
            filt,
            early_threshold=early_threshold,
            flux_max=100.0,
            filtid=2,  # r filter
        )

        # Create light curve data
        lc_early, lc_peak = ZTFDataProcessor.create_light_curve_data(
            phase, flux, flux_err, fcqfid, filt, t_g_early, t_r_early
        )

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid, t0_err=t0_err)


class ZTFIaDR2(SNLightCurve):
    """
    Rigault et al. 2025
    ZTF SN Ia DR2
    """

    dr2_dir: str = "./data/ztf_snia_dr2/"
    tab_info_path: str = "tables/snia_data_basic_normal.csv"
    tab_lc_path: str = "lightcurves_preproc/*lc.csv"

    def __init__(
        self, ztfid: str, early_threshold: float = 0.4, sn_type: str = "normal"
    ) -> None:
        tab_info = Table.read(
            self.dr2_dir + self.tab_info_path.replace("normal", sn_type)
        )
        lc_list = sorted(glob.glob(self.dr2_dir + self.tab_lc_path))
        ztfid_list = [lc.split("/")[-1].split("_")[0] for lc in lc_list]
        if ztfid not in ztfid_list:
            raise ValueError(f"ZTF ID {ztfid} not found in DR2 light curves.")
        tab_lc = Table.read(
            lc_list[ztfid_list.index(ztfid)], format="ascii", comment="#"
        )

        # Prepare filter id and fcqf id
        # tab_lc.add_column(
        #     np.select(
        #         [
        #             tab_lc["filter"] == "ztfg",
        #             tab_lc["filter"] == "ztfr",
        #             tab_lc["filter"] == "ztfi",
        #         ],
        #         [1, 2, 3],
        #         default=0,  # Or some other default value if needed
        #     ),
        #     name="filter_id",
        # )
        # RCID = 4(CCDID – 1) + QID – 1
        # tab_lc.add_column(
        #     np.int64(tab_lc["field_id"]) * 10000
        #     + np.int64(tab_lc["rcid"] // 4 + 1) * 100  # CCDID
        #     + np.int64(tab_lc["rcid"] % 4 + 1) * 10  # QID
        #     + np.int64(tab_lc["filter_id"]),
        #     name="fcqfid",
        # )

        info = tab_info[tab_info["ztfname"] == ztfid]
        t0 = info["t0"].data[0]
        t0_err = info["t0_err"].data[0]
        z = info["z"].data[0]

        # bad photometry: Rigault et al. 2025
        bad_bitmask = [1, 2, 4, 8, 16]
        mask = (
            np.bitwise_and(tab_lc["flag"].data[:, None], bad_bitmask).sum(axis=1) == 0
        )
        # outliers in the baseline
        phase = (tab_lc["mjd"].data - t0) / (1 + z)
        idx_baseline = (phase < -30) & (phase > -100)
        if idx_baseline.sum() > 0:
            bad_baseline = (
                np.abs(
                    tab_lc["flux"].data.astype("<f4")
                    / (
                        tab_lc["flux_err"].data.astype("<f4")
                        * tab_lc["err_scale"].data.astype("<f4")
                    )
                )
                > 10
            ) & idx_baseline

        for _fcqfid in np.unique(tab_lc["fcqfid"].data):
            idx_fcqfid = tab_lc["fcqfid"] == _fcqfid
            # If there are baseline data in certain fields
            if (idx_fcqfid & idx_baseline).sum() > 0:
                # More than 50% bad points in a single fcqfID -> mask the entire field
                if (bad_baseline & idx_fcqfid).sum() > 0.5 * (
                    idx_fcqfid & idx_baseline
                ).sum():
                    mask &= ~idx_fcqfid
                    print(
                        f"{ztfid}: {(bad_baseline & idx_fcqfid).sum()} bad points ({(idx_fcqfid & bad_baseline).sum() / (idx_fcqfid & idx_baseline).sum() * 100:.1f}%) in {_fcqfid}: mask the field"
                    )
                # Normal random outliers -> mask bad points
                else:
                    mask &= ~(bad_baseline & idx_fcqfid)

        data = tab_lc[mask]

        flux = data["flux"].data.astype("<f4")
        flux_err = data["flux_err"].data.astype("<f4")
        phase = (data["mjd"].data - t0) / (1 + z)
        beta = data["err_scale"].data

        fcqfid = data["fcqfid"].data
        filt = data["filter_id"].data

        # Calculate X% times and max flux from data
        t_g_early, flux_g_max = ZTFDataProcessor.calculate_early_times(
            phase,
            flux,
            flux_err,
            filt,
            early_threshold=early_threshold,
            flux_max=info["flux_peak_ztfg"].data,
            filtid=1,  # g filter
        )
        t_r_early, flux_r_max = ZTFDataProcessor.calculate_early_times(
            phase,
            flux,
            flux_err,
            filt,
            early_threshold=early_threshold,
            flux_max=info["flux_peak_ztfr"].data,
            filtid=2,  # r filter
        )

        # Normalize flux
        flux, flux_err = ZTFDataProcessor.process_flux_normalization(
            flux, flux_err, filt, flux_g_max, flux_r_max
        )

        # Create light curve data
        lc_early, lc_peak = ZTFDataProcessor.create_light_curve_data(
            phase, flux, flux_err, fcqfid, filt, t_g_early, t_r_early, beta=beta
        )

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid, t0_err=t0_err)


class ZTFIaEDR(SNLightCurve):
    """
    Yao et al. 2019
    127 ZTF SNe Ia with early light curves
    """

    edr_dir: str = "./data/ztf_snia_edr/"
    tab_info_path: str = "snia_data_basic_normal.csv"
    tab_lc_path: str = "ztf_early_Ia_lc_Yao2019.fit"

    def __init__(self, ztfid: str, early_threshold: float = 0.4) -> None:
        """
        Parameters
        ----------
        ztfid : str
            ZTF ID of the object.

        Returns
        -------
        None
        """

        tab_info = Table.read(self.edr_dir + self.tab_info_path)
        tab_lc = Table.read(self.edr_dir + self.tab_lc_path)
        tab_lc.add_column(
            np.int64(tab_lc["Field"]) * 10000
            + np.int64(tab_lc["CCD"]) * 100
            + np.int64(tab_lc["qID"]) * 10
            + np.int64(tab_lc["Filt"]),
            name="fcqfid",
        )

        info = tab_info[tab_info["name"] == ztfid]
        dat = tab_lc[tab_lc["ZTF"] == ztfid]

        # phase = dat["phase"].value.astype("<f4")
        zp = dat["ZP"].value.astype("<f4")
        flux = dat["Flux"].value.astype("<f4") / (10 ** (0.4 * zp))
        flux_err = dat["e_Flux"].value.astype("<f4") / (10 ** (0.4 * zp))

        t0 = info["t0_B_salt2"].value[0]
        t0_unc = info["t0_salt2_unc"].value[0]
        # t0_g = info["t0_g_adopted"].value[0]
        # t0_g_unc = info["t0_g_adopted_unc"].value[0]

        z = info["z_adopt"].value[0]

        phase = (dat["JD"].value - t0) / (1 + z)

        fcqfid = dat["fcqfid"].value
        filt = fcqfid % 10

        # max flux from SALT2 fit
        flux_g_max = info["fratio_gmax_2adam"].value[0]
        flux_r_max = info["fratio_rmax_2adam"].value[0]

        # Calculate 40% times and max flux from data
        t_g_early, flux_g_max = ZTFDataProcessor.calculate_early_times(
            phase,
            flux,
            flux_err,
            filt,
            early_threshold=early_threshold,
            flux_max=flux_g_max,
            filtid=1,  # g filter
        )
        t_r_early, flux_r_max = ZTFDataProcessor.calculate_early_times(
            phase,
            flux,
            flux_err,
            filt,
            early_threshold=early_threshold,
            flux_max=flux_r_max,
            filtid=2,  # r filter
        )

        # Normalize flux
        flux, flux_err = ZTFDataProcessor.process_flux_normalization(
            flux, flux_err, filt, flux_g_max, flux_r_max
        )

        # Create light curve data
        lc_early, lc_peak = ZTFDataProcessor.create_light_curve_data(
            phase, flux, flux_err, fcqfid, filt, t_g_early, t_r_early
        )

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid, t0_err=t0_unc)


@dataclass
class SampleConfig:
    source: str
    volume_complete: bool = False
    early_coverage: bool = False
    baseline_coverage: bool = False
    no_t0_err: bool = False
    x1_subsample: str | None = None
    sn_type: str = "normal"
    include_external_gr: bool = False
    external_noise_floor: float = 0.05

    def get_filename_suffix(self) -> str:
        suffix = ""
        if self.volume_complete:
            suffix += "_volume_complete"
        if self.early_coverage:
            suffix += "_early"
        if self.baseline_coverage:
            suffix += "_baseline"
        if self.no_t0_err:
            suffix += "_no_t0_err"
        if self.x1_subsample is not None:
            suffix += f"_{self.x1_subsample}"
        if self.sn_type != "normal":
            suffix += f"_{self.sn_type}"
        if self.include_external_gr:
            suffix += f"_external_gr_nf{int(round(self.external_noise_floor * 100))}"
        if self.source.lower() == "early_late":
            suffix += "_phot_ztf_norm"
        return suffix


class ZTFLib(SNLightCurveLib):
    def __init__(
        self,
        ztfid_lib: list = None,
        config: SampleConfig = None,
        early_threshold: float = 0.4,
        rise_model: str = "power_law",
        sampling_model: str = "hierarchical_mvn",
        pop_prior_config: str | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        ztfid_list : list
            List of ZTF IDs of the objects.
        source : str
            Source of the data: "EDR", "DR2", or "Early_Late".
        early_threshold : float
            Fraction of maximum luminosity to truncate light curves.
        rise_model : str
            Rise model: "power_law" or "curved_power_law".
        sampling_model : str
            Sampling model: "pooled", "unpooled", "hierarchical", "hierarchical_trise", or "hierarchical_mvn".
        pop_prior_config : str or None
            Stem of the YAML population prior config file, or None.
        x1_subsample : str or None
            Optional tag appended to the output filename to identify the x1
            subsample, e.g. ``"x1lo"`` or ``"x1hi"`` (default: None, no tag).
        sn_type : str
            SN type to fit (default: "normal"; options: "normal", "03fg"). Note: 03fg-like SNe are only available in DR2.

        Returns
        -------
        None
        """

        import os
        from pathlib import Path

        if ztfid_lib is None or config is None:
            super().__init__(sampling_model=sampling_model)
            return

        lc_early_lib = []
        lc_peak_lib = []

        t0_err_lib = []

        ztfid_lib_processed = []

        for ztfid in ztfid_lib:
            try:
                if config.source.lower() == "edr":
                    ztf_sn = ZTFIaEDR(ztfid=ztfid, early_threshold=early_threshold)
                elif config.source.lower() == "dr2":
                    ztf_sn = ZTFIaDR2(
                        ztfid=ztfid,
                        early_threshold=early_threshold,
                        sn_type=config.sn_type,
                    )
                elif config.source.lower() == "early_late":
                    ztf_sn = ZTFIaEarlyLate(
                        ztfid=ztfid,
                        early_threshold=early_threshold,
                        include_external_gr=config.include_external_gr,
                        external_noise_floor=config.external_noise_floor,
                    )
                else:
                    raise ValueError("Source must be 'EDR', 'DR2', or 'Early_Late'.")
                ztfid_lib_processed.append(ztfid)
            except ValueError as e:
                print(f"Skipping {ztfid} due to error: {e}")
                raise e
                # continue

            lc_early_lib.append(ztf_sn.lc_early)
            lc_peak_lib.append(ztf_sn.lc_peak)
            t0_err_lib.append(ztf_sn.t0_err)

        post_sample_dir = Path(
            f"./data/ztf_snia_{config.source.lower()}/results/frac{int(early_threshold * 100)}_{rise_model}"
        )
        filename = f"post_sample_{sampling_model}{config.get_filename_suffix()}.nc"
        if pop_prior_config:
            filename = filename.replace(
                sampling_model, f"{sampling_model}_pop_prior_{pop_prior_config}"
            )
        post_sample_file = post_sample_dir / filename

        if os.path.exists(post_sample_file):
            print("Loading existing .nc file...")
            post_sample = xr.load_dataset(post_sample_file)
        else:
            post_sample = None

        super().__init__(
            lc_early_lib=lc_early_lib,
            lc_peak_lib=lc_peak_lib,
            ztfid_lib=ztfid_lib_processed,
            t0_err=None if config.no_t0_err else t0_err_lib,
            sampling_model=sampling_model,
            **kwargs,
        )
        if post_sample is not None:
            expected_dims = {
                "obj": len(self.lc_library),
                "filt": self.n_filt_global,
                "fcqfid": len(np.unique(self.idx_fcqfid)),
            }
            mismatches = []
            for dim, expected in expected_dims.items():
                if dim in post_sample.sizes and post_sample.sizes[dim] != expected:
                    mismatches.append(
                        f"{dim}: file={post_sample.sizes[dim]}, current={expected}"
                    )
            if mismatches:
                print(
                    f"Ignoring incompatible posterior file {post_sample_file} "
                    f"({'; '.join(mismatches)})."
                )
                post_sample = None
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
        if self.post_sample is not None:
            self.decode_post_sample()
