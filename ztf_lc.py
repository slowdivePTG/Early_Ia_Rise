import glob
import numpy as np

from astropy.table import Table
from fit_early_lc import SNLightCurve, SNLightCurveLib
from _utils import data_binning


class ZTFDataProcessor:
    """Helper class to handle common ZTF data processing operations."""

    @staticmethod
    def process_flux_normalization(flux, flux_err, filt, flux_g_max, flux_r_max):
        """Normalize flux data by maximum flux values."""
        flux = flux.copy()
        flux_err = flux_err.copy()

        flux[filt == 1] /= flux_g_max / 100
        flux_err[filt == 1] /= flux_g_max / 100
        flux[filt == 2] /= flux_r_max / 100
        flux_err[filt == 2] /= flux_r_max / 100

        return flux, flux_err

    @staticmethod
    def calculate_40_percent_times(
        phase, flux, flux_err, filt, flux_g_max=None, flux_r_max=None
    ):
        """Calculate 40% flux times for g and r bands."""
        t_g, f_g, _ = data_binning(
            np.array([phase, flux, flux_err]).T[filt == 1], 0.5
        ).T
        t_r, f_r, _ = data_binning(
            np.array([phase, flux, flux_err]).T[filt == 2], 0.5
        ).T

        if flux_g_max is None:
            flux_g_max = np.max(f_g[np.abs(t_g) < 5])
        if flux_r_max is None:
            flux_r_max = np.max(f_r[np.abs(t_r) < 5])

        t_g_40 = t_g[np.where((f_g <= 0.4 * flux_g_max) & (t_g < -5))[0][-1]] + 0.25
        t_r_40 = t_r[np.where((f_r <= 0.4 * flux_r_max) & (t_r < -5))[0][-1]] + 0.25

        return t_g_40, t_r_40, flux_g_max, flux_r_max

    @staticmethod
    def create_light_curve_data(phase, flux, flux_err, fcqfid, filt, t_g_40, t_r_40):
        """Create early and peak light curve dictionaries."""
        # Filter out observations < 40% of max flux
        idx_i = filt == 3
        idx_rise = (phase < 0) & (phase > -100) & ~idx_i
        idx_g = (filt == 1) & (phase < t_g_40)
        idx_r = (filt == 2) & (phase < t_r_40)
        idx = idx_rise & (idx_g | idx_r)

        lc_early = {
            "phase": phase[idx],
            "flux": flux[idx],
            "flux_err": flux_err[idx],
            "fcqfid": fcqfid[idx],
            "filt": filt[idx],
        }

        lc_peak = {
            "phase": phase[idx_rise],
            "flux": flux[idx_rise],
            "flux_err": flux_err[idx_rise],
            "fcqfid": fcqfid[idx_rise],
            "filt": filt[idx_rise],
        }

        return lc_early, lc_peak


class ZTFIaDR2(SNLightCurve):
    """
    Rigault et al. 2025
    ZTF SN Ia DR2
    """

    dr2_dir: str = "./Data/ztf_snia_dr2/"
    tab_info_path: str = "tables/snia_data.csv"
    tab_lc_path: str = "lightcurves/*lc.csv"

    def __init__(self, ztfid: str) -> None:
        """
        Initialize the class instance.
        """
        tab_info = Table.read(self.dr2_dir + self.tab_info_path)
        lst_lc = sorted(glob.glob(self.dr2_dir + self.tab_lc_path))
        lst_ztdif = [lc.split("/")[-1].split("_")[0] for lc in lst_lc]
        if ztfid not in lst_ztdif:
            raise ValueError(f"ZTF ID {ztfid} not found in DR2 light curves.")
        tab_lc = Table.read(lst_lc[lst_ztdif.index(ztfid)], format="ascii", comment="#")

        # Prepare filter id and fcqf id
        tab_lc.add_column(
            np.where(
                tab_lc["filter"] == "ztfg",
                1,
                np.where(tab_lc["filter"] == "ztfr", 2, 3),
            ),
            name="filter_id",
        )
        # RCID = 4(CCDID – 1) + QID – 1
        tab_lc.add_column(
            np.int64(tab_lc["field_id"]) * 10000
            + np.int64(tab_lc["rcid"] // 4 + 1) * 100  # CCDID
            + np.int64(tab_lc["rcid"] % 4 + 1) * 10  # QID
            + np.int64(tab_lc["filter_id"]),
            name="fcqfid",
        )

        info = tab_info[tab_info["ztfname"] == ztfid]
        # bad photometry: Rigault et al. 2025
        bad_bitmask = [0, 1, 2, 3, 4]
        mask = (
            np.bitwise_and(tab_lc["flag"].data[:, None], bad_bitmask).sum(axis=1) == 0
        )
        data = tab_lc[mask]

        t0 = info["t0"].data[0]
        z = info["redshift"].data[0]
        flux = data["flux"].data.astype("<f4")
        flux_err = data["flux_err"].data.astype("<f4")
        phase = (data["mjd"].data - t0) / (1 + z)

        fcqfid = data["fcqfid"].data
        filt = data["filter_id"].data

        # Calculate 40% times and max flux from data
        t_g_40, t_r_40, flux_g_max, flux_r_max = (
            ZTFDataProcessor.calculate_40_percent_times(phase, flux, flux_err, filt)
        )

        # Normalize flux
        flux, flux_err = ZTFDataProcessor.process_flux_normalization(
            flux, flux_err, filt, flux_g_max, flux_r_max
        )

        # Create light curve data
        lc_early, lc_peak = ZTFDataProcessor.create_light_curve_data(
            phase, flux, flux_err, fcqfid, filt, t_g_40, t_r_40
        )

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid)


class ZTFIaEDR(SNLightCurve):
    """
    Yao et al. 2019
    127 ZTF SNe Ia with early light curves
    """

    edr_dir: str = "./Data/ztf_snia_edr/"
    tab_info_path: str = "Nobs_cut_salt2_spec_subtype_pec.csv"
    tab_lc_path: str = "ztf_early_Ia_lc_Yao2019.fit"

    def __init__(self, ztfid: str) -> None:
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

        self.t0_g = info["t0_g_adopted"].value[0]
        self.t0_B = info["t0_B_salt2"].value[0]
        self.t0_g_unc = info["t0_g_adopted_unc"].value[0]
        self.t0_B_unc = info["t0_salt2_unc"].value[0]

        z = info["z_adopt"].value[0]

        phase = (dat["JD"].value - self.t0_B) / (1 + z)

        fcqfid = dat["fcqfid"].value
        filt = fcqfid % 10

        # max flux from SALT2 fit
        flux_g_max = info["fratio_gmax_2adam"].value[0]
        flux_r_max = info["fratio_rmax_2adam"].value[0]

        # Calculate 40% times and max flux from data
        t_g_40, t_r_40, _, _ = ZTFDataProcessor.calculate_40_percent_times(
            phase, flux, flux_err, filt, flux_g_max=flux_g_max, flux_r_max=flux_r_max
        )

        # Normalize flux
        flux, flux_err = ZTFDataProcessor.process_flux_normalization(
            flux, flux_err, filt, flux_g_max, flux_r_max
        )

        # Create light curve data
        lc_early, lc_peak = ZTFDataProcessor.create_light_curve_data(
            phase, flux, flux_err, fcqfid, filt, t_g_40, t_r_40
        )

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid)


class ZTFLib(SNLightCurveLib):
    def __init__(self, ztfid_lib: list, source: str) -> None:
        """
        Parameters
        ----------
        ztfid_list : list
            List of ZTF IDs of the objects.
        source : str
            Source of the data, either "EDR" or "DR2".

        Returns
        -------
        None
        """

        lc_early_lib = []
        lc_peak_lib = []

        for ztfid in ztfid_lib:
            if source in ["EDR", "edr"]:
                ztf_sn = ZTFIaEDR(ztfid=ztfid)
            elif source in ["DR2", "dr2"]:
                ztf_sn = ZTFIaDR2(ztfid=ztfid)
            else:
                raise ValueError("Source must be 'EDR' or 'DR2'.")
            lc_early_lib.append(ztf_sn.lc_early)
            lc_peak_lib.append(ztf_sn.lc_peak)

        self.ztfid_lib = ztfid_lib
        super().__init__(
            lc_early_lib=lc_early_lib, lc_peak_lib=lc_peak_lib, ztfid_lib=ztfid_lib
        )
