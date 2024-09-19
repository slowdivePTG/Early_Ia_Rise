import numpy as np
from fit_early_lc import Ia_lc

class ZTF_SN_Ia(Ia_lc):

    def __init__(self, tab_info, tab_lc, ztfid: str) -> None:
        """
        Initialize the class instance.

        Parameters
        ----------
        tab_info : astropy.table.Table
            DataFrame containing information about the object.
        tab_lc : astropy.table.Table
            DataFrame containing light curve data.
        ztfid : str
            ZTF ID of the object.

        Returns
        -------
        None
        """

        info = tab_info[tab_info["name"] == ztfid]
        dat = tab_lc[tab_lc["ZTF"] == ztfid]

        # phase = dat["phase"].value.astype("<f4")
        ZP = dat["ZP"].value.astype("<f4")
        flux = dat["Flux"].value.astype("<f4") / (10 ** (0.4 * ZP))
        flux_err = dat["e_Flux"].value.astype("<f4") / (10 ** (0.4 * ZP))

        self.t0_g = info["t0_g_adopted"].value[0]
        self.t0_B = info["t0_B_salt2"].value[0]
        self.t0_g_unc = info["t0_g_adopted_unc"].value[0]
        self.t0_B_unc = info["t0_salt2_unc"].value[0]

        z = info["z_adopt"].value[0]

        phase = (dat["JD"].value - self.t0_B) / (1 + z)

        fcqfid = dat["fcqfid"].value
        filt = fcqfid % 10

        from spec_tool.data_binning import data_binning

        t_g, f_g, _ = data_binning(np.array([phase, flux, flux_err]).T[filt == 1], 0.5).T
        t_r, f_r, _ = data_binning(np.array([phase, flux, flux_err]).T[filt == 2], 0.5).T

        # max flux from data
        # flux_g_max = np.max(f_g)
        # flux_r_max = np.max(f_r)

        # max flux from SALT2 fit
        flux_g_max = info["fratio_gmax_2adam"].value[0]
        flux_r_max = info["fratio_rmax_2adam"].value[0]

        t_g_40 = t_g[np.where((f_g < 0.4 * flux_g_max) & (t_g < 0))[0][-1]] + 0.25
        t_r_40 = t_r[np.where((f_r < 0.4 * flux_r_max) & (t_r < 0))[0][-1]] + 0.25

        # normalization
        flux[filt == 1] /= flux_g_max / 100
        flux_err[filt == 1] /= flux_g_max / 100
        flux[filt == 2] /= flux_r_max / 100
        flux_err[filt == 2] /= flux_r_max / 100

        # filter out observations < 40% of max flux
        idx_rise = (phase < 0) & (phase > -100)
        idx_g = (filt == 1) & (phase < t_g_40)
        idx_r = (filt == 2) & (phase < t_r_40)
        idx = idx_rise & (idx_g | idx_r)

        # observations between 40% and 100% of max flux
        lc_early = {
            "phase": phase[idx],
            "flux": flux[idx],
            "flux_err": flux_err[idx],
            "fcqfid": fcqfid[idx],
            "filt": filt[idx],
        }

        # observations between -100 days and peak
        lc_peak = {
            "phase": phase[idx_rise],
            "flux": flux[idx_rise],
            "flux_err": flux_err[idx_rise],
            "fcqfid": fcqfid[idx_rise],
            "filt": filt[idx_rise],
        }

        super().__init__(lc_early=lc_early, lc_peak=lc_peak, ztfid=ztfid)