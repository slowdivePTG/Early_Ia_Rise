from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
from astropy.table import Table
from scipy.optimize import least_squares


DATA_DIR = Path("data/ztf_snia_early_late")
SALT2_SOURCE_NAME = "salt2"
SALT2_SOURCE_VERSION = "T21"
SALT2_LOCAL_DIR = DATA_DIR / "salt2_models" / "salt2-T21"
DIAG_DIR = DATA_DIR / "salt2_diagnostics"
MCMC_CHAIN_DIR = DATA_DIR / "salt_mcmc_chains"
EXTERNAL_DIR = DATA_DIR / "light_curve_external"
LOCAL_ZTF_FILTER_DIR = DATA_DIR / "sncosmo_filters" / "ztf"
LOCAL_SDSS_FILTER_DIR = DATA_DIR / "sncosmo_filters" / "sdss"
LOCAL_PS1_FILTER_DIR = DATA_DIR / "sncosmo_filters" / "ps1"
LOCAL_BAYESN_FILTER_DIR = DATA_DIR / "sncosmo_filters" / "bayesn"
MODEL_PHASE_MIN = -10.0
MODEL_PHASE_MAX = 40.0
MIN_OBS_PER_FILTER = 3
SALT2_PEAK_FLUX_BANDS = {
    "ztfg": "ztfg",
    "ztfr": "ztfr",
    "sdssg_AB": "sdssg",
    "sdssr_AB": "sdssr",
    "g_PS1": "ps1::g",
    "r_PS1": "ps1::r",
    "g_CSP2_AB": "swope2g",
    "r_CSP2_AB": "swope2r",
}
SALT2_X1_ROUND1_BOUNDS = (-3.0, 3.0)
SALT2_X1_FINAL_BOUNDS = (-3.0, 10.0)
SALT2_X1_MCMC_BOUNDS = (-5.0, 10.0)
SALT2_C_BOUNDS = (-1.0, 2.0)
SALT2_OUTLIER_SIGMA = 30.0
SALT2_OUTLIER_MAX_ITER = 2
EXCLUDED_FILTERS = {"swope2u", "sdssu", "ps1::u", "bessellu", "u"}
SALT2_ALLOWED_FILTERS = {
    "ztfg", "ztfr", "ztfi",
    "sdssg", "sdssr", "sdssi",
    "ps1::g", "ps1::r", "ps1::i",
    "swope2g", "swope2r", "swope2i",
}


def log(message: str) -> None:
    """Print a timestamped progress message to the terminal."""

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def parse_ztf_lc(filename: Path) -> pd.DataFrame:
    """Parse one ZTF forced-photometry file into sncosmo-compatible flux rows."""

    error_floor = {"ZTF_g": 0.025, "ZTF_r": 0.035, "ZTF_i": 0.06}

    lc = pd.read_csv(filename, sep=r"\s+", comment="#")
    lc = lc.rename(columns={key: key.replace(",", "") for key in lc.columns})

    lc_dat = lc[
        np.isfinite(lc["forcediffimflux"])
        & (lc["forcediffimfluxunc"] > 0)
        & (lc["infobitssci"] <= 33554432)
    ].copy()

    lc_dat["mjd"] = lc_dat["jd"] - 2400000.5
    lc_dat["zp"] = 30.0
    lc_dat["magsys"] = "ab"

    delta_zp = lc_dat["zpdiff"] - lc_dat["zp"]
    lc_dat["flux"] = lc_dat["forcediffimflux"] * 10 ** (-0.4 * delta_zp)
    floor = lc_dat["filter"].map(error_floor).fillna(0.0).to_numpy()
    lc_dat["fluxerr"] = np.sqrt(
        lc_dat["forcediffimfluxunc"] ** 2 + (floor * lc_dat["forcediffimflux"]) ** 2
    ) * 10 ** (-0.4 * delta_zp)

    filter_map = {"ZTF_g": "ztfg", "ZTF_r": "ztfr", "ZTF_i": "ztfi"}
    lc_dat["filter"] = lc_dat["filter"].replace(filter_map)

    return lc_dat[["mjd", "filter", "flux", "fluxerr", "zp", "magsys"]]


def parse_external_lc(objid: str) -> pd.DataFrame:
    """Load normalized external photometry for one object when available."""

    filename = EXTERNAL_DIR / f"{objid}_external.csv"
    columns = ["mjd", "filter", "flux", "fluxerr", "zp", "magsys"]
    if not filename.exists():
        return pd.DataFrame(columns=columns)

    lc = pd.read_csv(filename)
    lc = lc[np.isfinite(lc["flux"]) & np.isfinite(lc["fluxerr"]) & (lc["fluxerr"] > 0)].copy()
    before = len(lc)
    lc = lc[~lc["filter"].isin(EXCLUDED_FILTERS)].copy()
    if before > len(lc):
        log(f"{objid}: excluded {before - len(lc)} external u-band rows from SALT2 input")
    if not lc.empty:
        counts = lc.groupby(["source", "raw_filter", "filter"]).size().to_dict()
        log(f"{objid}: loaded external photometry from {filename} ({len(lc)} rows; {counts})")
    return lc[columns]


def register_external_filters() -> None:
    """Register local sncosmo filters needed by normalized external photometry."""

    register_local_ztf_filters()
    try:
        sys.path.insert(0, str(EXTERNAL_DIR))
        from register_bayesn_sncosmo_filters import register_bayesn_sncosmo_filters
    except ImportError as exc:
        log(f"Could not import external filter registration helper: {exc}")
        return
    try:
        manifest = register_bayesn_sncosmo_filters()
    except FileNotFoundError:
        log("No local BayeSN-derived sncosmo filter manifest found; skipping external filter registration")
        register_local_swope_aliases()
        return
    log(f"Registered {len(manifest)} local sncosmo filters for external photometry")


def register_local_bandpass(name: str, path: Path) -> None:
    """Register a local two-column wavelength/transmission file as a sncosmo bandpass."""

    if not path.exists():
        raise FileNotFoundError(f"Missing local sncosmo filter file: {path}")
    wave, trans = np.loadtxt(path, unpack=True)
    sncosmo.registry.register(sncosmo.Bandpass(wave, trans, name=name), name=name, force=True)


def register_local_swope_aliases() -> None:
    """Register normalized Swope filter aliases from local files."""

    aliases = {
        "swope2g": LOCAL_BAYESN_FILTER_DIR / "swope2g.dat",
        "swope2r": LOCAL_BAYESN_FILTER_DIR / "swope2r.dat",
        "swope2i": LOCAL_BAYESN_FILTER_DIR / "swope2i.dat",
    }
    for name, path in aliases.items():
        register_local_bandpass(name, path)
    log(f"Registered {len(aliases)} local Swope sncosmo aliases")


def register_local_ztf_filters() -> None:
    """Register local g/r/i passbands to avoid sncosmo network downloads."""

    filter_files = {
        "ztfg": LOCAL_ZTF_FILTER_DIR / "ztfg.dat",
        "ztfr": LOCAL_ZTF_FILTER_DIR / "ztfr.dat",
        "ztfi": LOCAL_ZTF_FILTER_DIR / "ztfi.dat",
        "sdssg": LOCAL_SDSS_FILTER_DIR / "sdssg.dat",
        "sdssr": LOCAL_SDSS_FILTER_DIR / "sdssr.dat",
        "sdssi": LOCAL_SDSS_FILTER_DIR / "sdssi.dat",
        "ps1::g": LOCAL_PS1_FILTER_DIR / "ps1_g.dat",
        "ps1::r": LOCAL_PS1_FILTER_DIR / "ps1_r.dat",
        "ps1::i": LOCAL_PS1_FILTER_DIR / "ps1_i.dat",
    }
    for name, path in filter_files.items():
        register_local_bandpass(name, path)
    log(f"Registered {len(filter_files)} local g/r/i sncosmo filters")


def drop_sparse_filters_after_phase_cut(sn: Table, objid: str) -> Table:
    """Drop sparse external filters while retaining all ZTF filters."""

    filters = np.asarray(sn["filter"])
    if "_survey" in sn.colnames:
        survey = np.asarray(sn["_survey"]).astype(str)
        external = survey != "ZTF"
        if not np.any(external):
            return sn
        counts = pd.Series(filters[external]).value_counts()
    else:
        survey = np.full(len(sn), "external")
        counts = pd.Series(filters).value_counts()
    dropped = counts[counts < MIN_OBS_PER_FILTER]
    if dropped.empty:
        return sn

    dropped_msg = ", ".join(f"{band}={count}" for band, count in dropped.items())
    log(f"{objid}: dropping external SALT2 filters with <{MIN_OBS_PER_FILTER} points after phase cut: {dropped_msg}")
    keep = (survey == "ZTF") | ~np.isin(filters, dropped.index.to_numpy())
    return sn[keep]


def keep_salt2_gri_filters(lc: pd.DataFrame, objid: str) -> pd.DataFrame:
    """Restrict SALT2 fitting photometry to g/r/i-like filters only."""

    before = len(lc)
    if before == 0:
        return lc
    dropped = lc.loc[~lc["filter"].isin(SALT2_ALLOWED_FILTERS), "filter"].value_counts()
    lc = lc[lc["filter"].isin(SALT2_ALLOWED_FILTERS)].copy()
    if not dropped.empty:
        dropped_msg = ", ".join(f"{band}={count}" for band, count in dropped.items())
        log(f"{objid}: excluding non-gri filters from SALT2 input: {dropped_msg}")
    return lc


def select_salt2_modeling_photometry(sn_tot_raw: Table, z: float, t0_ref: float, objid: str) -> Table:
    """Apply the SALT2 phase and sparse-filter cuts for a reference peak date."""

    sn = sn_tot_raw[
        (sn_tot_raw["mjd"] > t0_ref + MODEL_PHASE_MIN * (1 + z))
        & (sn_tot_raw["mjd"] < t0_ref + MODEL_PHASE_MAX * (1 + z))
    ]
    sn = drop_sparse_filters_after_phase_cut(sn, objid)
    if len(sn) == 0:
        raise ValueError("No photometry remains after phase and sparse-filter cuts.")
    return sn


def drop_salt2_flux_outliers(sn: Table, model: sncosmo.Model, objid: str) -> Table:
    """Drop catastrophic SALT2 flux residual outliers while preserving filter viability."""

    pred = model.bandflux(sn["filter"], sn["mjd"], zp=sn["zp"], zpsys=sn["magsys"])
    flux = np.asarray(sn["flux"], dtype=float)
    fluxerr = np.asarray(sn["fluxerr"], dtype=float)
    resid_sigma = (flux - pred) / fluxerr
    bad = np.isfinite(resid_sigma) & (np.abs(resid_sigma) > SALT2_OUTLIER_SIGMA)
    if not np.any(bad):
        return sn

    details = []
    for idx in np.flatnonzero(bad):
        details.append(
            f"{sn['filter'][idx]}@{float(sn['mjd'][idx]):.5f}={resid_sigma[idx]:.1f}sigma"
        )
    log(f"{objid}: dropping SALT2 flux outliers >{SALT2_OUTLIER_SIGMA:g} sigma: {', '.join(details)}")
    sn = sn[~bad]
    return drop_sparse_filters_after_phase_cut(sn, objid)


def load_sample() -> Table:
    """Load the observed early/late SN Ia sample after removing unobserved targets."""

    ztf_early = Table.read(DATA_DIR / "ztf_early_Ia.csv", format="ascii.csv")
    return ztf_early[ztf_early["not_obs"] != 1]


def make_salt2_source():
    """Return the explicitly selected local T21 SALT2 source."""

    if SALT2_LOCAL_DIR.exists():
        source = sncosmo.SALT2Source(str(SALT2_LOCAL_DIR), name=SALT2_SOURCE_NAME, version=SALT2_SOURCE_VERSION)
        source_label = str(SALT2_LOCAL_DIR)
    else:
        source = sncosmo.get_source(SALT2_SOURCE_NAME, version=SALT2_SOURCE_VERSION)
        source_label = f"{SALT2_SOURCE_NAME} version {SALT2_SOURCE_VERSION} from sncosmo registry"
    log(
        f"Using SALT2 source '{source_label}' "
        f"with wavelength range {source.minwave():.1f}-{source.maxwave():.1f} A"
    )
    return source


def _as_lc_dataframe(lc: Table | pd.DataFrame) -> pd.DataFrame:
    """Convert an astropy table or pandas DataFrame light curve to a DataFrame."""

    if isinstance(lc, pd.DataFrame):
        return lc.copy()
    return lc.to_pandas()


def _estimate_x0(lc: pd.DataFrame, model: sncosmo.Model, t0: float) -> float:
    """Estimate a positive SALT2 x0 amplitude by weighted linear projection."""

    model.set(t0=t0, x0=1.0, x1=0.0, c=0.0)
    pred = model.bandflux(
        lc["filter"].to_numpy(),
        lc["mjd"].to_numpy(),
        zp=lc["zp"].to_numpy(),
        zpsys=lc["magsys"].to_numpy(),
    )
    good = np.isfinite(pred) & np.isfinite(lc["flux"]) & np.isfinite(lc["fluxerr"]) & (lc["fluxerr"] > 0)
    if not np.any(good):
        return 1e-3
    w = 1.0 / lc.loc[good, "fluxerr"].to_numpy() ** 2
    numerator = np.sum(lc.loc[good, "flux"].to_numpy() * pred[good] * w)
    denominator = np.sum(pred[good] ** 2 * w)
    if denominator <= 0 or not np.isfinite(numerator):
        return 1e-3
    return float(np.clip(numerator / denominator, 1e-12, 10.0))


def fit_salt2_lc(
    lc: Table | pd.DataFrame,
    model: sncosmo.Model,
    t0_bounds: tuple[float, float],
    x1_bounds: tuple[float, float],
    t0_init: float | None = None,
) -> tuple[dict, sncosmo.Model]:
    """Fit SALT2 parameters with scipy least squares to initialize MCMC walkers."""

    lc_df = _as_lc_dataframe(lc)
    lc_df = lc_df[np.isfinite(lc_df["flux"]) & np.isfinite(lc_df["fluxerr"]) & (lc_df["fluxerr"] > 0)]
    if lc_df.empty:
        raise ValueError("No finite photometry available for SALT2 fit.")

    if t0_init is None:
        t0_init = lc_df.loc[lc_df["flux"].idxmax(), "mjd"]
    t0_init = float(np.clip(t0_init, t0_bounds[0], t0_bounds[1]))
    x0_init = _estimate_x0(lc_df, model, t0_init)

    def residual(theta):
        """Return normalized photometric residuals for least-squares fitting."""

        t0, log_x0, x1, color = theta
        model.set(t0=t0, x0=np.exp(log_x0), x1=x1, c=color)
        pred = model.bandflux(
            lc_df["filter"].to_numpy(),
            lc_df["mjd"].to_numpy(),
            zp=lc_df["zp"].to_numpy(),
            zpsys=lc_df["magsys"].to_numpy(),
        )
        return (pred - lc_df["flux"].to_numpy()) / lc_df["fluxerr"].to_numpy()

    p0 = np.array([t0_init, np.log(x0_init), 0.0, 0.0])
    bounds = (
        np.array([t0_bounds[0], np.log(1e-12), x1_bounds[0], SALT2_C_BOUNDS[0]]),
        np.array([t0_bounds[1], np.log(10.0), x1_bounds[1], SALT2_C_BOUNDS[1]]),
    )
    opt = least_squares(residual, p0, bounds=bounds, max_nfev=10000)
    t0, log_x0, x1, color = opt.x
    x0 = float(np.exp(log_x0))
    model.set(t0=t0, x0=x0, x1=x1, c=color)

    errors = {"t0": np.nan, "x0": np.nan, "x1": np.nan, "c": np.nan}
    ndof = max(len(lc_df) - len(opt.x), 1)
    try:
        cov = np.linalg.inv(opt.jac.T @ opt.jac) * (2 * opt.cost / ndof)
        diag = np.sqrt(np.diag(cov))
        errors = {"t0": diag[0], "x0": x0 * diag[1], "x1": diag[2], "c": diag[3]}
    except np.linalg.LinAlgError:
        pass

    result = {
        "success": bool(opt.success),
        "message": opt.message,
        "chisq": float(2 * opt.cost),
        "ndof": ndof,
        "param_names": model.param_names,
        "parameters": np.array([model.get(name) for name in model.param_names]),
        "errors": errors,
    }
    return result, model


def _sample_summary(samples: np.ndarray, names: list[str]) -> dict:
    """Summarize posterior samples with median and 16th/84th percentile errors."""

    summary = {}
    for i, name in enumerate(names):
        values = samples[:, i]
        p16, p50, p84 = np.nanpercentile(values, [16, 50, 84])
        summary[f"{name}_median"] = p50
        summary[f"{name}_err_minus"] = p50 - p16
        summary[f"{name}_err_plus"] = p84 - p50
    return summary


def run_salt2_mcmc(
    objid: str,
    sn_tot: Table,
    fitted_model: sncosmo.Model,
    nwalkers: int,
    nburn: int,
    nsamples: int,
    thin: int,
    t0_window: float,
) -> tuple[dict, sncosmo.Model, dict]:
    """Run sncosmo's emcee SALT2 sampler and persist raw posterior samples."""

    t0 = fitted_model.get("t0")
    x0 = fitted_model.get("x0")
    bounds = {
        "t0": (t0 - t0_window, t0 + t0_window),
        "x0": (max(x0 / 100.0, 1e-12), min(x0 * 100.0, 10.0)),
        "x1": SALT2_X1_MCMC_BOUNDS,
        "c": SALT2_C_BOUNDS,
    }
    log(
        f"{objid}: starting SALT2 MCMC "
        f"(nwalkers={nwalkers}, nburn={nburn}, nsamples={nsamples}, thin={thin})"
    )
    res, model = sncosmo.mcmc_lc(
        sn_tot,
        fitted_model,
        ["t0", "x0", "x1", "c"],
        bounds=bounds,
        guess_amplitude=False,
        guess_t0=False,
        nwalkers=nwalkers,
        nburn=nburn,
        nsamples=nsamples,
        thin=thin,
    )

    MCMC_CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MCMC_CHAIN_DIR / f"{objid}_salt2_mcmc.npz",
        samples=res.samples,
        vparam_names=np.array(res.vparam_names),
        covariance=res.covariance,
        parameters=res.parameters,
        param_names=np.array(res.param_names),
        data_mask=res.data_mask,
    )
    summary = _sample_summary(res.samples, list(res.vparam_names))
    summary["mean_acceptance_fraction"] = res.mean_acceptance_fraction
    summary["mcmc_ndof"] = res.ndof
    log(
        f"{objid}: finished SALT2 MCMC "
        f"(acceptance={res.mean_acceptance_fraction:.3f}, samples={len(res.samples)})"
    )
    log(f"{objid}: saved SALT2 MCMC chain to {MCMC_CHAIN_DIR / f'{objid}_salt2_mcmc.npz'}")
    return res, model, summary


def fit_one(
    objid: str,
    sn_info: pd.DataFrame,
    t21_source,
    save_diagnostics: bool,
    run_mcmc: bool,
    nwalkers: int,
    nburn: int,
    nsamples: int,
    thin: int,
    t0_window: float,
    use_ztf: bool,
    use_external: bool,
) -> dict:
    """Fit one SN with SALT2 and return the final parameter summary.

    A deterministic least-squares fit is always run first to initialize the MCMC
    walkers, but those initialization values are not written to the output row.
    """

    log(f"{objid}: starting SALT2 fit")
    sn_ztf = parse_ztf_lc(DATA_DIR / "light_curve_fps_ztf" / f"{objid}_fnu.csv") if use_ztf else pd.DataFrame()
    sn_external = parse_external_lc(objid) if use_external else pd.DataFrame()
    if use_external and sn_external.empty:
        raise ValueError("External photometry requested but none is available after cuts.")
    if not sn_ztf.empty:
        sn_ztf = sn_ztf.copy()
        sn_ztf["_survey"] = "ZTF"
    if not sn_external.empty:
        sn_external = sn_external.copy()
        sn_external["_survey"] = "external"
    sn_tot_df = pd.concat([sn_ztf, sn_external], ignore_index=True)
    sn_tot_df = keep_salt2_gri_filters(sn_tot_df, objid)
    if sn_tot_df.empty:
        raise ValueError("No g/r/i photometry available for SALT2 fit.")
    sn_tot_raw = Table.from_pandas(sn_tot_df)
    log(
        f"{objid}: loaded photometry "
        f"(ZTF={len(sn_ztf)}, external={len(sn_external)}, SALT2_gri={len(sn_tot_raw)})"
    )

    info_idx = sn_info.objid == objid
    if not np.any(info_idx):
        raise ValueError(f"No metadata row found for {objid}")
    z = sn_info.loc[info_idx, "z"].values[0]
    log(f"{objid}: metadata z={z:.6f}")

    t_peak_obs = sn_tot_df["mjd"].values[np.argmax(sn_tot_df["flux"])]
    sn_round1 = sn_tot_raw[
        (sn_tot_raw["mjd"] > t_peak_obs - 20) & (sn_tot_raw["mjd"] < t_peak_obs + 50)
    ]

    model_r1 = sncosmo.Model(source=t21_source)
    model_r1.set(z=z)
    log(f"{objid}: starting SALT2 round 1 deterministic fit with {len(sn_round1)} points")
    res1, _ = fit_salt2_lc(
        sn_round1,
        model_r1,
        t0_bounds=(sn_round1["mjd"].min(), sn_round1["mjd"].max()),
        x1_bounds=SALT2_X1_ROUND1_BOUNDS,
    )
    t0_guess = res1["parameters"][1]
    log(f"{objid}: round 1 complete, t0_guess={t0_guess:.5f}")

    sn_tot = select_salt2_modeling_photometry(sn_tot_raw, z, t0_guess, objid)
    model_r2 = sncosmo.Model(source=t21_source)
    model_r2.set(z=z, t0=t0_guess)
    log(f"{objid}: starting SALT2 round 2 deterministic fit with {len(sn_tot)} points")
    result, fitted_model = fit_salt2_lc(
        sn_tot,
        model_r2,
        t0_bounds=(sn_tot["mjd"].min(), sn_tot["mjd"].max()),
        x1_bounds=SALT2_X1_FINAL_BOUNDS,
        t0_init=t0_guess,
    )

    final_sn_tot = select_salt2_modeling_photometry(sn_tot_raw, z, result["parameters"][1], objid)
    if len(final_sn_tot) != len(sn_tot) or set(final_sn_tot["filter"]) != set(sn_tot["filter"]):
        log(
            f"{objid}: refitting SALT2 after final-t0 phase/sparse-filter selection "
            f"({len(sn_tot)} -> {len(final_sn_tot)} points)"
        )
        sn_tot = final_sn_tot
        model_r2 = sncosmo.Model(source=t21_source)
        model_r2.set(z=z, t0=result["parameters"][1])
        result, fitted_model = fit_salt2_lc(
            sn_tot,
            model_r2,
            t0_bounds=(sn_tot["mjd"].min(), sn_tot["mjd"].max()),
            x1_bounds=SALT2_X1_FINAL_BOUNDS,
            t0_init=result["parameters"][1],
        )

    for iteration in range(1, SALT2_OUTLIER_MAX_ITER + 1):
        clipped_sn_tot = drop_salt2_flux_outliers(sn_tot, fitted_model, objid)
        if len(clipped_sn_tot) == len(sn_tot):
            break
        log(
            f"{objid}: refitting SALT2 after outlier clipping iteration {iteration} "
            f"({len(sn_tot)} -> {len(clipped_sn_tot)} points)"
        )
        sn_tot = clipped_sn_tot
        model_r2 = sncosmo.Model(source=t21_source)
        model_r2.set(z=z, t0=result["parameters"][1])
        result, fitted_model = fit_salt2_lc(
            sn_tot,
            model_r2,
            t0_bounds=(sn_tot["mjd"].min(), sn_tot["mjd"].max()),
            x1_bounds=SALT2_X1_FINAL_BOUNDS,
            t0_init=result["parameters"][1],
        )
    log(
        f"{objid}: round 2 complete, "
        f"t0={result['parameters'][1]:.5f}, x0={result['parameters'][2]:.6g}, "
        f"x1={result['parameters'][3]:.4f}, c={result['parameters'][4]:.4f}, "
        f"chisq/ndof={result.get('chisq', np.nan):.2f}/{result.get('ndof', np.nan)}"
    )

    mcmc_summary = {}
    sampler = "least_squares"
    if run_mcmc:
        result, fitted_model, mcmc_summary = run_salt2_mcmc(
            objid,
            sn_tot,
            fitted_model,
            nwalkers=nwalkers,
            nburn=nburn,
            nsamples=nsamples,
            thin=thin,
            t0_window=t0_window,
        )
        sampler = "emcee_ensemble"
    else:
        log(f"{objid}: skipping SALT2 MCMC (--no-mcmc)")

    t_max = result["parameters"][1]
    t_max_err = result["errors"].get("t0", np.nan)
    sn_tot["phase"] = (sn_tot["mjd"] - t_max) / (1 + z)

    peak_flux = {}
    peak_mag = {}
    for column_band, sncosmo_band in SALT2_PEAK_FLUX_BANDS.items():
        peak_mag[column_band] = fitted_model.source_peakmag(sncosmo_band, magsys="ab")
        peak_flux[f"{column_band}_flux_max"] = 10 ** (-0.4 * (peak_mag[column_band] - 30.0))
    log(
        f"{objid}: peak mags "
        f"ztfg={peak_mag['ztfg']:.2f}, ztfr={peak_mag['ztfr']:.2f}, "
        f"sdssg_AB={peak_mag['sdssg_AB']:.2f}, sdssr_AB={peak_mag['sdssr_AB']:.2f}, "
        f"g_PS1={peak_mag['g_PS1']:.2f}, r_PS1={peak_mag['r_PS1']:.2f}, "
        f"g_CSP2_AB={peak_mag['g_CSP2_AB']:.2f}, r_CSP2_AB={peak_mag['r_CSP2_AB']:.2f}"
    )

    if save_diagnostics:
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        diag_phase_mask = (sn_tot["phase"] >= MODEL_PHASE_MIN) & (sn_tot["phase"] <= MODEL_PHASE_MAX)
        sn_diag = sn_tot[diag_phase_mask]
        log(
            f"{objid}: plotting {len(sn_diag)}/{len(sn_tot)} SALT2 points "
            f"within {MODEL_PHASE_MIN:g} to +{MODEL_PHASE_MAX:g} rest-frame days"
        )
        sncosmo.plot_lc(
            sn_diag,
            model=fitted_model,
            show=False,
            label=objid,
            markersize=3,
            linestyle="None",
            marker="o",
            errorbar=True,
            xerr=t_max_err,
        )
        fig = plt.gcf()
        source_label = photometry_config_label(use_ztf=use_ztf, use_external=use_external)
        suffix = "salt2_mcmc_fit" if run_mcmc else "salt2_fit"
        suffix = f"{source_label}_{suffix}"
        fig_path = DIAG_DIR / f"{objid}_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"{objid}: saved SALT2 diagnostic figure to {fig_path}")

    row = {
        "ztfid": objid,
        "t0": result["parameters"][1],
        "x0": result["parameters"][2],
        "x1": result["parameters"][3],
        "c": result["parameters"][4],
        "t0_err": result["errors"].get("t0", np.nan),
        "x0_err": result["errors"].get("x0", np.nan),
        "x1_err": result["errors"].get("x1", np.nan),
        "c_err": result["errors"].get("c", np.nan),
        "peak_flux_zp": 30.0,
        "sampler": sampler,
        "status": "ok",
        "error": "",
    }
    row.update(peak_flux)
    row.update(mcmc_summary)
    log(f"{objid}: completed SALT2 fit with sampler={sampler}")
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line options for SALT2 deterministic and MCMC fitting."""

    parser = argparse.ArgumentParser(description="Fit early/late ZTF SN Ia light curves with SALT2.")
    parser.add_argument("--limit", type=int, default=None, help="Fit only the first N objects.")
    parser.add_argument("--objects", nargs="*", default=None, help="Specific ZTF IDs to fit.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not save diagnostic figures.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to the full-sample SALT CSV.")
    parser.add_argument("--no-mcmc", action="store_true", help="Skip sncosmo.mcmc_lc and save least-squares fits only.")
    parser.add_argument("--no-ztf", action="store_true", help="Exclude ZTF photometry from SALT2 fits.")
    parser.add_argument("--external", action="store_true", help="Include normalized external photometry and fit only objects with external data.")
    parser.add_argument("--nwalkers", type=int, default=32, help="Number of emcee walkers for sncosmo.mcmc_lc.")
    parser.add_argument("--nburn", type=int, default=500, help="Burn-in steps for sncosmo.mcmc_lc.")
    parser.add_argument("--nsamples", type=int, default=2000, help="Production steps for sncosmo.mcmc_lc.")
    parser.add_argument("--thin", type=int, default=5, help="Thinning factor for sncosmo.mcmc_lc.")
    parser.add_argument("--t0-window", type=float, default=5.0, help="Half-width of the MCMC t0 bound around the deterministic fit.")
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    """Return the default SALT2 output path for this source configuration."""

    if args.limit is not None or args.objects:
        return DATA_DIR / "ztf_early_Ia_salt_subset.csv"
    if args.external and args.no_ztf:
        return DATA_DIR / "ztf_early_Ia_salt_external_only.csv"
    if args.external:
        return DATA_DIR / "ztf_early_Ia_salt_ztf_external.csv"
    return DATA_DIR / "ztf_early_Ia_salt.csv"


def photometry_config_label(use_ztf: bool, use_external: bool) -> str:
    """Return a compact label for the photometry sources used in a fit."""

    if use_ztf and use_external:
        return "ztf_external"
    if use_external:
        return "external_only"
    return "ztf"


def main() -> None:
    """Run SALT2 fitting for the requested sample and write the summary CSV."""

    args = parse_args()
    if args.no_ztf and not args.external:
        raise SystemExit("--no-ztf requires --external so at least one photometry source is enabled.")
    log("Starting SALT2 fitting script")
    sample = load_sample()
    objids = [str(x) for x in sample["ztfid"]]
    if args.objects:
        objids = [objid for objid in objids if objid in set(args.objects)]
    if args.external:
        external_objids = {path.stem.removesuffix("_external") for path in EXTERNAL_DIR.glob("*_external.csv")}
        before_external = len(objids)
        objids = [objid for objid in objids if objid in external_objids]
        log(f"--external selected: {len(objids)}/{before_external} requested objects have external files")
    if args.limit is not None:
        objids = objids[: args.limit]

    sn_info = pd.read_csv(DATA_DIR / "ztf_early_Ia_meta.csv")
    register_external_filters()
    t21_source = make_salt2_source()
    log(
        f"Loaded sample: {len(sample)} available objects, fitting {len(objids)} objects; "
        f"MCMC={'off' if args.no_mcmc else 'on'}; external={args.external}; use_ztf={not args.no_ztf}"
    )

    rows = []
    for i, objid in enumerate(objids, start=1):
        log(f"Object {i}/{len(objids)}: {objid}")
        try:
            rows.append(
                fit_one(
                    objid,
                    sn_info,
                    t21_source,
                    save_diagnostics=not args.no_diagnostics,
                    run_mcmc=not args.no_mcmc,
                    nwalkers=args.nwalkers,
                    nburn=args.nburn,
                    nsamples=args.nsamples,
                    thin=args.thin,
                    t0_window=args.t0_window,
                    use_ztf=not args.no_ztf,
                    use_external=args.external,
                )
            )
        except Exception as exc:
            log(f"{objid}: failed with error: {exc}")
            rows.append({"ztfid": objid, "status": "failed", "error": str(exc)})

    if args.output is not None:
        output = Path(args.output)
    else:
        output = default_output_path(args)
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False, float_format="%.6f")
    n_ok = int((out.get("status") == "ok").sum()) if "status" in out else 0
    log(f"Saved {output} ({n_ok}/{len(out)} successful fits)")


if __name__ == "__main__":
    main()
