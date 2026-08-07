from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# BayeSN's public fit() uses 4 parallel NumPyro chains. Configure CPU host
# devices before importing BayeSN/JAX so this works on machines with one CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table


DATA_DIR = Path("data/ztf_snia_early_late")
BAYESN_FILTER_YAML = DATA_DIR / "bayesn_filters" / "external_filters.yaml"
BAYESN_MODEL_NAME = "W22_model"
DIAG_DIR = DATA_DIR / "bayesn_diagnostics"
EXTERNAL_DIR = DATA_DIR / "light_curve_external"
LOCAL_FILTER_DIR = DATA_DIR / "sncosmo_filters"
LOCAL_BAYESN_FILTER_DIR = LOCAL_FILTER_DIR / "bayesn"
BAYESN_ZPT = 27.5
MODEL_PHASE_MIN = -10.0
MODEL_PHASE_MAX = 40.0
PEAK_PHASE_STEP = 0.25
PEAK_POSTERIOR_SAMPLES = 80
MIN_OBS_PER_FILTER = 3
EXCLUDED_FILTERS = {
    "swope2u", "sdssu", "ps1::u", "bessellu", "u",
    "sdssz", "ps1::z", "z",
    "ps1::y", "y",
}
SALT2_SCREEN_ALLOWED_FILTERS = {
    "ztfg", "ztfr", "ztfi",
    "sdssg", "sdssr", "sdssi",
    "ps1::g", "ps1::r", "ps1::i",
    "swope2g", "swope2r", "swope2i",
}
EXCLUDED_BAYESN_FILTERS = {"u_CSP2", "u_prime", "u", "z_prime", "z_PS1", "y_PS1", "z", "y"}
BAYESN_FILTER_BY_SYSTEM = {
    ("ztfg", "ab"): "p48g",
    ("ztfr", "ab"): "p48r",
    ("ztfi", "ab"): "p48i",
    ("bessellb", "vega"): "B",
    ("bessellv", "vega"): "V",
    ("bessellb", "ab"): "B_AB",
    ("bessellv", "ab"): "V_AB",
    ("sdssg", "bd17"): "g_prime",
    ("sdssr", "bd17"): "r_prime",
    ("sdssi", "bd17"): "i_prime",
    ("sdssg", "ab"): "sdssg_AB",
    ("sdssr", "ab"): "sdssr_AB",
    ("sdssi", "ab"): "sdssi_AB",
    ("sdssz", "bd17"): "z_prime",
    ("ps1::g", "ab"): "g_PS1",
    ("ps1::r", "ab"): "r_PS1",
    ("ps1::i", "ab"): "i_PS1",
    ("ps1::z", "ab"): "z_PS1",
    ("ps1::y", "ab"): "y_PS1",
    ("swope2u", "bd17"): "u_CSP2",
    ("swope2b", "vega"): "B_CSP2",
    ("swope2v", "vega"): "V_CSP2",
    ("swope2g", "bd17"): "g_CSP2",
    ("swope2r", "bd17"): "r_CSP2",
    ("swope2i", "bd17"): "i_CSP2",
    ("swope2b", "ab"): "B_CSP2_AB",
    ("swope2v", "ab"): "V_CSP2_AB",
    ("swope2g", "ab"): "g_CSP2_AB",
    ("swope2r", "ab"): "r_CSP2_AB",
    ("swope2i", "ab"): "i_CSP2_AB",
}
PLOT_BANDS = [
    "p48g", "p48r", "p48i",
    "B", "V", "B_AB", "V_AB", "g_prime", "r_prime", "i_prime",
    "sdssg_AB", "sdssr_AB", "sdssi_AB", "z_prime",
    "g_PS1", "r_PS1", "i_PS1", "z_PS1", "y_PS1",
    "u_CSP2", "B_CSP2", "V_CSP2", "g_CSP2", "r_CSP2", "i_CSP2",
    "B_CSP2_AB", "V_CSP2_AB", "g_CSP2_AB", "r_CSP2_AB", "i_CSP2_AB",
]
PLOT_COLORS = {
    "p48g": "tab:green",
    "p48r": "tab:red",
    "p48i": "tab:orange",
    "B": "tab:blue",
    "V": "tab:purple",
    "B_AB": "tab:blue",
    "V_AB": "tab:purple",
    "g_prime": "tab:green",
    "r_prime": "tab:red",
    "i_prime": "tab:orange",
    "sdssg_AB": "tab:green",
    "sdssr_AB": "tab:red",
    "sdssi_AB": "tab:orange",
    "z_prime": "saddlebrown",
    "g_PS1": "limegreen",
    "r_PS1": "crimson",
    "i_PS1": "darkorange",
    "z_PS1": "saddlebrown",
    "y_PS1": "black",
    "u_CSP2": "tab:cyan",
    "B_CSP2": "royalblue",
    "V_CSP2": "mediumpurple",
    "g_CSP2": "seagreen",
    "r_CSP2": "firebrick",
    "i_CSP2": "peru",
    "B_CSP2_AB": "royalblue",
    "V_CSP2_AB": "mediumpurple",
    "g_CSP2_AB": "seagreen",
    "r_CSP2_AB": "firebrick",
    "i_CSP2_AB": "peru",
}
PEAK_FLUX_BANDS = {
    "ztfg": {"bayesn_filter": "p48g", "zp": 30.0},
    "ztfr": {"bayesn_filter": "p48r", "zp": 30.0},
    "sdssg_AB": {"bayesn_filter": "sdssg_AB", "zp": 30.0},
    "sdssr_AB": {"bayesn_filter": "sdssr_AB", "zp": 30.0},
    "g_PS1": {"bayesn_filter": "g_PS1", "zp": 30.0},
    "r_PS1": {"bayesn_filter": "r_PS1", "zp": 30.0},
    "g_CSP2_AB": {"bayesn_filter": "g_CSP2_AB", "zp": 30.0},
    "r_CSP2_AB": {"bayesn_filter": "r_CSP2_AB", "zp": 30.0},
}


def log(message: str) -> None:
    """Print a timestamped progress message to the terminal."""

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def register_salt2_screen_filters() -> None:
    """Register all local sncosmo filters needed by the pre-BayeSN SALT2 screen."""

    import sncosmo

    filter_files = {
        "ztfg": LOCAL_FILTER_DIR / "ztf" / "ztfg.dat",
        "ztfr": LOCAL_FILTER_DIR / "ztf" / "ztfr.dat",
        "ztfi": LOCAL_FILTER_DIR / "ztf" / "ztfi.dat",
        "sdssg": LOCAL_FILTER_DIR / "sdss" / "sdssg.dat",
        "sdssr": LOCAL_FILTER_DIR / "sdss" / "sdssr.dat",
        "sdssi": LOCAL_FILTER_DIR / "sdss" / "sdssi.dat",
        "ps1::g": LOCAL_FILTER_DIR / "ps1" / "ps1_g.dat",
        "ps1::r": LOCAL_FILTER_DIR / "ps1" / "ps1_r.dat",
        "ps1::i": LOCAL_FILTER_DIR / "ps1" / "ps1_i.dat",
    }
    missing = [str(path) for path in filter_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local SALT2-screen filter files: " + ", ".join(missing))

    for name, path in filter_files.items():
        wave, trans = np.loadtxt(path, unpack=True)
        sncosmo.registry.register(sncosmo.Bandpass(wave, trans, name=name), name=name, force=True)
    swope_aliases = {
        "swope2g": LOCAL_BAYESN_FILTER_DIR / "swope2g.dat",
        "swope2r": LOCAL_BAYESN_FILTER_DIR / "swope2r.dat",
        "swope2i": LOCAL_BAYESN_FILTER_DIR / "swope2i.dat",
    }
    missing_swope = [str(path) for path in swope_aliases.values() if not path.exists()]
    if missing_swope:
        raise FileNotFoundError("Missing local Swope alias filter files: " + ", ".join(missing_swope))
    for name, path in swope_aliases.items():
        wave, trans = np.loadtxt(path, unpack=True)
        sncosmo.registry.register(sncosmo.Bandpass(wave, trans, name=name), name=name, force=True)
    log(f"Registered {len(filter_files) + len(swope_aliases)} local SALT2-screen sncosmo filters")


def resolve_bayesn_filter(filter_name: str, magsys: str) -> str:
    """Return the BayeSN filter alias matching both passband and magnitude system."""

    key = (str(filter_name).strip().lower(), str(magsys).strip().lower())
    try:
        return BAYESN_FILTER_BY_SYSTEM[key]
    except KeyError as exc:
        raise ValueError(f"No BayeSN filter mapping for filter={filter_name!r}, magsys={magsys!r}") from exc


def assign_bayesn_filters(lc: pd.DataFrame) -> pd.DataFrame:
    """Attach system-aware BayeSN filter aliases to parsed photometry rows."""

    lc = lc.copy()
    lc["bayesn_filter"] = [
        resolve_bayesn_filter(filter_name, magsys)
        for filter_name, magsys in zip(lc["filter"], lc["magsys"], strict=True)
    ]
    return lc


def parse_ztf_lc(filename: Path) -> pd.DataFrame:
    """Parse one ZTF forced-photometry file into the common light-curve schema."""

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

    lc_dat["filter"] = lc_dat["filter"].replace({"ZTF_g": "ztfg", "ZTF_r": "ztfr", "ZTF_i": "ztfi"})
    lc_dat = assign_bayesn_filters(lc_dat)
    return lc_dat[["mjd", "filter", "flux", "fluxerr", "zp", "magsys", "bayesn_filter"]]


def parse_external_lc(objid: str, external_sources: set[str] | None = None) -> pd.DataFrame:
    """Load normalized external photometry for one object when available."""

    filename = EXTERNAL_DIR / f"{objid}_external.csv"
    columns = ["mjd", "filter", "flux", "fluxerr", "zp", "magsys", "bayesn_filter"]
    if not filename.exists():
        return pd.DataFrame(columns=columns)

    lc = pd.read_csv(filename)
    lc = lc[np.isfinite(lc["flux"]) & np.isfinite(lc["fluxerr"]) & (lc["fluxerr"] > 0)].copy()
    lc = assign_bayesn_filters(lc)
    before = len(lc)
    exclude = lc["filter"].isin(EXCLUDED_FILTERS)
    if "bayesn_filter" in lc:
        exclude |= lc["bayesn_filter"].isin(EXCLUDED_BAYESN_FILTERS)
    lc = lc[~exclude].copy()
    if before > len(lc):
        log(f"{objid}: excluded {before - len(lc)} external u/z/y-band rows from BayeSN input")
    if external_sources is not None:
        before_source = len(lc)
        lc = lc[lc["source"].isin(external_sources)].copy()
        if before_source > len(lc):
            log(
                f"{objid}: excluded {before_source - len(lc)} external rows outside requested "
                f"sources={sorted(external_sources)}"
            )
    if not lc.empty:
        counts = lc.groupby(["source", "raw_filter", "filter", "bayesn_filter"]).size().to_dict()
        log(f"{objid}: loaded external photometry from {filename} ({len(lc)} rows; {counts})")
    return lc[columns]


def load_sample() -> Table:
    """Load the observed early/late SN Ia sample after removing unobserved targets."""

    ztf_early = Table.read(DATA_DIR / "ztf_early_Ia.csv", format="ascii.csv")
    return ztf_early[ztf_early["not_obs"] != 1]


def to_bayesn_fluxcal(lc: pd.DataFrame) -> pd.DataFrame:
    """Convert parsed photometry to BayeSN's fixed 27.5 zeropoint flux scale."""

    lc = lc.copy()
    scale = 10 ** (0.4 * (BAYESN_ZPT - lc["zp"]))
    lc["bayesn_flux"] = lc["flux"] * scale
    lc["bayesn_fluxerr"] = lc["fluxerr"] * scale
    lc = assign_bayesn_filters(lc)
    return lc


def drop_sparse_bayesn_filters_after_phase_cut(sn: pd.DataFrame, objid: str) -> pd.DataFrame:
    """Drop sparse external filters while retaining all ZTF filters."""

    if "_survey" not in sn:
        counts = sn["bayesn_filter"].value_counts()
        dropped = counts[counts < MIN_OBS_PER_FILTER]
        if dropped.empty:
            return sn
        dropped_msg = ", ".join(f"{band}={count}" for band, count in dropped.items())
        log(f"{objid}: dropping BayeSN filters with <{MIN_OBS_PER_FILTER} points after phase cut: {dropped_msg}")
        return sn[sn["bayesn_filter"].isin(counts[counts >= MIN_OBS_PER_FILTER].index)].copy()

    external = sn[sn["_survey"] != "ZTF"]
    if external.empty:
        return sn
    counts = external["bayesn_filter"].value_counts()
    dropped = counts[counts < MIN_OBS_PER_FILTER]
    if dropped.empty:
        return sn

    dropped_msg = ", ".join(f"{band}={count}" for band, count in dropped.items())
    log(f"{objid}: dropping external BayeSN filters with <{MIN_OBS_PER_FILTER} points after phase cut: {dropped_msg}")
    return sn[(sn["_survey"] == "ZTF") | ~sn["bayesn_filter"].isin(dropped.index)].copy()


def posterior_summary(samples: dict, key: str) -> tuple[float, float, float]:
    """Return median and asymmetric 16th/84th percentile errors for a sample key."""

    values = np.asarray(samples[key]).reshape(-1)
    med = np.nanmedian(values)
    p16, p84 = np.nanpercentile(values, [16, 84])
    return med, med - p16, p84 - med


def posterior_median_mad_std(samples: dict, key: str) -> tuple[float, float]:
    """Return posterior median and Gaussian-equivalent standard deviation from MAD."""

    values = np.asarray(samples[key]).reshape(-1)
    values = values[np.isfinite(values)]
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return med, 1.4826 * mad


def summarize_bayesn_observed_peaks(
    model: Any,
    samples: dict,
    z: float,
    ebv_mw: float,
) -> dict[str, float]:
    """Summarize observed-frame filter peak fluxes from BayeSN posterior draws."""

    phase = np.arange(MODEL_PHASE_MIN, MODEL_PHASE_MAX + 0.5 * PEAK_PHASE_STEP, PEAK_PHASE_STEP)
    output_bands = list(PEAK_FLUX_BANDS)
    bayesn_bands = [PEAK_FLUX_BANDS[band]["bayesn_filter"] for band in output_bands]
    peak_samples = subset_chains_for_plot(samples, n_samples=PEAK_POSTERIOR_SAMPLES)
    flux_grid = np.asarray(
        model.get_flux_from_chains(
            phase,
            bayesn_bands,
            peak_samples,
            np.array([float(z)]),
            np.array([float(ebv_mw)]),
            mag=False,
        )
    )[0]

    row = {}
    for i, band in enumerate(output_bands):
        native_zp = PEAK_FLUX_BANDS[band]["zp"]
        native_scale = 10 ** (0.4 * (native_zp - BAYESN_ZPT))
        band_flux = flux_grid[:, i, :] * native_scale
        peak_idx = np.nanargmax(band_flux, axis=1)
        peak_flux = band_flux[np.arange(band_flux.shape[0]), peak_idx]

        row[f"bayesn_{band}_flux_max"] = np.nanmedian(peak_flux)

    row["bayesn_peak_flux_zp"] = 30.0
    return row


def subset_chains_for_plot(samples: dict, n_samples: int = 80) -> dict:
    """Return a small posterior subset preserving BayeSN's chain/draw shape."""

    subset = {}
    for key, values in samples.items():
        arr = np.asarray(values)
        if arr.ndim < 2:
            subset[key] = values
            continue
        n_chains, n_draws = arr.shape[:2]
        n_keep = min(n_samples, n_chains * n_draws)
        flat = arr.reshape((n_chains * n_draws, *arr.shape[2:]))
        flat_idx = np.linspace(0, len(flat) - 1, n_keep, dtype=int)
        subset[key] = flat[flat_idx, ...].reshape((1, n_keep, *arr.shape[2:]))
    return subset


def save_diagnostic_plot(
    model: Any,
    objid: str,
    sn_bayesn: pd.DataFrame,
    samples: dict,
    z: float,
    ebv_mw: float,
    model_name: str,
    config_label: str,
) -> None:
    """Save an observed-vs-posterior BayeSN light-curve diagnostic plot."""

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    phase = np.linspace(MODEL_PHASE_MIN, MODEL_PHASE_MAX, 160)
    bands = [band for band in PLOT_BANDS if np.any(sn_bayesn["bayesn_filter"] == band)]
    if not bands:
        log(f"{objid}: no BayeSN bands available for diagnostic plot")
        return

    diagnostic_samples = subset_chains_for_plot(samples)
    n_diagnostic_samples = diagnostic_samples["theta"].shape[0] * diagnostic_samples["theta"].shape[1]
    log(
        f"{objid}: generating BayeSN posterior model curves for diagnostic plot "
        f"from {n_diagnostic_samples} samples"
    )
    flux_grid = np.asarray(
        model.get_flux_from_chains(
            phase,
            bands,
            diagnostic_samples,
            np.array([float(z)]),
            np.array([float(ebv_mw)]),
            mag=False,
        )
    )[0]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    offsets = {band: 0.0 for band in bands}
    if len(bands) > 1:
        scale = np.nanpercentile(np.abs(sn_bayesn["bayesn_flux"]), 95)
        step = 0.25 * scale if np.isfinite(scale) and scale > 0 else 1.0
        offsets = {band: i * step for i, band in enumerate(bands)}

    y_for_limits = []
    for i, band in enumerate(bands):
        color = PLOT_COLORS.get(band, f"C{i}")
        offset = offsets[band]
        idx = sn_bayesn["bayesn_filter"] == band
        data_y = sn_bayesn.loc[idx, "bayesn_flux"].to_numpy() + offset
        ax.errorbar(
            sn_bayesn.loc[idx, "phase"],
            data_y,
            yerr=sn_bayesn.loc[idx, "bayesn_fluxerr"],
            fmt="o",
            ms=3,
            color=color,
            alpha=0.8,
            label=f"{band} data",
        )
        med = np.nanmedian(flux_grid[:, i, :], axis=0)
        p16, p84 = np.nanpercentile(flux_grid[:, i, :], [16, 84], axis=0)
        ax.plot(phase, med + offset, color=color, lw=1.5, label=f"{band} BayeSN")
        ax.fill_between(phase, p16 + offset, p84 + offset, color=color, alpha=0.15, lw=0)
        y_for_limits.extend([data_y, med + offset, p16 + offset, p84 + offset])

    finite_y = np.concatenate([np.ravel(y) for y in y_for_limits])
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size > 1:
        y_low, y_high = np.nanpercentile(finite_y, [1, 99])
        if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
            y_pad = 0.1 * (y_high - y_low)
            ax.set_ylim(y_low - y_pad, y_high + y_pad)
            n_clipped = int(np.sum((finite_y < y_low) | (finite_y > y_high)))
            if n_clipped:
                log(f"{objid}: clipped BayeSN diagnostic y-axis around {n_clipped} extreme plotted values")

    av_med, av_lo, av_hi = posterior_summary(samples, "AV")
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Rest-frame phase from SALT2/BayeSN peak [day]")
    ax.set_ylabel("BayeSN flux scale + offsets")
    ax.set_title(f"{objid}: BayeSN {model_name}, AV={av_med:.3f} -{av_lo:.3f}/+{av_hi:.3f} mag")
    ax.legend(fontsize=8, ncol=2)
    fig_path = DIAG_DIR / f"{objid}_bayesn_{config_label}_fit.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"{objid}: saved BayeSN diagnostic figure to {fig_path}")


def drop_salt2_screened_outliers(sn_raw: pd.DataFrame, z: float, t0_salt: float, objid: str) -> pd.DataFrame:
    """Use a quick SALT2 g/r/i fit to remove catastrophic outliers before BayeSN."""

    import ztf_early_late_lc_salt as salt2_fit

    register_salt2_screen_filters()
    salt_df = sn_raw[sn_raw["filter"].isin(SALT2_SCREEN_ALLOWED_FILTERS)].copy()
    if salt_df.empty:
        log(f"{objid}: skipping pre-BayeSN SALT2 outlier screen; no g/r/i data available")
        return sn_raw

    removed_ids: set[int] = set()
    salt_source = salt2_fit.make_salt2_source()
    for iteration in range(1, salt2_fit.SALT2_OUTLIER_MAX_ITER + 1):
        salt_table = Table.from_pandas(salt_df)
        sn_fit = salt2_fit.select_salt2_modeling_photometry(salt_table, z, t0_salt, objid)
        model = salt2_fit.sncosmo.Model(source=salt_source)
        model.set(z=z, t0=t0_salt)
        result, fitted_model = salt2_fit.fit_salt2_lc(
            sn_fit,
            model,
            t0_bounds=(sn_fit["mjd"].min(), sn_fit["mjd"].max()),
            x1_bounds=salt2_fit.SALT2_X1_FINAL_BOUNDS,
            t0_init=t0_salt,
        )
        final_sn_fit = salt2_fit.select_salt2_modeling_photometry(
            salt_table, z, result["parameters"][1], objid
        )
        if len(final_sn_fit) != len(sn_fit) or set(final_sn_fit["filter"]) != set(sn_fit["filter"]):
            model = salt2_fit.sncosmo.Model(source=salt_source)
            model.set(z=z, t0=result["parameters"][1])
            sn_fit = final_sn_fit
            result, fitted_model = salt2_fit.fit_salt2_lc(
                sn_fit,
                model,
                t0_bounds=(sn_fit["mjd"].min(), sn_fit["mjd"].max()),
                x1_bounds=salt2_fit.SALT2_X1_FINAL_BOUNDS,
                t0_init=result["parameters"][1],
            )

        pred = fitted_model.bandflux(
            sn_fit["filter"], sn_fit["mjd"], zp=sn_fit["zp"], zpsys=sn_fit["magsys"]
        )
        resid_sigma = (np.asarray(sn_fit["flux"], dtype=float) - pred) / np.asarray(sn_fit["fluxerr"], dtype=float)
        bad = np.isfinite(resid_sigma) & (np.abs(resid_sigma) > salt2_fit.SALT2_OUTLIER_SIGMA)
        if not np.any(bad):
            break

        bad_ids = {int(row_id) for row_id in np.asarray(sn_fit["_row_id"])[bad]}
        details = []
        for idx in np.flatnonzero(bad):
            details.append(
                f"{sn_fit['filter'][idx]}@{float(sn_fit['mjd'][idx]):.5f}={resid_sigma[idx]:.1f}sigma"
            )
        log(
            f"{objid}: pre-BayeSN SALT2 screen dropping outliers "
            f">{salt2_fit.SALT2_OUTLIER_SIGMA:g} sigma iteration {iteration}: {', '.join(details)}"
        )
        removed_ids.update(bad_ids)
        salt_df = salt_df[~salt_df["_row_id"].isin(bad_ids)].copy()

    if not removed_ids:
        return sn_raw

    clipped = sn_raw[~sn_raw["_row_id"].isin(removed_ids)].copy()
    log(f"{objid}: removed {len(sn_raw) - len(clipped)} SALT2-screened outliers before BayeSN sampling")
    return clipped


def fit_one(
    objid: str,
    sn_info: pd.DataFrame,
    salt_fit: pd.DataFrame,
    model: Any,
    model_name: str,
    save_diagnostics: bool,
    use_ztf: bool,
    use_external: bool,
    external_sources: set[str] | None,
    rv: float | None,
) -> dict:
    """Fit one SN with BayeSN and return host-extinction posterior summaries."""

    log(f"{objid}: starting BayeSN fit")
    row = {"ztfid": objid, "status": "ok", "error": "", "bayesn_model": model_name}

    info_idx = sn_info.objid == objid
    if not np.any(info_idx):
        raise ValueError("No metadata row found.")
    salt_idx = salt_fit.ztfid == objid
    if not np.any(salt_idx):
        raise ValueError("No SALT2 fit row found.")

    z = sn_info.loc[info_idx, "z"].values[0]
    ebv_mw = sn_info.loc[info_idx, "ebv"].values[0]
    t0_salt = salt_fit.loc[salt_idx, "t0"].values[0]
    log(f"{objid}: metadata z={z:.6f}, ebv_mw={ebv_mw:.5f}, t0_salt={t0_salt:.5f}")

    sn_ztf = parse_ztf_lc(DATA_DIR / "light_curve_fps_ztf" / f"{objid}_fnu.csv") if use_ztf else pd.DataFrame()
    sn_external = parse_external_lc(objid, external_sources=external_sources) if use_external else pd.DataFrame()
    if use_external and sn_external.empty:
        raise ValueError("External photometry requested but none is available after cuts.")
    frames = []
    for survey, frame in [("ZTF", sn_ztf), ("external", sn_external)]:
        if not frame.empty:
            frame = frame.copy()
            frame["_survey"] = survey
            frames.append(frame)
    if not frames:
        raise ValueError("No photometry available for BayeSN fit.")
    sn_raw = pd.concat(frames, ignore_index=True)
    sn_raw = sn_raw[np.isfinite(sn_raw["flux"]) & np.isfinite(sn_raw["fluxerr"])]
    sn_raw = sn_raw[sn_raw["fluxerr"] > 0]
    sn_raw = sn_raw.reset_index(drop=True)
    sn_raw["_row_id"] = np.arange(len(sn_raw))
    sn_raw = drop_salt2_screened_outliers(sn_raw, z, t0_salt, objid)
    sn_bayesn = to_bayesn_fluxcal(sn_raw)
    sn_bayesn["phase"] = (sn_bayesn["mjd"] - t0_salt) / (1 + z)
    phase_mask = (sn_bayesn["phase"] >= MODEL_PHASE_MIN) & (sn_bayesn["phase"] <= MODEL_PHASE_MAX)
    sn_bayesn = sn_bayesn[phase_mask].copy()
    sn_bayesn = drop_sparse_bayesn_filters_after_phase_cut(sn_bayesn, objid)
    if sn_bayesn.empty:
        raise ValueError("No photometry remains after sparse-filter cut.")
    usable_counts = sn_bayesn["_survey"].value_counts().to_dict()
    log(
        f"{objid}: loaded photometry "
        f"(ZTF={usable_counts.get('ZTF', 0)}, external={usable_counts.get('external', 0)}, "
        f"usable={len(sn_bayesn)} "
        f"within {MODEL_PHASE_MIN:g} to +{MODEL_PHASE_MAX:g} rest-frame days)"
    )

    fit_kwargs = {}
    if rv is not None:
        fit_kwargs["RV"] = rv

    log(f"{objid}: starting BayeSN MCMC with model={model_name}, rv_kwargs={fit_kwargs}")
    samples, _ = model.fit(
        t=sn_bayesn["mjd"].values,
        flux=sn_bayesn["bayesn_flux"].values,
        flux_err=sn_bayesn["bayesn_fluxerr"].values,
        filters=sn_bayesn["bayesn_filter"].values,
        z=z,
        ebv_mw=ebv_mw,
        peak_mjd=t0_salt,
        filt_map={},
        print_summary=False,
        **fit_kwargs,
    )
    log(f"{objid}: finished BayeSN MCMC")

    row.update({"z": z, "ebv_mw": ebv_mw, "t0_salt": t0_salt})
    t0_bayesn, t0_err_bayesn = posterior_median_mad_std(samples, "peak_MJD")
    if "RV" in samples:
        r_v_host = np.nanmedian(np.asarray(samples["RV"]).reshape(-1))
    elif "Rv" in samples:
        r_v_host = np.nanmedian(np.asarray(samples["Rv"]).reshape(-1))
    else:
        r_v_host = float(np.asarray(model.RV).reshape(-1)[0])
    row.update({
        "t0_bayesn": t0_bayesn,
        "t0_err_bayesn": t0_err_bayesn,
        "R_V_host": r_v_host,
    })
    log(f"{objid}: t0_bayesn={t0_bayesn:.5f}, t0_err_bayesn={t0_err_bayesn:.5f}, R_V_host={r_v_host:.3f}")

    for key, prefix in [
        ("AV", "AV"),
        ("theta", "theta"),
        ("mu", "mu"),
        ("delM", "delM"),
        ("tmax", "tmax_offset"),
    ]:
        med, err_minus, err_plus = posterior_summary(samples, key)
        row[f"{prefix}_median"] = med
        row[f"{prefix}_err_minus"] = err_minus
        row[f"{prefix}_err_plus"] = err_plus
        if key in {"AV", "theta"}:
            log(f"{objid}: {prefix}={med:.5f} -{err_minus:.5f}/+{err_plus:.5f}")

    log(
        f"{objid}: estimating BayeSN observed-filter peaks from "
        f"{PEAK_POSTERIOR_SAMPLES} posterior draws on a {PEAK_PHASE_STEP:g}-day phase grid"
    )
    peak_summary = summarize_bayesn_observed_peaks(model, samples, z, ebv_mw)
    row.update(peak_summary)
    log(
        f"{objid}: BayeSN observed peak fluxes "
        f"ztfg={row['bayesn_ztfg_flux_max']:.6g}, "
        f"ztfr={row['bayesn_ztfr_flux_max']:.6g}, "
        f"sdssg_AB={row['bayesn_sdssg_AB_flux_max']:.6g}, "
        f"sdssr_AB={row['bayesn_sdssr_AB_flux_max']:.6g}, "
        f"g_PS1={row['bayesn_g_PS1_flux_max']:.6g}, "
        f"r_PS1={row['bayesn_r_PS1_flux_max']:.6g}, "
        f"g_CSP2_AB={row['bayesn_g_CSP2_AB_flux_max']:.6g}, "
        f"r_CSP2_AB={row['bayesn_r_CSP2_AB_flux_max']:.6g}"
    )

    if save_diagnostics:
        rv_label = f"RV{r_v_host:.2f}".replace(".", "p")
        survey_label = "_".join(sorted(str(survey) for survey in sn_bayesn["_survey"].unique()))
        config_label = f"{model_name}_{survey_label}_{rv_label}"
        save_diagnostic_plot(model, objid, sn_bayesn, samples, z, ebv_mw, model_name, config_label)

    log(f"{objid}: completed BayeSN fit")
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line options for BayeSN fitting."""

    parser = argparse.ArgumentParser(description="Fit early/late ZTF SN Ia light curves with BayeSN.")
    parser.add_argument("--limit", type=int, default=None, help="Fit only the first N objects.")
    parser.add_argument("--objects", nargs="*", default=None, help="Specific ZTF IDs to fit.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not save diagnostic figures.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to the full-sample BayeSN CSV.")
    parser.add_argument("--model", default=BAYESN_MODEL_NAME, help="Built-in or custom BayeSN model to load.")
    parser.add_argument("--no-ztf", action="store_true", help="Exclude ZTF photometry from BayeSN fits.")
    parser.add_argument("--external", action="store_true", help="Include normalized external photometry and fit only objects with external data.")
    parser.add_argument(
        "--external-sources",
        nargs="*",
        default=None,
        help="Restrict external photometry to these source labels, e.g. 'Las Cumbres 1m'.",
    )
    parser.add_argument("--rv", type=float, default=None, help="Fit with this fixed host R_V; omit to use the BayeSN model default.")
    return parser.parse_args()


def main() -> None:
    """Run BayeSN fitting for the requested sample and write the summary CSV."""

    args = parse_args()
    if args.no_ztf and not args.external:
        raise SystemExit("--no-ztf requires --external so at least one photometry source is enabled.")
    log("Starting BayeSN fitting script")
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
    salt_fit = pd.read_csv(DATA_DIR / "ztf_early_Ia_salt.csv")
    log(
        f"Loaded sample: {len(sample)} available objects, fitting {len(objids)} objects; "
        f"filter_yaml={BAYESN_FILTER_YAML}; model={args.model}; external={args.external}; "
        f"use_ztf={not args.no_ztf}; jax_platform=cpu; num_devices=4"
    )
    log("Initializing BayeSN SEDmodel")
    try:
        from bayesn import SEDmodel
    except ImportError as exc:
        log(
            "Failed to import BayeSN SEDmodel. Check that the bayesn package is "
            "installed in .venv and that site-packages/bayesn contains bayesn_model.py."
        )
        raise exc
    model = SEDmodel(num_devices=4, load_model=args.model, filter_yaml=str(BAYESN_FILTER_YAML))
    log("BayeSN SEDmodel initialized")

    external_sources = set(args.external_sources) if args.external_sources is not None else None

    rows = []
    for i, objid in enumerate(objids, start=1):
        log(f"Object {i}/{len(objids)}: {objid}")
        try:
            rows.append(
                fit_one(
                    objid,
                    sn_info,
                    salt_fit,
                    model,
                    args.model,
                    not args.no_diagnostics,
                    not args.no_ztf,
                    args.external,
                    external_sources,
                    args.rv,
                )
            )
        except Exception as exc:
            log(f"{objid}: failed with error: {exc}")
            rows.append({"ztfid": objid, "status": "failed", "error": str(exc), "bayesn_model": args.model})

    if args.output is not None:
        output = Path(args.output)
    elif args.limit is not None or args.objects:
        output = DATA_DIR / "ztf_early_Ia_bayesn_subset.csv"
    elif args.external and args.no_ztf:
        output = DATA_DIR / "ztf_early_Ia_bayesn_external_only.csv"
    elif args.external:
        output = DATA_DIR / "ztf_early_Ia_bayesn_ztf_external.csv"
    else:
        output = DATA_DIR / "ztf_early_Ia_bayesn.csv"
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False, float_format="%.6f")
    n_ok = int((out.get("status") == "ok").sum()) if "status" in out else 0
    log(f"Saved {output} ({n_ok}/{len(out)} successful fits)")


if __name__ == "__main__":
    main()
