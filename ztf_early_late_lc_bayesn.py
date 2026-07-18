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
BAYESN_FILTER_YAML = DATA_DIR / "bayesn_filters" / "atlas_filters.yaml"
DIAG_DIR = DATA_DIR / "bayesn_diagnostics"
EXTERNAL_DIR = DATA_DIR / "light_curve_external"
BAYESN_ZPT = 27.5
MODEL_PHASE_MIN = -15.0
MODEL_PHASE_MAX = 40.0
PEAK_PHASE_STEP = 0.25
PEAK_POSTERIOR_SAMPLES = 80
BAYESN_FILT_MAP = {
    "ztfg": "p48g",
    "ztfr": "p48r",
    "ztfi": "p48i",
    "atlaso": "atlaso",
    "atlasc": "atlasc",
    "bessellb": "B",
    "bessellv": "V",
    "sdssg": "g_prime",
    "sdssr": "r_prime",
    "sdssi": "i_prime",
    "sdssz": "z_prime",
    "ps1::g": "g_PS1",
    "ps1::r": "r_PS1",
    "ps1::i": "i_PS1",
    "ps1::z": "z_PS1",
    "ps1::y": "y_PS1",
    "swope2u": "u_CSP2",
    "swope2b": "B_CSP2",
    "swope2v": "V_CSP2",
    "swope2g": "g_CSP2",
    "swope2r": "r_CSP2",
    "swope2i": "i_CSP2",
}
PLOT_BANDS = [
    "p48g", "p48r", "p48i", "atlaso", "atlasc",
    "B", "V", "g_prime", "r_prime", "i_prime", "z_prime",
    "g_PS1", "r_PS1", "i_PS1", "z_PS1", "y_PS1",
    "u_CSP2", "B_CSP2", "V_CSP2", "g_CSP2", "r_CSP2", "i_CSP2",
]
PLOT_COLORS = {
    "p48g": "tab:green",
    "p48r": "tab:red",
    "p48i": "tab:orange",
    "atlaso": "tab:brown",
    "atlasc": "tab:cyan",
    "B": "tab:blue",
    "V": "tab:purple",
    "g_prime": "tab:green",
    "r_prime": "tab:red",
    "i_prime": "tab:orange",
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
}
PEAK_FLUX_BANDS = {
    "ztfg": {"bayesn_filter": "p48g", "zp": 30.0},
    "ztfr": {"bayesn_filter": "p48r", "zp": 30.0},
    "atlaso": {"bayesn_filter": "atlaso", "zp": 2.5 * np.log10(3631.0) + 15.0},
    "atlasc": {"bayesn_filter": "atlasc", "zp": 2.5 * np.log10(3631.0) + 15.0},
}


def log(message: str) -> None:
    """Print a timestamped progress message to the terminal."""

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


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
    return lc_dat[["mjd", "filter", "flux", "fluxerr", "zp", "magsys"]]


def parse_atlas_lc(filename: Path) -> pd.DataFrame:
    """Parse one ATLAS forced-photometry file into the common light-curve schema."""

    from astropy.stats import mad_std

    if not filename.exists():
        return pd.DataFrame(columns=["mjd", "filter", "flux", "fluxerr", "zp", "magsys"])

    lc = pd.read_csv(filename)
    lc_dat = lc[lc["err"] == 0].copy()
    lc_dat["magsys"] = "ab"

    mjd = lc_dat["MJD"].values
    flux = lc_dat["uJy"].values
    mask = np.ones(len(lc_dat), dtype=bool)
    i = 0
    while i < len(lc_dat):
        j = np.searchsorted(mjd, mjd[i] + 1.8, side="left")
        if j - i > 1:
            bin_flux = flux[i:j]
            median_flux = np.median(bin_flux)
            robust_std = mad_std(bin_flux)
            if robust_std > 0:
                mask[i:j] = np.abs(bin_flux - median_flux) <= (3 * robust_std)
        i = j

    lc_dat = lc_dat.rename(
        columns={"MJD": "mjd", "uJy": "flux", "duJy": "fluxerr", "F": "filter"}
    )
    lc_dat["zp"] = 2.5 * np.log10(3631.0) + 15.0
    lc_dat["filter"] = lc_dat["filter"].replace({"o": "atlaso", "c": "atlasc"})
    return lc_dat[["mjd", "filter", "flux", "fluxerr", "zp", "magsys"]][mask]


def parse_external_lc(objid: str) -> pd.DataFrame:
    """Load normalized external photometry for one object when available."""

    filename = EXTERNAL_DIR / f"{objid}_external.csv"
    columns = ["mjd", "filter", "flux", "fluxerr", "zp", "magsys"]
    if not filename.exists():
        return pd.DataFrame(columns=columns)

    lc = pd.read_csv(filename)
    lc = lc[np.isfinite(lc["flux"]) & np.isfinite(lc["fluxerr"]) & (lc["fluxerr"] > 0)].copy()
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
    lc["bayesn_filter"] = lc["filter"].replace(BAYESN_FILT_MAP)
    return lc


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
        if n_keep <= n_draws:
            subset[key] = arr[:1, :n_keep, ...]
        else:
            n_draws_keep = int(np.ceil(n_keep / n_chains))
            subset[key] = arr[:, :n_draws_keep, ...]
    return subset


def save_diagnostic_plot(
    model: Any,
    objid: str,
    sn_bayesn: pd.DataFrame,
    samples: dict,
    z: float,
    ebv_mw: float,
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
    ax.set_title(f"{objid}: BayeSN fit, AV={av_med:.3f} -{av_lo:.3f}/+{av_hi:.3f} mag")
    ax.legend(fontsize=8, ncol=2)
    fig_path = DIAG_DIR / f"{objid}_bayesn_fit.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"{objid}: saved BayeSN diagnostic figure to {fig_path}")


def fit_one(
    objid: str,
    sn_info: pd.DataFrame,
    salt_fit: pd.DataFrame,
    model: Any,
    save_diagnostics: bool,
) -> dict:
    """Fit one SN with BayeSN and return host-extinction posterior summaries."""

    log(f"{objid}: starting BayeSN fit")
    row = {"ztfid": objid, "status": "ok", "error": ""}

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

    sn_ztf = parse_ztf_lc(DATA_DIR / "light_curve_fps_ztf" / f"{objid}_fnu.csv")
    sn_atlas = parse_atlas_lc(DATA_DIR / "light_curve_fps_atlas" / f"{objid}_fnu.csv")
    sn_external = parse_external_lc(objid)
    sn_raw = pd.concat([sn_ztf, sn_atlas, sn_external], ignore_index=True)
    sn_raw = sn_raw[np.isfinite(sn_raw["flux"]) & np.isfinite(sn_raw["fluxerr"])]
    sn_raw = sn_raw[sn_raw["fluxerr"] > 0]
    sn_bayesn = to_bayesn_fluxcal(sn_raw)
    sn_bayesn["phase"] = (sn_bayesn["mjd"] - t0_salt) / (1 + z)
    phase_mask = (sn_bayesn["phase"] >= MODEL_PHASE_MIN) & (sn_bayesn["phase"] <= MODEL_PHASE_MAX)
    sn_bayesn = sn_bayesn[phase_mask].copy()
    log(
        f"{objid}: loaded photometry "
        f"(ZTF={len(sn_ztf)}, ATLAS={len(sn_atlas)}, external={len(sn_external)}, usable={len(sn_bayesn)} "
        f"within {MODEL_PHASE_MIN:g} to +{MODEL_PHASE_MAX:g} rest-frame days)"
    )

    log(f"{objid}: starting BayeSN MCMC")
    samples, _ = model.fit(
        t=sn_bayesn["mjd"].values,
        flux=sn_bayesn["bayesn_flux"].values,
        flux_err=sn_bayesn["bayesn_fluxerr"].values,
        filters=sn_bayesn["filter"].values,
        z=z,
        ebv_mw=ebv_mw,
        peak_mjd=t0_salt,
        filt_map=BAYESN_FILT_MAP,
        print_summary=False,
    )
    log(f"{objid}: finished BayeSN MCMC")

    row.update({"z": z, "ebv_mw": ebv_mw, "t0_salt": t0_salt})
    t0_bayesn, t0_err_bayesn = posterior_median_mad_std(samples, "peak_MJD")
    if "RV" in samples:
        r_v_host = np.nanmedian(np.asarray(samples["RV"]).reshape(-1))
    else:
        r_v_host = float(np.asarray(model.RV).reshape(-1)[0])
    row.update({"t0_bayesn": t0_bayesn, "t0_err_bayesn": t0_err_bayesn, "R_V_host": r_v_host})
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
        f"atlaso={row['bayesn_atlaso_flux_max']:.6g}, "
        f"atlasc={row['bayesn_atlasc_flux_max']:.6g}"
    )

    if save_diagnostics:
        save_diagnostic_plot(model, objid, sn_bayesn, samples, z, ebv_mw)

    log(f"{objid}: completed BayeSN fit")
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line options for BayeSN fitting."""

    parser = argparse.ArgumentParser(description="Fit early/late ZTF SN Ia light curves with BayeSN.")
    parser.add_argument("--limit", type=int, default=None, help="Fit only the first N objects.")
    parser.add_argument("--objects", nargs="*", default=None, help="Specific ZTF IDs to fit.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not save diagnostic figures.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to the full-sample BayeSN CSV.")
    return parser.parse_args()


def main() -> None:
    """Run BayeSN fitting for the requested sample and write the summary CSV."""

    args = parse_args()
    log("Starting BayeSN fitting script")
    sample = load_sample()
    objids = [str(x) for x in sample["ztfid"]]
    if args.objects:
        objids = [objid for objid in objids if objid in set(args.objects)]
    if args.limit is not None:
        objids = objids[: args.limit]

    sn_info = pd.read_csv(DATA_DIR / "ztf_early_Ia_meta.csv")
    salt_fit = pd.read_csv(DATA_DIR / "ztf_early_Ia_salt.csv")
    log(
        f"Loaded sample: {len(sample)} available objects, fitting {len(objids)} objects; "
        f"filter_yaml={BAYESN_FILTER_YAML}; jax_platform=cpu; num_devices=4"
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
    model = SEDmodel(num_devices=4, filter_yaml=str(BAYESN_FILTER_YAML))
    log("BayeSN SEDmodel initialized")

    rows = []
    for i, objid in enumerate(objids, start=1):
        log(f"Object {i}/{len(objids)}: {objid}")
        try:
            rows.append(fit_one(objid, sn_info, salt_fit, model, not args.no_diagnostics))
        except Exception as exc:
            log(f"{objid}: failed with error: {exc}")
            rows.append({"ztfid": objid, "status": "failed", "error": str(exc)})

    default_output = DATA_DIR / "ztf_early_Ia_bayesn.csv"
    if args.output is not None:
        output = Path(args.output)
    elif args.limit is not None or args.objects:
        output = DATA_DIR / "ztf_early_Ia_bayesn_subset.csv"
    else:
        output = default_output
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False, float_format="%.6f")
    n_ok = int((out.get("status") == "ok").sum()) if "status" in out else 0
    log(f"Saved {output} ({n_ok}/{len(out)} successful fits)")


if __name__ == "__main__":
    main()
