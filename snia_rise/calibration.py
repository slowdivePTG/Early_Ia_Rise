"""SALT2/BayeSN calibration and peak-flux normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table

from .io import LightCurveRecord


BAYESN_ZPT = 27.5
DEFAULT_FLUX_ZP = 30.0
MODEL_PHASE_MIN = -10.0
MODEL_PHASE_MAX = 40.0
PEAK_PHASE_STEP = 0.25
PEAK_POSTERIOR_SAMPLES = 80
MIN_OBS_PER_FILTER = 3

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
EXCLUDED_BAYESN_FILTERS = {"u_CSP2", "u_prime", "u", "z_prime", "z_PS1", "y_PS1", "z", "y"}
SALT2_ALLOWED_FILTERS = {
    "ztfg", "ztfr", "ztfi",
    "sdssg", "sdssr", "sdssi",
    "ps1::g", "ps1::r", "ps1::i",
    "swope2g", "swope2r", "swope2i",
}


@dataclass
class CalibrationResult:
    """Outputs from SALT2/BayeSN calibration."""

    record: LightCurveRecord
    salt2_summary: dict[str, Any]
    bayesn_summary: dict[str, Any]
    peak_fluxes: Table


def resolve_bayesn_filter(filter_name: str, magsys: str) -> str:
    """Return the BayeSN filter alias for a native filter and magnitude system."""

    key = (str(filter_name).lower(), str(magsys).lower())
    try:
        return BAYESN_FILTER_BY_SYSTEM[key]
    except KeyError as exc:
        raise ValueError(f"No BayeSN filter mapping for filter={filter_name!r}, magsys={magsys!r}") from exc


def assign_bayesn_filters(photometry: pd.DataFrame) -> pd.DataFrame:
    """Attach BayeSN filter aliases to a photometry table."""

    photometry = photometry.copy()
    filter_col = "native_filter" if "native_filter" in photometry else "filter"
    photometry["bayesn_filter"] = [
        resolve_bayesn_filter(name, magsys)
        for name, magsys in zip(photometry[filter_col], photometry["magsys"], strict=True)
    ]
    return photometry


def normalize_record_with_peak_fluxes(
    record: LightCurveRecord,
    peak_fluxes: Table,
    *,
    t0: float,
    t0_err: float | None,
    redshift: float,
    early_threshold: float = 0.4,
    output_flux_zp: float = DEFAULT_FLUX_ZP,
) -> LightCurveRecord:
    """Add normalized rise-model columns using BayeSN peak-flux medians."""

    record.validate()
    table = record.photometry.copy()
    peaks = peak_fluxes.to_pandas().copy()
    peaks["native_filter"] = peaks["native_filter"].astype(str)
    peaks["magsys"] = peaks["magsys"].astype(str).str.lower()
    peak_lookup = {
        (row.native_filter, row.magsys): float(row.peak_flux_median)
        for row in peaks.itertuples(index=False)
    }

    native_filters = np.asarray(table["native_filter"], dtype=str)
    magsys = np.asarray(table["magsys"], dtype=str)
    peak = np.array(
        [peak_lookup[(flt, system.lower())] for flt, system in zip(native_filters, magsys, strict=True)],
        dtype=float,
    )
    if np.any(~np.isfinite(peak)) or np.any(peak <= 0):
        raise ValueError("BayeSN peak-flux medians must be finite and positive")

    flux_scale = 10 ** (0.4 * (output_flux_zp - np.asarray(table["zp"], dtype=float)))
    table["phase"] = (np.asarray(table["mjd"], dtype=float) - float(t0)) / (1.0 + float(redshift))
    table["normalization_filter"] = native_filters
    table["normalization_flux"] = peak
    table["normalization_flux_zp"] = np.full(len(table), output_flux_zp)
    table["normalized_flux"] = 100.0 * np.asarray(table["flux"], dtype=float) * flux_scale / peak
    table["normalized_flux_err"] = 100.0 * np.asarray(table["flux_err"], dtype=float) * flux_scale / peak

    phase = np.asarray(table["phase"], dtype=float)
    model_filter = np.asarray(table["model_filter"], dtype=str)
    norm_flux = np.asarray(table["normalized_flux"], dtype=float)
    norm_flux_err = np.asarray(table["normalized_flux_err"], dtype=float)
    table["in_peak_plot"] = (phase < 0) & (phase > -100) & np.isin(model_filter, record.filter_order)
    table["in_early_fit"] = _early_fit_mask(
        phase,
        norm_flux,
        norm_flux_err,
        model_filter,
        record.filter_order,
        early_threshold,
    )

    metadata = dict(record.metadata)
    metadata.update(
        {
            "t0": float(t0),
            "t0_err": None if t0_err is None else float(t0_err),
            "redshift": float(redshift),
            "normalization": "bayesn_peak_flux_median",
            "normalization_flux_zp": float(output_flux_zp),
            "early_threshold": float(early_threshold),
        }
    )
    out = LightCurveRecord(record.object_id, table, metadata, record.filter_order)
    out.validate()
    return out


def estimate_bayesn_peak_fluxes(
    model: Any,
    samples: dict[str, Any],
    peak_filters: Table | pd.DataFrame,
    *,
    redshift: float,
    ebv_mw: float,
    output_flux_zp: float = DEFAULT_FLUX_ZP,
    phase_min: float = MODEL_PHASE_MIN,
    phase_max: float = MODEL_PHASE_MAX,
    phase_step: float = PEAK_PHASE_STEP,
    posterior_samples: int = PEAK_POSTERIOR_SAMPLES,
) -> Table:
    """Estimate observed-frame peak fluxes from BayeSN posterior draws."""

    peaks = peak_filters.to_pandas() if isinstance(peak_filters, Table) else peak_filters.copy()
    if "bayesn_filter" not in peaks:
        peaks = assign_bayesn_filters(peaks)
    peaks = peaks.drop_duplicates(["native_filter", "magsys", "bayesn_filter"])
    bayesn_filters = peaks["bayesn_filter"].astype(str).tolist()
    phase = np.arange(phase_min, phase_max + 0.5 * phase_step, phase_step)
    subset = _subset_chains(samples, posterior_samples)
    flux_grid = np.asarray(
        model.get_flux_from_chains(
            phase,
            bayesn_filters,
            subset,
            np.array([float(redshift)]),
            np.array([float(ebv_mw)]),
            mag=False,
        )
    )[0]

    rows = []
    scale = 10 ** (0.4 * (output_flux_zp - BAYESN_ZPT))
    for i, peak_row in enumerate(peaks.itertuples(index=False)):
        band_flux = flux_grid[:, i, :] * scale
        peak_idx = np.nanargmax(band_flux, axis=1)
        peak_flux = band_flux[np.arange(band_flux.shape[0]), peak_idx]
        p16, p50, p84 = np.nanpercentile(peak_flux, [16, 50, 84])
        rows.append(
            [
                peak_row.native_filter,
                str(peak_row.magsys).lower(),
                peak_row.bayesn_filter,
                output_flux_zp,
                p16,
                p50,
                p84,
                len(peak_flux),
            ]
        )
    return Table(
        rows=rows,
        names=[
            "native_filter",
            "magsys",
            "bayesn_filter",
            "flux_zp",
            "peak_flux_p16",
            "peak_flux_median",
            "peak_flux_p84",
            "posterior_draws",
        ],
    )


def run_salt2_calibration(
    record: LightCurveRecord,
    *,
    salt2_model_dir: str | Path | None = None,
    sncosmo_filter_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a deterministic SALT2 fit to estimate peak MJD for BayeSN."""

    try:
        import sncosmo
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError("SALT2 calibration requires sncosmo and scipy") from exc

    _register_sncosmo_filters(sncosmo, sncosmo_filter_dir)
    phot = _raw_photometry_dataframe(record)
    phot = phot[phot["native_filter"].isin(SALT2_ALLOWED_FILTERS)].copy()
    if phot.empty:
        raise ValueError(f"{record.object_id}: no SALT2-compatible g/r/i photometry")

    redshift = _metadata_float(record, "redshift", "z")
    t_peak_obs = float(phot.loc[phot["flux"].idxmax(), "mjd"])
    source = _make_salt2_source(sncosmo, salt2_model_dir)

    def fit_subset(df: pd.DataFrame, t0_init: float | None, x1_bounds: tuple[float, float]):
        model = sncosmo.Model(source=source)
        model.set(z=redshift)
        lc = df[np.isfinite(df["flux"]) & np.isfinite(df["flux_err"]) & (df["flux_err"] > 0)].copy()
        if lc.empty:
            raise ValueError("No finite photometry available for SALT2 fit")
        if t0_init is None:
            t0_init = float(lc.loc[lc["flux"].idxmax(), "mjd"])
        t0_init = float(np.clip(t0_init, lc["mjd"].min(), lc["mjd"].max()))
        x0_init = _estimate_salt2_x0(lc, model, t0_init)

        def residual(theta):
            t0, log_x0, x1, color = theta
            model.set(t0=t0, x0=np.exp(log_x0), x1=x1, c=color)
            pred = model.bandflux(
                lc["native_filter"].to_numpy(),
                lc["mjd"].to_numpy(),
                zp=lc["zp"].to_numpy(),
                zpsys=lc["magsys"].to_numpy(),
            )
            return (pred - lc["flux"].to_numpy()) / lc["flux_err"].to_numpy()

        p0 = np.array([t0_init, np.log(x0_init), 0.0, 0.0])
        bounds = (
            np.array([lc["mjd"].min(), np.log(1e-12), x1_bounds[0], -1.0]),
            np.array([lc["mjd"].max(), np.log(10.0), x1_bounds[1], 2.0]),
        )
        opt = least_squares(residual, p0, bounds=bounds, max_nfev=10000)
        t0, log_x0, x1, color = opt.x
        x0 = float(np.exp(log_x0))
        model.set(t0=t0, x0=x0, x1=x1, c=color)
        t0_err = np.nan
        ndof = max(len(lc) - len(opt.x), 1)
        try:
            cov = np.linalg.inv(opt.jac.T @ opt.jac) * (2 * opt.cost / ndof)
            t0_err = float(np.sqrt(np.diag(cov))[0])
        except np.linalg.LinAlgError:
            pass
        return model, {"t0": float(t0), "t0_err": t0_err, "x0": x0, "x1": float(x1), "c": float(color), "chisq": float(2 * opt.cost), "ndof": ndof}

    round1 = phot[(phot["mjd"] > t_peak_obs - 20) & (phot["mjd"] < t_peak_obs + 50)]
    _, summary1 = fit_subset(round1, None, (-3.0, 3.0))
    phase_mask = _salt2_phase_mask(phot, redshift, summary1["t0"])
    model, summary = fit_subset(phot[phase_mask], summary1["t0"], (-3.0, 10.0))
    phase_mask = _salt2_phase_mask(phot, redshift, summary["t0"])
    if int(phase_mask.sum()) != len(phot[phase_mask]):
        model, summary = fit_subset(phot[phase_mask], summary["t0"], (-3.0, 10.0))
    summary.update({"sampler": "least_squares", "redshift": redshift})
    return summary


def run_bayesn_calibration(
    record: LightCurveRecord,
    salt2_summary: dict[str, Any],
    *,
    filter_yaml: str | Path,
    model_name: str = "W22_model",
    num_devices: int = 4,
    rv: float | None = None,
    output_flux_zp: float = DEFAULT_FLUX_ZP,
) -> tuple[dict[str, Any], Table, dict[str, Any]]:
    """Run BayeSN and return summary, peak fluxes, and raw samples."""

    try:
        from bayesn import SEDmodel
    except ImportError as exc:
        raise ImportError("BayeSN calibration requires the bayesn package") from exc

    redshift = _metadata_float(record, "redshift", "z")
    ebv_mw = _metadata_float(record, "ebv_mw", default=0.0)
    phot = assign_bayesn_filters(_raw_photometry_dataframe(record))
    phot = phot[~phot["bayesn_filter"].isin(EXCLUDED_BAYESN_FILTERS)].copy()
    phot["phase"] = (phot["mjd"] - float(salt2_summary["t0"])) / (1.0 + redshift)
    phot = phot[(phot["phase"] >= MODEL_PHASE_MIN) & (phot["phase"] <= MODEL_PHASE_MAX)].copy()
    phot = _drop_sparse_bayesn_filters(phot)
    if phot.empty:
        raise ValueError(f"{record.object_id}: no photometry remains for BayeSN")
    phot["bayesn_flux"] = phot["flux"] * 10 ** (0.4 * (BAYESN_ZPT - phot["zp"]))
    phot["bayesn_fluxerr"] = phot["flux_err"] * 10 ** (0.4 * (BAYESN_ZPT - phot["zp"]))

    model = SEDmodel(num_devices=num_devices, load_model=model_name, filter_yaml=str(filter_yaml))
    fit_kwargs = {"RV": rv} if rv is not None else {}
    samples, _ = model.fit(
        t=phot["mjd"].values,
        flux=phot["bayesn_flux"].values,
        flux_err=phot["bayesn_fluxerr"].values,
        filters=phot["bayesn_filter"].values,
        z=redshift,
        ebv_mw=ebv_mw,
        peak_mjd=float(salt2_summary["t0"]),
        filt_map={},
        print_summary=False,
        **fit_kwargs,
    )

    peak_filters = _peak_filter_table(record)
    peak_fluxes = estimate_bayesn_peak_fluxes(
        model,
        samples,
        peak_filters,
        redshift=redshift,
        ebv_mw=ebv_mw,
        output_flux_zp=output_flux_zp,
    )
    summary = {
        "model": model_name,
        "redshift": redshift,
        "ebv_mw": ebv_mw,
        "t0_salt2": float(salt2_summary["t0"]),
        "t0_bayesn": _sample_median(samples, "peak_MJD"),
        "t0_err_bayesn": _sample_mad_std(samples, "peak_MJD"),
    }
    for key in ("AV", "theta", "mu", "delM", "tmax"):
        if key in samples:
            summary[f"{key}_median"] = _sample_median(samples, key)
    return summary, peak_fluxes, samples


def calibrate_record(
    record: LightCurveRecord,
    *,
    filter_yaml: str | Path,
    bayesn_model: str = "W22_model",
    bayesn_num_devices: int = 4,
    salt2_model_dir: str | Path | None = None,
    sncosmo_filter_dir: str | Path | None = None,
    rv: float | None = None,
    early_threshold: float = 0.4,
    output_flux_zp: float = DEFAULT_FLUX_ZP,
) -> tuple[CalibrationResult, dict[str, Any]]:
    """Run SALT2, BayeSN, and normalize a record for early-rise fitting."""

    salt2_summary = run_salt2_calibration(
        record,
        salt2_model_dir=salt2_model_dir,
        sncosmo_filter_dir=sncosmo_filter_dir,
    )
    bayesn_summary, peak_fluxes, samples = run_bayesn_calibration(
        record,
        salt2_summary,
        filter_yaml=filter_yaml,
        model_name=bayesn_model,
        num_devices=bayesn_num_devices,
        rv=rv,
        output_flux_zp=output_flux_zp,
    )
    redshift = _metadata_float(record, "redshift", "z")
    calibrated = normalize_record_with_peak_fluxes(
        record,
        peak_fluxes,
        t0=float(salt2_summary["t0"]),
        t0_err=salt2_summary.get("t0_err"),
        redshift=redshift,
        early_threshold=early_threshold,
        output_flux_zp=output_flux_zp,
    )
    return CalibrationResult(calibrated, salt2_summary, bayesn_summary, peak_fluxes), samples


def _raw_photometry_dataframe(record: LightCurveRecord) -> pd.DataFrame:
    df = record.photometry.to_pandas()
    if "native_filter" not in df and "filter" in df:
        df["native_filter"] = df["filter"]
    required = {"mjd", "native_filter", "flux", "flux_err", "zp", "magsys"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{record.object_id}: raw photometry missing columns {sorted(missing)}")
    df = df[np.isfinite(df["mjd"]) & np.isfinite(df["flux"]) & np.isfinite(df["flux_err"]) & (df["flux_err"] > 0)].copy()
    df["magsys"] = df["magsys"].astype(str).str.lower()
    return df


def _peak_filter_table(record: LightCurveRecord) -> Table:
    df = _raw_photometry_dataframe(record)
    rows = df[["native_filter", "magsys"]].drop_duplicates()
    rows = assign_bayesn_filters(rows)
    rows = rows[~rows["bayesn_filter"].isin(EXCLUDED_BAYESN_FILTERS)]
    return Table.from_pandas(rows, index=False)


def _early_fit_mask(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    model_filter: np.ndarray,
    filter_order: list[str],
    early_threshold: float,
) -> np.ndarray:
    peak_mask = (phase < 0) & (phase > -100) & np.isin(model_filter, filter_order)
    early_mask = np.zeros(len(phase), dtype=bool)
    for filt in filter_order:
        idx = (model_filter == filt) & peak_mask
        if not np.any(idx):
            continue
        t_early = _calculate_early_time(phase[idx], flux[idx], flux_err[idx], early_threshold)
        early_mask |= idx & (phase < t_early)
    return early_mask


def _calculate_early_time(phase: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, early_threshold: float) -> float:
    order = np.argsort(phase)
    t = phase[order]
    f = flux[order]
    if len(t) == 0:
        return -np.inf
    flux_max = 100.0
    below = f <= early_threshold * flux_max if early_threshold < 1 else np.ones_like(f, dtype=bool)
    if len(below) > 1:
        below[1:] &= below[:-1]
    idx = below & (t < 0)
    return float(t[idx][-1] + 0.25) if np.any(idx) else -np.inf


def _subset_chains(samples: dict[str, Any], n_samples: int) -> dict[str, Any]:
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


def _metadata_float(record: LightCurveRecord, *keys: str, default: float | None = None) -> float:
    for key in keys:
        if key in record.metadata and record.metadata[key] is not None:
            return float(record.metadata[key])
    if default is not None:
        return float(default)
    raise ValueError(f"{record.object_id}: metadata missing one of {keys}")


def _sample_median(samples: dict[str, Any], key: str) -> float:
    return float(np.nanmedian(np.asarray(samples[key]).reshape(-1)))


def _sample_mad_std(samples: dict[str, Any], key: str) -> float:
    values = np.asarray(samples[key]).reshape(-1)
    values = values[np.isfinite(values)]
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med)))


def _drop_sparse_bayesn_filters(phot: pd.DataFrame) -> pd.DataFrame:
    counts = phot["bayesn_filter"].value_counts()
    keep_filters = counts[counts >= MIN_OBS_PER_FILTER].index
    return phot[phot["bayesn_filter"].isin(keep_filters)].copy()


def _salt2_phase_mask(phot: pd.DataFrame, redshift: float, t0: float) -> np.ndarray:
    phase = (phot["mjd"].to_numpy(dtype=float) - float(t0)) / (1.0 + float(redshift))
    return (phase > MODEL_PHASE_MIN) & (phase < MODEL_PHASE_MAX)


def _estimate_salt2_x0(lc: pd.DataFrame, model: Any, t0: float) -> float:
    model.set(t0=t0, x0=1.0, x1=0.0, c=0.0)
    pred = model.bandflux(
        lc["native_filter"].to_numpy(),
        lc["mjd"].to_numpy(),
        zp=lc["zp"].to_numpy(),
        zpsys=lc["magsys"].to_numpy(),
    )
    good = np.isfinite(pred) & np.isfinite(lc["flux"]) & np.isfinite(lc["flux_err"]) & (lc["flux_err"] > 0)
    if not np.any(good):
        return 1e-3
    w = 1.0 / lc.loc[good, "flux_err"].to_numpy() ** 2
    numerator = np.sum(lc.loc[good, "flux"].to_numpy() * pred[good] * w)
    denominator = np.sum(pred[good] ** 2 * w)
    if denominator <= 0 or not np.isfinite(numerator):
        return 1e-3
    return float(np.clip(numerator / denominator, 1e-12, 10.0))


def _make_salt2_source(sncosmo: Any, salt2_model_dir: str | Path | None):
    if salt2_model_dir is not None:
        return sncosmo.SALT2Source(str(salt2_model_dir), name="salt2", version="local")
    return sncosmo.get_source("salt2", version="T21")


def _register_sncosmo_filters(sncosmo: Any, filter_dir: str | Path | None) -> None:
    if filter_dir is None:
        return
    filter_dir = Path(filter_dir)
    for path in filter_dir.rglob("*.dat"):
        name = path.stem
        if name.startswith("ps1_"):
            name = "ps1::" + name.split("_", 1)[1]
        wave, trans = np.loadtxt(path, unpack=True)
        sncosmo.registry.register(sncosmo.Bandpass(wave, trans, name=name), name=name, force=True)
