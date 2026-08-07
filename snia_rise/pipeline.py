"""Package-level single-supernova fitting pipeline.

This module is the public, script-independent interface for exporting raw ZTF
photometry, calibrating it with SALT2/BayeSN, and fitting the early-rise model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table

from .calibration import calibrate_record
from .fitting import fit_light_curve_record, save_single_fit_result
from .io import LightCurveRecord, read_light_curve_bundle, write_light_curve_bundle
from .prior_registry import resolve_prior_config


ZTF_FILTER_MAP = {"ZTF_g": "ztfg", "ZTF_r": "ztfr", "ztfg": "ztfg", "ztfr": "ztfr"}
EXTERNAL_GR_MODEL_FILTERS = {
    "sdssg": "ztfg",
    "swope2g": "ztfg",
    "ps1::g": "ztfg",
    "sdssr": "ztfr",
    "swope2r": "ztfr",
    "ps1::r": "ztfr",
}


def export_ztf_early_late_record(
    data_root: str | Path,
    object_id: str,
    *,
    include_external_gr: bool = False,
) -> LightCurveRecord:
    """Export one early/late ZTF object into a raw portable record."""

    data_root = Path(data_root)
    meta = pd.read_csv(data_root / "ztf_early_Ia_meta.csv")
    meta_row = meta.loc[meta["objid"].astype(str).eq(object_id)]
    if len(meta_row) != 1:
        raise ValueError(f"No unique metadata row for {object_id}")
    meta_row = meta_row.iloc[0]

    ztf = _read_ztf_forced_photometry(data_root / "light_curve_fps_ztf" / f"{object_id}_fnu.csv")
    frames = [ztf]
    if include_external_gr:
        external_path = data_root / "light_curve_external" / f"{object_id}_external.csv"
        if external_path.exists():
            frames.append(_read_external_gr(external_path))
    phot = pd.concat(frames, ignore_index=True)
    phot = phot[np.isfinite(phot["flux"]) & np.isfinite(phot["flux_err"]) & (phot["flux_err"] > 0)].copy()
    if phot.empty:
        raise ValueError(f"{object_id}: no usable g/r photometry")

    record = LightCurveRecord(
        object_id=object_id,
        photometry=Table.from_pandas(phot, index=False),
        metadata={
            "name": meta_row.get("name", ""),
            "redshift": float(meta_row["z"]),
            "ebv_mw": float(meta_row.get("ebv", meta_row.get("ebv_mw", 0.0))),
            "source": "ztf_early_late",
        },
        filter_order=["ztfg", "ztfr"],
    )
    record.validate()
    return record


def export_ztf_early_late_bundle(
    data_root: str | Path,
    object_id: str,
    output: str | Path,
    *,
    include_external_gr: bool = False,
) -> LightCurveRecord:
    """Export one early/late ZTF object and write a raw bundle."""

    record = export_ztf_early_late_record(
        data_root,
        object_id,
        include_external_gr=include_external_gr,
    )
    write_light_curve_bundle(
        output,
        [record],
        manifest={
            "generated_by": "snia_rise.pipeline.export_ztf_early_late_bundle",
            "source": "ztf_early_late",
            "include_external_gr": bool(include_external_gr),
        },
    )
    return record


def calibrate_single_sn_bundle(
    bundle: str | Path,
    output: str | Path,
    *,
    object_id: str | None = None,
    filter_yaml: str | Path,
    bayesn_model: str = "W22_model",
    bayesn_num_devices: int = 4,
    salt2_model_dir: str | Path | None = None,
    sncosmo_filter_dir: str | Path | None = None,
    rv: float | None = None,
    early_threshold: float = 0.4,
    output_flux_zp: float = 30.0,
):
    """Run SALT2/BayeSN calibration for one object in a raw bundle."""

    record = select_record(read_light_curve_bundle(bundle), object_id)
    result, samples = calibrate_record(
        record,
        filter_yaml=filter_yaml,
        bayesn_model=bayesn_model,
        bayesn_num_devices=bayesn_num_devices,
        salt2_model_dir=salt2_model_dir,
        sncosmo_filter_dir=sncosmo_filter_dir,
        rv=rv,
        early_threshold=early_threshold,
        output_flux_zp=output_flux_zp,
    )

    output = Path(output)
    write_light_curve_bundle(
        output,
        [result.record],
        manifest={
            "generated_by": "snia_rise.pipeline.calibrate_single_sn_bundle",
            "source_bundle": str(bundle),
            "calibration": "salt2_bayesn",
        },
    )

    import yaml

    cal_dir = output / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "salt2_summary.yaml", "w") as f:
        yaml.safe_dump(result.salt2_summary, f, sort_keys=False)
    with open(cal_dir / "bayesn_summary.yaml", "w") as f:
        yaml.safe_dump(result.bayesn_summary, f, sort_keys=False)
    result.peak_fluxes.write(cal_dir / "peak_fluxes.ecsv", format="ascii.ecsv", overwrite=True)
    np.savez_compressed(cal_dir / "bayesn_samples.npz", **{key: np.asarray(value) for key, value in samples.items()})
    return result


def fit_single_sn_bundle(
    bundle: str | Path,
    output: str | Path,
    *,
    object_id: str | None = None,
    prior: str | None = None,
    prior_config: str | None = None,
    rise_model: str = "power_law",
    sample_beta: bool = False,
    prior_type: str = "maximum_entropy",
    prior_kwargs: dict[str, Any] | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    nuts_params: dict[str, Any] | None = None,
    command: list[str] | None = None,
):
    """Fit one calibrated bundle with the unpooled early-rise model."""

    record = select_record(read_light_curve_bundle(bundle), object_id)
    prior_kwargs = dict(prior_kwargs or {})
    prior_config_dict, prior_profile = resolve_prior_config(
        prior=prior,
        prior_config=prior_config,
        rise_model=rise_model,
        sample_beta=sample_beta,
        filter_order=record.filter_order,
        prior_type=prior_type,
        **prior_kwargs,
    )
    sampling = {
        "num_samples": 1000,
        "num_warmup": 3000,
        "num_chains": 2,
        "thinning": 1,
        "random_seed": 11,
        "prior_pred_samples": 500,
    }
    sampling.update(sampling_kwargs or {})
    light_curve = fit_light_curve_record(
        record,
        prior_config_dict,
        nuts_params=nuts_params or {},
        **sampling,
    )
    run_config = {
        "command": command or [],
        "object_id": record.object_id,
        "bundle": str(bundle),
        "filter_order": record.filter_order,
        "model": rise_model,
        "sampling": {**sampling, "nuts_params": nuts_params or {}},
        "prior": prior_profile.get("name"),
    }
    save_single_fit_result(
        output,
        record,
        light_curve,
        prior_config_dict,
        prior_profile,
        run_config,
    )
    return light_curve


def select_record(records: dict[str, LightCurveRecord], object_id: str | None) -> LightCurveRecord:
    """Select one record from a bundle mapping."""

    if object_id is None:
        if len(records) != 1:
            raise ValueError("object_id is required when the bundle contains multiple objects")
        return next(iter(records.values()))
    if object_id not in records:
        raise ValueError(f"Object {object_id!r} not found in bundle")
    return records[object_id]


def _read_ztf_forced_photometry(path: Path) -> pd.DataFrame:
    lc = pd.read_csv(path, sep=r"\s+", comment="#")
    lc = lc.rename(columns={key: key.replace(",", "") for key in lc.columns})
    lc = lc[lc["filter"].isin(["ZTF_g", "ZTF_r"])].copy()
    lc["native_filter"] = lc["filter"].map(ZTF_FILTER_MAP)
    lc["model_filter"] = lc["native_filter"]
    lc["mjd"] = lc["jd"] - 2400000.5
    lc["zp"] = 30.0
    lc["magsys"] = "ab"
    delta_zp = lc["zpdiff"] - lc["zp"]
    lc["flux"] = lc["forcediffimflux"] * 10 ** (-0.4 * delta_zp)
    floor = lc["filter"].map({"ZTF_g": 0.025, "ZTF_r": 0.035}).to_numpy()
    lc["flux_err"] = np.sqrt(lc["forcediffimfluxunc"] ** 2 + (floor * lc["forcediffimflux"]) ** 2) * 10 ** (-0.4 * delta_zp)
    lc["origin"] = "ztf"
    lc["source"] = "ZTF"
    lc["stream_id"] = (
        lc["field"].astype(str)
        + "_"
        + lc["ccdid"].astype(str)
        + "_"
        + lc["qid"].astype(str)
        + "_"
        + lc["native_filter"].astype(str)
    )
    return lc[["mjd", "native_filter", "model_filter", "flux", "flux_err", "zp", "magsys", "origin", "source", "stream_id"]]


def _read_external_gr(path: Path) -> pd.DataFrame:
    lc = pd.read_csv(path)
    lc = lc[lc["filter"].isin(EXTERNAL_GR_MODEL_FILTERS)].copy()
    lc["native_filter"] = lc["filter"].astype(str)
    lc["model_filter"] = lc["native_filter"].map(EXTERNAL_GR_MODEL_FILTERS)
    lc["flux_err"] = lc["fluxerr"]
    lc["origin"] = "external"
    lc["stream_id"] = lc["source"].astype(str) + "_" + lc["native_filter"].astype(str)
    return lc[["mjd", "native_filter", "model_filter", "flux", "flux_err", "zp", "magsys", "origin", "source", "stream_id"]]
