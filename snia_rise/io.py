"""Portable light-curve bundle IO for single-object fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.table import Table, vstack


SCHEMA_VERSION = 1
REQUIRED_PHOTOMETRY_COLUMNS = {"model_filter"}
NORMALIZED_COLUMNS = {"phase", "normalized_flux", "normalized_flux_err"}
LEGACY_NORMALIZED_COLUMNS = {"phase", "flux", "flux_err"}
RAW_PHOTOMETRY_COLUMNS = {"mjd", "native_filter", "flux", "flux_err", "zp", "magsys"}


@dataclass
class LightCurveRecord:
    """Model-ready light curve and metadata for one supernova."""

    object_id: str
    photometry: Table
    metadata: dict[str, Any]
    filter_order: list[str]

    def validate(self) -> None:
        missing = REQUIRED_PHOTOMETRY_COLUMNS - set(self.photometry.colnames)
        if missing:
            raise ValueError(f"photometry is missing required columns: {sorted(missing)}")
        has_normalized = NORMALIZED_COLUMNS <= set(self.photometry.colnames)
        has_legacy_normalized = LEGACY_NORMALIZED_COLUMNS <= set(self.photometry.colnames)
        has_raw = RAW_PHOTOMETRY_COLUMNS <= set(self.photometry.colnames)
        if not (has_normalized or has_legacy_normalized or has_raw):
            raise ValueError(
                "photometry must contain either normalized columns "
                f"{sorted(NORMALIZED_COLUMNS)} or raw calibrated columns "
                f"{sorted(RAW_PHOTOMETRY_COLUMNS)}"
            )
        if not self.filter_order:
            raise ValueError("filter_order must contain at least one filter")

        filt = np.asarray(self.photometry["model_filter"], dtype=str)
        unknown = sorted(set(filt) - set(self.filter_order))
        if unknown:
            raise ValueError(
                f"{self.object_id}: model_filter values {unknown} are not in filter_order"
            )

        for col in ("phase", "normalized_flux", "normalized_flux_err", "flux", "flux_err", "mjd", "zp"):
            if col not in self.photometry.colnames:
                continue
            values = np.asarray(self.photometry[col], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{self.object_id}: column {col} contains non-finite values")
        for col in ("normalized_flux_err", "flux_err"):
            if col in self.photometry.colnames and np.any(np.asarray(self.photometry[col], dtype=float) <= 0):
                raise ValueError(f"{self.object_id}: {col} must be positive")

    def has_normalized_photometry(self) -> bool:
        """Return whether the record can be passed to the rise-time model."""

        cols = set(self.photometry.colnames)
        return NORMALIZED_COLUMNS <= cols or LEGACY_NORMALIZED_COLUMNS <= cols

    def to_light_curve(self, include_peak: bool = True):
        """Convert the record into the existing ``SNLightCurve`` model input."""

        self.validate()
        if not self.has_normalized_photometry():
            raise ValueError(
                f"{self.object_id}: no normalized photometry available. "
                "Run the SALT2/BayeSN calibration stage first."
            )
        table = self.photometry
        flux_col = "normalized_flux" if "normalized_flux" in table.colnames else "flux"
        flux_err_col = (
            "normalized_flux_err" if "normalized_flux_err" in table.colnames else "flux_err"
        )

        if "in_early_fit" in table.colnames:
            early_mask = np.asarray(table["in_early_fit"], dtype=bool)
        else:
            early_mask = np.ones(len(table), dtype=bool)
        if not np.any(early_mask):
            raise ValueError(f"{self.object_id}: no rows selected for early-time fitting")

        if include_peak and "in_peak_plot" in table.colnames:
            peak_mask = np.asarray(table["in_peak_plot"], dtype=bool)
        else:
            peak_mask = early_mask

        filter_id = {name: idx + 1 for idx, name in enumerate(self.filter_order)}
        filt = np.array([filter_id[str(name)] for name in table["model_filter"]], dtype=int)

        if "stream_id" in table.colnames:
            raw_stream = np.asarray(table["stream_id"], dtype=str)
        elif "fcqfid" in table.colnames:
            raw_stream = np.asarray(table["fcqfid"], dtype=str)
        else:
            raw_stream = np.asarray(table["model_filter"], dtype=str)
        stream_labels = {name: idx + 1 for idx, name in enumerate(sorted(set(raw_stream)))}
        fcqfid = np.array([stream_labels[name] for name in raw_stream], dtype=int)

        beta = (
            np.asarray(table["beta"], dtype=float)
            if "beta" in table.colnames
            else np.ones(len(table), dtype=float)
        )

        def _package(mask: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "phase": np.asarray(table["phase"], dtype=float)[mask],
                "flux": np.asarray(table[flux_col], dtype=float)[mask],
                "flux_err": np.asarray(table[flux_err_col], dtype=float)[mask],
                "fcqfid": fcqfid[mask],
                "filt": filt[mask],
                "beta": beta[mask],
            }

        t0_err = self.metadata.get("t0_err")
        if t0_err is not None:
            t0_err = float(t0_err)

        from .model.lightcurve import SNLightCurve

        return SNLightCurve(
            lc_early=_package(early_mask),
            lc_peak=_package(peak_mask) if include_peak else None,
            t0_err=t0_err,
            ztfid=self.object_id,
            filt_classes_global=list(range(1, len(self.filter_order) + 1)),
        )


def read_light_curve_bundle(path: str | Path) -> dict[str, LightCurveRecord]:
    """Read a portable light-curve bundle directory."""

    path = Path(path)
    with open(path / "manifest.yaml", "r") as f:
        manifest = yaml.safe_load(f) or {}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema_version={manifest.get('schema_version')!r}"
        )

    filter_order = list(manifest.get("filter_order") or [])
    objects = Table.read(path / "objects.ecsv", format="ascii.ecsv")
    photometry = Table.read(path / "photometry.ecsv", format="ascii.ecsv")
    if "object_id" not in objects.colnames or "object_id" not in photometry.colnames:
        raise ValueError("objects.ecsv and photometry.ecsv must contain object_id")

    records: dict[str, LightCurveRecord] = {}
    for row in objects:
        object_id = str(row["object_id"])
        metadata = {
            name: row[name].item() if hasattr(row[name], "item") else row[name]
            for name in objects.colnames
            if name != "object_id"
        }
        rows = photometry[np.asarray(photometry["object_id"], dtype=str) == object_id]
        if len(rows) == 0:
            raise ValueError(f"{object_id}: no photometry rows in bundle")
        record = LightCurveRecord(object_id, rows, metadata, filter_order)
        record.validate()
        records[object_id] = record
    return records


def write_light_curve_bundle(
    path: str | Path,
    records: list[LightCurveRecord] | dict[str, LightCurveRecord],
    manifest: dict[str, Any] | None = None,
) -> None:
    """Write records to a portable light-curve bundle directory."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    record_list = list(records.values()) if isinstance(records, dict) else list(records)
    if not record_list:
        raise ValueError("At least one LightCurveRecord is required")

    filter_order = record_list[0].filter_order
    for record in record_list:
        if record.filter_order != filter_order:
            raise ValueError("All records in a bundle must share filter_order")
        record.validate()

    manifest_out = dict(manifest or {})
    manifest_out.update({"schema_version": SCHEMA_VERSION, "filter_order": filter_order})
    with open(path / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest_out, f, sort_keys=False)

    metadata_keys = sorted({key for record in record_list for key in record.metadata})
    objects = Table(
        rows=[
            [record.object_id, *[record.metadata.get(key) for key in metadata_keys]]
            for record in record_list
        ],
        names=["object_id", *metadata_keys],
    )
    objects.write(path / "objects.ecsv", format="ascii.ecsv", overwrite=True)

    tables = []
    for record in record_list:
        table = record.photometry.copy()
        if "object_id" not in table.colnames:
            table.add_column([record.object_id] * len(table), name="object_id", index=0)
        tables.append(table)
    photometry = tables[0] if len(tables) == 1 else vstack(tables, metadata_conflicts="silent")
    photometry.write(path / "photometry.ecsv", format="ascii.ecsv", overwrite=True)
