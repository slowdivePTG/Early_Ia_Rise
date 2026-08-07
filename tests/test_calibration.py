from __future__ import annotations

import numpy as np
from astropy.table import Table

from snia_rise.cli.calibrate_single import build_parser
from snia_rise.cli.export_ztf_early_late import build_parser as build_export_parser
from snia_rise.calibration import (
    estimate_bayesn_peak_fluxes,
    normalize_record_with_peak_fluxes,
    resolve_bayesn_filter,
)
from snia_rise.io import LightCurveRecord


class FakeBayeSNModel:
    def get_flux_from_chains(self, phase, filters, samples, z, ebv_mw, mag=False):
        n_sample = np.asarray(samples["peak_MJD"]).reshape(-1).size
        out = np.zeros((n_sample, len(filters), len(phase)))
        for i in range(n_sample):
            for j, _ in enumerate(filters):
                amp = (j + 1) * (10.0 + i)
                out[i, j, :] = amp * np.exp(-0.5 * (phase / 5.0) ** 2)
        return out[None, ...]


def _raw_record() -> LightCurveRecord:
    phot = Table(
        {
            "mjd": [100.0, 101.0, 102.0, 100.5, 101.5, 102.5],
            "native_filter": ["ztfg", "ztfg", "ztfg", "ztfr", "ztfr", "ztfr"],
            "model_filter": ["ztfg", "ztfg", "ztfg", "ztfr", "ztfr", "ztfr"],
            "flux": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            "flux_err": [0.1, 0.1, 0.2, 0.2, 0.2, 0.3],
            "zp": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            "magsys": ["ab", "ab", "ab", "ab", "ab", "ab"],
            "stream_id": ["g", "g", "g", "r", "r", "r"],
        }
    )
    return LightCurveRecord(
        object_id="TEST",
        photometry=phot,
        metadata={"redshift": 0.0, "ebv_mw": 0.01},
        filter_order=["ztfg", "ztfr"],
    )


def test_raw_bundle_record_validates_but_requires_normalization_for_model() -> None:
    record = _raw_record()
    record.validate()
    assert record.has_normalized_photometry() is False

    try:
        record.to_light_curve()
    except ValueError as exc:
        assert "calibration" in str(exc)
    else:
        raise AssertionError("raw record unexpectedly converted to SNLightCurve")


def test_bayesn_filter_mapping_is_system_aware() -> None:
    assert resolve_bayesn_filter("ztfg", "ab") == "p48g"
    assert resolve_bayesn_filter("sdssg", "ab") == "sdssg_AB"


def test_estimate_bayesn_peak_fluxes_reports_percentiles() -> None:
    peak_filters = Table(
        {
            "native_filter": ["ztfg", "ztfr"],
            "magsys": ["ab", "ab"],
        }
    )
    peaks = estimate_bayesn_peak_fluxes(
        FakeBayeSNModel(),
        {"peak_MJD": np.arange(4).reshape(1, 4)},
        peak_filters,
        redshift=0.0,
        ebv_mw=0.0,
        output_flux_zp=27.5,
        posterior_samples=4,
    )

    assert list(peaks["bayesn_filter"]) == ["p48g", "p48r"]
    np.testing.assert_allclose(peaks["peak_flux_median"], [11.5, 23.0], rtol=1e-3)
    assert np.all(peaks["peak_flux_p84"] > peaks["peak_flux_p16"])


def test_normalize_record_uses_peak_median_and_preserves_raw_flux() -> None:
    record = _raw_record()
    peaks = Table(
        {
            "native_filter": ["ztfg", "ztfr"],
            "magsys": ["ab", "ab"],
            "bayesn_filter": ["p48g", "p48r"],
            "flux_zp": [30.0, 30.0],
            "peak_flux_p16": [9.0, 18.0],
            "peak_flux_median": [10.0, 20.0],
            "peak_flux_p84": [11.0, 22.0],
            "posterior_draws": [4, 4],
        }
    )
    calibrated = normalize_record_with_peak_fluxes(
        record,
        peaks,
        t0=103.0,
        t0_err=0.2,
        redshift=0.0,
        early_threshold=0.4,
    )

    assert calibrated.has_normalized_photometry() is True
    np.testing.assert_allclose(calibrated.photometry["flux"], record.photometry["flux"])
    np.testing.assert_allclose(calibrated.photometry["normalized_flux"], [10, 20, 30, 10, 20, 30])
    assert "in_early_fit" in calibrated.photometry.colnames


def test_calibration_cli_parser() -> None:
    args = build_parser().parse_args(
        ["--bundle", "raw", "--output", "cal", "--filter-yaml", "filters.yaml"]
    )

    assert str(args.bundle) == "raw"
    assert str(args.output) == "cal"


def test_export_cli_parser() -> None:
    args = build_export_parser().parse_args(["--object", "ZTF24aahgaov", "--output", "bundle"])

    assert args.object_id == "ZTF24aahgaov"
    assert str(args.output) == "bundle"
