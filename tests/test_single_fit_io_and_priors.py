from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from astropy.table import Table

from snia_rise.cli.fit_single import build_parser
from snia_rise.io import LightCurveRecord, read_light_curve_bundle, write_light_curve_bundle
from snia_rise.prior_registry import (
    list_builtin_priors,
    read_prior_profile,
    resolve_prior_config,
)


def _record(object_id: str = "TEST") -> LightCurveRecord:
    phot = Table(
        {
            "phase": [-12.0, -10.0, -8.0],
            "flux": [1.0, 2.0, 4.0],
            "flux_err": [0.2, 0.2, 0.3],
            "model_filter": ["ztfr", "ztfr", "ztfr"],
            "stream_id": ["field-r", "field-r", "field-r"],
            "beta": [1.0, 1.0, 1.0],
            "in_early_fit": [True, True, True],
            "in_peak_plot": [True, True, True],
        }
    )
    return LightCurveRecord(
        object_id=object_id,
        photometry=phot,
        metadata={"t0_err": 0.1, "z": 0.02},
        filter_order=["ztfg", "ztfr"],
    )


def test_bundle_roundtrip_preserves_filter_order_for_single_band_object() -> None:
    record = _record()
    with tempfile.TemporaryDirectory() as tmp:
        write_light_curve_bundle(tmp, [record])
        loaded = read_light_curve_bundle(tmp)["TEST"]

    assert loaded.filter_order == ["ztfg", "ztfr"]
    light_curve = loaded.to_light_curve()
    assert np.unique(light_curve.idx_filt).tolist() == [1]


def test_builtin_population_prior_validates_filter_order() -> None:
    config, profile = resolve_prior_config(
        prior="ztf-dr2-normal-frac40-mvn",
        filter_order=["ztfg", "ztfr"],
    )

    assert profile["name"] == "ztf-dr2-normal-frac40-mvn"
    assert "population_priors" in config


def test_prior_filter_order_mismatch_fails() -> None:
    try:
        resolve_prior_config(
            prior="ztf-dr2-normal-frac40-mvn",
            filter_order=["ztfr", "ztfg"],
        )
    except ValueError as exc:
        assert "filter_order" in str(exc)
    else:
        raise AssertionError("filter order mismatch did not fail")


def test_all_builtin_prior_profiles_load() -> None:
    names = list_builtin_priors()
    assert "ztf-dr2-normal-frac40-mvn" in names
    for name in names:
        profile = read_prior_profile(name)
        resolve_prior_config(prior=name, filter_order=profile.get("filter_order"))


def test_cli_parser_supports_listing_priors() -> None:
    args = build_parser().parse_args(["--list-priors"])

    assert args.list_priors is True


def test_custom_prior_config_path_loads() -> None:
    profile = read_prior_profile("ztf-dr2-normal-frac40-t-rise")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prior.yaml"
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(profile, f)
        config, loaded = resolve_prior_config(
            prior_config=str(path),
            filter_order=["ztfg", "ztfr"],
        )

    assert loaded["name"] == "ztf-dr2-normal-frac40-t-rise"
    assert config["population_priors"]["t_rise"]["mean"] == 18.55


def test_public_pipeline_api_imports_from_package_root() -> None:
    import snia_rise

    assert callable(snia_rise.export_ztf_early_late_bundle)
    assert callable(snia_rise.calibrate_single_sn_bundle)
    assert callable(snia_rise.fit_single_sn_bundle)
