from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from astropy.table import Table

from snia_rise.cli.fit_single import build_parser
from snia_rise.fitting import save_single_fit_result, summarize_posterior
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


def test_summary_includes_single_object_rise_time() -> None:
    class LightCurve:
        post_sample = xr.Dataset(
            {
                "t_rise": (("chain", "draw", "obj"), np.array([[[15.0], [17.0], [19.0]]])),
                "alpha_0": (
                    ("chain", "draw", "obj", "filt"),
                    np.array([[[[2.0, 2.5]], [[3.0, 3.5]], [[4.0, 4.5]]]]),
                ),
            },
            coords={"obj": ["TEST"], "filt": ["ztfg", "ztfr"]},
        )

    summary = summarize_posterior(LightCurve())
    assert summary is not None
    rows = {(row["parameter"], row["filter"]): row for row in summary}

    assert ("t_rise", "") in rows
    assert rows[("t_rise", "")]["median"] == 17.0
    assert ("alpha_0", "ztfg") in rows
    assert ("alpha_0", "ztfr") in rows


def test_save_single_fit_result_writes_light_curve_diagnostic() -> None:
    class LightCurve:
        def __init__(self) -> None:
            self.inf_data = None
            self.post_sample = xr.Dataset(
                {"t_rise": (("chain", "draw", "obj"), np.array([[[16.0], [18.0]]]))},
                coords={"obj": [0]},
            )
            self.plot_calls = []

        def plot_lc(self, *, save: bool = False, filename: str | None = None, **kwargs) -> None:
            self.plot_calls.append({"save": save, "filename": filename, **kwargs})
            if save and filename is not None:
                Path(filename + ".pdf").write_bytes(b"%PDF-1.4\n")

    light_curve = LightCurve()
    with tempfile.TemporaryDirectory() as tmp:
        save_single_fit_result(
            tmp,
            _record(),
            light_curve,
            prior_config={},
            prior_profile={"name": "test"},
            run_config={},
        )

        assert light_curve.plot_calls == [{"save": True, "filename": str(Path(tmp) / "light_curve")}]
        assert (Path(tmp) / "light_curve.pdf").exists()
        assert (Path(tmp) / "summary.ecsv").exists()
