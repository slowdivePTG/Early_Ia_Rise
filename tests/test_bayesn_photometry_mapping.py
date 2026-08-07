from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from astropy.table import Table

import ztf_early_late_lc_bayesn as bayesn_fit
import ztf_early_late_lc_salt as salt_fit


def test_resolve_bayesn_filter_uses_magsys() -> None:
    assert bayesn_fit.resolve_bayesn_filter("ztfg", "ab") == "p48g"
    assert bayesn_fit.resolve_bayesn_filter("bessellb", "vega") == "B"
    assert bayesn_fit.resolve_bayesn_filter("bessellb", "ab") == "B_AB"
    assert bayesn_fit.resolve_bayesn_filter("sdssg", "ab") == "sdssg_AB"
    assert bayesn_fit.resolve_bayesn_filter("sdssg", "bd17") == "g_prime"
    assert bayesn_fit.resolve_bayesn_filter("swope2r", "ab") == "r_CSP2_AB"
    assert bayesn_fit.resolve_bayesn_filter("swope2r", "bd17") == "r_CSP2"


def test_resolve_bayesn_filter_rejects_unknown_system() -> None:
    try:
        bayesn_fit.resolve_bayesn_filter("sdssg", "vega")
    except ValueError as exc:
        assert "No BayeSN filter mapping" in str(exc)
    else:
        raise AssertionError("unknown sdssg/vega mapping did not fail")


def test_to_bayesn_fluxcal_preserves_system_specific_aliases() -> None:
    lc = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0, 4.0],
            "filter": ["ztfg", "ztfr", "bessellb", "bessellb"],
            "flux": [10.0, 20.0, 30.0, 40.0],
            "fluxerr": [1.0, 2.0, 3.0, 4.0],
            "zp": [30.0, 30.0, 25.0, 25.0],
            "magsys": ["ab", "ab", "ab", "vega"],
        }
    )

    out = bayesn_fit.to_bayesn_fluxcal(lc)

    assert out["bayesn_filter"].tolist() == ["p48g", "p48r", "B_AB", "B"]
    expected_scale = 10 ** (0.4 * (bayesn_fit.BAYESN_ZPT - lc["zp"]))
    np.testing.assert_allclose(out["bayesn_flux"], lc["flux"] * expected_scale)
    np.testing.assert_allclose(out["bayesn_fluxerr"], lc["fluxerr"] * expected_scale)


def test_parse_args_defaults_to_ztf_only() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["ztf_early_late_lc_bayesn.py"]
        args = bayesn_fit.parse_args()
    finally:
        sys.argv = original_argv

    assert args.external is False
    assert args.no_ztf is False
    assert args.rv is None


def test_parse_args_supports_external_only() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["ztf_early_late_lc_bayesn.py", "--external", "--no-ztf"]
        args = bayesn_fit.parse_args()
    finally:
        sys.argv = original_argv

    assert args.external is True
    assert args.no_ztf is True


def test_bayesn_sparse_cut_keeps_two_ztf_rows_but_drops_external() -> None:
    lc = pd.DataFrame(
        {
            "bayesn_filter": ["p48r", "p48r", "sdssr_AB", "sdssr_AB", "sdssg_AB", "sdssg_AB", "sdssg_AB"],
            "_survey": ["ZTF", "ZTF", "external", "external", "external", "external", "external"],
        }
    )

    out = bayesn_fit.drop_sparse_bayesn_filters_after_phase_cut(lc, "TEST")

    assert out["bayesn_filter"].tolist() == ["p48r", "p48r", "sdssg_AB", "sdssg_AB", "sdssg_AB"]


def test_salt_sparse_cut_keeps_two_ztf_rows_but_drops_external() -> None:
    lc = Table(
        {
            "filter": ["ztfr", "ztfr", "sdssr", "sdssr", "sdssg", "sdssg", "sdssg"],
            "_survey": ["ZTF", "ZTF", "external", "external", "external", "external", "external"],
        }
    )

    out = salt_fit.drop_sparse_filters_after_phase_cut(lc, "TEST")

    assert list(out["filter"]) == ["ztfr", "ztfr", "sdssg", "sdssg", "sdssg"]


if __name__ == "__main__":
    test_resolve_bayesn_filter_uses_magsys()
    test_resolve_bayesn_filter_rejects_unknown_system()
    test_to_bayesn_fluxcal_preserves_system_specific_aliases()
    test_parse_args_defaults_to_ztf_only()
    test_parse_args_supports_external_only()
    test_bayesn_sparse_cut_keeps_two_ztf_rows_but_drops_external()
    test_salt_sparse_cut_keeps_two_ztf_rows_but_drops_external()
