from __future__ import annotations

import importlib.util
from pathlib import Path
import pandas as pd
import warnings


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "data" / "ztf_snia_early_late" / "ztf_early_late_phot.py"
SPEC = importlib.util.spec_from_file_location("ztf_early_late_phot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
phot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phot)


def _rows(values: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(values, orient="index")


def test_salt_selection_uses_external_only_above_half_day_threshold() -> None:
    salt_ztf = _rows(
        {
            "GOOD": {"ztfid": "GOOD", "t0_err": 0.5, "t0": 1.0, "x1": 0.0, "x1_err": 0.1},
            "BAD": {"ztfid": "BAD", "t0_err": 0.51, "t0": 2.0, "x1": 0.0, "x1_err": 0.1},
        }
    )
    salt_external = _rows(
        {"BAD": {"ztfid": "BAD", "t0_err": 0.2, "t0": 3.0, "x1": 1.0, "x1_err": 0.2}}
    )

    row, used_external = phot.select_salt_row("GOOD", salt_ztf, salt_external)
    assert used_external is False
    assert row["t0"] == 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row, used_external = phot.select_salt_row("BAD", salt_ztf, salt_external)
    assert used_external is True
    assert row["t0"] == 3.0


def test_bayesn_selection_uses_external_for_flux_and_extinction() -> None:
    bayesn_ztf = _rows(
        {"BAD": {"ztfid": "BAD", "t0_err_bayesn": 0.51, "AV_median": 0.3}}
    )
    bayesn_external = _rows(
        {"BAD": {"ztfid": "BAD", "t0_err_bayesn": 0.2, "AV_median": 0.6}}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row, used_external = phot.select_bayesn_row("BAD", bayesn_ztf, bayesn_external)

    assert used_external is True
    assert row["AV_median"] == 0.6


def test_build_phot_table_has_required_final_flags() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        table = phot.build_phot_table()

    assert list(table.columns[-2:]) == ["salt2_external", "bayesn_external"]
    assert table["ztfid"].is_unique
    assert (table["flux_zp"] > 0).all()
    assert (table[["ztfg_flux_max", "ztfr_flux_max"]] > 0).all().all()
    assert bool(table.loc[table["ztfid"].eq("ZTF24aahgaov"), "salt2_external"].item()) is True


if __name__ == "__main__":
    test_salt_selection_uses_external_only_above_half_day_threshold()
    test_bayesn_selection_uses_external_for_flux_and_extinction()
    test_build_phot_table_has_required_final_flags()
