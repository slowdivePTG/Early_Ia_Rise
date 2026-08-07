from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from snia_rise.ztf_lc import SampleConfig, ZTFIaEarlyLate
from snia_rise.ztf_lc import ZTFDataProcessor


def test_early_late_suffix_marks_ztf_normalization() -> None:
    config = SampleConfig(source="early_late")

    assert config.get_filename_suffix().endswith("_phot_ztf_norm")


def test_early_late_loader_has_no_atlas_path() -> None:
    assert not hasattr(ZTFIaEarlyLate, "atlas_lc_path")


def test_external_gr_normalization_uses_native_peak_flux() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "external.csv"
        pd.DataFrame(
            {
                "mjd": [10.0, 11.0],
                "filter": ["sdssg", "ps1::g"],
                "bayesn_filter": ["sdssg_AB", "g_PS1"],
                "flux": [20.0, 40.0],
                "fluxerr": [2.0, 4.0],
                "zp": [30.0, 30.0],
                "source": ["A", "A"],
            }
        ).to_csv(path, index=False)

        loaded, _ = ZTFDataProcessor.load_external_gr_photometry(
            path,
            t0=10.0,
            z=0.0,
            peak_flux_by_filter={"sdssg_AB": 20.0, "g_PS1": 80.0},
            noise_floor=0.0,
        )

    np.testing.assert_allclose(loaded["flux"].to_numpy(), [100.0, 50.0])
    assert loaded["fcqfid"].nunique() == 2


if __name__ == "__main__":
    test_early_late_suffix_marks_ztf_normalization()
    test_early_late_loader_has_no_atlas_path()
    test_external_gr_normalization_uses_native_peak_flux()
