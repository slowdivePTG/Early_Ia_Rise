from __future__ import annotations

import sys

import ztf_early_late_lc_salt as salt_fit


def test_salt_parse_args_defaults_to_ztf_only() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["ztf_early_late_lc_salt.py"]
        args = salt_fit.parse_args()
    finally:
        sys.argv = original_argv

    assert args.external is False
    assert args.no_ztf is False


def test_salt_parse_args_supports_external_only() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["ztf_early_late_lc_salt.py", "--external", "--no-ztf"]
        args = salt_fit.parse_args()
    finally:
        sys.argv = original_argv

    assert args.external is True
    assert args.no_ztf is True


def _parse_salt_args(argv: list[str]):
    original_argv = sys.argv[:]
    try:
        sys.argv = ["ztf_early_late_lc_salt.py", *argv]
        return salt_fit.parse_args()
    finally:
        sys.argv = original_argv


def test_salt_default_output_paths_are_source_specific() -> None:
    cases = [
        ([], "ztf_early_Ia_salt.csv"),
        (["--external"], "ztf_early_Ia_salt_ztf_external.csv"),
        (["--external", "--no-ztf"], "ztf_early_Ia_salt_external_only.csv"),
        (["--objects", "ZTF24aahgaov"], "ztf_early_Ia_salt_subset.csv"),
        (["--limit", "1"], "ztf_early_Ia_salt_subset.csv"),
    ]
    for argv, expected_name in cases:
        args = _parse_salt_args(argv)
        assert salt_fit.default_output_path(args) == salt_fit.DATA_DIR / expected_name


def test_salt_photometry_config_labels_are_source_specific() -> None:
    assert salt_fit.photometry_config_label(use_ztf=True, use_external=False) == "ztf"
    assert salt_fit.photometry_config_label(use_ztf=True, use_external=True) == "ztf_external"
    assert salt_fit.photometry_config_label(use_ztf=False, use_external=True) == "external_only"


if __name__ == "__main__":
    test_salt_parse_args_defaults_to_ztf_only()
    test_salt_parse_args_supports_external_only()
    test_salt_default_output_paths_are_source_specific()
    test_salt_photometry_config_labels_are_source_specific()
