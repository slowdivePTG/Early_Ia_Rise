"""Compatibility wrapper for the package-owned single-SN calibration CLI."""

from snia_rise.cli.calibrate_single import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
