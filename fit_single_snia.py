"""Compatibility wrapper for the package-owned single-SN fitter CLI."""

from snia_rise.cli.fit_single import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
