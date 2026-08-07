"""Compatibility wrapper for the package-owned early/late ZTF bundle exporter."""

from snia_rise.cli.export_ztf_early_late import build_parser, main
from snia_rise.pipeline import export_ztf_early_late_record as export_record


if __name__ == "__main__":
    raise SystemExit(main())
