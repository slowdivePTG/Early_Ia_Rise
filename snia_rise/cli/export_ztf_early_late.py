"""CLI for exporting one early/late ZTF SN to a raw bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from snia_rise.pipeline import export_ztf_early_late_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an early/late ZTF SN to a raw snia_rise bundle.")
    parser.add_argument("--data-root", type=Path, default=Path("data/ztf_snia_early_late"))
    parser.add_argument("--object", dest="object_id", required=True, help="ZTF ID to export")
    parser.add_argument("--output", type=Path, required=True, help="Output bundle directory")
    parser.add_argument("--include-external-gr", action="store_true", help="Include external g/r photometry when available")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_ztf_early_late_bundle(
        args.data_root,
        args.object_id,
        args.output,
        include_external_gr=args.include_external_gr,
    )
    print(f"Saved raw bundle to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
