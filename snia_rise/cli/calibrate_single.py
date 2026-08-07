"""CLI for SALT2/BayeSN calibration of one bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from snia_rise.pipeline import calibrate_single_sn_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SALT2 then BayeSN to normalize one portable light-curve bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Raw portable light-curve bundle")
    parser.add_argument("--object", dest="object_id", help="Object ID inside the bundle")
    parser.add_argument("--output", type=Path, required=True, help="Output calibrated bundle directory")
    parser.add_argument("--filter-yaml", type=Path, required=True, help="BayeSN filter YAML")
    parser.add_argument("--bayesn-model", default="W22_model", help="BayeSN model name")
    parser.add_argument("--bayesn-num-devices", type=int, default=4, help="CPU devices for BayeSN")
    parser.add_argument("--salt2-model-dir", type=Path, default=None, help="Optional local SALT2 model directory")
    parser.add_argument("--sncosmo-filter-dir", type=Path, default=None, help="Optional local sncosmo filter directory")
    parser.add_argument("--rv", type=float, default=None, help="Optional fixed host R_V for BayeSN")
    parser.add_argument("--early-threshold", type=float, default=0.4, help="Fractional flux threshold for early-rise fit")
    parser.add_argument("--output-flux-zp", type=float, default=30.0, help="Zeropoint for peak fluxes and normalized photometry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibrate_single_sn_bundle(
        args.bundle,
        args.output,
        object_id=args.object_id,
        filter_yaml=args.filter_yaml,
        bayesn_model=args.bayesn_model,
        bayesn_num_devices=args.bayesn_num_devices,
        salt2_model_dir=args.salt2_model_dir,
        sncosmo_filter_dir=args.sncosmo_filter_dir,
        rv=args.rv,
        early_threshold=args.early_threshold,
        output_flux_zp=args.output_flux_zp,
    )
    print(f"Saved calibrated bundle to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
