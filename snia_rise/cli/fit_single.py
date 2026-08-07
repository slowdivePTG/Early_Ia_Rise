"""CLI for fitting one calibrated SN Ia bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snia_rise.prior_registry import list_builtin_priors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit one SN Ia light curve from a portable snia_rise bundle."
    )
    parser.add_argument("--bundle", type=Path, help="Portable light-curve bundle directory")
    parser.add_argument("--object", dest="object_id", help="Object ID inside the bundle")
    parser.add_argument("--output", type=Path, help="Output result directory")
    parser.add_argument("--list-priors", action="store_true", help="List installed population priors and exit")
    parser.add_argument("--prior", help="Installed population-prior profile name")
    parser.add_argument("--prior-config", help="Path to a user population-prior YAML profile")
    parser.add_argument(
        "--prior-type",
        choices=["uniform", "maximum_entropy", "miller", "normal"],
        default="maximum_entropy",
        help="Base unpooled prior to use when --prior/--prior-config is not set",
    )
    parser.add_argument("--mean-alpha-0", type=float)
    parser.add_argument("--sigma-alpha-0", type=float)
    parser.add_argument("--min-alpha-0", type=float)
    parser.add_argument("--max-alpha-0", type=float)
    parser.add_argument("--mean-t-rise", type=float)
    parser.add_argument("--sigma-t-rise", type=float)
    parser.add_argument("--t-rise-min", type=float)
    parser.add_argument("--t-rise-max", type=float)
    parser.add_argument(
        "--model",
        choices=["power_law", "curved_power_law"],
        default="power_law",
        help="Rise model",
    )
    parser.add_argument("--sample-beta", action="store_true", help="Sample beta uncertainty scaling")
    parser.add_argument("--num-warmup", type=int, default=3000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--prior-pred-samples", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=12)
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="JAX platform",
    )
    parser.add_argument("--num-devices", type=int, default=2, help="CPU host device count")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_priors:
        for name in list_builtin_priors():
            print(name)
        return 0
    if args.bundle is None or args.output is None:
        parser.error("--bundle and --output are required unless --list-priors is used")

    import jax
    import numpyro

    from snia_rise._utils import set_best_platform
    from snia_rise.pipeline import fit_single_sn_bundle

    numpyro.enable_x64()
    if args.num_devices:
        numpyro.set_host_device_count(args.num_devices)
    if args.platform == "auto":
        platform = set_best_platform(prefer_gpu=True)
    elif args.platform == "cpu":
        platform = set_best_platform(prefer_gpu=False)
    else:
        platform = set_best_platform(prefer_gpu=True)
    print(f"Using platform: {platform}; devices: {jax.device_count()}")

    prior_kwargs = {
        "mean_alpha_0": args.mean_alpha_0,
        "sigma_alpha_0": args.sigma_alpha_0,
        "min_alpha_0": args.min_alpha_0,
        "max_alpha_0": args.max_alpha_0,
        "mean_t_rise": args.mean_t_rise,
        "sigma_t_rise": args.sigma_t_rise,
        "t_rise_min": args.t_rise_min,
        "t_rise_max": args.t_rise_max,
    }
    prior_kwargs = {key: value for key, value in prior_kwargs.items() if value is not None}
    sampling_kwargs = {
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "thinning": args.thinning,
        "random_seed": args.random_seed,
        "prior_pred_samples": args.prior_pred_samples,
    }
    nuts_params = {
        "target_accept_prob": args.target_accept_prob,
        "max_tree_depth": args.max_tree_depth,
    }
    fit_single_sn_bundle(
        args.bundle,
        args.output,
        object_id=args.object_id,
        prior=args.prior,
        prior_config=args.prior_config,
        rise_model=args.model,
        sample_beta=args.sample_beta,
        prior_type=args.prior_type,
        prior_kwargs=prior_kwargs,
        sampling_kwargs=sampling_kwargs,
        nuts_params=nuts_params,
        command=[Path(sys.argv[0]).name, *(argv or sys.argv[1:])],
    )
    print(f"Saved single-SN fit to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
