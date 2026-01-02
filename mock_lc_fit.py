import os
from pathlib import Path

import jax
import numpyro
import xarray as xr

from snia_rise._utils import set_best_platform
from snia_rise.simulate.mock_lc import RedbackLightCurveLib

numpyro.enable_x64()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["power_law", "curved_power_law"],
        help="Select model to fit the data: 'power_law' or 'curved_power_law'",
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="JAX platform to use (default: auto - detects NVIDIA GPU if available)",
    )
    parser.add_argument(
        "--num-host-devices",
        type=int,
        default=4,
        help="Number of CPU devices for parallel chains (e.g., 4)",
    )
    parser.add_argument(
        "--true_model",
        choices=[
            "power_law",
            "curved_power_law",
            "broken_power_law",
            "snf_2011fe",
            None,
        ],
        default=None,
        help="Select the underlying true model: 'power_law' or 'curved_power_law'. Default is the same as the fitting model.",
    )
    parser.add_argument(
        "--sampling_model",
        choices=[
            "pooled",
            "unpooled",
            "hierarchical",
            "hierarchical_trise",
            "hierarchical_mvn",
        ],
        default="hierarchical_mvn",
        help="Select sampling model: 'pooled', 'unpooled', or 'hierarchical (including _trise and _mvn)' (default: 'hierarchical_mvn')",
    )
    parser.add_argument(
        "--prior_type",
        choices=[
            "miller",
            "uniform",
            "maximum_entropy",
            "normal",
        ],
        default="uniform",
        help="Select prior type for alpha_0 in 'independent' correlation structure or non-hierarchical models (default: 'uniform')",
    )
    parser.add_argument(
        "--num_lc",
        type=int,
        default=100,
        help="Number of mock light curves to simulate (default: 100)",
    )
    parser.add_argument(
        "--early_threshold",
        nargs="+",
        type=float,
        default=[0.4],
        help="fraction of maximum luminosity to truncate light curves (default: [0.4])",
    )
    parser.add_argument(
        "--z_fixed",
        type=float,
        default=None,
        help="Fixed redshift for all mock light curves (default: None, i.e., draw from distribution)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2000,
        help="Number of warmup steps (default: 2000)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)",
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=4,
        help="Number of chains (default: 4)",
    )
    parser.add_argument(
        "--thinning",
        type=int,
        default=2,
        help="Thinning factor for MCMC samples (default: 2)",
    )

    args = parser.parse_args()

    # Configure JAX/NumPyro platform
    print("\n" + "=" * 70)
    print("PLATFORM CONFIGURATION")
    print("=" * 70)

    # Set host device count (must be before platform selection)
    # Only useful if platform is CPU
    if args.num_host_devices:
        numpyro.set_host_device_count(args.num_host_devices)

    # Set platform based on auto-detection or user preference
    if args.platform == "auto":
        # Auto-detect: prefer GPU (NVIDIA CUDA) if available
        platform = set_best_platform(prefer_gpu=True)
    elif args.platform == "cpu":
        platform = set_best_platform(prefer_gpu=False)
    elif args.platform == "gpu":
        # Force GPU (will use it if available, otherwise fall back to CPU)
        platform = set_best_platform(prefer_gpu=True)
    else:
        platform = set_best_platform(prefer_gpu=True)

    print(f"Using platform: {platform}")
    print("Precision: float64 (x64 mode)")
    print(f"Number of {platform.upper()} devices: {jax.device_count()}")
    print("=" * 70 + "\n")

    for early_threshold in args.early_threshold:
        model = args.model
        true_model = model if args.true_model is None else args.true_model

        true_model = (
            true_model
            if args.z_fixed is None
            else f"{true_model}_z_{args.z_fixed:.2f}".replace(".", "_")
        )

        lib = RedbackLightCurveLib(
            n_lc=args.num_lc,
            early_threshold=early_threshold,
            model=model,
            true_model=true_model,
            sampling_model=args.sampling_model,
            prior_type=args.prior_type.lower(),
        )

        result_dir = Path(
            f"./data/mock/{true_model}/{model}_frac{int(early_threshold * 100)}"
        )
        os.makedirs(result_dir, exist_ok=True)

        # Sampling
        lib.sampling(
            prior_config={
                "rise_model": model,
                "prior_type": args.prior_type.lower(),
            },
            num_warmup=args.num_warmup,
            num_samples=args.num_samples * args.thinning,
            num_chains=args.num_chains,
            thinning=args.thinning,
        )

        # Save the posterior for the hierarchical model
        post_sample = xr.Dataset(lib.post_sample)
        if args.sampling_model in ["hierarchical_trise", "unpooled", "pooled"]:
            sampling_model_str = f"{args.sampling_model}_{args.prior_type.lower()}"
        else:
            sampling_model_str = args.sampling_model
        post_sample.to_netcdf(
            result_dir / f"post_sample_{sampling_model_str}_{args.num_lc}.nc"
        )
