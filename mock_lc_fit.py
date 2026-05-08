import os
from pathlib import Path

import jax
import numpyro

from snia_rise._utils import load_population_prior_config, set_best_platform
from snia_rise.simulate.simulator import RedbackLightCurveLib

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
        "--true_model",
        default=None,
        help="Select the underlying true model (e.g. 'power_law', 'power_law_bump', 'shen', '2011fe'). Default is the same as the fitting model.",
    )
    parser.add_argument(
        "--true_param_dependence",
        choices=["independent", "correlated", None],
        default=None,
        help="Parameter dependence of the true simulation for power-law families: 'independent' or 'correlated' (default: None).",
    )
    parser.add_argument(
        "--template_model_id",
        default=None,
        help="Template model ID (e.g. '2011fe' or '1.0_2e5') used for simulation. Required if true_model implies a template model.",
    )
    parser.add_argument(
        "--sampling_model",
        choices=[
            "pooled",
            "unpooled",
            "hierarchical",
            "hierarchical_mvn",
        ],
        default="hierarchical_mvn",
        help="Select sampling model: 'pooled', 'unpooled', or 'hierarchical (including _mvn)' (default: 'hierarchical_mvn')",
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
        default=None,
        help="Number of mock light curves to use (default: all available)",
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
    parser.add_argument(
        "--num_devices",
        type=int,
        default=4,
        help="Number of devices for parallel processing (default: local device count)",
    )
    parser.add_argument(
        "--early_coverage",
        action="store_true",
        help="Only fit light curves that pass the early-time coverage cut",
    )
    parser.add_argument(
        "--baseline_coverage",
        action="store_true",
        help="Only fit light curves that pass the baseline-time coverage cut",
    )
    parser.add_argument(
        "--prior-config",
        type=str,
        default=None,
        help=(
            "YAML file with population prior configuration for the "
            "unpooled model (see docs/ for schema)"
        ),
    )

    args = parser.parse_args()

    # Configure JAX/NumPyro platform
    print("\n" + "=" * 70)
    print("PLATFORM CONFIGURATION")
    print("=" * 70)

    # Set platform based on auto-detection or user preference
    if args.platform == "auto":
        # Auto-detect: prefer GPU (NVIDIA CUDA) if available
        platform = set_best_platform(prefer_gpu=True)
    elif args.platform == "cpu":
        numpyro.set_host_device_count(args.num_devices)
        platform = set_best_platform(prefer_gpu=False)
    elif args.platform == "gpu":
        # Force GPU (will use it if available, otherwise fall back to CPU)
        platform = set_best_platform(prefer_gpu=True)

    print(f"Using platform: {platform}")
    print("Precision: float64 (x64 mode)")
    print(f"Number of {platform.upper()} devices: {jax.device_count()}")
    print("=" * 70 + "\n")

    for early_threshold in args.early_threshold:
        model = args.model
        true_model = model if args.true_model is None else args.true_model

        if args.template_model_id is not None:
            sanitized_id = args.template_model_id.replace(":", "_")
            true_model = f"{args.true_model}_{sanitized_id}"

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
            true_param_dependence=args.true_param_dependence,
            early_coverage=args.early_coverage,
            baseline_coverage=args.baseline_coverage,
            pop_prior=(args.prior_config is not None),
        )

        # Align result directory with input directory structure that may include param dependence
        suffix = (
            f"_{args.true_param_dependence}"
            if (
                args.true_param_dependence is not None
                and "power_law" in true_model
                and "power_law_bump" not in true_model
            )
            else ""
        )
        result_dir = Path(
            f"./data/mock/{true_model}{suffix}/{model}_frac{int(early_threshold * 100)}"
        )
        os.makedirs(result_dir, exist_ok=True)

        # Load population prior config if provided
        prior_config = {
            "rise_model": model,
            "prior_type": args.prior_type.lower(),
        }
        if args.prior_config is not None:
            config_path = Path(args.prior_config)
            if not config_path.is_absolute():
                config_path = result_dir / "config" / config_path
            pop_config = load_population_prior_config(str(config_path))
            prior_config.update(pop_config)

        # Sampling
        lib.sampling(
            prior_config=prior_config,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            thinning=args.thinning,
            nuts_params=dict(max_tree_depth=12, target_accept_prob=0.8),
        )

        # Save the posterior for the hierarchical model
        if args.sampling_model in ["unpooled", "pooled"]:
            sampling_model_str = f"{args.sampling_model}_{args.prior_type.lower()}"
        else:
            sampling_model_str = args.sampling_model
        if lib.pop_prior:
            sampling_model_str += "_pop_prior"
        lib.save_post_sample(
            result_dir / f"post_sample_{sampling_model_str}_{args.num_lc}.nc"
        )
