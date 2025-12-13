from ast import arg
import os
import numpyro
import xarray as xr

numpyro.set_host_device_count(4)
numpyro.enable_x64()

from pathlib import Path

from snia_rise.simulate.mock_lc import RedbackLightCurveLib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["power_law", "curved_power_law"],
        help="Select model to fit the data: 'power_law' or 'curved_power_law'",
    )
    parser.add_argument(
        "--true_model",
        choices=["power_law", "curved_power_law", None],
        default=None,
        help="Select the underlying true model: 'power_law' or 'curved_power_law'. Default is the same as the fitting model.",
    )
    parser.add_argument(
        "--sampling_model",
        choices=[
            "pooled",
            "unpooled",
            "hierarchical",
            "hierarchical_tfl",
            "hierarchical_mvn",
        ],
        default="hierarchical_mvn",
        help="Select sampling model: 'pooled', 'unpooled', or 'hierarchical (including _tfl and _mvn)' (default: 'hierarchical_mvn')",
    )
    parser.add_argument(
        "--prior_type",
        choices=[
            "miller",
            "uniform",
            "maximum_entropy",
            "normal",
        ],
        default="flat",
        help="Select prior type for alpha_0 in 'independent' correlation structure or non-hierarchical models (default: 'flat')",
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
        "--num_warmup",
        type=int,
        default=3000,
        help="Number of warmup steps (default: 3000)",
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

    args = parser.parse_args()

    for early_threshold in args.early_threshold:
        model = args.model
        true_model = model if args.true_model is None else args.true_model

        lib = RedbackLightCurveLib(
            n_lc=args.num_lc,
            early_threshold=early_threshold,
            model=model,
            true_model=true_model,
        )

        result_dir = Path(
            f"./data/mock/{true_model}/{model}_frac{int(early_threshold * 100)}"
        )
        os.makedirs(result_dir, exist_ok=True)

        # Sampling
        lib.sampling(
            prior_config={
                "curved_power_law": args.model == "curved_power_law",
                "prior_type": args.prior_type.lower(),
            },
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            model_structure=args.sampling_model,
        )

        # Save the posterior for the hierarchical model
        post_sample = xr.Dataset(lib.post_sample)
        if args.sampling_model in ["hierarchical_tfl", "unpooled", "pooled"]:
            sampling_model_str = f"{args.sampling_model}_{args.prior_type.lower()}"
        else:
            sampling_model_str = args.sampling_model
        post_sample.to_netcdf(
            result_dir / f"post_sample_{sampling_model_str}_{args.num_lc}.nc"
        )
