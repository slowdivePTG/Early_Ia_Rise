import os
from pathlib import Path

import jax
import numpy as np
import numpyro
import xarray as xr
from astropy.table import Table

from snia_rise._utils import set_best_platform
from snia_rise.ztf_lc import ZTFLib

numpyro.enable_x64()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dr",
        type=str,
        default="dr2",
        help="data release to use (default: dr2; options: dr2, edr, early_late)",
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
        "--volume-complete",
        default=False,
        action="store_true",
        help="Use volume-complete sample from ZTF DR2 or EDR",
    )
    parser.add_argument(
        "--early-coverage",
        default=False,
        action="store_true",
        help="Use sample with early light curve coverage",
    )
    parser.add_argument(
        "--model",
        choices=["power_law", "curved_power_law"],
        help="Select model to fit the data: 'power_law' or 'curved_power_law'",
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
        "--early_threshold",
        nargs="+",
        type=float,
        default=[0.4],
        help="fraction of maximum luminosity to truncate light curves (default: [0.4])",
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
        help="Thinning factor for samples (default: 2)",
    )
    parser.add_argument(
        "--no-t0-err",
        default=False,
        action="store_true",
        help="Disable t0_err usage (force t0_err=None) and adjust auto-loaded filenames",
    )
    parser.add_argument(
        "--sample-beta",
        default=False,
        action="store_true",
        help="Sample beta (uncertainty scaling) as a free parameter with log(beta) ~ HalfNormal (default: False, fixed at 1.0)",
    )
    args = parser.parse_args()
    dr = args.dr.lower()

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

    dr_dir = None

    for early_threshold in args.early_threshold:
        print(f"\nProcessing DRs: {dr} with early threshold: {early_threshold}")
        print(
            f"Volume-complete: {args.volume_complete}, Early-coverage: {args.early_coverage}"
        )
        print(f"Model: {args.model}, Sampling model: {args.sampling_model}")

        if dr == "dr2":
            dr_dir = "ztf_snia_dr2"
            tab_info = Table.read(
                "./data/ztf_snia_dr2/tables/snia_data_basic_normal.csv",
                format="ascii.csv",
            )
            # normal = tab_early_info["sn_type"] != "snia-pec"
            idx = np.ones(len(tab_info), dtype=bool)
            if args.volume_complete:
                idx &= tab_info["volume_limited"] == 1
            if args.early_coverage:
                idx &= tab_info["early_coverage"] == 1
            ztflib = ZTFLib(
                ztfid_lib=tab_info["ztfname"][idx],
                source="DR2",
                early_threshold=early_threshold,
                volume_complete=args.volume_complete,
                early_coverage=args.early_coverage,
                rise_model=args.model,
                sampling_model=args.sampling_model,
                no_t0_err=args.no_t0_err,
            )

        elif dr == "edr":
            dr_dir = "ztf_snia_edr"
            tab_info = Table.read(
                "./data/ztf_snia_edr/snia_data_basic_normal.csv", format="ascii.csv"
            )
            # normal = ~pd.array(tab_info["Ia subtype"]).isin(
            #     ["Ia-CSM", "SC", "SC*", "86G-like", "02cx-like"]
            # )
            idx = np.ones(len(tab_info), dtype=bool)
            if args.volume_complete:
                idx &= tab_info["volume_limited"] == 1
            args.early_coverage = True  # set to True for EDR since all have early data
            ztflib = ZTFLib(
                ztfid_lib=tab_info["name"][idx],
                source="EDR",
                early_threshold=early_threshold,
                volume_complete=args.volume_complete,
                early_coverage=args.early_coverage,
                rise_model=args.model,
                sampling_model=args.sampling_model,
                no_t0_err=args.no_t0_err,
            )

        elif dr == "early_late":
            dr_dir = "ztf_early_late"
            tab_early_info = Table.read("./data/ztf_early_late/ztf_early_Ia.csv")
            ztflib = ZTFLib(
                tab_early_info[tab_early_info["not_obs"].mask]["ztfid"],
                source="early_late",
                early_threshold=early_threshold,
                volume_complete=args.volume_complete,
                early_coverage=args.early_coverage,
                rise_model=args.model,
                sampling_model=args.sampling_model,
                no_t0_err=args.no_t0_err,
            )

        file_dir = Path(
            f"./data/{dr_dir}/results/frac{int(early_threshold * 100)}_{args.model}"
        )

        # if os.path.exists(file_dir):
        #     # Remove existing results to avoid conflicts
        #     shutil.rmtree(file_dir)
        os.makedirs(file_dir, exist_ok=True)
        if args.no_t0_err:
            ztflib.t0_err = None
        # debug_basename = f"post_sample_{args.sampling_model}"
        # if args.volume_complete:
        #     debug_basename += "_volume_complete"
        # if args.early_coverage:
        #     debug_basename += "_early_coverage"
        # if args.no_t0_err:
        #     debug_basename += "_no_t0_err"

        ztflib.sampling(
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            thinning=args.thinning,
            nuts_params=dict(max_tree_depth=12, target_accept_prob=0.85),
            random_seed=114514,
            prior_config=dict(
                rise_model=args.model,
                sample_beta=args.sample_beta,
            ),
            debug_save=False,
            # debug_dir=str(file_dir),
            # debug_basename=debug_basename,
        )

        # Save the posterior for the hierarchical model
        post_sample = xr.Dataset(ztflib.post_sample)
        outfile = f"post_sample_{args.sampling_model}"
        if args.volume_complete:
            outfile += "_volume_complete"
        if args.early_coverage:
            outfile += "_early_coverage"
        # suffix handling delegated to ZTFLib
        outfile += ".nc"
        post_sample.to_netcdf(file_dir / outfile)
