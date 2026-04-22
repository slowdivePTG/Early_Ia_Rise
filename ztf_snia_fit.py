import os
from pathlib import Path

import jax
import numpy as np
import numpyro
import xarray as xr
from astropy.table import Table

from snia_rise._utils import set_best_platform
from snia_rise.ztf_lc import SampleConfig, ZTFLib

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
        "--num_devices",
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
        "--baseline-coverage",
        default=False,
        action="store_true",
        help="Use sample with baseline light curve coverage",
    )
    parser.add_argument(
        "--x1-cut",
        type=float,
        default=None,
        help="Apply hierarchical model to two subsamples based on x1 cut (default: None, no cut; set to e.g., 0.5 to separate into x1 < 0.5 and x1 >= 0.5 subsamples)",
    )
    parser.add_argument(
        "--sn_type",
        choices=["normal", "03fg"],
        default="normal",
        help="Select SN type to fit (default: normal; options: normal, 03fg). Note: 03fg-like SNe are only available in DR2",
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
    if args.num_devices:
        numpyro.set_host_device_count(args.num_devices)

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
            f"Volume-complete: {args.volume_complete}, Early-coverage: {args.early_coverage}, Baseline-coverage: {args.baseline_coverage}"
        )
        print(f"Model: {args.model}, Sampling model: {args.sampling_model}")

        # ------------------------------------------------------------------ #
        # Load the metadata table and build the base boolean selection mask.  #
        # For DR2 and EDR we also extract the per-object x1 values so that   #
        # the x1-cut subsample loop (below) can split them correctly.        #
        # ------------------------------------------------------------------ #

        x1_values = None  # will be set for DR2 / EDR

        if dr == "dr2":
            dr_dir = "ztf_snia_dr2"
            tab_info = Table.read(
                f"./data/ztf_snia_dr2/tables/snia_data_basic_{args.sn_type}.csv",
                format="ascii.csv",
            )
            idx = np.ones(len(tab_info), dtype=bool)
            if args.volume_complete:
                idx &= tab_info["volume_limited"] == 1
            if args.early_coverage:
                idx &= tab_info["early_coverage"] == 1
            if args.baseline_coverage:
                idx &= tab_info["baseline_coverage"] == 1
            ztfid_col = "ztfname"
            x1_col = "x1"  # column name in DR2 CSV
            source = "DR2"

        elif dr == "edr":
            dr_dir = "ztf_snia_edr"
            tab_info = Table.read(
                "./data/ztf_snia_edr/snia_data_basic_normal.csv", format="ascii.csv"
            )
            idx = np.ones(len(tab_info), dtype=bool)
            if args.volume_complete:
                idx &= tab_info["volume_limited"] == 1
            if args.baseline_coverage:
                idx &= tab_info["baseline_coverage"] == 1
            args.early_coverage = True  # set to True for EDR since all have early data
            ztfid_col = "name"
            x1_col = "x1_salt2"  # column name in EDR CSV
            source = "EDR"

        elif dr == "early_late":
            dr_dir = "ztf_early_late"
            tab_info = Table.read("./data/ztf_early_late/ztf_early_Ia.csv")
            idx = tab_info["not_obs"].mask  # boolean mask already
            ztfid_col = "ztfid"
            x1_col = None  # x1-cut not supported for early_late
            source = "early_late"

        # ------------------------------------------------------------------ #
        # Build the list of (subsample_tag, boolean_mask) pairs to iterate.  #
        # When x1_cut is None we run a single pass with no subsample tag.    #
        # ------------------------------------------------------------------ #

        if args.x1_cut is not None and x1_col is not None:
            x1_arr = np.array(tab_info[x1_col], dtype=float)
            x1_cut_str = str(args.x1_cut).replace(".", "p").replace("-", "m")
            subsamples = [
                (f"x1lo_cut{x1_cut_str}", idx & (x1_arr < args.x1_cut)),
                (f"x1hi_cut{x1_cut_str}", idx & (x1_arr >= args.x1_cut)),
            ]
        elif args.x1_cut is not None and x1_col is None:
            print(
                f"WARNING: --x1-cut is set but the '{dr}' data release does not "
                "provide x1 values in its metadata table. Ignoring x1-cut."
            )
            subsamples = [(None, idx)]
        else:
            subsamples = [(None, idx)]

        # ------------------------------------------------------------------ #
        # Main loop over subsamples                                           #
        # ------------------------------------------------------------------ #

        file_dir = Path(
            f"./data/{dr_dir}/results/frac{int(early_threshold * 100)}_{args.model}"
        )
        os.makedirs(file_dir, exist_ok=True)

        for x1_subsample, idx_sub in subsamples:
            n_sub = int(np.sum(idx_sub))
            if x1_subsample is not None:
                print(f"\n--- x1 subsample: {x1_subsample}  ({n_sub} objects) ---")
            else:
                print(f"\n--- Full sample ({n_sub} objects) ---")

            config = SampleConfig(
                source=source,
                volume_complete=args.volume_complete,
                early_coverage=args.early_coverage,
                baseline_coverage=args.baseline_coverage,
                no_t0_err=args.no_t0_err,
                x1_subsample=x1_subsample,
                sn_type=args.sn_type,
            )
            ztflib = ZTFLib(
                ztfid_lib=tab_info[ztfid_col][idx_sub],
                config=config,
                early_threshold=early_threshold,
                rise_model=args.model,
                sampling_model=args.sampling_model,
            )

            if args.no_t0_err:
                ztflib.t0_err = None

            target_accept_prob = 0.85 if args.baseline_coverage else 0.6

            ztflib.sampling(
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                thinning=args.thinning,
                nuts_params=dict(
                    max_tree_depth=12, target_accept_prob=target_accept_prob
                ),
                random_seed=114514,
                prior_config=dict(
                    rise_model=args.model,
                    sample_beta=args.sample_beta,
                ),
                debug_save=False,
            )

            # Save the posterior
            post_sample = xr.Dataset(ztflib.post_sample)
            outfile = (
                f"post_sample_{args.sampling_model}{config.get_filename_suffix()}.nc"
            )
            post_sample.to_netcdf(file_dir / outfile)
            print(f"Saved posterior samples to: {file_dir / outfile}")
