import os
import shutil
import numpy as np
import numpyro
import xarray as xr

numpyro.set_host_device_count(4)
numpyro.enable_x64()

from pathlib import Path
from astropy.table import Table
from snia_rise.ztf_lc import ZTFLib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dr",
        nargs="+",
        type=str,
        default=["dr2"],
        help="data release to use (default: dr2; options: dr2, edr, early_late)",
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
    drs = [dr.lower() for dr in args.dr]

    ztflib = ZTFLib()

    dr_dir = None

    for early_threshold in args.early_threshold:
        for dr in drs:
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
                ztflib.append(
                    ZTFLib(
                        ztfid_lib=tab_info["ztfname"][idx],
                        source="DR2",
                        early_threshold=early_threshold,
                    )
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
                args.early_coverage = (
                    True  # set to True for EDR since all have early data
                )
                ztflib.append(
                    ZTFLib(
                        ztfid_lib=tab_info["name"][idx],
                        source="EDR",
                        early_threshold=early_threshold,
                    )
                )

            elif dr == "early_late":
                dr_dir = "ztf_early_late"
                tab_early_info = Table.read("./data/ztf_early_late/ztf_early_Ia.csv")
                ztflib.append(
                    ZTFLib(
                        tab_early_info[tab_early_info["not_obs"].mask]["ztfid"],
                        source="early_late",
                        early_threshold=early_threshold,
                    )
                )

        file_dir = Path(
            f"./data/{dr_dir}/results/frac{int(early_threshold * 100)}_{args.model}"
        )

        if os.path.exists(file_dir):
            # Remove existing results to avoid conflicts
            shutil.rmtree(file_dir)
        os.makedirs(file_dir)

        ztflib.sampling(
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            nuts_params=dict(max_tree_depth=12),
            random_seed=114514,
            prior_config={"curved_power_law": args.model == "curved_power_law"},
            model_structure=args.sampling_model,
        )

        # Save the posterior for the hierarchical model
        post_sample = xr.Dataset(ztflib.post_sample)
        outfile = f"post_sample_{args.sampling_model}"
        if args.volume_complete:
            outfile += "_volume_complete"
        if args.early_coverage:
            outfile += "_early_coverage"
        outfile += ".nc"
        post_sample.to_netcdf(file_dir / outfile)
