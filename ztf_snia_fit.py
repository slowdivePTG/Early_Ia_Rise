import os
import pandas as pd
import numpyro

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
        "--model",
        choices=["power_law", "curved_power_law"],
        help="Select model to fit the data: 'power_law' or 'curved_power_law'",
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
                tab_early_info = Table.read(
                    "./data/ztf_snia_dr2/tables/snia_early_data.csv", format="ascii.csv"
                )
                normal = tab_early_info["sn_type"] != "snia-pec"
                ztflib.append(
                    ZTFLib(
                        ztfid_lib=tab_early_info["ztfname"][normal],
                        source="DR2",
                        early_threshold=early_threshold,
                    )
                )

            elif dr == "edr":
                dr_dir = "ztf_snia_edr"
                tab_salt = Table.read(
                    "./data/ztf_snia_edr/Nobs_cut_salt2_spec_subtype_pec.csv"
                )
                normal = ~pd.array(tab_salt["Ia subtype"]).isin(
                    ["Ia-CSM", "SC", "SC*", "86G-like", "02cx-like"]
                )
                ztflib.append(
                    ZTFLib(
                        ztfid_lib=tab_salt["name"][normal],
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

        file_dir = Path(f"./data/{dr_dir}/results/frac{int(early_threshold * 100)}_{args.model}")

        if os.path.exists(file_dir):
            # Remove existing results to avoid conflicts
            os.removedirs(file_dir)
        os.makedirs(file_dir, exist_ok=True)

        ztflib.sampling(
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            nuts_params=dict(max_tree_depth=12),
            random_seed=114514,
            prior_params={"curved_power_law": args.model == "curved_power_law"},
        )

        # Save the posterior for the hierarchical model
        ztflib.post_sample.to_netcdf(file_dir / "posterior_hierarchical.nc")

        # Save the posterior samples for each light curve
        for k in range(len(ztflib.lc_library)):
            lc = ztflib.lc_library[k]
            posterior = lc.post_sample
            # save the posterior
            posterior.to_netcdf(file_dir / f"posterior_{ztflib.ztfid_lib[k]}.nc")
