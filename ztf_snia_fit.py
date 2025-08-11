import os
import pandas as pd
import jax.numpy as jnp
import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()

from astropy.table import Table
from ztf_lc import ZTFLib
from numpyro import infer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dr",
        type=str,
        default="dr2",
        help="Data release to use (default: dr2)",
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

    args = parser.parse_args()
    dr = args.dr.lower()

    if dr == "dr2":
        dr_dir = "ztf_snia_dr2"
        tab_early_info = Table.read(
            "./Data/ztf_snia_dr2/tables/snia_early_data.csv", format="ascii.csv"
        )
        normal = tab_early_info["sn_type"] != "snia-pec"
        ztflib = ZTFLib(ztfid_lib=tab_early_info["ztfname"][normal], source="DR2")

    elif dr == "edr":
        dr_dir = "ztf_snia_edr"
        tab_salt = Table.read("./Data/ztf_snia_edr/Nobs_cut_salt2_spec_subtype_pec.csv")
        normal = ~pd.array(tab_salt["Ia subtype"]).isin(
            ["Ia-CSM", "SC", "SC*", "86G-like", "02cx-like"]
        )
        ztflib = ZTFLib(ztfid_lib=tab_salt["name"][normal], source="EDR")

    print(f"Number of normal Ia in {dr}: {len(ztflib.lc_library)}")

    ztflib.sampling(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=4,
        nuts_params={
            "init_strategy": infer.init_to_value(
                values={
                    "mean_alpha": jnp.ones(2, dtype=float) * 2.0,
                    "std_alpha": jnp.ones(2, dtype=float) * 0.1,
                    "mean_t_fl": -20.0,
                    "std_t_fl": 1.0,
                }
            ),
        },
        random_seed=114514,
    )

    if os.path.exists(f"./Data/{dr_dir}/results/"):
        # Remove existing results to avoid conflicts
        os.system(f"rm -rf ./Data/{dr_dir}/results/*nc")
    os.makedirs(f"./Data/{dr_dir}/results/", exist_ok=True)

    # Save the posterior for the hierarchical model
    ztflib.post_sample.to_netcdf(f"./Data/{dr_dir}/results/posterior_hierarchical.nc")

    # Save the posterior samples for each light curve
    for k in range(len(ztflib.lc_library)):
        lc = ztflib.lc_library[k]
        posterior = lc.post_sample
        # save the posterior
        posterior.to_netcdf(
            f"./Data/{dr_dir}/results/posterior_{ztflib.ztfid_lib[k]}.nc"
        )
