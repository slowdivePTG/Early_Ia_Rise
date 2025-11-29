import os
import pandas as pd
import jax.numpy as jnp
import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()

from astropy.table import Table
from numpyro import infer

from snia_rise._utils import plt
from snia_rise.simulate.mock_lc import RedbackLightCurveLib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["power_law", "curved_power_law"],
        help="Select model: 'power_law' or 'curved_power_law'",
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

        save_dir = f"./data/mock/{args.model}_frac{int(early_threshold * 100)}"
        os.makedirs(save_dir, exist_ok=True)

        lib = RedbackLightCurveLib(
            n_lc=args.num_lc,
            early_threshold=early_threshold,
        )

        # Save the simulated light curves and parameters
        pd.DataFrame(lib.params_true).to_csv(
            f"./data/mock/{args.model}_frac{int(early_threshold * 100)}/simulated_lc_params.csv",
            index=False,
        )

        for k, lc_library in enumerate(lib.lc_library):
            pd.DataFrame(lc_library.lc_early).to_csv(
                f"./data/mock/{args.model}_frac{int(early_threshold * 100)}/lc_early_{k}.csv",
                index=False,
            )
            pd.DataFrame(lc_library.lc_peak).to_csv(
                f"./data/mock/{args.model}_frac{int(early_threshold * 100)}/lc_peak_{k}.csv",
                index=False,
            )

        # Sampling
        lib.sampling(
            prior_params={"curved_power_law": args.model == "curved_power_law"},
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
        )

        # Save the posterior for the hierarchical model
        lib.post_sample.to_netcdf(
            f"./data/mock/{args.model}_frac{int(early_threshold * 100)}/posterior_hierarchical.nc"
        )
        # Save the posterior samples for each light curve
        for k in range(len(lib.lc_library)):
            lc = lib.lc_library[k]
            posterior = lc.post_sample
            # save the posterior
            posterior.to_netcdf(
                f"./data/mock/{args.model}_frac{int(early_threshold * 100)}/posterior_{k}.nc"
            )
