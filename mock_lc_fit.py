import os
import numpyro
import glob

numpyro.set_host_device_count(4)
numpyro.enable_x64()

from pathlib import Path

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
        file_dir = Path(f"./data/mock/{args.model}_frac{int(early_threshold * 100)}")

        if not os.path.exists(file_dir):
            raise FileNotFoundError(
                f"{file_dir} does not exist. Please run the simulation first."
            )

        early_files = sorted(glob.glob(str(file_dir / "lc_early*.csv")))
        peak_files = sorted(glob.glob(str(file_dir / "lc_peak*.csv")))

        if len(early_files) == 0 or len(peak_files) == 0:
            raise FileNotFoundError(
                f"No light curve files found in {file_dir}. Please run the simulation first."
            )

        if not (len(early_files) == len(peak_files) >= args.num_lc):
            raise ValueError(
                f"Insufficient light curve files in {file_dir}: found {len(early_files)} simulated light curves, but {args.num_lc} are required."
            )

        if os.path.exists(file_dir / "posterior_hierarchical.nc"):
            print("Removing existing .nc files...")
            os.system(f"rm -rf {str(file_dir / '*.nc')}")

        lib = RedbackLightCurveLib.from_files(
            early_files=early_files[: args.num_lc],
            peak_files=peak_files[: args.num_lc],
            params_file=file_dir / "simulated_lc_params.csv",
        )

        # Sampling
        lib.sampling(
            prior_params={"curved_power_law": args.model == "curved_power_law"},
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
        )

        # Save the posterior for the hierarchical model
        lib.post_sample.to_netcdf(file_dir / "posterior_hierarchical.nc")
        # Save the posterior samples for each light curve
        for k in range(len(lib.lc_library)):
            lc = lib.lc_library[k]
            posterior = lc.post_sample
            # save the posterior
            posterior.to_netcdf(file_dir / f"posterior_{k}.nc")
