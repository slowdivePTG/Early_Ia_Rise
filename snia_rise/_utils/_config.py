"""Population prior configuration loader."""

import os
from pathlib import Path

import numpy as np


def load_population_prior_config(file_path: str) -> dict:
    """
    Load a YAML population prior configuration file.

    Parameters
    ----------
    file_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        A ``prior_config``-compatible dictionary containing ``rise_model``
        and ``population_priors``.
    """
    import yaml

    with open(file_path, "r") as f:
        raw = yaml.safe_load(f)

    prior_config = {}

    if "rise_model" in raw:
        prior_config["rise_model"] = raw["rise_model"]

    pop = raw.get("population_priors", None)
    if pop is not None:
        _validate_population_priors(pop)
        _print_population_priors(pop, file_path)
        prior_config["population_priors"] = pop

    return prior_config


def _validate_population_priors(pop: dict):
    """Validate the structure of a ``population_priors`` dictionary."""

    n_total = 0
    param_keys = ["t_rise", "alpha_0", "log_Aprime"]

    for key in param_keys:
        spec = pop.get(key)
        if spec is not None:
            mean = spec.get("mean")
            sigma = spec.get("sigma")
            if mean is None or sigma is None:
                raise ValueError(
                    f"population_priors.{key} must provide both 'mean' and 'sigma'."
                )
            if key == "t_rise":
                if np.ndim(mean) != 0 or np.ndim(sigma) != 0:
                    raise ValueError(
                        f"population_priors.t_rise: 'mean' and 'sigma' must be scalars."
                    )
                n_total += 1
            else:
                mean_arr = np.atleast_1d(np.array(mean))
                sigma_arr = np.atleast_1d(np.array(sigma))
                if mean_arr.ndim != 1 or sigma_arr.ndim != 1:
                    raise ValueError(
                        f"population_priors.{key}: 'mean' and 'sigma' must be 1-D arrays."
                    )
                if len(mean_arr) != len(sigma_arr):
                    raise ValueError(
                        f"population_priors.{key}: 'mean' and 'sigma' must have the same length."
                    )
                n_total += len(mean_arr)

    corr = pop.get("corr")
    if corr is not None:
        corr_arr = np.array(corr)
        if corr_arr.ndim != 2 or corr_arr.shape[0] != corr_arr.shape[1]:
            raise ValueError("population_priors.corr must be a square 2-D matrix.")
        if corr_arr.shape[0] != n_total:
            raise ValueError(
                f"population_priors.corr shape ({corr_arr.shape[0]}, "
                f"{corr_arr.shape[1]}) does not match the total number of "
                f"specified parameter elements ({n_total}). "
                f"Parameters contribute in order: "
                f"[t_rise, alpha_0 (per filter), log_Aprime (per filter)]."
            )


def _print_population_priors(pop: dict, source: str):
    """Print a summary of the loaded population prior hyperparameters."""
    print(f"\n{'=' * 60}")
    print(f"  Population prior config: {source}")
    print(f"{'=' * 60}")

    for key in ["t_rise", "alpha_0", "log_Aprime"]:
        spec = pop.get(key)
        if spec is not None:
            mean = np.asarray(spec["mean"])
            sigma = np.asarray(spec["sigma"])
            mean_str = np.array2string(mean, precision=4, suppress_small=True)
            sigma_str = np.array2string(sigma, precision=4, suppress_small=True)
            print(f"  {key}:  mean = {mean_str}   sigma = {sigma_str}")

    corr = pop.get("corr")
    if corr is not None:
        corr_arr = np.array(corr)
        print(f"  corr:  shape = {list(corr_arr.shape)}")
        print(
            f"         corr =\n{np.array2string(corr_arr, precision=3, suppress_small=True)}"
        )
    else:
        print(f"  corr:  none (independent Normals)")

    print(f"{'=' * 60}\n")
