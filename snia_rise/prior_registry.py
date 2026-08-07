"""Resolve built-in and user-supplied priors for light-curve fitting."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from ._utils._config import _validate_population_priors


RESOURCE_PACKAGE = "snia_rise.resources.priors"
BASE_PRIORS = {"uniform", "maximum_entropy", "miller", "normal"}


def list_builtin_priors() -> list[str]:
    """Return names of installed population-prior profiles."""

    try:
        root = resources.files(RESOURCE_PACKAGE)
    except ModuleNotFoundError:
        return []
    return sorted(path.stem for path in root.iterdir() if path.name.endswith(".yaml"))


def read_prior_profile(name_or_path: str) -> dict[str, Any]:
    """Read a built-in prior profile name or YAML file path."""

    path = Path(name_or_path)
    if path.exists():
        with open(path, "r") as f:
            profile = yaml.safe_load(f) or {}
        profile.setdefault("name", path.stem)
        return profile

    resource = resources.files(RESOURCE_PACKAGE) / f"{name_or_path}.yaml"
    if not resource.is_file():
        available = ", ".join(list_builtin_priors()) or "none"
        raise ValueError(f"Unknown prior {name_or_path!r}. Available built-ins: {available}")
    with resource.open("r") as f:
        return yaml.safe_load(f) or {}


def build_base_prior_config(
    *,
    rise_model: str = "power_law",
    prior_type: str = "maximum_entropy",
    sample_beta: bool = False,
    mean_alpha_0: float | None = None,
    sigma_alpha_0: float | None = None,
    min_alpha_0: float | None = None,
    max_alpha_0: float | None = None,
    mean_t_rise: float | None = None,
    sigma_t_rise: float | None = None,
    t_rise_min: float | None = None,
    t_rise_max: float | None = None,
) -> dict[str, Any]:
    """Build a prior_config dict for non-population unpooled priors."""

    prior_type = prior_type.lower()
    if prior_type not in BASE_PRIORS and prior_type not in {"gaussian"}:
        raise ValueError(f"Unsupported prior_type {prior_type!r}")
    if prior_type == "normal" and sigma_alpha_0 is None:
        raise ValueError("--sigma-alpha-0 is required when --prior-type normal")

    config: dict[str, Any] = {
        "rise_model": rise_model,
        "prior_type": prior_type,
        "sample_beta": sample_beta,
    }
    optional = {
        "mean_alpha_0": mean_alpha_0,
        "sigma_alpha_0": sigma_alpha_0,
        "min_alpha_0": min_alpha_0,
        "max_alpha_0": max_alpha_0,
        "mean_t_rise": mean_t_rise,
        "sigma_t_rise": sigma_t_rise,
        "t_rise_min": t_rise_min,
        "t_rise_max": t_rise_max,
    }
    config.update({key: value for key, value in optional.items() if value is not None})
    return config


def build_population_prior_config(
    profile: dict[str, Any],
    *,
    rise_model: str | None = None,
    sample_beta: bool = False,
    filter_order: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a prior profile into a NumPyro prior_config dict."""

    if "population_priors" not in profile:
        raise ValueError("Population prior profile must contain population_priors")

    profile_filter_order = profile.get("filter_order")
    if filter_order is not None and profile_filter_order is not None:
        if list(filter_order) != list(profile_filter_order):
            raise ValueError(
                "Prior filter_order does not match input data: "
                f"prior={profile_filter_order}, data={filter_order}"
            )

    pop = profile["population_priors"]
    _validate_population_priors(pop)
    if filter_order is not None:
        n_filter = len(filter_order)
        for key in ("alpha_0", "log_Aprime"):
            if key in pop and len(list(pop[key]["mean"])) != n_filter:
                raise ValueError(
                    f"population_priors.{key} length does not match filter_order "
                    f"({len(list(pop[key]['mean']))} != {n_filter})"
                )

    return {
        "rise_model": rise_model or profile.get("rise_model", "power_law"),
        "sample_beta": sample_beta,
        "population_priors": pop,
        "config_stem": profile.get("name", "population_prior"),
    }


def resolve_prior_config(
    *,
    prior: str | None = None,
    prior_config: str | None = None,
    rise_model: str = "power_law",
    sample_beta: bool = False,
    filter_order: list[str] | None = None,
    prior_type: str = "maximum_entropy",
    mean_alpha_0: float | None = None,
    sigma_alpha_0: float | None = None,
    min_alpha_0: float | None = None,
    max_alpha_0: float | None = None,
    mean_t_rise: float | None = None,
    sigma_t_rise: float | None = None,
    t_rise_min: float | None = None,
    t_rise_max: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve CLI prior options into ``prior_config`` and metadata."""

    if prior and prior_config:
        raise ValueError("Use only one of --prior and --prior-config")
    if prior or prior_config:
        profile = read_prior_profile(prior or prior_config or "")
        config = build_population_prior_config(
            profile,
            rise_model=rise_model,
            sample_beta=sample_beta,
            filter_order=filter_order,
        )
        return config, profile

    config = build_base_prior_config(
        rise_model=rise_model,
        prior_type=prior_type,
        sample_beta=sample_beta,
        mean_alpha_0=mean_alpha_0,
        sigma_alpha_0=sigma_alpha_0,
        min_alpha_0=min_alpha_0,
        max_alpha_0=max_alpha_0,
        mean_t_rise=mean_t_rise,
        sigma_t_rise=sigma_t_rise,
        t_rise_min=t_rise_min,
        t_rise_max=t_rise_max,
    )
    return config, {"name": prior_type, "kind": "base"}
