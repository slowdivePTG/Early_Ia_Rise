"""Single-object fitting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.table import Table

from .io import LightCurveRecord, write_light_curve_bundle


def fit_light_curve_record(
    record: LightCurveRecord,
    prior_config: dict[str, Any],
    *,
    num_samples: int = 1000,
    num_warmup: int = 3000,
    num_chains: int = 2,
    thinning: int = 1,
    random_seed: int = 11,
    prior_pred_samples: int = 500,
    nuts_params: dict[str, Any] | None = None,
):
    """Fit one portable light-curve record with the existing unpooled model."""

    config = dict(prior_config)
    config["n_global_filt"] = len(record.filter_order)
    light_curve = record.to_light_curve()
    light_curve.sampling(
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        thinning=thinning,
        random_seed=random_seed,
        prior_pred_samples=prior_pred_samples,
        prior_config=config,
        nuts_params=nuts_params or {},
    )
    _assign_named_coords(light_curve, record)
    return light_curve


def save_single_fit_result(
    output_dir: str | Path,
    record: LightCurveRecord,
    light_curve,
    prior_config: dict[str, Any],
    prior_profile: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    """Write a self-contained single-SN fit result directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_light_curve_bundle(
        output_dir / "input",
        [record],
        manifest={"generated_by": "snia-rise-fit", "selection": [record.object_id]},
    )

    with open(output_dir / "resolved_prior.yaml", "w") as f:
        yaml.safe_dump(
            {"profile": prior_profile, "prior_config": prior_config},
            f,
            sort_keys=False,
        )
    with open(output_dir / "run.yaml", "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    if getattr(light_curve, "inf_data", None) is not None:
        light_curve.inf_data.to_netcdf(output_dir / "inference.nc")
    elif getattr(light_curve, "post_sample", None) is not None:
        light_curve.post_sample.to_netcdf(output_dir / "posterior.nc")

    plot_lc = getattr(light_curve, "plot_lc", None)
    if callable(plot_lc):
        plot_lc(save=True, filename=str(output_dir / "light_curve"))
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass

    summary = summarize_posterior(light_curve)
    if summary is not None:
        summary.write(output_dir / "summary.ecsv", format="ascii.ecsv", overwrite=True)


def summarize_posterior(light_curve) -> Table | None:
    """Create a compact posterior median and 16/84 percent summary table."""

    posterior = getattr(light_curve, "post_sample", None)
    if posterior is None:
        return None

    rows = []
    for var in posterior.data_vars:
        values = np.asarray(posterior[var].values, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            continue
        if values.ndim <= 2:
            flat = values.reshape(-1)
            rows.append([var, "", *np.percentile(flat, [16, 50, 84])])
        elif "filt" in posterior[var].dims:
            axis = posterior[var].dims.index("filt")
            if "filt" in posterior[var].coords:
                filt_values = posterior[var].coords["filt"].values
            else:
                filt_values = range(values.shape[axis])
            for i, filt_name in enumerate(filt_values):
                flat = np.take(values, i, axis=axis).reshape(-1)
                rows.append([var, str(filt_name), *np.percentile(flat, [16, 50, 84])])
        elif "obj" in posterior[var].dims and posterior[var].sizes.get("obj") == 1:
            flat = values.reshape(-1)
            rows.append([var, "", *np.percentile(flat, [16, 50, 84])])
    if not rows:
        return None
    return Table(rows=rows, names=["parameter", "filter", "p16", "median", "p84"])


def _assign_named_coords(light_curve, record: LightCurveRecord) -> None:
    """Attach object/filter names to xarray groups produced by ArviZ."""

    filt_coords = record.filter_order
    for group_name in ("posterior", "prior", "posterior_predictive"):
        group = getattr(getattr(light_curve, "inf_data", None), group_name, None)
        if group is None:
            continue
        updates = {}
        if "obj" in group.dims and group.sizes["obj"] == 1:
            updates["obj"] = [record.object_id]
        if "filt" in group.dims and group.sizes["filt"] == len(filt_coords):
            updates["filt"] = filt_coords
        if updates:
            setattr(light_curve.inf_data, group_name, group.assign_coords(updates))
    if getattr(light_curve, "post_sample", None) is not None:
        updates = {}
        if "obj" in light_curve.post_sample.dims and light_curve.post_sample.sizes["obj"] == 1:
            updates["obj"] = [record.object_id]
        if "filt" in light_curve.post_sample.dims and light_curve.post_sample.sizes["filt"] == len(filt_coords):
            updates["filt"] = filt_coords
        if updates:
            light_curve.post_sample = light_curve.post_sample.assign_coords(updates)
