"""Utility functions for working with NumPyro models and ArviZ."""

from typing import Literal

import jax.random as random
import numpyro
from numpyro import handlers


def set_best_platform(prefer_gpu=True) -> Literal["cpu", "gpu", "tpu"]:
    """
    Automatically detect and set the best available JAX/NumPyro platform.

    This function checks available platforms and sets the best one:
    - On Linux/Windows with NVIDIA GPU: gpu (CUDA)
    - Fallback: cpu

    Parameters
    ----------
    prefer_gpu : bool, optional
        If True (default), prefer GPU/accelerator platforms over CPU.
        If False, use CPU even if GPU is available.

    Returns
    -------
    str
        The platform that was set.

    Notes
    -----
    - NVIDIA GPUs (CUDA) fully support float64 operations.
    - On systems without NVIDIA GPUs, CPU will be used.

    Examples
    --------
    >>> from snia_rise._utils import set_best_platform
    >>> platform = set_best_platform()
    >>> print(f"Using platform: {platform}")
    """
    import jax

    # Get available platforms
    try:
        from jax.extend import backend

        backends = backend.backends()
    except (ImportError, AttributeError):
        # Fallback for older JAX versions
        try:
            backends = jax.lib.xla_bridge.backends()
        except AttributeError:
            # Very old JAX version
            backends = {"cpu": None}

    available = list(backends.keys())

    # Determine platform: prioritize NVIDIA GPU
    if not prefer_gpu or len(available) == 1:
        platform = "cpu"
    else:
        # Priority: gpu (NVIDIA CUDA) > tpu > cpu
        if "gpu" in available or "cuda" in available:
            platform = "cuda" if "cuda" in available else "gpu"
            # Check if it's actually CUDA (NVIDIA)
            try:
                devices = jax.devices("gpu")
                if devices:
                    device_kind = devices[0].device_kind
                    print(f"Detected GPU: {device_kind}")
            except Exception:
                pass
        elif "tpu" in available:
            platform = "tpu"
        else:
            platform = "cpu"

    print(f"Available platforms: {available}")
    print(f"Setting platform to: {platform}")
    numpyro.set_platform(platform)

    return platform


def get_recommended_chain_method(platform=None, num_chains=4):
    """
    Get the recommended chain_method for MCMC based on the platform.

    Different platforms have different optimal strategies for running multiple chains:
    - GPU/TPU: "parallel" - runs chains in parallel on the accelerator
    - CPU with many cores: "vectorized" - vectorizes across chains for efficiency
    - CPU with few cores: "sequential" - runs chains one after another

    Parameters
    ----------
    platform : str, optional
        The platform being used. If None, will auto-detect from JAX.
        Options: "gpu", "METAL", "tpu", "cpu"
    num_chains : int, optional
        Number of chains to run (default: 4). Used to determine if vectorization
        is beneficial.

    Returns
    -------
    str
        Recommended chain_method: "parallel", "vectorized", or "sequential"

    Examples
    --------
    >>> from snia_rise._utils import get_recommended_chain_method
    >>> chain_method = get_recommended_chain_method()
    >>> print(f"Using chain_method: {chain_method}")
    """
    import os

    import jax

    # Auto-detect platform if not provided
    if platform is None:
        try:
            from jax.extend import backend

            current_backend = backend.get_backend()
            platform = current_backend.platform
        except (ImportError, AttributeError):
            try:
                current_backend = jax.lib.xla_bridge.get_backend()
                platform = current_backend.platform
            except AttributeError:
                platform = "cpu"

    platform = platform.upper()

    # GPU/TPU platforms: use vectorized
    if platform in ["GPU", "TPU", "CUDA"]:
        return "parallel"

    # CPU: decide between vectorized and sequential based on available cores
    # and number of chains
    cpu_count = os.cpu_count() or 4

    if num_chains <= cpu_count and num_chains > 1:
        return "parallel"
    elif num_chains == 1:
        return "sequential"
    else:
        # More chains than cores - sequential might be better to avoid overhead
        return "sequential"


def extract_coords_dims_from_model(
    model, model_args=(), model_kwargs=None, num_samples=None
):
    """
    Extract coordinate names and dimensions from a NumPyro model by tracing it.

    This function traces a NumPyro model to extract plate (conditional independence)
    information. It also samples from the model to infer actual array shapes, which
    helps capture dimensions that arise from broadcasting operations not explicitly
    wrapped in plates.

    Parameters
    ----------
    model : callable
        A NumPyro model function
    model_args : tuple, optional
        Positional arguments to pass to the model (default: ())
    model_kwargs : dict, optional
        Keyword arguments to pass to the model (default: None)
    num_samples : int, optional
        Number of samples to draw when inferring shapes. If None, uses 1.

    Returns
    -------
    coords : dict
        Dictionary mapping coordinate names to their values (ranges).
        Example: {'obj': [0, 1, 2, 3, 4], 'filt': [0, 1]}
    dims : dict
        Dictionary mapping variable names to their dimension names.
        Example: {'alpha_0': ['obj', 'filt'], 't_rise': ['obj']}

    Notes
    -----
    - The 'data' plate (typically used for likelihood) is automatically excluded
      from the returned coordinates
    - Both sampled and deterministic sites are included
    - Observed variables are excluded
    - Dimensions are inferred from both the plate stack and actual array shapes
      to handle broadcasting cases

    Why is this needed?
    -------------------
    When using `az.from_numpyro()`:
    - WITH an MCMC sampler: ArviZ extracts plate names from the sampler, giving
      clean dimension names like 'obj', 'filt', 'fcqfid'
    - WITH only prior samples (from Predictive): ArviZ cannot access plate info,
      resulting in generic names like 'alpha_0_dim_0', 'alpha_0_dim_1'

    This function bridges the gap by tracing the model to extract plate information,
    allowing prior samples to have the same clean dimension names as posterior samples.

    Examples
    --------
    >>> def my_model(n_obj=5, n_filt=2):
    ...     with numpyro.plate('obj', n_obj, dim=-2):
    ...         with numpyro.plate('filt', n_filt, dim=-1):
    ...             alpha = numpyro.sample('alpha', dist.Normal(0, 1))
    >>> coords, dims = extract_coords_dims_from_model(
    ...     my_model, model_kwargs={'n_obj': 5, 'n_filt': 2}
    ... )
    >>> print(coords)
    {'obj': [0, 1, 2, 3, 4], 'filt': [0, 1]}
    >>> print(dims)
    {'alpha': ['obj', 'filt']}

    >>> # Use with ArviZ:
    >>> prior_pred = infer.Predictive(my_model, num_samples=100)(...)
    >>> # WITHOUT coords/dims: dimensions are 'alpha_dim_0', 'alpha_dim_1'
    >>> # WITH coords/dims: dimensions are 'obj', 'filt'
    >>> idata = az.from_numpyro(prior=prior_pred, coords=coords, dims=dims)
    """
    if model_kwargs is None:
        model_kwargs = {}
    if num_samples is None:
        num_samples = 1

    # Trace the model to get plate information
    rng_key = random.PRNGKey(0)
    trace = handlers.trace(handlers.seed(model, rng_key)).get_trace(
        *model_args, **model_kwargs
    )

    # Also sample from the model to get actual shapes
    from numpyro.infer import Predictive

    predictive = Predictive(model, num_samples=num_samples)
    samples = predictive(rng_key, *model_args, **model_kwargs)

    coords = {}
    dims = {}

    # First pass: collect all plates from trace
    all_plates = {}
    for site_name, site_value in trace.items():
        site_type = site_value.get("type")
        if site_type in ["sample", "deterministic"]:
            is_obs = site_value.get("is_observed", False)
            if not is_obs:
                cond_indep_stack = site_value.get("cond_indep_stack", [])
                for frame in cond_indep_stack:
                    plate_name = frame.name
                    plate_size = frame.size
                    if plate_name != "data" and plate_name not in all_plates:
                        all_plates[plate_name] = plate_size

    # Create coords for all discovered plates
    for plate_name, plate_size in all_plates.items():
        coords[plate_name] = list(range(plate_size))

    # Second pass: infer dims for each variable from actual shapes
    for site_name, site_value in trace.items():
        site_type = site_value.get("type")
        if site_type in ["sample", "deterministic"]:
            is_obs = site_value.get("is_observed", False)
            if not is_obs and site_name in samples:
                # Get the shape without the num_samples dimension
                actual_shape = samples[site_name].shape[1:]  # Skip first dim (samples)

                # Get plates from stack (these should be matched first)
                cond_indep_stack = site_value.get("cond_indep_stack", [])
                stack_plates = [
                    frame.name for frame in cond_indep_stack if frame.name != "data"
                ]
                stack_sizes = [
                    frame.size for frame in cond_indep_stack if frame.name != "data"
                ]

                # Infer dimension names from shape and available plates
                # Priority: plates in cond_indep_stack > other plates > generic names
                inferred_dims = []
                used_plates = set()

                for dim_size in actual_shape:
                    matched = False

                    # First, try to match with plates from the stack
                    for plate_name, plate_size in zip(stack_plates, stack_sizes):
                        if dim_size == plate_size and plate_name not in used_plates:
                            inferred_dims.append(plate_name)
                            used_plates.add(plate_name)
                            matched = True
                            break

                    # If not matched, try other known plates
                    if not matched:
                        for plate_name, plate_size in all_plates.items():
                            if dim_size == plate_size and plate_name not in used_plates:
                                inferred_dims.append(plate_name)
                                used_plates.add(plate_name)
                                matched = True
                                break

                    # If still no match, use generic name
                    if not matched:
                        generic_name = f"{site_name}_dim_{len(inferred_dims)}"
                        inferred_dims.append(generic_name)
                        if generic_name not in coords:
                            coords[generic_name] = list(range(dim_size))

                if inferred_dims:
                    dims[site_name] = inferred_dims

    return coords, dims
