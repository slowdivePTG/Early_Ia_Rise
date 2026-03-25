#!/usr/bin/env python
"""
Diagnostic script to identify why MCMC sampling scales linearly with num_chains.

This script will:
1. Verify chain_method configuration
2. Check for vmap incompatibilities
3. Profile GPU utilization
4. Time each component separately
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro import infer

# Import your model
from snia_rise._utils import get_recommended_chain_method
from snia_rise.model.model_structure import hierarchical_model

# Load a small test dataset
from snia_rise.simulate.simulator import RedbackLightCurveLib

# Enable GPU
numpyro.set_platform("gpu")
numpyro.enable_x64()

print("=" * 70)
print("JAX/NumPyro Configuration")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print("=" * 70 + "\n")


print("Loading test data (10 light curves)...")
lib = RedbackLightCurveLib(
    n_lc=10,
    early_threshold=0.4,
    model="power_law",
    true_model="power_law",
    sampling_model="hierarchical_mvn",
)

running_params = {
    "t": lib.phase,
    "flux": lib.flux,
    "flux_err": lib.flux_err,
    "t0_err": lib.t0_err if lib.t0_err is not None else jnp.zeros_like(lib.phase),
    "idx_obj": lib.idx_obj,
    "idx_fcqfid": lib.idx_fcqfid,
    "idx_filt": lib.idx_filt,
}

prior_config = {
    "rise_model": "power_law",
    "correlation_structure": "mvn",
}

print(f"Data loaded:")
print(f"  n_obj: {len(np.unique(lib.idx_obj))}")
print(f"  n_filt: {len(np.unique(lib.idx_filt))}")
print(f"  n_obs: {len(lib.phase)}")
print()

# Test 1: Verify chain_method selection
print("=" * 70)
print("TEST 1: Chain Method Selection")
print("=" * 70)

for num_chains in [1, 2, 4]:
    platform = jax.default_backend()
    chain_method = get_recommended_chain_method(platform, num_chains)
    print(f"num_chains={num_chains} → chain_method='{chain_method}'")
print()

# Test 2: Test vmap compatibility explicitly
print("=" * 70)
print("TEST 2: Vmap Compatibility Check")
print("=" * 70)

try:
    # Try to vmap the model
    print("Attempting to vmap the model...")

    def single_chain_run(rng_key):
        """Run a single MCMC chain"""
        sampler = infer.MCMC(
            infer.NUTS(hierarchical_model, target_accept_prob=0.8),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            chain_method="sequential",
        )
        sampler.run(rng_key, **running_params, prior_config=prior_config)
        return sampler.get_samples()

    # Try to vmap it
    rng_keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmapped_run = jax.vmap(single_chain_run)

    print("Testing vmap with 2 chains...")
    start = time.time()
    result = vmapped_run(rng_keys)
    jax.block_until_ready(result)
    elapsed = time.time() - start
    print(f"✓ Vmap successful! Took {elapsed:.2f}s")
    print()

except Exception as e:
    print(f"✗ Vmap failed with error:")
    print(f"  {type(e).__name__}: {e}")
    print()

# Test 3: Time MCMC with different chain counts
print("=" * 70)
print("TEST 3: Timing MCMC with Different Chain Counts")
print("=" * 70)

results = []

for num_chains in [1, 2, 4]:
    print(f"\nTesting with {num_chains} chain(s)...")

    platform = jax.default_backend()
    chain_method = get_recommended_chain_method(platform, num_chains)

    print(f"  Chain method: {chain_method}")

    sampler = infer.MCMC(
        infer.NUTS(
            hierarchical_model,
            target_accept_prob=0.8,
        ),
        num_warmup=50,
        num_samples=50,
        num_chains=num_chains,
        chain_method=chain_method,
    )

    # Print actual chain_method being used
    print(f"  Sampler._chain_method: {sampler._chain_method}")

    # Time the sampling
    start = time.time()
    sampler.run(
        jax.random.PRNGKey(42 + num_chains),
        **running_params,
        prior_config=prior_config,
    )

    # Block until complete (important for GPU timing!)
    samples = sampler.get_samples()
    for v in samples.values():
        jax.block_until_ready(v)

    elapsed = time.time() - start

    results.append(
        {
            "num_chains": num_chains,
            "chain_method": chain_method,
            "time": elapsed,
            "time_per_chain": elapsed / num_chains,
        }
    )

    print(f"  ✓ Completed in {elapsed:.2f}s")
    print(f"  Time per chain (if sequential): {elapsed / num_chains:.2f}s")

# Analyze results
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\nTiming Summary:")
print(
    f"{'Chains':<10} {'Method':<12} {'Total (s)':<12} {'Per Chain (s)':<15} {'Speedup'}"
)
print("-" * 70)

baseline_time = results[0]["time"]
for r in results:
    speedup = baseline_time * r["num_chains"] / r["time"]
    print(
        f"{r['num_chains']:<10} {r['chain_method']:<12} {r['time']:<12.2f} "
        f"{r['time_per_chain']:<15.2f} {speedup:.2f}x"
    )

print("\nInterpretation:")
print("- If speedup ≈ num_chains: ✓ Vectorization is working!")
print("- If speedup ≈ 1.0: ✗ Chains are running sequentially (problem!)")
print("- If speedup between 1-2: GPU may be memory/compute bound")

# Check if linear scaling
if len(results) >= 2:
    ratio_2_to_1 = results[1]["time"] / results[0]["time"]
    ratio_4_to_2 = results[2]["time"] / results[1]["time"] if len(results) > 2 else None

    print(f"\nScaling ratios:")
    print(f"  2 chains / 1 chain: {ratio_2_to_1:.2f}x")
    if ratio_4_to_2:
        print(f"  4 chains / 2 chains: {ratio_4_to_2:.2f}x")

    if ratio_2_to_1 > 1.7:  # Should be close to 2.0 if sequential
        print("\n⚠️  WARNING: Time scales ~linearly with chains!")
        print("   This suggests chains are running SEQUENTIALLY, not vectorized.")
        print("\nPossible causes:")
        print("1. Chain method not being set correctly")
        print("2. NumPyro version issue (need >= 0.12 for good vectorization)")
        print("3. JAX/CUDA configuration issue")
        print("4. Model has vmap-incompatible operations")
    else:
        print("\n✓ Good! Vectorization appears to be working.")

print("\n" + "=" * 70)
print("Next steps:")
print("=" * 70)
print("1. Check NumPyro version: pip list | grep numpyro")
print("2. If vmap test failed, inspect the error for incompatible operations")
print("3. Run: nvidia-smi -l 1  (while sampling to check GPU utilization)")
print("4. Check if XLA_FLAGS are set: echo $XLA_FLAGS")
print()
