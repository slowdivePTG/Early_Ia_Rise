# Tests and Diagnostics

This directory contains test scripts and diagnostic tools for the `snia_rise` package.

## Available Tests

### `test_gpu_detection.py`
Verifies GPU/TPU detection and platform configuration.

**Usage:**
```bash
python tests/test_gpu_detection.py
```

**What it checks:**
- JAX backend detection (CPU/GPU/TPU)
- Available devices
- Recommended chain method for MCMC sampling
- Platform-specific optimizations

---

### `diagnose_vectorization.py`
Diagnoses MCMC chain vectorization performance and identifies bottlenecks.

**Usage:**
```bash
python tests/diagnose_vectorization.py
```

**What it does:**
1. Verifies chain_method configuration for different chain counts
2. Tests vmap compatibility with your model
3. Times MCMC sampling with 1, 2, and 4 chains
4. Analyzes scaling behavior to detect vectorization issues

**Expected output:**
- ✓ Good vectorization: 4 chains takes ~same time as 1 chain
- ✗ Sequential execution: 4 chains takes ~4× longer than 1 chain

**Interpreting results:**
- **Speedup ≈ num_chains**: Vectorization is working correctly
- **Speedup ≈ 1.0**: Chains running sequentially (problem!)
- **Speedup 1.5-2.0**: GPU may be memory/compute bound

---

## Running Tests

From the project root:

```bash
# Run GPU detection test
python tests/test_gpu_detection.py

# Run vectorization diagnostics
python tests/diagnose_vectorization.py
```

## Troubleshooting

If `diagnose_vectorization.py` shows linear scaling with chains:

1. **Check NumPyro version** (need ≥ 0.12 for good vectorization):
   ```bash
   pip list | grep numpyro
   ```

2. **Monitor GPU utilization** during sampling:
   ```bash
   nvidia-smi -l 1
   ```
   - Should see ~80-100% GPU usage during main sampling
   - If low usage, chains may be running sequentially

3. **Check for vmap incompatibilities**: The diagnostic will report if the model has operations that can't be vectorized

4. **Verify JAX/CUDA setup**:
   ```bash
   python -c "import jax; print(jax.devices())"
   ```

---

## Adding New Tests

When adding new test scripts:

1. Place them in this `tests/` directory
2. Use descriptive names: `test_*.py` or `diagnose_*.py`
3. Add usage instructions to this README
4. Make scripts runnable from project root