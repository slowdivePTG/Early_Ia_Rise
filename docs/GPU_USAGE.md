# GPU Usage Guide

This guide explains how to use GPU acceleration (NVIDIA CUDA) with the Early Ia Rise fitting pipeline.

## Overview

The pipeline automatically detects and configures the best available computing platform:

- **NVIDIA GPUs (CUDA)**: Automatically detected and used for maximum performance
- **CPU**: Used as fallback when no NVIDIA GPU is available

**Note**: The pipeline always uses float64 (x64) precision as required for NUTS sampling.

## Quick Start

### Automatic Detection (Recommended)

The simplest way to run the pipeline is to let it auto-detect the best platform:

```bash
# Auto-detect platform (uses NVIDIA GPU if available)
uv run ztf_snia_fit.py edr --model curved_power_law --early_threshold 1 --volume-complete

# Force CPU (if needed)
uv run ztf_snia_fit.py edr --model curved_power_law --early_threshold 1 --volume-complete --platform cpu
```

### Manual Platform Selection

You can explicitly specify the platform:

```bash
# Force GPU (NVIDIA CUDA)
uv run ztf_snia_fit.py edr --model curved_power_law --platform gpu

# Force CPU
uv run ztf_snia_fit.py edr --model curved_power_law --platform cpu
```

## GPU Detection Test

Before running a long job, test GPU detection:

```bash
uv run python test_gpu_detection.py
```

This script will:
- Detect available JAX backends
- Identify NVIDIA GPU devices
- Test float64 compatibility
- Benchmark simple computations
- Provide recommendations

## Platform Comparison

| Platform | Float64 (x64) | Performance | Availability |
|----------|---------------|-------------|--------------|
| **NVIDIA GPU (CUDA)** | ✅ | Fastest | Linux/Windows clusters |
| **CPU** | ✅ | Moderate | All systems |

### Key Points

1. **NVIDIA GPUs fully support float64**: Required for NUTS sampling
2. **Automatic selection**: The pipeline automatically uses GPU if available
3. **No code changes needed**: Same scripts work on CPU or GPU
4. **Cluster-ready**: Designed for deployment on HPC clusters with NVIDIA GPUs
5. **x64 always enabled**: Float64 precision is always used (required for numerical stability)

## Command Line Options

### All Scripts (`ztf_snia_fit.py`)

```bash
--platform {cpu,gpu,auto}
    JAX platform to use (default: auto - detects NVIDIA GPU if available)

--num-host-devices N
    Number of CPU devices for parallel chains (e.g., 4)
    Only relevant when using CPU platform
```

## Usage Examples

### Example 1: Cluster Job with GPU

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Let the script auto-detect the GPU
uv run ztf_snia_fit.py edr \
    --model curved_power_law \
    --early_threshold 1 \
    --volume-complete \
    --num_warmup 2000 \
    --num_samples 2000 \
    --num_chains 4
```

### Example 2: CPU-only Job

```bash
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Force CPU and use multiple cores
uv run ztf_snia_fit.py edr \
    --model curved_power_law \
    --early_threshold 1 \
    --volume-complete \
    --platform cpu \
    --num-host-devices 4
```

## Technical Details

### How Auto-Detection Works

1. **Platform Detection**: Checks for available JAX backends (`gpu`, `cpu`, `tpu`)
2. **Priority Order**: `gpu` (NVIDIA CUDA) > `tpu` > `cpu`
3. **Device Query**: For GPU, queries actual devices to confirm CUDA availability

### Chain Method Selection

The pipeline automatically selects the optimal MCMC chain method based on platform:

- **GPU/TPU**: `parallel` - runs chains in parallel on accelerator
- **CPU**: `vectorized` or `sequential` based on core count

### Precision

- **Float64 (x64)**: Always enabled (required for NUTS sampling numerical stability)
- **Performance**: ~2x memory usage, slightly slower, but essential for accurate results

## Troubleshooting

### No GPU Detected on Cluster

If running on a cluster with GPUs but they're not detected:

1. Check GPU allocation: `nvidia-smi`
2. Verify CUDA is available: `nvcc --version`
3. Check JAX installation: `python -c "import jax; print(jax.devices())"`
4. Try forcing GPU: `--platform gpu`

### Out of Memory Errors

GPU memory is limited. If you get OOM errors:

```bash
# Use CPU instead
uv run ztf_snia_fit.py edr --model curved_power_law --platform cpu
```

Or reduce the dataset size/batch size.

### Slow Performance on GPU

Some operations may be slower on GPU than expected:

1. Small datasets: GPU overhead may dominate
2. First run: JIT compilation takes time (subsequent runs are faster)
3. Try CPU comparison: `--platform cpu`

## Performance Tips

1. **Use GPU for large datasets**: 100+ light curves benefit most from GPU acceleration
2. **Set num-host-devices on CPU**: Match to available cores for parallel chains
3. **Warm start**: Save and reuse posterior samples for faster re-runs
4. **Monitor GPU usage**: Use `nvidia-smi` to check utilization

## Environment Setup for Clusters

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=snia_fit
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=snia_fit_%j.log

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.12

# Activate environment
source .venv/bin/activate

# Run with auto-detection
uv run ztf_snia_fit.py edr \
    --model curved_power_law \
    --early_threshold 1 \
    --volume-complete \
    --num_warmup 2000 \
    --num_samples 2000 \
    --num_chains 4
```

### Checking GPU Utilization

Monitor GPU usage during runs:

```bash
# In another terminal/session
watch -n 1 nvidia-smi
```

## API Reference

### `set_best_platform(prefer_gpu=True)`

Automatically detect and configure JAX platform.

**Parameters:**
- `prefer_gpu` (bool): Use GPU if available (default: True)

**Returns:**
- `str`: Platform name ('gpu', 'cpu', etc.)

**Example:**
```python
from snia_rise._utils import set_best_platform
import numpyro

# Enable x64 (required for NUTS)
numpyro.enable_x64()

# Detect and set platform
platform = set_best_platform(prefer_gpu=True)
```

### `get_recommended_chain_method(platform=None, num_chains=4)`

Get optimal MCMC chain method for platform.

**Parameters:**
- `platform` (str, optional): Platform name (auto-detected if None)
- `num_chains` (int): Number of chains (default: 4)

**Returns:**
- `str`: Chain method ('parallel', 'vectorized', or 'sequential')

**Example:**
```python
from snia_rise._utils import get_recommended_chain_method
method = get_recommended_chain_method(platform='gpu', num_chains=4)
# Returns: 'parallel'
```

## FAQ

**Q: Do I need to change my code to use GPU?**  
A: No, the scripts automatically detect and use GPU if available.

**Q: Can I use multiple GPUs?**  
A: Currently, the pipeline uses a single GPU. Multi-GPU support may be added in the future.

**Q: Why is x64 always enabled?**  
A: Float64 precision is required for NUTS sampling to ensure numerical stability and accurate results.

**Q: Does x64 slow down GPU?**  
A: Yes, float64 operations use more memory and are slightly slower than float32, but NVIDIA GPUs handle float64 well and it's necessary for reliable inference.

**Q: How much faster is GPU?**  
A: Typically 5-20x faster for large MCMC sampling jobs, depending on the model and data size.

**Q: What if I'm on macOS?**  
A: The pipeline will automatically use CPU. This works well for development and testing.

## Additional Resources

- [JAX GPU Documentation](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
- [NumPyro Documentation](https://num.pyro.ai/en/stable/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

---

**Last Updated**: 2025-12-31  
**Tested With**: JAX 0.7.0, CUDA 11.8, NumPyro 0.15.3
