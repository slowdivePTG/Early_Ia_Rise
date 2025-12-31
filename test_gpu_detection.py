#!/usr/bin/env python3
"""
Test script to verify GPU (NVIDIA CUDA) detection and configuration.
This script tests automatic platform detection and configuration for cluster deployment.
"""

import sys


def test_gpu_detection():
    """Test GPU detection and platform configuration."""
    print("=" * 70)
    print("GPU DETECTION TEST")
    print("=" * 70)

    # Test 1: Check JAX before configuration
    print("\nTest 1: Checking available JAX backends...")
    try:
        import jax

        # Get available platforms
        try:
            from jax.extend import backend

            backends = backend.backends()
        except (ImportError, AttributeError):
            try:
                backends = jax.lib.xla_bridge.backends()
            except AttributeError:
                backends = {"cpu": None}

        available = list(backends.keys())
        print(f"✓ Available JAX backends: {available}")

        # Check for GPU devices
        if "gpu" in available or "cuda" in available:
            gpu_kind = "cuda" if "cuda" in available else "gpu"
            try:
                gpu_devices = jax.devices(gpu_kind)
                print(f"✓ GPU devices found: {len(gpu_devices)}")
                for i, device in enumerate(gpu_devices):
                    print(f"  - Device {i}: {device}")
                    print(f"    Kind: {device.device_kind}")
                    print(f"    Platform: {device.platform}")
            except Exception as e:
                print(f"  Warning: Could not query GPU devices: {e}")
        else:
            print("  No GPU backend available")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: Test platform auto-detection
    print("\nTest 2: Testing auto-detection with prefer_gpu=True...")
    try:
        from snia_rise._utils import set_best_platform

        platform = set_best_platform(prefer_gpu=True)
        print(f"✓ Selected platform: {platform}")

        # Check actual devices being used
        devices = jax.devices()
        print(f"✓ Active devices: {len(devices)}")
        for device in devices[:3]:  # Show first 3
            print(f"  - {device}")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Test chain method recommendation
    print("\nTest 3: Testing chain method recommendation...")
    try:
        from snia_rise._utils import get_recommended_chain_method

        chain_method = get_recommended_chain_method(platform=platform, num_chains=4)
        print(f"✓ Recommended chain_method for {platform}: {chain_method}")

        if platform == "gpu":
            assert chain_method == "parallel", (
                f"GPU should use 'parallel', got '{chain_method}'"
            )
            print("  ✓ Correct: GPU uses parallel chains")
        elif platform == "cpu":
            print(f"  ✓ CPU uses '{chain_method}' chains")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: Simple JAX computation
    print("\nTest 4: Running simple JAX computation...")
    try:
        import jax.numpy as jnp

        # Matrix multiplication benchmark
        size = 1000
        A = jnp.ones((size, size))
        B = jnp.ones((size, size))

        # Warmup
        _ = jnp.dot(A, B).block_until_ready()

        # Timed run
        import time

        start = time.time()
        C = jnp.dot(A, B).block_until_ready()
        elapsed = time.time() - start

        print(f"✓ Matrix multiplication ({size}x{size}): {elapsed:.4f} seconds")
        print(f"  Result shape: {C.shape}")
        print(f"  Device: {C.devices()}")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Platform: {platform}")
    print(f"GPU available: {'gpu' in available or 'cuda' in available}")
    if "gpu" in available:
        print("  ✓ NVIDIA GPU detected - cluster deployment ready!")
        print("  ✓ Supports float64 (x64) precision (always enabled)")
        print("  ✓ Recommended: Use --platform auto (default)")
    else:
        print("  CPU only - no NVIDIA GPU detected")
        print("  Recommended: Use --platform cpu")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_gpu_detection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
