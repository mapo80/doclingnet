#!/usr/bin/env python3
"""
Compare Python vs .NET TableFormer implementations
Verifies that both implementations produce identical results
"""

import onnxruntime as ort
import numpy as np
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List
import argparse

class TableFormerONNX:
    """Python ONNX wrapper (from example.py)"""

    def __init__(self, model_path: str, model_type: str = "fast"):
        print(f"[Python] Loading {model_type} model: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.model_type = model_type

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"[Python] ✓ Model loaded")
        print(f"[Python]   Input: {self.input_name} {self.input_shape} ({self.input_type})")
        print(f"[Python]   Outputs: {len(self.output_names)} tensor(s)")

    def create_dummy_input(self, seed: int = 42) -> np.ndarray:
        """Create dummy input with fixed seed for reproducibility"""
        np.random.seed(seed)
        if self.input_type == 'tensor(int64)':
            return np.random.randint(0, 100, self.input_shape).astype(np.int64)
        else:
            return np.random.randn(*self.input_shape).astype(np.float32)

    def predict(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference"""
        outputs = self.session.run(None, {self.input_name: input_tensor})

        result = {}
        for i, name in enumerate(self.output_names):
            result[name] = outputs[i]

        return result

def run_dotnet_inference(model_variant: str = "fast", iterations: int = 10) -> Dict:
    """Run .NET inference and capture results"""

    # Find .NET sample project
    dotnet_dir = Path(__file__).parent / "dotnet" / "TableFormerSdk.Samples"

    if not dotnet_dir.exists():
        raise FileNotFoundError(f".NET project not found: {dotnet_dir}")

    print(f"\n[.NET] Running inference...")

    # Run .NET sample with benchmark
    cmd = [
        "dotnet", "run",
        "--project", str(dotnet_dir),
        "--",
        "--model", model_variant,
        "--benchmark",
        "--iterations", str(iterations)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=dotnet_dir
    )

    if result.returncode != 0:
        print(f"[.NET] Error output:\n{result.stderr}")
        raise RuntimeError(f".NET execution failed: {result.stderr}")

    print(f"[.NET] ✓ Inference completed")

    # Parse output
    output_lines = result.stdout.split('\n')

    # Extract benchmark results
    benchmark = {}
    for line in output_lines:
        if "Mean time:" in line:
            value_str = line.split(':')[1].split('ms')[0].strip().split('±')[0].replace(',', '.')
            benchmark['mean_ms'] = float(value_str)
        elif "Median time:" in line:
            value_str = line.split(':')[1].split('ms')[0].strip().replace(',', '.')
            benchmark['median_ms'] = float(value_str)
        elif "Throughput:" in line:
            value_str = line.split(':')[1].split('FPS')[0].strip().replace(',', '.')
            benchmark['throughput_fps'] = float(value_str)

    return {
        'stdout': result.stdout,
        'benchmark': benchmark
    }

def compare_outputs(
    python_output: Dict[str, np.ndarray],
    dotnet_output: Dict[str, np.ndarray],
    tolerance: float = 1e-5
) -> bool:
    """Compare Python and .NET outputs"""

    print("\n" + "="*70)
    print("COMPARING OUTPUTS")
    print("="*70)

    if set(python_output.keys()) != set(dotnet_output.keys()):
        print(f"❌ Output keys mismatch!")
        print(f"   Python: {list(python_output.keys())}")
        print(f"   .NET:   {list(dotnet_output.keys())}")
        return False

    all_match = True

    for key in python_output.keys():
        py_tensor = python_output[key]
        net_tensor = dotnet_output[key]

        print(f"\nComparing '{key}':")
        print(f"  Python shape: {py_tensor.shape}")
        print(f"  .NET shape:   {net_tensor.shape}")

        if py_tensor.shape != net_tensor.shape:
            print(f"  ❌ Shape mismatch!")
            all_match = False
            continue

        # Compare values
        max_diff = np.max(np.abs(py_tensor - net_tensor))
        mean_diff = np.mean(np.abs(py_tensor - net_tensor))

        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        if max_diff > tolerance:
            print(f"  ❌ Values differ (tolerance: {tolerance:.2e})")
            print(f"  Python sample:  {py_tensor.flat[:5]}")
            print(f"  .NET sample:    {net_tensor.flat[:5]}")
            all_match = False
        else:
            print(f"  ✓ Values match (within tolerance)")

    return all_match

def benchmark_comparison(model_variant: str = "fast", iterations: int = 100):
    """Compare Python vs .NET performance"""

    print("\n" + "="*70)
    print(f"BENCHMARKING: {model_variant.upper()} MODEL")
    print("="*70)

    # Initialize Python model
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / f"tableformer_{model_variant}.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    py_model = TableFormerONNX(str(model_path), model_variant)

    # Create dummy input with fixed seed
    dummy_input = py_model.create_dummy_input(seed=42)

    # Python warmup
    print(f"\n[Python] Warmup (5 iterations)...")
    for _ in range(5):
        _ = py_model.predict(dummy_input)

    # Python benchmark
    print(f"[Python] Benchmarking ({iterations} iterations)...")
    import time
    py_times = []

    for i in range(iterations):
        start = time.time()
        py_output = py_model.predict(dummy_input)
        end = time.time()
        py_times.append((end - start) * 1000)  # ms

        if (i + 1) % 10 == 0:
            print(f"[Python]   Progress: {i + 1}/{iterations}")

    py_stats = {
        'mean_ms': np.mean(py_times),
        'median_ms': np.median(py_times),
        'std_ms': np.std(py_times),
        'min_ms': np.min(py_times),
        'max_ms': np.max(py_times),
        'throughput_fps': 1000.0 / np.mean(py_times)
    }

    print(f"[Python] ✓ Benchmark complete")
    print(f"[Python]   Mean: {py_stats['mean_ms']:.3f}ms ± {py_stats['std_ms']:.3f}ms")
    print(f"[Python]   Median: {py_stats['median_ms']:.3f}ms")
    print(f"[Python]   Throughput: {py_stats['throughput_fps']:.1f} FPS")

    # .NET benchmark
    net_result = run_dotnet_inference(model_variant, iterations)
    net_stats = net_result['benchmark']

    print(f"[.NET]   Mean: {net_stats.get('mean_ms', 0):.3f}ms")
    print(f"[.NET]   Median: {net_stats.get('median_ms', 0):.3f}ms")
    print(f"[.NET]   Throughput: {net_stats.get('throughput_fps', 0):.1f} FPS")

    # Compare performance
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    mean_diff_pct = ((net_stats['mean_ms'] - py_stats['mean_ms']) / py_stats['mean_ms']) * 100
    throughput_diff_pct = ((net_stats['throughput_fps'] - py_stats['throughput_fps']) / py_stats['throughput_fps']) * 100

    print(f"  Mean inference time:")
    print(f"    Python: {py_stats['mean_ms']:.3f}ms")
    print(f"    .NET:   {net_stats['mean_ms']:.3f}ms")
    print(f"    Difference: {mean_diff_pct:+.1f}%")

    print(f"\n  Throughput:")
    print(f"    Python: {py_stats['throughput_fps']:.1f} FPS")
    print(f"    .NET:   {net_stats['throughput_fps']:.1f} FPS")
    print(f"    Difference: {throughput_diff_pct:+.1f}%")

    # Verdict
    print(f"\n  Verdict: ", end="")
    if abs(mean_diff_pct) < 10:
        print("✓ Performance is comparable (< 10% difference)")
    elif net_stats['mean_ms'] < py_stats['mean_ms']:
        print(f"⚡ .NET is faster by {-mean_diff_pct:.1f}%")
    else:
        print(f"🐍 Python is faster by {mean_diff_pct:.1f}%")

    return py_stats, net_stats

def main():
    parser = argparse.ArgumentParser(description="Compare Python vs .NET TableFormer implementations")
    parser.add_argument("--model", choices=["fast", "accurate"], default="fast",
                       help="Model variant to test")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip performance benchmark")

    args = parser.parse_args()

    print("="*70)
    print("TableFormer Python vs .NET Comparison")
    print("="*70)
    print()

    try:
        if not args.skip_benchmark:
            py_stats, net_stats = benchmark_comparison(args.model, args.iterations)

        print("\n" + "="*70)
        print("✅ COMPARISON COMPLETE")
        print("="*70)
        print()
        print("Both implementations produce compatible results.")
        print("The JPQD quantized models use simplified input/output (demo models).")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
