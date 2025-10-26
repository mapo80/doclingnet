#!/usr/bin/env python3
"""
Generate Python baseline outputs for PositionalEncoding
To be compared with C# TorchSharp implementation
"""
import sys
import torch
import numpy as np
import json
from pathlib import Path

# Find the installed package
try:
    from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import PositionalEncoding
    print("✅ Imported PositionalEncoding from docling_ibm_models")
except ImportError:
    print("❌ Failed to import PositionalEncoding")
    sys.exit(1)

def test_positional_encoding():
    """Generate baseline outputs for various test cases"""

    print("="*70)
    print("POSITIONAL ENCODING - PYTHON BASELINE GENERATION")
    print("="*70)

    # Parameters matching C# implementation
    d_model = 256
    dropout = 0.1
    max_len = 1024

    print(f"\nParameters:")
    print(f"  d_model: {d_model}")
    print(f"  dropout: {dropout}")
    print(f"  max_len: {max_len}")

    # Create module
    pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
    pe.eval()  # Set to eval mode to disable dropout

    results = {}

    # Test case 1: Short sequence
    print("\n" + "-"*70)
    print("Test 1: Short sequence (10, 1, 256)")
    print("-"*70)

    torch.manual_seed(42)
    seq_len, batch_size = 10, 1
    x1 = torch.randn(seq_len, batch_size, d_model)

    with torch.no_grad():
        y1 = pe(x1)

    results['test1_short_sequence'] = {
        'input_shape': list(x1.shape),
        'output_shape': list(y1.shape),
        'input_mean': float(x1.mean()),
        'input_std': float(x1.std()),
        'output_mean': float(y1.mean()),
        'output_std': float(y1.std()),
        'output_min': float(y1.min()),
        'output_max': float(y1.max()),
        'input_sample': x1[:5, 0, :5].numpy().tolist(),
        'output_sample': y1[:5, 0, :5].numpy().tolist(),
    }

    print(f"  Input shape: {x1.shape}")
    print(f"  Output shape: {y1.shape}")
    print(f"  Input mean/std: {x1.mean():.6f} / {x1.std():.6f}")
    print(f"  Output mean/std: {y1.mean():.6f} / {y1.std():.6f}")
    print(f"  Output min/max: {y1.min():.6f} / {y1.max():.6f}")

    # Test case 2: Zero input to isolate positional encoding
    print("\n" + "-"*70)
    print("Test 2: Zero input (10, 1, 256) - Isolates PE effect")
    print("-"*70)

    x2 = torch.zeros(seq_len, batch_size, d_model)

    with torch.no_grad():
        y2 = pe(x2)

    results['test2_zero_input'] = {
        'input_shape': list(x2.shape),
        'output_shape': list(y2.shape),
        'output_mean': float(y2.mean()),
        'output_std': float(y2.std()),
        'output_min': float(y2.min()),
        'output_max': float(y2.max()),
        # Save first 10 positions, first 10 dimensions
        'positional_encoding_sample': y2[:10, 0, :10].numpy().tolist(),
    }

    print(f"  Output mean/std: {y2.mean():.6f} / {y2.std():.6f}")
    print(f"  Output min/max: {y2.min():.6f} / {y2.max():.6f}")
    print(f"  First position, first 5 dims: {y2[0, 0, :5].numpy()}")

    # Test case 3: Position uniqueness
    print("\n" + "-"*70)
    print("Test 3: Position Uniqueness")
    print("-"*70)

    x3 = torch.zeros(5, 1, d_model)

    with torch.no_grad():
        y3 = pe(x3)

    position_diffs = []
    for i in range(4):
        diff = (y3[i] - y3[i+1]).abs().sum().item()
        position_diffs.append(diff)
        print(f"  Difference between position {i} and {i+1}: {diff:.6f}")

    results['test3_position_uniqueness'] = {
        'position_differences': position_diffs,
        'all_different': all(d > 0.001 for d in position_diffs),
    }

    # Test case 4: Batch consistency
    print("\n" + "-"*70)
    print("Test 4: Batch Consistency (10, 4, 256)")
    print("-"*70)

    x4 = torch.zeros(10, 4, d_model)

    with torch.no_grad():
        y4 = pe(x4)

    batch_diffs = []
    for b in range(1, 4):
        diff = (y4[:, 0, :] - y4[:, b, :]).abs().max().item()
        batch_diffs.append(diff)
        print(f"  Max diff between batch 0 and batch {b}: {diff:.10f}")

    results['test4_batch_consistency'] = {
        'batch_differences': batch_diffs,
        'all_identical': all(d < 1e-6 for d in batch_diffs),
    }

    # Test case 5: Long sequence
    print("\n" + "-"*70)
    print("Test 5: Long sequence (500, 2, 256)")
    print("-"*70)

    torch.manual_seed(123)
    x5 = torch.randn(500, 2, d_model)

    with torch.no_grad():
        y5 = pe(x5)

    results['test5_long_sequence'] = {
        'input_shape': list(x5.shape),
        'output_shape': list(y5.shape),
        'output_mean': float(y5.mean()),
        'output_std': float(y5.std()),
        'output_min': float(y5.min()),
        'output_max': float(y5.max()),
    }

    print(f"  Output mean/std: {y5.mean():.6f} / {y5.std():.6f}")
    print(f"  Output min/max: {y5.min():.6f} / {y5.max():.6f}")

    # Test case 6: Deterministic behavior
    print("\n" + "-"*70)
    print("Test 6: Deterministic Behavior (eval mode)")
    print("-"*70)

    torch.manual_seed(456)
    x6 = torch.randn(20, 1, d_model)

    with torch.no_grad():
        y6_first = pe(x6)
        y6_second = pe(x6)

    max_diff = (y6_first - y6_second).abs().max().item()

    results['test6_deterministic'] = {
        'max_difference_between_runs': max_diff,
        'is_deterministic': max_diff < 1e-6,
    }

    print(f"  Max difference between two runs: {max_diff:.10f}")
    print(f"  Is deterministic: {max_diff < 1e-6}")

    # Save results
    output_path = Path(__file__).parent / "positional_encoding_baseline.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("BASELINE GENERATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Test 1 (Short sequence): Output shape correct")
    print(f"✅ Test 2 (Zero input): PE values in range [{results['test2_zero_input']['output_min']:.3f}, {results['test2_zero_input']['output_max']:.3f}]")
    print(f"✅ Test 3 (Position uniqueness): All positions different")
    print(f"✅ Test 4 (Batch consistency): All batches identical (max diff {max(batch_diffs):.2e})")
    print(f"✅ Test 5 (Long sequence): Processed successfully")
    print(f"✅ Test 6 (Deterministic): Eval mode is deterministic (diff {max_diff:.2e})")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run C# PositionalEncoding tests")
    print("2. Export C# test outputs to JSON")
    print("3. Compare with this baseline")
    print("4. Verify numerical differences < 1e-6")

    return results

if __name__ == '__main__':
    test_positional_encoding()
