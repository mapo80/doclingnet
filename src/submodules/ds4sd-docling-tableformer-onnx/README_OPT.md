# ONNX Runtime + .NET Optimization

This project demonstrates dynamic INT8 quantization and .NET inference for the TableFormer models (`fast` and `accurate`).

## Setup

```bash
pip install -r requirements.txt
make convert-int8
make to-ort
```

## Benchmark & Validation

```bash
make validate
make bench-py
make bench-dotnet  # requires .NET 8 SDK
```

## KPI Summary

| variant | size(MB) | mean ms | p95 ms | IoU mean | IoU@0.5 | Δ box % |
|:--|--:|--:|--:|--:|--:|--:|
| baseline | 327.4 | 1753.15 | 2086.39 | 1.00 | 1.00 | 0.0 |
| onnx-optimized | 163.5 | 903.99 | 976.21 | 1.00 | 1.00 | 0.0 |
| onnx-int8-dynamic | 143.7 | 675.54 | 719.63 | 0.983 | 0.985 | -5.85 |

Models are ignored via `.gitignore` (use Git LFS for binaries >100 MB).
