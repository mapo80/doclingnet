.PHONY: convert optimize infer-baseline infer-onnx-converted infer-onnx-optimized infer-all compare convert-int8 to-ort validate bench-py bench-dotnet

convert:
	python convert_to_onnx.py

optimize: convert
	python optimize_onnx.py

infer-baseline:
	python infer_baseline.py

infer-onnx-converted: convert
	python infer_onnx.py --model models/heron-converted.onnx --variant onnx-converted

infer-onnx-optimized: optimize
	python infer_onnx.py --model models/heron-optimized.onnx --variant onnx-optimized

infer-all: infer-baseline infer-onnx-converted infer-onnx-optimized

compare: infer-all
	python compare_kpi.py

convert-int8:
	python scripts/quantize_dynamic.py

to-ort:
	python scripts/convert_to_ort.py

validate:
	python scripts/validate_quality.py --dataset ./dataset

bench-py:
	python scripts/bench_python.py --dataset ./dataset --model ./models/heron-int8-dynamic.onnx --out ./results/onnx-int8-dynamic

bench-dotnet:
	dotnet run --project dotnet/OrtRunner/OrtRunner.csproj --configuration Release
