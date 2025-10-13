# How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

## Convert model to ONNX
```bash
make convert
```

## Optimize ONNX model
```bash
make optimize
```

## Run inference for all variants
```bash
make infer-all
```

## Compare KPIs
```bash
make compare
```
