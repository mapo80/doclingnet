import onnx, os, hashlib
from onnxconverter_common.float16 import convert_float_to_float16
m = onnx.load("models/heron-optimized.onnx")
m_fp16 = convert_float_to_float16(m, keep_io_types=True)
os.makedirs("models", exist_ok=True)
onnx.save(m_fp16, "models/heron-optimized-fp16.onnx")
print("fp16-bytes", os.path.getsize("models/heron-optimized-fp16.onnx"))

