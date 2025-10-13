import os, json
import openvino as ov

os.makedirs("models", exist_ok=True)

model = ov.convert_model("models/heron-converted.onnx")
xml_path = "models/heron-converted.xml"
ov.save_model(model, xml_path)
bin_path = "models/heron-converted.bin"
info = {
    "model": os.path.basename(xml_path),
    "size_bytes": os.path.getsize(xml_path) + os.path.getsize(bin_path)
}
json.dump(info, open(xml_path + ".json", "w"), indent=2)
print("openvino-bytes", info["size_bytes"])
