import subprocess, os, json, pathlib
os.makedirs("models", exist_ok=True)
def conv(src, dst):
    subprocess.check_call([
        "python","-m","onnxruntime.tools.convert_onnx_models_to_ort",
        f"models/{src}","--output_dir","models"
    ])
    out = pathlib.Path("models")/ (pathlib.Path(src).stem+".ort")
    out.rename(pathlib.Path("models")/dst)
    info={"model":dst,"size_bytes":os.path.getsize("models/"+dst)}
    json.dump(info, open(f"models/{dst}.json","w"), indent=2)
conv("heron-optimized.onnx", "heron-optimized.ort")
conv("heron-optimized-fp16.onnx", "heron-optimized-fp16.ort")

