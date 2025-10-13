import argparse, os, time, json, hashlib, platform
from datetime import datetime
import numpy as np, cv2
import onnxruntime as ort

p=argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--images", default="./dataset")
p.add_argument("--variant-name", required=True)
p.add_argument("--output", default="results")
p.add_argument("--target-h", type=int, default=1024)
p.add_argument("--target-w", type=int, default=1024)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--runs-per-image", type=int, default=1)
p.add_argument("--threads-intra", type=int, default=0)
p.add_argument("--threads-inter", type=int, default=1)
p.add_argument("--sequential", action="store_true")
args=p.parse_args()

ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(args.output, args.variant_name, f"run-{ts}")
os.makedirs(run_dir, exist_ok=True)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if args.sequential: so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.intra_op_num_threads = args.threads_intra
so.inter_op_num_threads = args.threads_inter
sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
inp_name = inputs[0].name
N,C,H,W = 1,3,args.target_h,args.target_w

def load_and_prep(pth):
    im = cv2.imread(pth, cv2.IMREAD_COLOR)
    if im is None: return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (W,H), interpolation=cv2.INTER_AREA)
    arr = (im.astype(np.float32)/255.0).transpose(2,0,1)[None, ...]  # NCHW
    return arr

# immagini
files = [os.path.join(args.images,f) for f in os.listdir(args.images) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
files.sort()
timings=[]

# warmup
dummy = np.zeros((N,C,H,W), dtype=np.float32)
for _ in range(max(1,args.warmup)):
    sess.run(None, {inp_name: dummy})

# misure
import csv, time
with open(os.path.join(run_dir,"timings.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["filename","ms"])
    for pth in (files or ["__synthetic__"]):
        x = load_and_prep(pth) if pth!="__synthetic__" else dummy
        for _ in range(args.runs_per_image):
            t0=time.perf_counter(); sess.run(None, {inp_name: x}); dt=(time.perf_counter()-t0)*1000
            w.writerow([os.path.basename(pth), f"{dt:.3f}"]); timings.append(dt)

import numpy as np, statistics as st
summary = {
  "count": len(timings),
  "mean_ms": float(np.mean(timings)) if timings else None,
  "median_ms": float(np.median(timings)) if timings else None,
  "p95_ms": float(np.quantile(timings,0.95)) if timings else None
}
json.dump(summary, open(os.path.join(run_dir,"summary.json"),"w"), indent=2)

mi = {
  "model_path": args.model,
  "model_size_bytes": os.path.getsize(args.model),
  "ep": "CPU", "device": "CPU",
  "precision": "fp16" if "fp16" in os.path.basename(args.model).lower() else "fp32"
}
json.dump(mi, open(os.path.join(run_dir,"model_info.json"),"w"), indent=2)

env = {
  "python": platform.python_version(),
  "onnxruntime": ort.__version__,
  "os": platform.platform(),
}
json.dump(env, open(os.path.join(run_dir,"env.json"),"w"), indent=2)

cfg = vars(args)
json.dump(cfg, open(os.path.join(run_dir,"config.json"),"w"), indent=2)

import hashlib, glob
def sha(p): 
    h=hashlib.sha256(); h.update(open(p,"rb").read()); return h.hexdigest()
manifest=[]
for pth in ["timings.csv","summary.json","model_info.json","env.json","config.json"]:
    q=os.path.join(run_dir,pth); 
    if os.path.exists(q): manifest.append({"file":pth,"sha256":sha(q)})
json.dump({"files":manifest}, open(os.path.join(run_dir,"manifest.json"),"w"), indent=2)
open(os.path.join(run_dir,"logs.txt"),"w").write(f"RUN {args.variant_name} ok, N={summary['count']}\n")
print("OK:", run_dir)

