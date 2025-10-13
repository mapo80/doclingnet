# Model Artifacts

The heavy OCR models are not tracked in Git. Use the helper scripts to populate the
local cache when needed:

1. Download the TorchfreeEasyOCR ONNX models:

   ```bash
   python tools/download_torchfree_models.py
   ```

   This creates `models/cpu/*.onnx` (ignored by git).

2. Optionally convert them to OpenVINO IR for the OpenVINO execution provider:

   ```bash
   python tools/convert_to_openvino.py
   ```

   The converted files are stored under `models/openvino/`.

The original EasyOCR baseline models (`EasyOCRDetector.onnx`, `EasyOCRRecognizer.onnx`)
remain under `models/` for comparison.
