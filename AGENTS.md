# Repository guidelines

- Before building or testing ensure custom NuGet packages are available when you need the real ML implementations:
  1. `mkdir -p packages/custom`
  2. Download the EasyOCR, layout SDK, and TableFormer packages:
     ```bash
     curl -L -o packages/custom/EasyOcrNet.1.0.0.nupkg \
       https://github.com/mapo80/easyocrnet/releases/download/v2025.09.19/EasyOcrNet.1.0.0.nupkg
     curl -L -o packages/custom/Docling.LayoutSdk.1.0.2.nupkg \
       https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/download/models-2025-09-19/Docling.LayoutSdk.1.0.2.nupkg
     curl -L -o packages/custom/TableFormerSdk.1.0.0.nupkg \
       https://github.com/mapo80/ds4sd-docling-tableformer-onnx/releases/download/v1.0.0/TableFormerSdk.1.0.0.nupkg
     ```
  3. Run `dotnet restore` to hydrate the local cache.

  The runtime projects depend on the **real** EasyOcrNet and Docling.LayoutSdk packages. Do not rely on the legacy stubs: they are reserved for unit tests only and live in the test project when required. Always hydrate the packages before building or running tests.

- To validate changes locally execute `dotnet test`. Coverlet runs automatically and enforces a minimum **90% line coverage** threshold. Investigate any coverage regression before opening a pull request.

- Before running the CLI converter make sure you execute the command with workflow diagnostics enabled so the goldens stay in sync **and** persist a new iteration instead of overwriting existing artefacts:

  ```bash
  run_id="$(date -u +%Y-%m-%dT%H%M%SZ)"
  dotnet run --project src/Docling.Tooling -- convert \
    --input dataset/2305.03393v1-pg9-img.png \
    --output "dataset/golden/v0.12.0/2305.03393v1-pg9/dotnet-cli/${run_id}" \
    --markdown docling.md \
    --assets assets \
    --workflow-debug
  ```

  Always create a fresh timestamped subdirectory (e.g. `dotnet-cli/2025-09-30T102910Z/`) so previous iterations remain available for regression comparisons. After each run commit the new folder—including the Markdown, metadata, and `debug/workflow/*.json` snapshots—without deleting older iterations. The layout runner now auto-letterboxes every page to 640x640 before invoking the Heron model, so you no longer need to preprocess the images manually—the CLI command above will finish without the ONNX dimension fault once the NuGet models are restored.

- Immediately after generating a new `.NET` snapshot run the Markdown diff tool so the comparison artefacts stay versioned alongside the export:

  ```bash
  python3 eng/tools/compare_markdown.py \
    dataset/golden/v0.12.0/2305.03393v1-pg9/python-cli/docling.md \
    "dataset/golden/v0.12.0/2305.03393v1-pg9/dotnet-cli/${run_id}/docling.md" \
    "dataset/golden/v0.12.0/2305.03393v1-pg9/comparisons/python-vs-dotnet/${run_id}"
  ```

  Commit both the `.NET` Markdown and the generated `report.md`/`summary.json` so every iteration is traceable without re-running the Python pipeline.

- Image inputs now skip the aggressive deskew/contrast preprocessing pass so layout inference matches the raw page geometry. If you need those transforms (e.g. for PDF flows) convert from PDF sources or adjust the preprocessing flags before running the CLI.

- When layout analysis yields no regions, the OCR stage automatically performs a full-page fallback. The resulting Markdown and `debug/workflow/04_ocr.json` will include the recognised lines along with a `docling:fallback_reason` so debugging information is never empty.

- Keep the workspace clean: ensure `dotnet test` passes and leave the git tree without pending changes after completing a task.
