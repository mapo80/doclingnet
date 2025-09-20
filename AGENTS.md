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

- Keep the workspace clean: ensure `dotnet test` passes and leave the git tree without pending changes after completing a task.
