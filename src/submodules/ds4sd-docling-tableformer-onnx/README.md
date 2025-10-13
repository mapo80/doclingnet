# ds4sd-docling-tableformer-onnx

## Informazioni generali
Repository di valutazione per i modelli `ds4sd/docling-models` dedicati al riconoscimento di tabelle TableFormer.
L'obiettivo è convertire le varianti *fast* e *accurate* in diversi formati (ONNX, formato ORT, OpenVINO), verificarne la parità rispetto ai modelli originali e misurarne le prestazioni su CPU.

## Modelli di partenza
Gli artifact HuggingFace disponibili in [`ds4sd/docling-models`](https://huggingface.co/ds4sd/docling-models/tree/main/model_artifacts/tableformer) contengono due configurazioni principali:

- **TableFormer fast** – ottimizzata per la latenza
- **TableFormer accurate** – ottimizzata per la qualità dell'estrazione

Entrambe condividono la stessa pipeline di preprocessing (resize a 448×448, normalizzazione RGB) ma differiscono per dimensioni e numero di layer.

## Formati supportati
- **PyTorch**: modelli originali in formato `safetensors`
- **ONNX FP32**: esportazione del grafo encoder/decoder TableFormer
- **ONNX ottimizzato (ORT)**: grafo compatto per l'esecuzione con ONNX Runtime
- **OpenVINO IR**: modelli convertiti in `.xml`/`.bin` per CPU Intel

> La conversione in FP16 e INT8 è in corso di validazione; le istruzioni verranno aggiornate una volta completata la parità.

## Pipeline di conversione (bozza)
1. **Esportazione in ONNX**
   ```bash
   python convert_to_onnx.py --model-name tableformer-fast --output models/tableformer-fast.onnx --dataset dataset
   python convert_to_onnx.py --model-name tableformer-accurate --output models/tableformer-accurate.onnx --dataset dataset
   ```
2. **Ottimizzazione**
   ```bash
   python optimize_onnx.py --input models/tableformer-fast.onnx --output models/tableformer-fast-optimized.onnx
   python optimize_onnx.py --input models/tableformer-accurate.onnx --output models/tableformer-accurate-optimized.onnx
   ```
3. **Formato ORT**
   ```bash
   python scripts/convert_to_ort.py --input models/tableformer-fast-optimized.onnx --output models/tableformer-fast.ort
   python scripts/convert_to_ort.py --input models/tableformer-accurate-optimized.onnx --output models/tableformer-accurate.ort
   ```
4. **Conversione in OpenVINO**
   ```bash
   python scripts/convert_to_openvino.py --input models/tableformer-fast.onnx --output-dir models/ov-ir-fast
   python scripts/convert_to_openvino.py --input models/tableformer-accurate.onnx --output-dir models/ov-ir-accurate
   ```

## Benchmark
Gli script di benchmark generano cartelle `results/<variant>/run-YYYYMMDD-HHMMSS/` con:
- `timings.csv` – latenza per immagine
- `summary.json` – statistiche aggregate (media, mediana, p95)
- `model_info.json` – percorso, dimensione, precisione del modello
- `env.json`, `config.json`, `manifest.json`, `logs.txt` – contesto di esecuzione

Esempio di benchmark Python per la variante fast:
```bash
python scripts/bench_python.py --model models/tableformer-fast-optimized.onnx \
  --images ./dataset --variant-name onnx-fast-fp32-cpu \
  --sequential --threads-intra 0 --threads-inter 1 --target-h 448 --target-w 448
```

## SDK .NET 9
La libreria `TableFormerSdk` fornisce un'API modulare per orchestrare i diversi runtime supportati:

- **ONNX Runtime** (`.onnx`)
- **ONNX Runtime** con grafi ottimizzati (`.ort`)
- **OpenVINO CSharp API** (`.xml`/`.bin`)

La soluzione è stata migrata a .NET 9 per sfruttare i miglioramenti del runtime e delle librerie di base. Tutti i progetti sono ora allineati con le ultime versioni stabili dei pacchetti NuGet (ONNX Runtime 1.22.1, SkiaSharp 3.119.0, suite xUnit 2.9.x) per garantire compatibilità e patch di sicurezza aggiornate.

### Architettura della libreria
Il refactoring suddivide le responsabilità principali in componenti dedicate:

- `TableFormerSdk` coordina la pipeline di inferenza, applica la validazione degli input e restituisce `TableStructureResult`.
- `TableFormerSdkOptions` raccoglie i percorsi dei modelli (`TableFormerModelPaths`), la configurazione della visualizzazione (`TableVisualizationOptions`) e le lingue disponibili (`TableFormerLanguage`).
- `BackendRegistry` gestisce il ciclo di vita dei backend concreti, istanziati tramite `DefaultBackendFactory` a partire dalle opzioni fornite.
- Le classi in `TableFormerSdk.Constants` centralizzano messaggi e valori di default (colore/ampiezza degli overlay).
- `OverlayRenderer` incapsula la generazione della bitmap con le bounding box.

### Enumerazioni ufficiali
- `TableFormerRuntime` – runtime di esecuzione (`Auto`, ONNX, ORT, OpenVINO).
- `TableFormerModelVariant` – variante del modello (*Fast* / *Accurate*).
- `TableFormerLanguage` – lingua dell'input gestita dalla configurazione (English, Italian, French, German, Spanish, Portuguese, Chinese, Japanese).

### Ottimizzazione basata sui tempi di esecuzione
La release introduce un sottosistema di ottimizzazione che misura automaticamente le latenze di inferenza di ogni backend e sceglie la configurazione più veloce per ciascuna variante del modello. Il driver è il nuovo `TableFormerPerformanceAdvisor`, alimentato da una finestra scorrevole di metriche (`TableFormerPerformanceOptions.SlidingWindowSize`) e da soglie configurabili (`MinimumSamples`).

- La modalità `TableFormerRuntime.Auto` effettua dapprima un'esplorazione dei runtime disponibili (rispettando l'ordine preferenziale definito in `TableFormerPerformanceOptions.RuntimePriority`) per raccogliere un numero minimo di campioni.
- Una volta popolata la finestra di osservazione, l'SDK sceglie sempre il runtime con la latenza media più bassa, aggiornando l'informazione a ogni chiamata.
- I risultati sono esposti tramite `TableStructureResult.PerformanceSnapshot` e dagli helper `TableFormerSdk.GetPerformanceSnapshots` / `GetLatestSnapshot`, utili per esporre dashboard o telemetrie enterprise.

### Configurazione e uso
```csharp
using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Performance;

var modelRoot = Path.Combine(AppContext.BaseDirectory, "models", "tableformer-onnx");
var fastPaths = TableFormerVariantModelPaths.FromDirectory(modelRoot, "tableformer_fast");
TableFormerVariantModelPaths? accuratePaths = null;
try
{
    accuratePaths = TableFormerVariantModelPaths.FromDirectory(modelRoot, "tableformer_accurate");
}
catch (FileNotFoundException)
{
    Console.WriteLine("Accurate variant not available – falling back to Fast only.");
}

var options = new TableFormerSdkOptions(
    onnx: new TableFormerModelPaths(fastPaths, accuratePaths),
    supportedLanguages: new[]
    {
        TableFormerLanguage.English,
        TableFormerLanguage.Italian
    },
    visualizationOptions: new TableVisualizationOptions(
        strokeColor: SKColors.DeepSkyBlue,
        strokeWidth: 3f),
    performanceOptions: new TableFormerPerformanceOptions(
        enableAdaptiveRuntimeSelection: true,
        minimumSamples: 2,
        runtimePriority: new[]
        {
            TableFormerRuntime.Onnx
        }));

using var sdk = new TableFormerSdk(options);

var result = sdk.Process(
    imagePath: "dataset/gazette_de_france.jpg",
    overlay: true,
    runtime: TableFormerRuntime.Auto,
    variant: TableFormerModelVariant.Fast,
    language: TableFormerLanguage.Italian);

Console.WriteLine($"Detected regions: {result.Regions.Count}");
Console.WriteLine($"Language: {result.Language}");
Console.WriteLine($"Runtime: {result.Runtime} Avg={result.PerformanceSnapshot.AverageLatencyMilliseconds:F2}ms");

result.OverlayImage?.Encode(SKEncodedImageFormat.Png, 90)
    .SaveTo(File.OpenWrite("overlay.png"));
```
> Nota: i percorsi dei modelli vengono validati in fase di costruzione tramite `TableFormerModelPaths`,
> che verifica la presenza di tutti i componenti ONNX (`encoder`, `tag_transformer_*`, `bbox_decoder`, `config.json`, `wordmap.json`) per ciascuna variante.

I pacchetti NuGet di riferimento includono `Microsoft.ML.OnnxRuntime` e `SkiaSharp`.

Le medie aritmetiche (in millisecondi) sulle cinque misurazioni utili sono riassunte nella tabella seguente. I CSV generati includono ora il runtime realmente utilizzato (`filename,runtime,ms`) perché l'SDK restituisce la latenza nativa attraverso `TableStructureResult.InferenceTime`.

| Immagine | ONNX Runtime (ms) | OpenVINO (ms) |
| --- | ---: | ---: |
| HAL.2004.page_82.pdf_125315.png | 60,65 | 6,80 |
| HAL.2004.page_82.pdf_125317.png | 53,90 | 5,57 |

I valori derivano dai CSV generati dall'applicazione (`timings.csv`).【3c7576†L1-L11】【5d16ff†L1-L11】 L'IR OpenVINO riduce di circa 9 volte il tempo medio rispetto al runtime ONNX su questo campione ristretto, pur mantenendo la scalabilità sull'intero dataset grazie al preprocessing integrato nello SDK. La console di benchmark sfrutta direttamente `TableStructureResult.InferenceTime`, mantenendo la coerenza con la logica di auto-tuning disponibile in produzione.
### Test e benchmark .NET
- **Test unitari**: `dotnet test TableFormerSdk.sln`
- **Benchmark**: l'applicazione console `TableFormerSdk.Benchmarks` replica gli script Python e salva gli stessi artifact.

```bash
dotnet run --project dotnet/TableFormerSdk.Benchmarks/TableFormerSdk.Benchmarks.csproj -- \
  --engine Onnx --variant-name Fast \
  --onnx-fast models/heron-optimized.onnx \
  --images dataset/FinTabNet/benchmark --output results/benchmarks \
  --runs-per-image 5 --warmup 1 --target-h 640 --target-w 640
```

## Requisiti
Installare le dipendenze principali:
```bash
pip install -r requirements.txt
```
I modelli di grandi dimensioni vengono salvati nella cartella `models/` ed esclusi dal versionamento git.
