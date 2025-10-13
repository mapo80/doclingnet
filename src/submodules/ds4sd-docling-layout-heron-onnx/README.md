# ds4sd-docling-layout-heron-onnx

## Informazioni generali
Repository di valutazione per il modello `ds4sd/docling-layout-heron`, un detector di layout documentale. L'obiettivo è convertire il modello in diversi formati (ONNX, formato ORT, OpenVINO) e confrontarne le prestazioni su CPU.

## Modello di partenza
Il modello HuggingFace `ds4sd/docling-layout-heron` estrae strutture di layout da pagine di documento, rilevando componenti come paragrafi, titoli, tabelle e figure. Le utilità presenti in questa repo permettono di esportarlo e verificarne la parità rispetto alla versione PyTorch.

## Modelli supportati
- **PyTorch**: modello originale da HuggingFace.
- **ONNX FP32**: versione convertita.
- **ONNX FP32 ottimizzato**: generato con le ottimizzazioni di ONNX Runtime.
- **Formato ORT**: serializzazione del grafo ottimizzato in `.ort` (FP32 e, se supportato, FP16).
- **OpenVINO IR**: modello convertito in formato OpenVINO (FP32).

> La conversione in FP16 è possibile ma l'esecuzione con ONNX Runtime può fallire a causa di operatori non supportati.

## Modelli pronti all'uso
I convertiti più recenti sono pubblicati come asset nella release
[`models-2025-09-19`](https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/tag/models-2025-09-19).
Gli artifact sono stati generati con:

- `onnxruntime 1.22.1`
- `onnxruntime-tools 1.7.0`
- `torch 2.6.0`
- `transformers 4.56.1`
- `openvino 2025.3.0`

### Contenuto della release

| File | Dimensione (MB) | Descrizione |
|:--|--:|:--|
| `heron-converted.onnx` | 163.87 | Esportazione ONNX FP32 direttamente dal modello HuggingFace |
| `heron-optimized.onnx` | 163.53 | ONNX FP32 con ottimizzazioni ORT (graph level `ORT_ENABLE_ALL`) |
| `heron-optimized-fp16.onnx` | 82.05 | ONNX FP16 (conversione FP16 completa, conservando IO FP32) |
| `heron-optimized.ort` | 163.95 | Formato ORT con ottimizzazioni “Fixed” per CPU |
| `heron-optimized.with_runtime_opt.ort` | 163.95 | Formato ORT con ottimizzazioni “Runtime” |
| `heron-converted.xml` | 1.44 | OpenVINO IR (XML) |
| `heron-converted.bin` | 81.58 | OpenVINO IR weights |

> Nota: ONNX Runtime non riesce a caricare la variante FP16 nel formato `.ort`
per via di un mismatch di tipo sul nodo `/model/encoder/Cast_2`. Per l'uso con
ORT rimangono quindi disponibili le sole varianti FP32.

### Download rapido

Scarica i file nella cartella `models/` con `curl` (o `wget`):

```bash
mkdir -p models models/ov-ir
curl -L -o models/heron-optimized.onnx \
  https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/download/models-2025-09-19/heron-optimized.onnx
curl -L -o models/heron-optimized.ort \
  https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/download/models-2025-09-19/heron-optimized.ort
curl -L -o models/ov-ir/heron-optimized.xml \
  https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/download/models-2025-09-19/heron-optimized.xml
curl -L -o models/ov-ir/heron-optimized.bin \
  https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/download/models-2025-09-19/heron-optimized.bin
```

Sostituisci il nome del file per ottenere gli altri asset della
release. Dopo il download, gli script nella sezione successiva
possono essere eseguiti direttamente senza ulteriore conversione.

## Pacchetto NuGet `Docling.LayoutSdk`

È disponibile un pacchetto NuGet che contiene la libreria .NET (`LayoutSdk`) e **tutti** i modelli distribuiti nella release `models-2025-09-19`. Il pacchetto (`Docling.LayoutSdk.1.0.2.nupkg`, ~750 MB) è stato allegato alla stessa release su GitHub e può essere scaricato direttamente da:

<https://github.com/mapo80/ds4sd-docling-layout-heron-onnx/releases/download/models-2025-09-19/Docling.LayoutSdk.1.0.2.nupkg>

### Contenuto del pacchetto

All'interno del pacchetto sono inclusi i seguenti asset, copiati nella cartella `models/` (con sottocartella `ov-ir/` per il formato OpenVINO) durante il `restore` del progetto che fa riferimento alla libreria:

- `heron-converted.onnx`
- `heron-optimized.onnx`
- `heron-optimized-fp16.onnx`
- `heron-optimized.ort`
- `heron-optimized.with_runtime_opt.ort`
- `ov-ir/heron-converted.xml`
- `ov-ir/heron-converted.bin`

La libreria espone inoltre la classe di supporto `LayoutSdkBundledModels` che consente di risolvere i percorsi di questi file e creare rapidamente delle `LayoutSdkOptions` già pronte all'uso. La scelta del backend avviene passando il valore dell'enum `LayoutRuntime` (`Onnx`, `Ort` oppure `OpenVino`) al metodo `Process`, senza dover indicare manualmente il percorso del modello per ogni formato.

```csharp
using LayoutSdk.Configuration;

var options = LayoutSdkBundledModels.CreateOptions(validateModelPaths: true);
options.EnsureModelPaths();

Console.WriteLine(LayoutSdkBundledModels.GetOptimizedOnnxPath());
Console.WriteLine(LayoutSdkBundledModels.GetOpenVinoXmlPath());

using var sdk = new LayoutSdk.LayoutSdk(options);
var resultOnnx = sdk.Process("input.png", overlay: false, LayoutSdk.LayoutRuntime.Onnx);
var resultOrt = sdk.Process("input.png", overlay: false, LayoutSdk.LayoutRuntime.Ort);
var resultOpenVino = sdk.Process("input.png", overlay: false, LayoutSdk.LayoutRuntime.OpenVino);
```

> Il parametro opzionale `useRuntimeOptimizedOrt` di `CreateOptions` permette di scegliere la variante `.ort` da utilizzare (predefinita: `heron-optimized.with_runtime_opt.ort`). Impostandolo a `false` viene selezionato `heron-optimized.ort`.

### Come rigenerare il pacchetto

1. Scaricare gli asset della release nella cartella `dotnet/LayoutSdk/PackagedModels/models/` mantenendo la stessa struttura (`ov-ir/` incluso).
2. Eseguire il comando di packaging in Release:
   ```bash
   dotnet pack dotnet/LayoutSdk/LayoutSdk.csproj -c Release
   ```
   L'output viene salvato nella cartella `artifacts/`.
3. (Opzionale) Pubblicare il pacchetto sulla release GitHub esistente:
   ```bash
   export GITHUB_TOKEN=... # usare un token personale con permessi "repo"
   release_id=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/mapo80/ds4sd-docling-layout-heron-onnx/releases/tags/models-2025-09-19 | jq '.id')
   curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @artifacts/Docling.LayoutSdk.1.0.2.nupkg \
     "https://uploads.github.com/repos/mapo80/ds4sd-docling-layout-heron-onnx/releases/$release_id/assets?name=Docling.LayoutSdk.1.0.2.nupkg"
   ```

> **Nota:** non includere il token nel repository. In locale è sufficiente esportarlo nel proprio ambiente prima di eseguire i comandi `curl`.

### Progetto di esempio `LayoutSdk.PackageDemo`

La soluzione contiene il progetto console `dotnet/LayoutSdk.PackageDemo`, configurato per utilizzare esclusivamente il pacchetto NuGet (`Docling.LayoutSdk`). Il progetto usa la classe `LayoutSdkBundledModels` per:

- verificare che tutti i file inclusi nel pacchetto siano presenti;
- creare le `LayoutSdkOptions` predefinite;
- eseguire un'inferenza con il backend ONNX (o con la variante ORT) su un'immagine `640×640` generata (oppure, se disponibile, sulla pagina `dataset/gazette_de_france.jpg` ridimensionata automaticamente).

Per eseguire l'esempio senza scaricare manualmente i modelli è sufficiente:

```bash
# 1. Copiare il pacchetto nella cartella "packages" del repository
cp Docling.LayoutSdk.1.0.2.nupkg packages/

# 2. Ripristinare ed eseguire il progetto console
dotnet restore dotnet/LayoutSdk.PackageDemo/LayoutSdk.PackageDemo.csproj
dotnet run --project dotnet/LayoutSdk.PackageDemo/LayoutSdk.PackageDemo.csproj -c Release
```

La configurazione `nuget.config` già inclusa nel progetto aggiunge automaticamente la cartella locale `packages/` come feed, per cui non è necessario registrare sorgenti aggiuntive. L'output della console conferma che i file del modello vengono trovati direttamente dal pacchetto e riporta le tempistiche dell'inferenza.

## Performance
Benchmark su CPU con input `640×640`, eseguiti in sequenza su due immagini del folder `dataset/` con `--threads-intra 0` e `--threads-inter 1`.

### Tabella di confronto

| Variante                      | Runtime | Precisione | Median (ms) | p95 (ms) | Dimensione (MB) |
|-------------------------------|---------|------------|-------------|----------|-----------------|
| onnx-fp32-cpu                 | Python  | FP32       | 704.90      | 727.85   | 163.53          |
| onnx-fp32-ort                 | Python  | FP32       | 668.98      | 704.83   | 163.97          |
| openvino-fp32-cpu             | Python  | FP32       | 430.03      | 447.35   | 83.07           |
| dotnet-onnx-fp32-cpu          | .NET    | FP32       | 822.09      | 830.62   | 163.53          |
| dotnet-onnx-fp32-ort          | .NET    | FP32       | 926.53      | 935.15   | 163.97          |
| dotnet-openvino-fp32-cpu      | .NET    | FP32       | 576.59      | 594.77   | 83.07           |

I valori provengono dai file `summary.json` e `model_info.json` generati durante il benchmark.

### Dove trovare le misurazioni
Ogni esecuzione salva risultati autoconsistenti in `results/<variant>/run-YYYYMMDD-HHMMSS/` con i seguenti file principali:
- `timings.csv` – latenza per singola immagine.
- `summary.json` – statistiche aggregate (media, mediana, p95).
- `model_info.json` – percorso, dimensione e precisione del modello.
- `env.json`, `config.json`, `manifest.json`, `logs.txt` – contesto di esecuzione e manifest.

## Benchmark su DocLayNet
L'intero set di 20 immagini estratte da `icdar2023-doclaynet.parquet` è stato utilizzato per valutare le tre varianti principali del modello.
OpenVINO risulta il runtime più rapido, con una latenza mediana di ~430 ms e un throughput di oltre 2 immagini al secondo, mantenendo al contempo una dimensione del modello dimezzata rispetto alle versioni ONNX/ORT.
Le varianti ONNX e ORT mostrano prestazioni simili (mediana ~590 ms), ma ORT evidenzia una maggiore variabilità. OpenVINO rileva 3 box in meno rispetto agli altri due, un differenziale di ~0,3 % sul totale delle bounding box.

| Runtime | Model MB | Images | Boxes | Boxes/img | Mean ms | Median ms | P95 ms | Min ms | Max ms | Std ms | Img/s | Boxes/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onnx-fp32-cpu | 163.53 | 20 | 1046 | 52.30 | 593.10 | 590.10 | 656.92 | 522.15 | 752.97 | 64.25 | 1.69 | 88.18 |
| onnx-fp32-ort | 163.97 | 20 | 1046 | 52.30 | 603.14 | 592.92 | 711.99 | 514.61 | 957.76 | 104.70 | 1.66 | 86.71 |
| openvino-fp32-cpu | 83.03 | 20 | 1043 | 52.15 | 492.24 | 430.02 | 713.12 | 392.06 | 1183.23 | 165.96 | 2.03 | 105.94 |

## Conversione e benchmark in ONNX/ORT
1. **Esportazione in ONNX**
   ```bash
   python convert_to_onnx.py --output models/heron-converted.onnx --dataset dataset
   ```
2. **Ottimizzazione**
   ```bash
   python optimize_onnx.py --input models/heron-converted.onnx --output models/heron-optimized.onnx
   ```
3. **Conversione in FP16 (opzionale)**
   ```bash
   python scripts/convert_fp16.py
   ```
4. **Generazione del formato ORT**
   ```bash
   python scripts/convert_to_ort.py
   ```
5. **Benchmark**
   ```bash
   python scripts/bench_python.py --model models/heron-optimized.onnx \
     --images ./dataset --variant-name onnx-fp32-cpu \
     --sequential --threads-intra 0 --threads-inter 1 --target-h 640 --target-w 640

   python scripts/bench_python.py --model models/heron-optimized.ort \
     --images ./dataset --variant-name onnx-fp32-ort \
     --sequential --threads-intra 0 --threads-inter 1 --target-h 640 --target-w 640
   ```

## Conversione e benchmark in OpenVINO
1. **Conversione a IR**
   ```bash
   python scripts/convert_to_openvino.py
   ```
2. **Benchmark**
   ```bash
   python scripts/bench_openvino.py --model models/heron-converted.xml \
     --images ./dataset --variant-name openvino-fp32-cpu \
     --sequential --threads-intra 0 --target-h 640 --target-w 640
   ```

## Requisiti
Installare le dipendenze principali:
```bash
pip install -r requirements.txt
```
I file di modello di grandi dimensioni sono salvati nella cartella `models/` e sono esclusi dal versionamento git.

## SDK .NET 8
La libreria `LayoutSdk` è stata rifattorizzata con un'architettura modulare e professionale, organizzata in spazi dei nomi indipendenti:

- **Configuration** – `LayoutSdkOptions` centralizza i percorsi dei modelli (ONNX e `OpenVinoModelOptions` per la coppia `xml/bin`) e la lingua predefinita del documento tramite l'enum `DocumentLanguage`. Le opzioni possono verificare a runtime l'esistenza dei modelli (`ValidateModelPaths`).
- **Factories** – `ILayoutBackendFactory` e `LayoutBackendFactory` incapsulano la creazione dei backend per i diversi motori (ONNX Runtime e OpenVINO), favorendo l'iniezione di dipendenze e la personalizzazione enterprise.
- **Processing** – `IImagePreprocessor`, `SkiaImagePreprocessor`, `ImageTensor` e `LayoutPipeline` separano il preprocessing dei pixel, l'esecuzione del backend e la gestione del ciclo di vita delle risorse (`ArrayPool<float>` per ridurre le allocazioni).
- **Metrics** – `LayoutExecutionMetrics` espone la durata di preprocessing, inferenza e disegno overlay, fornendo un driver affidabile per benchmark e telemetria senza dover misurare manualmente i tempi con `Stopwatch`.
- **Rendering** – `IImageOverlayRenderer` e `ImageOverlayRenderer` generano l'overlay grafico applicando le costanti definite in `LayoutDefaults`.

Questo design consente di estendere facilmente la libreria (ad esempio aggiungendo un nuovo backend o una pipeline GPU) e rende la base di codice pronta per scenari enterprise (telemetria, osservabilità, riuso delle risorse).

### Utilizzo
```csharp
using LayoutSdk;
using LayoutSdk.Configuration;

var options = new LayoutSdkOptions(
    onnxModelPath: "models/heron-optimized.onnx",
    openVino: new OpenVinoModelOptions(
        modelXmlPath: "models/ov-ir/heron-optimized.xml",
        weightsBinPath: "models/ov-ir/heron-optimized.bin"),
    defaultLanguage: DocumentLanguage.English,
    validateModelPaths: true);

using var sdk = new LayoutSdk(options);
var result = sdk.Process("dataset/gazette_de_france.jpg", overlay: true, LayoutRuntime.OpenVino);
Console.WriteLine($"Detected: {result.Boxes.Count} elements in {result.Language}");
Console.WriteLine($"Latency (ms): {result.Metrics.TotalDuration.TotalMilliseconds:F2}");
result.OverlayImage?.Encode(SKEncodedImageFormat.Png, 90)
    .SaveTo(File.OpenWrite("overlay.png"));
```

### Costanti, runtime e lingue supportate
- Le costanti grafiche (colore e spessore dell'overlay, messaggi di errore) sono definite nel file `LayoutDefaults.cs` per garantirne la riusabilità e l'allineamento tra backend.
- I runtime di inferenza disponibili sono esposti dall'enum `LayoutRuntime` (`Onnx`, `Ort`, `OpenVino`), così da selezionare esplicitamente il modello ONNX, il formato serializzato `.ort` oppure il formato nativo OpenVINO IR.
- Le lingue disponibili sono esposte dall'enum `DocumentLanguage` (`English`, `French`, `German`, `Italian`, `Spanish`), in modo da poter configurare con chiarezza scenari multilingua.

### Personalizzazione enterprise
- **Preprocessing personalizzato** – È possibile iniettare un'implementazione di `IImagePreprocessor` per supportare pipeline hardware accelerate (GPU, FPGA) mantenendo invariata la logica di orchestrazione.
- **Backend specializzati** – Implementando `ILayoutBackend` è possibile collegare motori di inferenza aggiuntivi (TensorRT, DirectML, ecc.) riusando la pipeline e la raccolta di metriche.
- **Metriche centralizzate** – I dati di `LayoutExecutionMetrics` possono essere esportati verso sistemi di telemetria (Application Insights, OpenTelemetry) fornendo visibilità end-to-end sulla latenza.

### Test automatici
La soluzione include test di unità in `LayoutSdk.Tests`, eseguibili con:

```bash
dotnet test dotnet/LayoutSdk.Tests/LayoutSdk.Tests.csproj
```

I test coprono i casi d'errore della pipeline di elaborazione immagini, la generazione dell'overlay, la propagazione della lingua e la validazione dei percorsi modello. I pacchetti NuGet utilizzati sono allineati alle ultime versioni stabili disponibili (`Microsoft.ML.OnnxRuntime 1.22.1`, `OpenVINO.CSharp.API 2025.1.0.2`, `OpenVINO.runtime.ubuntu.24-x86_64 2024.4.0.1`, `SkiaSharp 3.119.0`).

### Benchmark .NET
L'applicazione console `LayoutSdk.Benchmarks` replica lo script Python generando le stesse cartelle di output:

```bash
dotnet run --project dotnet/LayoutSdk.Benchmarks/LayoutSdk.Benchmarks.csproj -- \
  --runtime Onnx --variant-name dotnet-onnx-fp32-cpu \
  --images dataset --target-h 640 --target-w 640

dotnet run --project dotnet/LayoutSdk.Benchmarks/LayoutSdk.Benchmarks.csproj -- \
  --runtime Ort --variant-name dotnet-ort-fp32-cpu \
  --images dataset --target-h 640 --target-w 640

dotnet run --project dotnet/LayoutSdk.Benchmarks/LayoutSdk.Benchmarks.csproj -- \
  --runtime OpenVino --variant-name dotnet-openvino-fp32-cpu \
  --images dataset --target-h 640 --target-w 640

# confronto automatico su due pagine del dataset
dotnet run --project dotnet/LayoutSdk.Benchmarks/LayoutSdk.Benchmarks.csproj -- \
  --compare --variant-name dotnet-runtime-comparison \
  --images dataset --target-h 640 --target-w 640
```
I risultati vengono salvati in `results/<variant>/run-YYYYMMDD-HHMMSS/` con gli stessi file `summary.json`, `model_info.json` e relativi manifest. In modalità `--compare` viene inoltre generato `comparison.json` con la sintesi affiancata delle metriche per ONNX Runtime e OpenVINO sfruttando le prime due immagini disponibili nella cartella `dataset/`.
