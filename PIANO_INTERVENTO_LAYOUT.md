# Piano Di Intervento – Allineamento Performance Layout (.NET vs Python)

## Contesto Attuale
- **Accuracy**: .NET e Python producono 12 detection identiche (IoU medio ≈ 0.99; scarto confidenze ≈ 0.01).
- **Performance**: su `dataset/2305.03393v1-pg9-img.png` la pipeline Python (HuggingFace + ONNX Runtime) impiega ~434.6 ms, la pipeline .NET ~465.1 ms (≈ +7 %, +30 ms).
- **Obiettivo**: ridurre il delta a ≤ 2 % senza regredire in accuratezza o stabilità.

## Strategia Di Ottimizzazione (FASE PER FASE)
### Fase 1 – Ottimizzazioni A Basso Rischio (T=~1.5h)
1. **Cache Bitmap Decodificata**
   - Spostare la decodifica PNG (`SKBitmap.Decode`) fuori dal ciclo di inferenza (sia nei benchmark sia nel pipeline).  
   - Introduzione di un wrapper `IBitmapSource` nel pipeline per riutilizzare l’immagine originaria lungo lo stesso documento.
2. **Riuso `ImageTensor`**
   - Estendere `ImageTensor` con un metodo statico che restituisce istanze da `ArrayPool<float>` (es. `ImageTensor.RentPooled`) e rilascia in `Dispose`.
   - Aggiornare `SkiaImagePreprocessor` a usare il buffer pooled; registrare metriche pre/post per verificare riduzione allocazioni.
3. **SessionOptions ottimizzate**
   - Impostare `GraphOptimizationLevel.ORT_ENABLE_ALL`.
   - Allineare `IntraOpNumThreads` alla macchina (1 su CPU mobile, >1 su server) e documentare i valori scelti.

**Output atteso fase 1**: riduzione delta di almeno 10 ms (target ~455 ms).

### Fase 2 – Gestione Tensioni Di Memoria (T=~2.5h)
1. **Eliminare `ToArray()`**
   - Modificare `LayoutBackendResult` per accettare `ReadOnlyMemory<float>` o wrapper custom (`TensorOwner`) che mantenga in vita `DisposableNamedOnnxValue`.
   - Aggiornare `LayoutPostprocessor` a lavorare su `ReadOnlySpan<float>` (no allocazioni).
2. **Pooling NamedOnnxValue**
   - Introdurre un helper `OnnxValueOwner` per creare/disporre parametri di input in `using` garantendo il rilascio e riuso del buffer `DenseTensor`.
3. **Benchmark incrementale**
   - Misurare il tempo di copia eliminato (profilare `dotnet-counters` o aggiungere `Stopwatch` attorno al post-process).

**Output atteso fase 2**: delta residuo ≤ 15 ms (target ~450 ms).

### Fase 3 – Micro Ottimizzazioni E Validazione Finale (T=~2h)
1. **Profiling fine**
   - Utilizzare `dotnet-trace` con eventi CPU per capire eventuali hotspot residui (es. `NamedOnnxValue` allocazioni).
   - Se necessario, introdurre `Unsafe.AsPointer`/`MemoryMarshal` per evitare conversioni multiple in loop caldo.
2. **Configurazione condizionale**
   - Permettere di disabilitare/abilitare NMS avanzata tramite opzione (per test comparativi).
3. **Regression suite completa**
   - Eseguire `dotnet test` (LayoutSdk.Tests), `infer_onnx.py`, e il benchmark .NET su almeno 3 immagini per generare statistica.

**Output atteso fase 3**: delta ≤ 10 ms (target ~440–445 ms) con detection identiche.

## Processo Di Misurazione
### Strumenti
- Python: `infer_onnx.py` (repo submodule) • output: `results/<run>/timings.csv`.
- .NET: tool console `/tmp/layoutdump` (o equivalente) compilato contro LayoutSdk.

### Metodologia
1. **Warm-up**: eseguire 1 run scartandone il risultato (per entrambe le pipeline).
2. **Misurazioni**: raccogliere almeno 5 campioni consecutivi (`Stopwatch` in ms); loggare in CSV/JSON.
3. **Metriche da registrare**:
   - Per run: `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms`.
   - Per dataset: media, deviazione standard, delta (.NET – Python), rapporto (.NET / Python).

### Script Di Supporto
- **Python summary**:
  ```bash
  python3 src/submodules/ds4sd-docling-layout-heron-onnx/infer_onnx.py \
    --model src/submodules/ds4sd-docling-layout-heron-onnx/dotnet/LayoutSdk/PackagedModels/models/heron-optimized.onnx \
    --variant perf-layout --dataset dataset
  ```
- **.NET summary**:
  ```bash
  dotnet run --project tools/LayoutPerfRunner \
    --input dataset/2305.03393v1-pg9-img.png \
    --runs 6 --discard 1 --output logs/layout_perf_net.json
  ```
  (Il runner dovrà esporre breakdown tempi.)

## Piano Di Validazione
1. **Unit test** (`dotnet test` LayoutSdk.Tests) – devono restare verdi.
2. **Parity check** con script python per verificare detection identiche (IoU ≥ 0.95 per ogni class).
3. **Benchmark** prima/dopo ogni fase con raccolta log (commit con `perf-before`/`perf-after`).

## Deliverables Finali
- Report con tabella comparativa (Python vs .NET) per almeno 3 immagini.
- Change log con elenco ottimizzazioni e impatto stimato.
- Aggiornamento documentazione interna (README se serve) indicando come eseguire benchmark.

