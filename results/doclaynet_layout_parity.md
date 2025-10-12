## Layout Parity – doclaynet\_parquet (Python vs .NET)

- **Dataset**: `src/submodules/ds4sd-docling-layout-heron-onnx/dataset/doclaynet_parquet` (20 PNG pages).
- **Baselines**:
  - Python (`pipeline_utils` + `AutoImageProcessor` + ORT): `/tmp/layout_python_doclaynet.py`
  - .NET (`LayoutSdk` ONNX backend): `/tmp/docling-bench-TSChUE/Program.cs`
- **Artifacts**:
  - Metrics: `/tmp/layout_compare/python/metrics.json`, `/tmp/layout_compare/dotnet/metrics.json`
  - Prediction JSON per pagina: nelle rispettive cartelle `predictions/`
  - Diff consolidato (IoU, delta per etichetta, mismatch): `/tmp/layout_compare/report.json`

### Timing – medie per pagina

| Stage | Python (ms) | .NET (ms) | Δ (.NET − Python) |
| --- | --- | --- | --- |
| Preprocess | 9.96 | 2.49 | −7.47 |
| Inference | 458.30 | 471.29 | +12.99 |
| Postprocess | 1.00 | 1.02 | +0.02 |
| **Total** | **469.25** | **474.80** | **+5.54** (+1.18 %) |

**Outliers**
- `.NET` molto più lenta: `6b0748bbee3583ef05429a3d43099fb74939058bb76d56465d802f21ce835120_0045.png` (+105 ms), `d811f9ce79742ff715dcc00f54edb24880553b50af2750e4d37198df1b3d6691_0066.png` (+35 ms).
- `.NET` più rapida: `9cbefdde84940335ea820a689ecd6ade03d5294facdf0d6f745ac95c6e0a8531_0054.png` (−102 ms), `43210a2cc24df60b30ae934c7f5d5ad908cd47e1895efce2f228e2599687f421_0010.png` (−40 ms).

### Detection parity

- **Totale box**: Python 1046 vs .NET 1071 (Δ +25).
- **IoU medio (box match)**: 0.9728 (min 0.9538) → geometria allineata.
- **Differenze etichette** (`dotnet` − `python`):
  - `Text` +22, `Key-Value Region` +13, `List-item` +6, `Picture` −10, `Section-header` −3, `Table` −2.

### File critici

| File | Δ box (.NET − Py) | Note su mismatch |
| --- | --- | --- |
| `cbdfa2821c5e3ff670df52fe31add3979ad78c84fbe98db20fa3e3986589a53a_0000.png` | +8 | `.NET` aggiunge 11 `Key-Value Region` e 3 `List-item`; Python ha 3 `Section-header`, 1 `Formula` in più. |
| `2ef60a7ecbc03e09be8437b7d3c7b099a25069d0be108968d93e3f92fbc0ac33_0044.png` | +5 | `.NET` incrementa `Text`/`Key-Value Region`/`List-item`, perde `Picture` e `Table`. |
| `5d50d1d6888d6e97d4cddddc90ff3681856c67b664c5eaee0d4f4a2c59959733_0099.png` | −5 | Python-only `Key-Value Region`×2, `Text`×3. |
| `6b0748bbee3583ef05429a3d43099fb74939058bb76d56465d802f21ce835120_0045.png` | +4 | Solo `.NET` rileva 4 `Text`; anche peggior regressione di tempo (+105 ms). |
| `d811f9ce79742ff715dcc00f54edb24880553b50af2750e4d37198df1b3d6691_0066.png` | +4 | `.NET`: +7 `Text`, +Page-footer/List/KV; Python ha 4 `Text` e 1 `Table` mancanti. |
| `9cbefdde84940335ea820a689ecd6ade03d5294facdf0d6f745ac95c6e0a8531_0054.png` | −2 | Python-only `Text`×8, `Section-header`×4, `Table`×2, `K/V`×1; `.NET` ha `Text`×8, `Picture`×1 ecc. |
| Altri con Δ±2/3 | `1c10f9f…_0013.png`, `ea90c8e…_0246.png`, `700fef0…_0160.png`, `43210a2…_0010.png`, `f2f98f4…_0147.png`, `fa62aed…_0036.png`, `0dc2c83…_0151.png`. |

Questi file coprono ~80 % delle differenze totali (20/25 box) e vanno analizzati per arrivare al target di 99.99 % identicità.

### Cause probabili

- **Soglie/NMS non allineate**: Python applica threshold per classe nel post-process; `.NET` usa `LayoutPostprocessOptions` statiche ⇒ eccesso di `Text`/`Key-Value Region`.
- **Implementazione softmax/ordinamento**: differenze di rounding o ordering possono spostare le predizioni marginali (es. `.NET` aggiunge box con score ~0.27).
- **Prestazioni**: il pooling riduce preprocess (−7 ms) ma costa +13 ms in inferenza ⇒ verificare `OnnxInputBuilder` per overhead su LINQ/array clearing; controllare `LayoutSdkProfilingSnapshot` nei job lenti.
- **Distribuzione etichette**: `.NET` converge verso classi ad alta frequenza (`Text`); soglie per classi minori (`Picture`, `Section-header`, `Table`) vanno riallineate.

### Piano per raggiungere 99.99 % parity

1. **Allineare threshold/NMS**
   - Estrarre da Python i `processor.config` per class thresholds e replicarli in `LayoutPostprocessOptions`.
   - Rendere configurabile `EnableAdvancedNonMaxSuppression` e testare entrambe le modalità sulle pagine problematiche.
   - Aggiungere test parametrico in `LayoutPostprocessorTests` che confronti output con JSON golden Python (per pagine campione).

2. **Strumento di parity**
   - Estendere `tools/LayoutPerfRunner` o aggiungere `LayoutParityRunner` per eseguire .NET, caricare predictions Python e calcolare IoU/conteggi (basato su `/tmp/layout_compare_analyze.py`).
   - Usare il report per gate CI: tolleranza max 0.01 % mismatch e IoU medio ≥0.99.

3. **Profiling mirato**
   - Profilare `6b0748…`, `d811f9…`, `5d50d1…` con `dotnet-trace` e `dotnet-counters` per verificare overhead in inference/postprocess.
   - Analizzare `LayoutSdkProfilingSnapshot` per vedere se `PostprocessMilliseconds` cresce durante i picchi.

4. **Normalizzare output**
   - Garantire ordinamento per score descending (`LayoutPipeline` prima di restituire `Boxes`).
  - Rounding identico a Python (doppia precisione) per le coordinate.
   - Verificare che `TensorOwner` fornisca buffer corretti senza stride differente.

5. **Test end-to-end**
   - Integrare in `LayoutSdk.Tests` un test di golden parity con 3–4 immagini campione (assert su conteggio, tipologia, IoU > 0.99).
   - Script in `eng/` per rigenerare golden e report Markdown per ogni run (Versioning in `results/`).

6. **Validazione finale**
   - Rerun comparativa completa; obiettivo: |box_delta| ≤ 0.0001 * totale (≤1 box su 10 000).
   - Documentare procedura nel README override e aggiungere controllo CI che blocchi regressioni di parity.

Target finale: comportamento `.NET` indistinguibile al 99.99 % dalla pipeline Python, sia in termini di detection (conteggio/label/IoU) sia di performance su base dataset.
