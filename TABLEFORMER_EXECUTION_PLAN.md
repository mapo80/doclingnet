# Piano Operativo TableFormer (.NET)

Questo piano descrive nel dettaglio le attività necessarie per completare l'integrazione del backend TableFormer basato sui modelli ufficiali Docling. Ogni fase è suddivisa in sotto-attività con prerequisiti, deliverable, strumenti e verifiche esplicite.

---

## Fase 0 · Preparazione Ambiente

- **Obiettivo**: garantire che l'ambiente locale disponga di tutte le dipendenze e degli artefatti necessari per le fasi successive.
- **Prerequisiti**: accesso ai repository principali (`doclingnet`, `ds4sd-docling-tableformer-onnx`, `docling-ibm-models`), tooling `dotnet 9`, `python3`, `onnxruntime`, `git`.
- **Durata stimata**: 0.5 h.

### 0.1 Verifica dipendenze .NET
- Controllare che `dotnet --info` riporti la versione SDK 9.x.
- Se mancante, installare SDK 9.0 e riavviare terminale/IDE.

### 0.2 Hydration pacchetti custom
- Creare directory `packages/custom` se non presente.
- Scaricare i pacchetti NuGet custom (EasyOcrNet, Docling.LayoutSdk, TableFormerSdk) con gli URL forniti nelle repository guidelines.
- Eseguire `dotnet restore` nella root della repo.
- Verifica: assenza di warning/errore nel restore, presenza dei pacchetti nella cache locale.

### 0.3 Validazione ambiente Python/ONNX
- Assicurarsi che `python3 -m onnxruntime` importi senza errori.
- Verificare che i pacchetti `torch`, `safetensors`, `onnx`, `onnxruntime` siano installati nelle versioni usate dagli script (`pip list`).

---

## Fase 1 · Consolidamento Artefatti Modello

- **Obiettivo**: allineare i file ONNX, config e word map generati dalla conversione component-wise.
- **Prerequisiti**: aver completato FASE2 conversione ONNX (file `.onnx` già presenti nella submodule).
- **Durata stimata**: 1 h.

### 1.1 Riesportazione Config & Wordmap
- Eseguire `python tools/convert_tableformer_components_to_onnx.py --model fast --output src/submodules/ds4sd-docling-tableformer-onnx/models`.
- Ripetere con `--model accurate`.
- Verifica: presenza dei file `tableformer_{variant}_config.json` e `tableformer_{variant}_wordmap.json` nella cartella modelli.

### 1.2 Controllo coerenza file
- Validare che i JSON contengano le chiavi `dataset_wordmap`, `model` e valori mean/std.
- Aggiornare `TABLEFORMER_MIGRATION_STATUS.md` con data/completion note.

### 1.3 Aggiornamento gestione path
- Adeguare `TableFormerVariantModelPaths` se i nomi file cambiano (es. suffisso).
- Scrivere test rapido (es. `TableFormerModelPathsTests`) che carica i nuovi JSON.

---

## Fase 2 · Normalizzazione & Preprocessing

- **Obiettivo**: applicare correttamente la normalizzazione PubTabNet e sincronizzare i preprocess tra backend e pipeline.
- **Prerequisiti**: Fase 1 completata, config disponibili.
- **Durata stimata**: 1.5 h.

### 2.1 Estrazione mean/std dinamici
- Leggere le statistiche normalizzazione dal config JSON (sezione `dataset_normalization`).
- Serializzare i valori in una struttura C# (es. record `NormalizationParameters`).

### 2.2 Aggiornamento `PreprocessImage`
- Modificare `TableFormerOnnxBackend.PreprocessImage` per:
  - effettuare convert-to-float,
  - normalizzare con mean/std,
  - evitare smaltimento prematuro della bitmap di destinazione.
- Aggiungere test in `TableFormerOnnxBackendTests` che confronta il tensore pre-normalizzazione.

### 2.3 Allineamento dimensioni input
- Verificare se serve letterboxing a 448 o se i modelli attesi sono 640 (cross-check con config).
- Aggiornare commenti/documentazione (`docs/TABLEFORMER_ARCHITECTURE.md`) dopo la conferma.

---

## Fase 3 · Autoregressivo & Decoder Step

- **Obiettivo**: ottenere la sequenza token reale sfruttando il decoder ONNX e generare i tensori necessari per il bbox decoder.
- **Prerequisiti**: Fase 2 completata; esecuzione `TableFormerOnnxComponents.RunTagTransformerDecoderStep` funzionante.
- **Durata stimata**: 3 h.

### 3.1 Adeguare input decoder
- Correggere la shape del tensore `decoded_tags` per rispettare `[seq_len, batch_size]`.
- Rimuovere l’uso dell’`encoder_mask` se il modello esportato non lo accetta; in alternativa creare il mask con dimensioni corrette.
- Validare con `onnxruntime` (script Python) che la chiamata restituisca output coerente.

### 3.2 Gestione word map
- Caricare il `wordmap` dal JSON e creare mapping ID→token / token→ID.
- Utilizzare il mapping nell’autoregressivo (stop token, tipi cella, header).

### 3.3 Loop autoregressivo robusto
- Implementare logica:
  - inizializzazione con `<start>`,
  - decoding greedy/beam (per ora greedy),
  - early-stop su `<end>` o step massimi,
  - raccolta `tag_hidden` solo per token cella.
- Gestire fallback se non vengono generati token cella (ritornare fallback region list).

### 3.4 Test unitari autoregressivo
- Mockare componenti ONNX con `ITableFormerOnnxComponents` fittizio per simulare logits.
- Aggiungere test per: stop `<end>`, correzione `xcel→lcel`, limit max step.

---

## Fase 4 · Bounding Box & Filtraggio Classi

- **Obiettivo**: usare `bbox_classes` per filtrare le celle e convertire correttamente le coordinate.
- **Prerequisiti**: Fase 3 completata e tag hidden disponibili.
- **Durata stimata**: 2 h.

### 4.1 Interpretazione classi
- Mappare gli indici classe (es. background, cella, header) dal config/wordmap.
- Applicare softmax/logit threshold (configurabile) per accettare bbox.

### 4.2 Conversione coordinate
- Convertire `[cx, cy, w, h]` in coordinate assolute rispetto al bounding box sorgente (`TableStructureRequest.BoundingBox`).
- Clamping e normalizzazione finale `TableRegion`.
- Testare con sample statico (mock `bbox_coords`).

### 4.3 Gestione spanning
- Implementare merge per token `lcel`/`xcel` orizzontali e `ucel` verticali.
- Aggiornare `OtslParser` per segnare celle nascoste non esportate.
- Test: assicurarsi che `TableRegion` count match rispetto a `RowSpan`/`ColSpan`.

---

## Fase 5 · Integrazione Backend & SDK

- **Obiettivo**: collegare il backend completo all’SDK e al servizio Docling.
- **Prerequisiti**: Fase 4 completata, output coerente.
- **Durata stimata**: 2 h.

### 5.1 Refactoring factory/backend
- Aggiornare `DefaultBackendFactory` per passare l’intero `TableFormerVariantModelPaths` al costruttore del backend.
- Consentire la selezione Accurate vs Fast (passare `TableFormerVariantModelPaths` giusto).

### 5.2 Gestione runtime/config
- Rimuovere costruttore transitorio `TableFormerOnnxBackend(string path)`.
- Aggiornare `TableFormerTableStructureService` per storicizzare fallback e incomplete models.
- Configurare environment variables (documentare in `TABLEFORMER_USER_GUIDE.md`).

### 5.3 Telemetria & logging
- Loggare path caricati, tempi sessioni, numero token generati, bounding box finali.
- Aggiornare `TableFormerMetrics` per registrare row/column counts, fallback usage.

---

## Fase 6 · Validazione e Test Suite

- **Obiettivo**: dimostrare che la pipeline produce output corretti e copre i principali scenari.
- **Prerequisiti**: Fase 5 completata.
- **Durata stimata**: 3 h.

### 6.1 Test unitari addizionali
- `TableFormerOnnxBackendTests`: aggiungere test per immagini sintetiche con overlay disabilitato, check di invarianti su `TableStructure`.
- `OtslParserTests`: introdurre casi per spanning multipli, header multipli, newline precoce.

### 6.2 Test integrazione golden
- Eseguire `dotnet test` assicurandosi che Coverlet ≥90%.
- Aggiungere test end-to-end su immagine `dataset/2305.03393v1-pg9-img.png` confrontando output con golden Markdown (tolleranza percentuale).

### 6.3 Benchmark baseline
- Usare `TableFormerBenchmark` per misurare latenza su fast vs accurate (CPU).
- Registrare risultati in `docs/TABLEFORMER_PERFORMANCE.md` (nuove sezioni).

---

## Fase 7 · CLI & Workflow Goldens

- **Obiettivo**: aggiornare i goldens CLI e gli artefatti di confronto Markdown.
- **Prerequisiti**: Fase 6 completata con output validato.
- **Durata stimata**: 2 h.

### 7.1 Generazione snapshot CLI
- Eseguire il comando `dotnet run --project src/Docling.Tooling -- convert ...` con nuovo backend.
- Salvare output in cartella timestampata `dataset/golden/.../dotnet-cli/<run_id>/`.

### 7.2 Diff Markdown
- Eseguire `python3 eng/tools/compare_markdown.py` per generare `report.md` e `summary.json`.
- Verificare differenze attese (nessun grosso delta vs Python se tutto corretto).

### 7.3 Commit artefatti
- Aggiornare `regression_parity_golden_catalog.md`.
- Commit atomic con messaggio `docs: update tableformer dotnet goldens`.

---

## Fase 8 · Documentazione & Chiusura

- **Obiettivo**: finalizzare la documentazione e completare la migrazione.
- **Prerequisiti**: tutte le fasi operative completate.
- **Durata stimata**: 1.5 h.

### 8.1 Aggiornamento documenti tecnici
- `docs/TABLEFORMER_ARCHITECTURE.md`: descrivere pipeline definitiva, includere diagramma aggiornato.
- `docs/TABLEFORMER_ONNX_CONVERSION.md`: aggiungere sezione “Post-export integration”.
- `docs/TABLEFORMER_USER_GUIDE.md`: istruzioni complete per installazione modelli, varianti, env vars.

### 8.2 Changelog e stato progetto
- Aggiornare `TABLEFORMER_MIGRATION_STATUS.md` indicando fase 3-8 completate.
- Preparare estratto per CHANGELOG/Release Notes (.NET).

### 8.3 Pulizia finale
- Eseguire `dotnet test` e assicurarsi git tree pulito.
- Taggare eventuali release interne o creare PR con checklist completata.

---

## Dipendenze Trasversali & Checkpoint

- **Checkpoint A (dopo Fase 3)**: loop autoregressivo produce almeno N celle sensible su immagine di test → abilita Fase 4.
- **Checkpoint B (dopo Fase 5)**: servizio `TableFormerTableStructureService` restituisce output non vuoti con backend ONNX → abilita Fase 6.
- **Checkpoint C (dopo Fase 7)**: goldens CLI aggiornati e confronti Markdown salvati → abilita Fase 8.

---

## Rischi & Mitigazioni Chiave

- **Rischio**: mismatch dimensioni tra decoder step e maschera → mitigare testando con script Python prima dell’integrazione C#.
- **Rischio**: regressione performance CPU → profilare durante Fase 6 e attivare ottimizzazioni ORT (session options già predisposti).
- **Rischio**: incoerenza word map tra fast/accurate → caricare mapping per variante e validare con test.

---

## Log dei Deliverable Principali

| Fase | Deliverable | Verifica |
| ---- | ----------- | -------- |
| 1 | `tableformer_{variant}_{config,wordmap}.json` versionati | `ls models/tableformer_*` |
| 2 | `PreprocessImage` aggiornato + test | `dotnet test` pass, snapshot tensor |
| 3 | `TableFormerAutoregressive` completo + test | `TableFormerAutoregressiveTests` |
| 4 | Conversione bbox + filtri classi | Test coordinate e merge spans |
| 5 | Backend integrato nel servizio | `TableFormerTableStructureService` usa backend reale |
| 6 | Suite test aggiornata, coverage ≥90% | Output `dotnet test` |
| 7 | Goldens CLI + diff Markdown | Commit artefatti nuovi |
| 8 | Documentazione aggiornata + status finale | PR summary / release note |

---

## Note Finali

- Tutte le modifiche devono mantenere la copertura Coverlet ≥90% (scenario build: `dotnet test` root).
- Evitare refusi: commenti e messaggi log in inglese, README/doc in italiano dove già localizzati.
- Coordinare con la squadra AI per eventuale validazione qualitativa su dataset aggiuntivi prima del merge in main branch.
