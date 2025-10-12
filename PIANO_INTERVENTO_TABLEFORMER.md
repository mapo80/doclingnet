# Piano di intervento – TableFormer

## Contesto
- Il layout engine .NET individua correttamente la tabella sulla pagina (`layout_analysis` produce 1 item `Table`).
- Lo stadio `table_structure` restituisce però `rowCount=0`, `columnCount=0`, `cellCount=0`, quindi il Markdown finale non contiene la tabella.
- Il golden Python (`dataset/golden/v0.12.0/2305.03393v1-pg9/python-cli/docling.md`) mostra che TableFormer lato Python produce celle coerenti; l’output .NET dev’essere riallineato.

## Divergenze osservate / ipotesi da verificare
1. **Pre-processing dell’immagine**
   - Backend .NET (ONNX e pipeline) usa il crop RAW della tabella ridimensionato a 448×448 con semplice normalizzazione 0–1.
   - Pipeline Python passa attraverso `AutoImageProcessor` (HuggingFace) con letterboxing e normalizzazione (mean/std dedicate). Serve confermare e replicare la stessa trasformazione.
2. **Soglie di confidenza**
   - `TableFormerDetectionParser` filtra con `ScoreThreshold = 0.5`. La pipeline Python potrebbe usare soglie più permissive (es. 0.25–0.35) o softmax anziché sigmoid.
3. **Modelli / percorsi**
   - Il servizio .NET punta a file ONNX hard-coded nel submodule. Dobbiamo verificare: a) che corrispondano ai pesi usati in Python, b) che siano caricati dal pacchetto NuGet `TableFormerSdk`.
4. **Post-processing delle box**
   - Verificare che la conversione da `TableRegion` a `TableCell` mantenga la stessa semantica (coordinate relative/assolute, clamping, gestione di rowspan/colspan). Ad oggi generiamo celle atomiche senza clustering.

## Piano operativo
1. **Raccogliere baseline Python**
   - Estrarre dal run Python i JSON grezzi con bounding box e confidence (se già disponibili) oppure rieseguire la pipeline Python su `2305.03393v1-pg9` catturando l’output intermedio TableFormer.
   - Documentare: dimensioni input al modello, distribuzione delle confidence, numero celle per tabella.
2. **Strumentare la pipeline .NET**
   - Abilitare log temporanei in `TableStructureInferenceStage` e `TableFormerTableStructureService` per salvare: dimensioni crop, prime 10 confidence, conteggio regioni prima del filtro.
   - Aggiungere flag (temporaneo) per esportare l’immagine pre-processata inviata al modello così da confrontarla con quella generata da Python.
3. **Allineare pre-processing**
   - Implementare lo stesso `image processor` della pipeline Python: letterboxing/resize ai valori attesi e normalizzazione con mean/std ufficiali (per entrambe le varianti Onnx/Pipeline).
   - Validare con test unitari che, dato un crop di prova, il tensore generato da .NET combaci (entro tolleranza) con quello prodotto da Python.
4. **Ricalibrare il filtro sulle detection**
   - Una volta replicato il pre-processing, confrontare le confidence: se restano più basse rispetto a Python, introdurre soglie configurabili (per variante modello) allineate alla reference.
   - Valutare se serve applicare softmax/argmax come in python prima della soglia.
5. **Verificare la conversione delle regioni**
   - Assicurarsi che `ConvertRegions` restituisca celle con coordinate corrette sul sistema originale. Se Python restituisce anche struttura (row/column span), considerare l’adozione di heuristics equivalenti.
6. **Testing e regressioni**
   - Introdurre test automatici per TableFormer (mockando input immagine nota) con snapshot delle regioni attese.
   - Rieseguire `dotnet test` e rigenerare i golden .NET + confronti Markdown.
   - Aggiornare documentazione e, se necessario, script di setup (es. percorsi modelli dal pacchetto NuGet).

## Deliverable finali
- Fix del servizio TableFormer .NET con comportamento allineato al riferimento Python.
- Nuovi test / snapshot che coprano la regressione.
- Aggiornamento markdown golden e report di confronto con differenze minime attese.
