# Piano di Verifica per Identificare la Divergenza Finale

## Problema
- C# genera sequenze ripetitive (lcel fcel fcel fcel...) senza mai generare `<end>`
- Python genera sequenze corrette (ecel ecel ecel... nl ... <end>)
- Logit di `<end>` sempre negativo in C# (-0.8 to -2.5)

## Ipotesi Possibili
1. ❓ Embedding layer: pesi non caricati o dimensioni sbagliate
2. ❓ Positional encoding: bug nell'implementazione o pesi
3. ❓ Transformer encoder: qualche layer ha pesi sbagliati
4. ❓ Transformer decoder: causal mask non corretto
5. ❓ Final FC layer (_fc): pesi per `<end>` token sbagliati
6. ❓ Input format: l'input al transformer è in formato sbagliato

## Verifiche da Eseguire (in ordine)

### 1. Verifica Embedding Layer ✅ PRIORITÀ ALTA
**Test**: Salvare output dell'embedding per lo stesso input token
**Comando**:
```
# Python: embedding(<start>) -> vector
# C#: embedding(<start>) -> vector
# Confrontare i primi 20 valori
```
**Atteso**: Vettori identici (diff < 1e-5)

### 2. Verifica Positional Encoding ✅ PRIORITÀ ALTA  
**Test**: Salvare PE per posizione 0 e posizione 50
**Comando**:
```
# Confrontare PE[0] e PE[50] tra Python e C#
```
**Atteso**: Valori identici

### 3. Verifica Final FC Layer Weights ✅ PRIORITÀ ALTA
**Test**: Confrontare pesi del layer _fc, specialmente per `<end>` token
**Comando**:
```
# Verificare _fc.weight[<end>_idx, :] tra Python e C#
# Verificare _fc.bias[<end>_idx]
```
**Atteso**: Pesi identici

### 4. Verifica Transformer Output (Step 0) ✅ PRIORITÀ MEDIA
**Test**: Salvare output del transformer decoder al primo step
**Comando**:
```
# Dopo TagTransformer.forward() con input=<start>
# Salvare decoded tensor e predictions tensor
```
**Atteso**: Output simili (può avere piccole differenze accumulate)

### 5. Verifica Causal Mask ✅ PRIORITÀ MEDIA
**Test**: Verificare che il decoder usi il causal mask corretto
**Comando**:
```
# Stampare la mask generata al primo step
```
**Atteso**: Triangular mask corretta

### 6. Verifica Accumulo Errori ✅ PRIORITÀ BASSA
**Test**: Teacher forcing - fornire la sequenza corretta e vedere l'output
**Comando**:
```
# Invece di autoregressive, dare sequenza Python corretta
# Verificare se genera <end> alla fine
```

## Ordine di Esecuzione
1. Embedding (più probabile - se sbagliato causa divergenza immediata)
2. Final FC (secondo più probabile - diretto impatto su logits <end>)
3. Positional Encoding (può causare pattern ripetitivi)
4. Transformer output
5. Causal mask
6. Teacher forcing test

## Output Atteso
- Identificare QUALE componente causa la divergenza
- Quantificare la differenza (se weights/output diversi)
- Proporre fix specifico

