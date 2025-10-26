# DoclingNet

Port .NET di [Docling](https://github.com/DS4SD/docling) - una soluzione per la conversione di documenti (immagini, PDF) in Markdown strutturato usando AI/ML.

## Caratteristiche

- 🔍 **Rilevamento Layout** - Analisi automatica del layout usando il modello Heron (ONNX)
- 📝 **OCR** - Estrazione testo con EasyOCR (CRAFT detection + CRNN recognition)
- 📊 **Riconoscimento Tabelle** - Analisi struttura tabelle con TableFormer (TorchSharp)
- 📄 **Export Markdown** - Conversione in formato Markdown strutturato
- 🚀 **API Unificata** - SDK semplice con un solo punto di ingresso

## Struttura Progetto

```
DoclingNet/
├── src/
│   ├── DoclingNetSdk/          # 🎯 SDK principale - punto di ingresso unificato
│   ├── Docling.Core/            # Modelli di documento (DoclingDocument, DocItem, ecc.)
│   ├── Docling.Export/          # Serializzazione Markdown
│   ├── Docling.Backends/        # Backend per immagini e PDF
│   └── submodules/              # Librerie AI/ML esterne
│       ├── ds4sd-docling-layout-heron-onnx/
│       ├── easyocrnet/
│       └── ds4sd-docling-tableformer-onnx/
└── examples/
    └── BasicUsage/              # Esempio di utilizzo dell'SDK
```

## Quick Start

### Prerequisiti

- .NET 9.0 SDK
- Sistema operativo: Windows, macOS, o Linux

### Installazione

```bash
git clone https://github.com/yourusername/doclingnet.git
cd doclingnet
git submodule update --init --recursive
dotnet build
```

### Utilizzo Base

```csharp
using DoclingNetSdk;

// 1. Crea la configurazione (auto-rileva i modelli)
var config = DoclingConfiguration.CreateDefault();

// 2. Inizializza il converter
using var converter = new DoclingConverter(config);

// 3. Converti un'immagine in markdown
var result = await converter.ConvertImageAsync("document.png");

// 4. Usa il risultato
Console.WriteLine(result.Markdown);
File.WriteAllText("output.md", result.Markdown);

// Statistiche
Console.WriteLine($"Layout elements: {result.LayoutElementCount}");
Console.WriteLine($"OCR elements: {result.OcrElementCount}");
Console.WriteLine($"Tables: {result.TableCount}");
```

### Esecuzione dell'Esempio

```bash
cd examples/BasicUsage
dotnet run path/to/your/image.png
```

## API DoclingNetSdk

### DoclingConverter

Classe principale per la conversione di documenti.

```csharp
public sealed class DoclingConverter : IDisposable
{
    // Costruttore
    public DoclingConverter(DoclingConfiguration config, ILogger? logger = null);

    // Conversione singola immagine
    public Task<DoclingConversionResult> ConvertImageAsync(
        string imagePath,
        CancellationToken cancellationToken = default);

    // Conversione batch
    public Task<Dictionary<string, DoclingConversionResult>> ConvertImagesAsync(
        IEnumerable<string> imagePaths,
        CancellationToken cancellationToken = default);
}
```

### DoclingConfiguration

Configurazione del converter.

```csharp
public sealed class DoclingConfiguration
{
    // Directory per modelli e cache
    public string ArtifactsPath { get; set; } = "./artifacts";

    // Lingua OCR (default: "en")
    public string OcrLanguage { get; set; } = "en";

    // Abilita/disabilita funzionalità
    public bool EnableTableRecognition { get; set; } = true;
    public bool EnableOcr { get; set; } = true;

    // Variante modello TableFormer (Fast o Accurate)
    public TableFormerVariant TableFormerVariant { get; set; } = TableFormerVariant.Fast;

    // Factory method con configurazione di default
    public static DoclingConfiguration CreateDefault();
}

// Nota: Il percorso del modello layout è rilevato automaticamente
```

### DoclingConversionResult

Risultato della conversione.

```csharp
public sealed class DoclingConversionResult
{
    public DoclingDocument Document { get; }  // Documento strutturato
    public string Markdown { get; }           // Export markdown
    public int LayoutElementCount { get; }    // Numero elementi layout
    public int OcrElementCount { get; }       // Numero elementi OCR
    public int TableCount { get; }            // Numero tabelle
    public int TotalItems { get; }            // Totale elementi documento
}
```

## Pipeline di Conversione

1. **Layout Detection** - Identifica regioni (titoli, paragrafi, tabelle, figure)
2. **OCR Extraction** - Estrae testo da elementi non-tabella
3. **Table Structure Recognition** - Analizza struttura delle tabelle
4. **Document Building** - Costruisce DoclingDocument strutturato
5. **Markdown Export** - Esporta in formato Markdown

## Modelli AI/ML

I modelli vengono scaricati automaticamente al primo utilizzo:

- **Heron Layout Model** - Rilevamento layout (~150MB)
- **EasyOCR Models** - Detection e recognition (~50MB)
- **TableFormer Models** - Analisi tabelle (variante Fast: ~30MB, Accurate: ~120MB)

I modelli sono salvati in `./artifacts/` (configurabile via `ArtifactsPath`).

## Performance

Tempi tipici per una pagina A4 (su CPU):

- Layout Detection: ~500ms
- OCR (10 regioni): ~2-3s
- Table Recognition (2 tabelle): ~1-2s
- **Totale**: ~4-6s per pagina

Su GPU i tempi possono ridursi significativamente.

## Progetti Correlati

- [Docling (Python)](https://github.com/DS4SD/docling) - Progetto originale
- [LayoutSdk](https://github.com/DS4SD/docling-layout-heron-onnx) - Rilevamento layout
- [EasyOcrNet](https://github.com/yourusername/easyocrnet) - OCR per .NET
- [TableFormer](https://github.com/DS4SD/docling-tableformer-onnx) - Riconoscimento struttura tabelle

## Build dalla Solution

```bash
# Build completo
dotnet build DoclingNet.sln

# Solo SDK
dotnet build src/DoclingNetSdk/DoclingNetSdk.csproj

# Run esempio
dotnet run --project examples/BasicUsage/BasicUsage.csproj image.png
```

## Documentazione Aggiuntiva

- [Stato del Porting](docs/progress.md)
- [Piano Implementazione](DOCLING_IMAGE_TO_MARKDOWN_PLAN.md)
- [Diagramma Architettura](ARCHITECTURE_DIAGRAM.md)

## Licenza

MIT License

## Contributi

I contributi sono benvenuti! Per favore apri una issue o una pull request.
