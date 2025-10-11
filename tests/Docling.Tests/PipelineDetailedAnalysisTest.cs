using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Abstractions;
using Docling.Backends.Image;
using Docling.Core.Documents;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Assembly;
using Docling.Pipelines.Export;
using Docling.Pipelines.Layout;
using Docling.Pipelines.Ocr;
using Docling.Pipelines.Options;
using Docling.Pipelines.Preprocessing;
using Docling.Pipelines.Serialization;
using Docling.Pipelines.Tables;
using LayoutSdk;
using Microsoft.Extensions.Logging;
using TableFormerSdk.Enums;
using Xunit;
using Xunit.Abstractions;

namespace Docling.Tests;

/// <summary>
/// Test dettagliato che esegue la pipeline completa sul file dataset
/// mostrando l'output di ogni singolo stage.
/// </summary>
public sealed class PipelineDetailedAnalysisTest
{
    private readonly ITestOutputHelper _output;

    public PipelineDetailedAnalysisTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task AnalyzePipelineStepByStep()
    {
        // Verifica che il file esista
        var inputPath = Path.Combine(
            Directory.GetCurrentDirectory(),
            "..", "..", "..", "..", "..",
            "dataset", "2305.03393v1-pg9-img.png"
        );
        inputPath = Path.GetFullPath(inputPath);

        if (!File.Exists(inputPath))
        {
            _output.WriteLine($"⚠️  File non trovato: {inputPath}");
            _output.WriteLine("Cercando in percorsi alternativi...");

            // Prova percorsi alternativi
            var alternatives = new[]
            {
                Path.Combine(Directory.GetCurrentDirectory(), "dataset", "2305.03393v1-pg9-img.png"),
                Path.Combine(Environment.CurrentDirectory, "dataset", "2305.03393v1-pg9-img.png"),
                "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/2305.03393v1-pg9-img.png"
            };

            foreach (var alt in alternatives)
            {
                if (File.Exists(alt))
                {
                    inputPath = alt;
                    break;
                }
            }

            if (!File.Exists(inputPath))
            {
                _output.WriteLine($"❌ File dataset non trovato in nessun percorso.");
                throw new FileNotFoundException("File dataset non trovato", inputPath);
            }
        }

        _output.WriteLine("═══════════════════════════════════════════════════════════");
        _output.WriteLine("🚀 ANALISI DETTAGLIATA PIPELINE DOCLING");
        _output.WriteLine("═══════════════════════════════════════════════════════════");
        _output.WriteLine($"📄 Input: {Path.GetFileName(inputPath)}");
        _output.WriteLine($"📁 Path completo: {inputPath}");
        _output.WriteLine();

        var cancellationToken = CancellationToken.None;

        // ══════════════════════════════════════════════════════════
        // STEP 1: BACKEND - Caricamento immagine
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 1: BACKEND - Caricamento Immagine                 │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var backendOptions = new ImageBackendOptions
        {
            DocumentId = "test-doc",
            SourceName = "2305.03393v1-pg9",
            DefaultDpi = 300,
            Sources = new[]
            {
                new ImageSourceDescriptor
                {
                    Identifier = "page-0",
                    FileName = Path.GetFileName(inputPath),
                    MediaType = "image/png",
                    Dpi = 300,
                    StreamFactory = _ => Task.FromResult<Stream>(File.OpenRead(inputPath))
                }
            }
        };

        var logger = new TestLogger(_output);
        var backend = new ImageBackend(backendOptions, logger);
        var document = await backend.LoadDocumentAsync(cancellationToken);

        _output.WriteLine($"✅ Documento caricato");
        _output.WriteLine($"   📊 ID: {document.Id}");
        _output.WriteLine($"   📊 Source: {document.SourceId}");
        _output.WriteLine($"   📊 Pagine: {document.Pages.Count}");
        _output.WriteLine($"   📊 Dimensioni pagina 1: {document.Pages[0].Size.Width} x {document.Pages[0].Size.Height}");
        _output.WriteLine();

        var imageStore = await backend.RenderPagesAsync(cancellationToken);
        var pageImage = imageStore.GetPage(0);

        _output.WriteLine($"✅ Immagine renderizzata");
        _output.WriteLine($"   📊 Dimensioni: {pageImage.Width} x {pageImage.Height} px");
        _output.WriteLine($"   📊 DPI: {pageImage.Dpi}");
        _output.WriteLine($"   📊 Formato: {pageImage.PixelFormat}");
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 2: PREPROCESSING - Normalizzazione immagine
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 2: PREPROCESSING - Normalizzazione                │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var preprocessingOptions = new PreprocessingOptions
        {
            TargetDpi = 300,
            EnableDeskew = false,
            NormalizeContrast = false
        };

        var preprocessor = new DefaultPagePreprocessor(preprocessingOptions, logger);
        var preprocessStage = new PagePreprocessingStage(preprocessor, logger);

        await preprocessStage.ExecuteAsync(document, imageStore, cancellationToken);

        var processedImage = imageStore.GetPage(0);
        _output.WriteLine($"✅ Preprocessing completato");
        _output.WriteLine($"   📊 Dimensioni finali: {processedImage.Width} x {processedImage.Height} px");
        _output.WriteLine($"   📊 DPI target: {preprocessingOptions.TargetDpi}");
        _output.WriteLine($"   📊 Deskew: {preprocessingOptions.EnableDeskew}");
        _output.WriteLine($"   📊 Normalize contrast: {preprocessingOptions.NormalizeContrast}");
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 3: LAYOUT ANALYSIS - Rilevamento struttura
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 3: LAYOUT ANALYSIS - Rilevamento Struttura        │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var layoutOptions = new LayoutOptions
        {
            Model = LayoutModelConfiguration.DoclingLayoutEgretMedium,
            CreateOrphanClusters = false,
            KeepEmptyClusters = true,
            GenerateDebugArtifacts = false
        };

        var layoutSdkOptions = new LayoutSdkDetectionOptions
        {
            ValidateModelFiles = true,
            MaxDegreeOfParallelism = 1,
            Runtime = LayoutRuntime.Ort
        };

        var layoutService = new LayoutSdkDetectionService(layoutSdkOptions, logger);
        var layoutStage = new LayoutAnalysisStage(layoutService, layoutOptions, logger, null);

        await layoutStage.ExecuteAsync(document, imageStore, cancellationToken);

        var layoutResult = document.TryGetProperty("__layout_result_0", out var layoutJson)
            ? System.Text.Json.JsonSerializer.Deserialize<dynamic>(layoutJson)
            : null;

        _output.WriteLine($"✅ Layout analysis completato");

        // Conta gli elementi rilevati per tipo
        var layoutItems = document.Items.Where(i => i.Metadata.ContainsKey("layout:cluster")).ToList();
        var byKind = layoutItems.GroupBy(i => i.Kind).ToDictionary(g => g.Key, g => g.Count());

        _output.WriteLine($"   📊 Elementi rilevati: {layoutItems.Count}");
        foreach (var (kind, count) in byKind.OrderByDescending(x => x.Value))
        {
            _output.WriteLine($"      • {kind}: {count}");
        }

        _output.WriteLine();
        _output.WriteLine("   📦 Dettaglio primi 5 elementi di layout:");
        foreach (var item in layoutItems.Take(5))
        {
            var bbox = item.BoundingBox;
            _output.WriteLine($"      [{item.Kind}] Box: ({bbox.Left:F1}, {bbox.Top:F1}) → ({bbox.Right:F1}, {bbox.Bottom:F1}) | Area: {bbox.Area:F0}");
        }
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 4: TABLE STRUCTURE - Analisi tabelle
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 4: TABLE STRUCTURE - Analisi Tabelle              │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var tableOptions = new TableStructureOptions
        {
            Mode = TableFormerMode.Accurate
        };

        var tableServiceOptions = new TableFormerStructureServiceOptions
        {
            Variant = TableFormerModelVariant.Accurate,
            GenerateOverlay = false,
            WorkingDirectory = Path.Combine(Path.GetTempPath(), $"docling-test-{Guid.NewGuid():N}")
        };

        var tableService = new TableFormerTableStructureService(tableServiceOptions, logger);
        var pipelineOptions = new PdfPipelineOptions { TableStructure = tableOptions };
        var tableStage = new TableStructureInferenceStage(tableService, pipelineOptions, logger);

        try
        {
            await tableStage.ExecuteAsync(document, imageStore, cancellationToken);
            _output.WriteLine($"✅ Table structure analysis completato");

            var tables = document.GetItemsOfKind(DocItemKind.Table);
            _output.WriteLine($"   📊 Tabelle rilevate: {tables.Count}");

            foreach (var table in tables.Cast<TableItem>())
            {
                _output.WriteLine($"      • Tabella: {table.RowCount} righe x {table.ColumnCount} colonne, {table.Cells.Count} celle");
            }
        }
        catch (Exception ex)
        {
            _output.WriteLine($"⚠️  Table analysis non completato: {ex.Message}");
        }
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 5: OCR - Riconoscimento testo
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 5: OCR - Riconoscimento Testo                     │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var ocrOptions = new EasyOcrOptions
        {
            Languages = new[] { "en" },
            ForceFullPageOcr = false,
            BitmapAreaThreshold = 0.0005,
            ModelStorageDirectory = null // Usa default
        };

        var ocrServiceFactory = new OcrServiceFactory();
        var ocrPipelineOptions = new PdfPipelineOptions
        {
            Ocr = ocrOptions
        };
        var ocrStage = new OcrStage(ocrServiceFactory, ocrPipelineOptions, logger);

        try
        {
            await ocrStage.ExecuteAsync(document, imageStore, cancellationToken);
            _output.WriteLine($"✅ OCR completato");

            var paragraphs = document.GetItemsOfKind(DocItemKind.Paragraph).Cast<ParagraphItem>().ToList();
            _output.WriteLine($"   📊 Paragrafi con testo: {paragraphs.Count}");
            _output.WriteLine($"   📊 Parole totali: {paragraphs.Sum(p => p.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length)}");

            _output.WriteLine();
            _output.WriteLine("   📝 Primi 3 paragrafi estratti:");
            foreach (var para in paragraphs.Take(3))
            {
                var preview = para.Text.Length > 80 ? para.Text.Substring(0, 80) + "..." : para.Text;
                _output.WriteLine($"      • {preview}");
            }
        }
        catch (Exception ex)
        {
            _output.WriteLine($"⚠️  OCR non completato: {ex.Message}");
            _output.WriteLine($"      Dettaglio: {ex.StackTrace}");
        }
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 6: ASSEMBLY - Composizione documento
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 6: ASSEMBLY - Composizione Documento              │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var assemblyStage = new PageAssemblyStage();
        await assemblyStage.ExecuteAsync(document, imageStore, cancellationToken);

        _output.WriteLine($"✅ Document assembly completato");
        _output.WriteLine($"   📊 Items totali nel documento: {document.Items.Count}");
        _output.WriteLine();
        _output.WriteLine("   📦 Distribuzione per tipo:");
        var itemsByKind = document.Items.GroupBy(i => i.Kind).ToDictionary(g => g.Key, g => g.Count());
        foreach (var (kind, count) in itemsByKind.OrderByDescending(x => x.Value))
        {
            _output.WriteLine($"      • {kind}: {count}");
        }
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 7: IMAGE EXPORT - Estrazione immagini
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 7: IMAGE EXPORT - Estrazione Immagini             │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var imageCropService = new ImageCropService();
        var imageExportOptions = new PdfPipelineOptions
        {
            GeneratePageImages = true,
            GeneratePictureImages = true,
            GenerateImageDebugArtifacts = false
        };
        var imageExportStage = new ImageExportStage(imageCropService, imageExportOptions, logger);

        await imageExportStage.ExecuteAsync(document, imageStore, cancellationToken);

        var figures = document.GetItemsOfKind(DocItemKind.Picture).Cast<PictureItem>().ToList();
        var figuresWithImage = figures.Where(f => f.ImageRef != null).ToList();

        _output.WriteLine($"✅ Image export completato");
        _output.WriteLine($"   📊 Figure totali: {figures.Count}");
        _output.WriteLine($"   📊 Figure con immagine esportata: {figuresWithImage.Count}");
        _output.WriteLine();

        // ══════════════════════════════════════════════════════════
        // STEP 8: MARKDOWN SERIALIZATION - Output finale
        // ══════════════════════════════════════════════════════════
        _output.WriteLine("┌─────────────────────────────────────────────────────────┐");
        _output.WriteLine("│ STEP 8: MARKDOWN SERIALIZATION - Output Finale         │");
        _output.WriteLine("└─────────────────────────────────────────────────────────┘");

        var serializerOptions = new MarkdownSerializerOptions
        {
            AssetsPath = "assets",
            ImageMode = ImageExportMode.Placeholder
        };

        var serializer = new MarkdownDocSerializer(serializerOptions);
        var serializationStage = new MarkdownSerializationStage(serializer);

        await serializationStage.ExecuteAsync(document, imageStore, cancellationToken);

        if (document.TryGetProperty("markdown:content", out var markdownContent))
        {
            _output.WriteLine($"✅ Markdown generato");
            _output.WriteLine($"   📊 Lunghezza: {markdownContent.Length} caratteri");
            _output.WriteLine($"   📊 Righe: {markdownContent.Split('\n').Length}");
            _output.WriteLine();
            _output.WriteLine("───────────────────────────────────────────────────────────");
            _output.WriteLine("📄 MARKDOWN OUTPUT:");
            _output.WriteLine("───────────────────────────────────────────────────────────");
            _output.WriteLine(markdownContent);
            _output.WriteLine("───────────────────────────────────────────────────────────");
        }
        else
        {
            _output.WriteLine($"⚠️  Markdown non disponibile");
        }

        _output.WriteLine();
        _output.WriteLine("═══════════════════════════════════════════════════════════");
        _output.WriteLine("✅ ANALISI PIPELINE COMPLETATA");
        _output.WriteLine("═══════════════════════════════════════════════════════════");
    }

    private class TestLogger : ILogger<ImageBackend>, ILogger<DefaultPagePreprocessor>,
        ILogger<LayoutSdkDetectionService>, ILogger<TableFormerTableStructureService>,
        ILogger<LayoutAnalysisStage>, ILogger<TableStructureInferenceStage>,
        ILogger<OcrStage>, ILogger<ImageExportStage>
    {
        private readonly ITestOutputHelper _output;

        public TestLogger(ITestOutputHelper output)
        {
            _output = output;
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            var message = formatter(state, exception);
            var prefix = logLevel switch
            {
                LogLevel.Error => "❌",
                LogLevel.Warning => "⚠️ ",
                LogLevel.Information => "ℹ️ ",
                LogLevel.Debug => "🔍",
                _ => "  "
            };
            _output.WriteLine($"{prefix} {message}");
            if (exception != null)
            {
                _output.WriteLine($"   Exception: {exception.Message}");
            }
        }
    }
}
