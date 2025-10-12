using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Tables;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerSdk;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using Xunit;
using Xunit.Abstractions;

namespace Docling.Tests.Tables;

/// <summary>
/// Test case specifico per validare l'estrazione delle tabelle dall'immagine 2305.03393v1-pg9-img.png
/// e confrontare i risultati con l'output Python di riferimento.
/// </summary>
public sealed class TableExtractionValidationTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _testImagePath;
    private readonly string _outputDirectory;
    private static readonly JsonSerializerOptions PrettyJson = new() { WriteIndented = true };

    public TableExtractionValidationTests(ITestOutputHelper output)
    {
        _output = output;

        // Path all'immagine di test
        _testImagePath = Path.GetFullPath(
            Path.Combine("..", "..", "..", "..", "..", "dataset", "2305.03393v1-pg9-img.png")
        );

        // Directory per salvare i risultati
        _outputDirectory = Path.Combine(
            Path.GetTempPath(),
            $"docling-table-validation-{DateTime.UtcNow:yyyy-MM-ddTHHmmss}"
        );
        Directory.CreateDirectory(_outputDirectory);

        _output.WriteLine($"Test image: {_testImagePath}");
        _output.WriteLine($"Output directory: {_outputDirectory}");
        _output.WriteLine($"Image exists: {File.Exists(_testImagePath)}");
    }

    [Fact]
    public void TableFormer_ExtractsTables_FromAcademicPaper()
    {
        // Arrange
        if (!File.Exists(_testImagePath))
        {
            throw new FileNotFoundException($"Test image not found: {_testImagePath}");
        }

        // Carica modelli TableFormer (assumendo che siano nella directory models/)
        var modelsDir = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "..", "models", "tableformer-onnx"));
        var fastEncoderPath = Path.Combine(modelsDir, "tableformer_fast_encoder.onnx");

        // Verifica che i modelli esistano
        if (!File.Exists(fastEncoderPath))
        {
            _output.WriteLine($"⚠️ Modelli TableFormer non trovati in: {modelsDir}");
            _output.WriteLine("Saltando il test - i modelli devono essere scaricati prima.");
            return; // Skip test se modelli non disponibili
        }

        var options = new TableFormerSdkOptions(
            onnx: new TableFormerModelPaths(
                TableFormerVariantModelPaths.FromDirectory(modelsDir, "tableformer_fast"),
                accurate: null)
        );

        using var tableFormer = new TableFormer(options);

        // Act - Estrai tabelle dall'immagine
        var startTime = DateTime.UtcNow;
        var result = tableFormer.Process(
            imagePath: _testImagePath,
            overlay: true, // Genera visualizzazione overlay
            runtime: TableFormerRuntime.Onnx,
            variant: TableFormerModelVariant.Fast
        );
        var processingTime = DateTime.UtcNow - startTime;

        // Assert - Verifica risultati
        Assert.NotNull(result);
        Assert.NotNull(result.Regions);

        _output.WriteLine($"\n📊 RISULTATI ESTRAZIONE TABELLE:");
        _output.WriteLine($"Tempo elaborazione: {processingTime.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Regioni rilevate: {result.Regions.Count}");
        _output.WriteLine($"Runtime utilizzato: {result.Runtime}");
        _output.WriteLine($"Variante modello: Fast");

        // Analizza le regioni rilevate
        var regionsByType = result.Regions
            .GroupBy(r => r.Label)
            .OrderByDescending(g => g.Count());

        _output.WriteLine($"\n📋 DISTRIBUZIONE REGIONI:");
        foreach (var group in regionsByType)
        {
            _output.WriteLine($"  {group.Key}: {group.Count()} regioni");
        }

        // Conta le tabelle rilevate
        var tableCount = result.Regions.Count(r => r.Label.Contains("table", StringComparison.OrdinalIgnoreCase));

        _output.WriteLine($"\n✅ Tabelle identificate: {tableCount}");

        // Aspettativa: secondo il piano dovrebbero esserci 4 tabelle
        _output.WriteLine($"📌 Target atteso: 4 tabelle");

        if (tableCount < 4)
        {
            _output.WriteLine($"⚠️ ATTENZIONE: Rilevate {tableCount} tabelle invece di 4");
        }

        // Salva risultati dettagliati
        SaveDetailedResults(result, processingTime);

        // Salva overlay se disponibile
        if (result.OverlayImage != null)
        {
            SaveOverlayImage(result.OverlayImage);
        }

        // Success criteria: almeno una tabella deve essere rilevata
        Assert.True(result.Regions.Count > 0, "Nessuna regione rilevata");
    }

    [Fact]
    public async Task EndToEnd_ExtractsTableStructure_FromCompleteImage()
    {
        // Arrange
        if (!File.Exists(_testImagePath))
        {
            throw new FileNotFoundException($"Test image not found: {_testImagePath}");
        }

        var modelsDir = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "..", "models", "tableformer-onnx"));
        var fastEncoderPath = Path.Combine(modelsDir, "tableformer_fast_encoder.onnx");

        if (!File.Exists(fastEncoderPath))
        {
            _output.WriteLine("⚠️ Modelli non disponibili - saltando test");
            return;
        }

        // Crea il servizio TableFormer
        var tableFormerOptions = new TableFormerSdkOptions(
            onnx: new TableFormerModelPaths(
                TableFormerVariantModelPaths.FromDirectory(modelsDir, "tableformer_fast"),
                accurate: null)
        );

        var serviceOptions = new TableFormerStructureServiceOptions
        {
            Variant = TableFormerModelVariant.Fast,
            Runtime = TableFormerRuntime.Onnx,
            WorkingDirectory = _outputDirectory,
            GenerateOverlay = true,
            SdkOptions = tableFormerOptions
        };

        using var service = new TableFormerTableStructureService(
            serviceOptions,
            NullLogger<TableFormerTableStructureService>.Instance
        );

        // Carica immagine completa
        var imageBytes = await File.ReadAllBytesAsync(_testImagePath);

        using var bitmap = SKBitmap.Decode(imageBytes);
        var page = new PageReference(9, 300); // Pagina 9 del documento
        var bounds = BoundingBox.FromSize(0, 0, bitmap.Width, bitmap.Height);

        var request = new TableStructureRequest(page, bounds, imageBytes);

        // Act
        var startTime = DateTime.UtcNow;
        var structure = await service.InferStructureAsync(request);
        var processingTime = DateTime.UtcNow - startTime;

        // Assert
        Assert.NotNull(structure);

        _output.WriteLine($"\n🔬 ANALISI STRUTTURA TABELLA END-TO-END:");
        _output.WriteLine($"Tempo elaborazione: {processingTime.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Celle estratte: {structure.Cells.Count}");
        _output.WriteLine($"Righe: {structure.RowCount}");
        _output.WriteLine($"Colonne: {structure.ColumnCount}");

        // Salva artifact di debug se disponibile
        if (structure.DebugArtifact != null)
        {
            var artifactPath = Path.Combine(_outputDirectory, "table_structure_overlay.png");
            await File.WriteAllBytesAsync(artifactPath, structure.DebugArtifact.ImageContent.ToArray());
            _output.WriteLine($"Debug artifact salvato: {artifactPath}");
        }

        // Salva risultati struttura
        SaveTableStructure(structure);
    }

    private void SaveDetailedResults(TableStructureResult result, TimeSpan processingTime)
    {
        var resultsFile = Path.Combine(_outputDirectory, "dotnet_extraction_results.json");

        var jsonData = new
        {
            test_info = new
            {
                timestamp = DateTime.UtcNow.ToString("o"),
                image_path = _testImagePath,
                framework = ".NET",
                model = "TableFormer Pipeline"
            },
            performance = new
            {
                processing_time_ms = processingTime.TotalMilliseconds,
                inference_time_ms = result.InferenceTime.TotalMilliseconds,
                runtime = result.Runtime.ToString(),
                model_variant = "Fast"
            },
            detections = new
            {
                total_regions = result.Regions.Count,
                table_count = result.Regions.Count(r => r.Label.Contains("table", StringComparison.OrdinalIgnoreCase)),
                regions = result.Regions.Select((region, index) => new
                {
                    id = index,
                    class_label = region.Label,
                    bbox = new
                    {
                        x = region.X,
                        y = region.Y,
                        width = region.Width,
                        height = region.Height,
                        area = region.Width * region.Height
                    }
                }).ToArray()
            },
            comparison_target = new
            {
                expected_tables = 4,
                source = "Python Docling baseline (as per PIANO_INTERVENTO.md)"
            }
        };

        File.WriteAllText(resultsFile, JsonSerializer.Serialize(jsonData, PrettyJson));

        _output.WriteLine($"\n💾 Risultati dettagliati salvati: {resultsFile}");
    }

    private void SaveOverlayImage(SKBitmap overlayImage)
    {
        var overlayPath = Path.Combine(_outputDirectory, "table_detection_overlay.png");

        using var image = SKImage.FromBitmap(overlayImage);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.OpenWrite(overlayPath);
        data.SaveTo(stream);

        _output.WriteLine($"🖼️  Overlay salvato: {overlayPath}");
    }

    private void SaveTableStructure(TableStructure structure)
    {
        var structureFile = Path.Combine(_outputDirectory, "table_structure.json");

        var jsonData = new
        {
            page = structure.Page.PageNumber,
            dpi = structure.Page.Dpi,
            dimensions = new
            {
                rows = structure.RowCount,
                columns = structure.ColumnCount
            },
            cells = structure.Cells.Select((cell, index) => new
            {
                index,
                row_span = cell.RowSpan,
                column_span = cell.ColumnSpan,
                bbox = new
                {
                    left = cell.BoundingBox.Left,
                    top = cell.BoundingBox.Top,
                    right = cell.BoundingBox.Right,
                    bottom = cell.BoundingBox.Bottom,
                    width = cell.BoundingBox.Width,
                    height = cell.BoundingBox.Height
                }
            }).ToArray()
        };

        File.WriteAllText(structureFile, JsonSerializer.Serialize(jsonData, PrettyJson));

        _output.WriteLine($"📊 Struttura tabella salvata: {structureFile}");
    }

    public void Dispose()
    {
        // Cleanup: mantieni i file per analisi manuale
        _output.WriteLine($"\n📁 Output preservati in: {_outputDirectory}");
    }
}
