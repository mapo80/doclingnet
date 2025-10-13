using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TableFormerSdk;
using TableFormerSdk.Backends;
using TableFormerSdk.Configuration;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;
using TableFormerClient = TableFormerSdk.TableFormer;
using Xunit;

namespace TableFormerSdk.Tests;

file sealed class FakeBackend : ITableFormerBackend
{
    private readonly TimeSpan _delay;

    public FakeBackend(TimeSpan? delay = null)
    {
        _delay = delay ?? TimeSpan.Zero;
    }

    public int InvocationCount { get; private set; }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        InvocationCount++;
        if (_delay > TimeSpan.Zero)
        {
            Thread.Sleep(_delay);
        }

        return new List<TableRegion> { new(1, 1, 2, 2, "cell") };
    }
}

public class TableFormerSdkTests
{
    private static TableFormerClient CreateSdkWithFakeBackend(TableFormerPerformanceOptions? performanceOptions = null)
    {
        var paths = CreateTempVariantPaths();
        var options = new TableFormerSdkOptions(new TableFormerModelPaths(paths), performanceOptions: performanceOptions);

        var sdk = new TableFormerClient(options);
        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, new FakeBackend());
        return sdk;
    }

    private static TableFormerVariantModelPaths CreateTempVariantPaths()
    {
        static string CreateTempFile(string extension, string? contents = null)
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + extension);
            if (contents is null)
            {
                File.WriteAllBytes(path, Array.Empty<byte>());
            }
            else
            {
                File.WriteAllText(path, contents);
            }

            return path;
        }

        var encoder = CreateTempFile(".onnx");
        var tagEncoder = CreateTempFile(".onnx");
        var decoderStep = CreateTempFile(".onnx");
        var bboxDecoder = CreateTempFile(".onnx");
        // Minimal JSON payloads to satisfy validation when parsed during tests
        var config = CreateTempFile(".json", "{}");
        var wordMap = CreateTempFile(".json", "{}");

        return new TableFormerVariantModelPaths(
            encoder,
            tagEncoder,
            decoderStep,
            bboxDecoder,
            config,
            wordMap);
    }

    private static string CreateSampleImage()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".png");
        using var bitmap = new SKBitmap(16, 16);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);
        using var paint = new SKPaint { Color = SKColors.Black, StrokeWidth = 1f };
        canvas.DrawRect(new SKRect(2, 2, 14, 14), paint);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.Open(path, FileMode.Create, FileAccess.Write);
        data.SaveTo(stream);
        return path;
    }

    [Fact]
    public void Process_EmptyPath_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<ArgumentException>(() => sdk.Process("", false, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx));
    }

    [Fact]
    public void Process_MissingImage_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<FileNotFoundException>(() => sdk.Process("missing.png", false, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx));
    }

    [Fact]
    public void Process_Overlay_ReturnsBitmap()
    {
        var sdk = CreateSdkWithFakeBackend();
        var imagePath = CreateSampleImage();
        var result = sdk.Process(imagePath, true, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx);
        Assert.NotNull(result.OverlayImage);
        Assert.Single(result.Regions);
        Assert.Equal(TableFormerRuntime.Onnx, result.Runtime);
        Assert.True(result.InferenceTime >= TimeSpan.Zero);
        Assert.Equal(TableFormerRuntime.Onnx, result.PerformanceSnapshot.Runtime);
    }

    [Fact]
    public void Process_NoOverlay_ReturnsNull()
    {
        var sdk = CreateSdkWithFakeBackend();
        var imagePath = CreateSampleImage();
        var result = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Onnx);
        Assert.Null(result.OverlayImage);
    }

    [Fact]
    public void Process_AutoRuntime_ExploresAndSelectsFastestBackend()
    {
        var performanceOptions = new TableFormerPerformanceOptions(minimumSamples: 1, slidingWindowSize: 8, runtimePriority: new[] { TableFormerRuntime.Onnx });
        var sdk = CreateSdkWithFakeBackend(performanceOptions);

        var onnxBackendSlow = new FakeBackend(TimeSpan.FromMilliseconds(15));
        var onnxBackendFast = new FakeBackend(TimeSpan.FromMilliseconds(2));

        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, onnxBackendSlow);
        sdk.RegisterBackend(TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate, onnxBackendFast);

        var imagePath = CreateSampleImage();

        var first = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        var second = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);
        var third = sdk.Process(imagePath, false, TableFormerModelVariant.Fast, TableFormerRuntime.Auto);

        Assert.Equal(TableFormerRuntime.Onnx, first.Runtime);
        Assert.Equal(TableFormerRuntime.Onnx, second.Runtime);
        Assert.Equal(TableFormerRuntime.Onnx, third.Runtime);

        Assert.True(second.InferenceTime <= first.InferenceTime);
        Assert.True(third.InferenceTime <= second.InferenceTime);

        var snapshots = sdk.GetPerformanceSnapshots(TableFormerModelVariant.Fast);
        Assert.NotEmpty(snapshots);
        Assert.All(snapshots, s => Assert.Equal(TableFormerRuntime.Onnx, s.Runtime));
    }

    [Fact]
    public void FromDirectory_LoadsModelPathsCorrectly()
    {
        var modelsDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "models");

        // Skip if models directory doesn't exist
        if (!Directory.Exists(modelsDir))
        {
            return;
        }

        // Try to load Fast variant
        var fastPaths = TableFormerVariantModelPaths.FromDirectory(modelsDir, "tableformer_fast");

        Assert.NotNull(fastPaths);
        Assert.True(File.Exists(fastPaths.EncoderPath), "Encoder ONNX file should exist");
        Assert.True(File.Exists(fastPaths.TagEncoderPath), "Tag encoder ONNX file should exist");
        Assert.True(File.Exists(fastPaths.DecoderStepPath), "Decoder step ONNX file should exist");
        Assert.True(File.Exists(fastPaths.BboxDecoderPath), "BBox decoder ONNX file should exist");
        Assert.True(File.Exists(fastPaths.ConfigPath), "Config JSON file should exist");
        Assert.True(File.Exists(fastPaths.WordMapPath), "WordMap JSON file should exist");

        // Try to load Accurate variant
        var accuratePaths = TableFormerVariantModelPaths.FromDirectory(modelsDir, "tableformer_accurate");

        Assert.NotNull(accuratePaths);
        Assert.True(File.Exists(accuratePaths.EncoderPath), "Accurate encoder ONNX file should exist");
        Assert.True(File.Exists(accuratePaths.TagEncoderPath), "Accurate tag encoder ONNX file should exist");
        Assert.True(File.Exists(accuratePaths.DecoderStepPath), "Accurate decoder step ONNX file should exist");
        Assert.True(File.Exists(accuratePaths.BboxDecoderPath), "Accurate BBox decoder ONNX file should exist");
        Assert.True(File.Exists(accuratePaths.ConfigPath), "Accurate config JSON file should exist");
        Assert.True(File.Exists(accuratePaths.WordMapPath), "Accurate wordMap JSON file should exist");
    }

    [Fact]
    public void TableFormerConfig_LoadsNormalizationParametersCorrectly()
    {
        var modelsDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "models");
        var configPath = Path.Combine(modelsDir, "tableformer_fast_config.json");

        // Skip if config doesn't exist
        if (!File.Exists(configPath))
        {
            return;
        }

        var config = TableFormerConfig.LoadFromFile(configPath);

        Assert.NotNull(config);
        Assert.NotNull(config.Dataset);
        Assert.NotNull(config.Dataset.ImageNormalization);

        var norm = config.NormalizationParameters;

        // Validate PubTabNet normalization values
        Assert.Equal(3, norm.Mean.Length);
        Assert.Equal(3, norm.Std.Length);

        // Check approximate values (PubTabNet dataset statistics)
        Assert.True(Math.Abs(norm.Mean[0] - 0.942f) < 0.01f, "Mean[0] should be ~0.942");
        Assert.True(Math.Abs(norm.Mean[1] - 0.942f) < 0.01f, "Mean[1] should be ~0.942");
        Assert.True(Math.Abs(norm.Mean[2] - 0.942f) < 0.01f, "Mean[2] should be ~0.942");

        Assert.True(Math.Abs(norm.Std[0] - 0.179f) < 0.01f, "Std[0] should be ~0.179");
        Assert.True(Math.Abs(norm.Std[1] - 0.179f) < 0.01f, "Std[1] should be ~0.179");
        Assert.True(Math.Abs(norm.Std[2] - 0.179f) < 0.01f, "Std[2] should be ~0.179");

        Assert.True(norm.Enabled, "Normalization should be enabled");

        // Validate target image size
        Assert.Equal(448, config.TargetImageSize);

        // Validate bbox classes
        Assert.Equal(2, config.BboxClasses);
    }

    [Fact]
    public void TableFormerWordMap_LoadsCorrectly()
    {
        var modelsDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "models");
        var wordMapPath = Path.Combine(modelsDir, "tableformer_fast_wordmap.json");

        // Skip if word map doesn't exist
        if (!File.Exists(wordMapPath))
        {
            return;
        }

        var wordMap = TableFormerWordMap.LoadFromFile(wordMapPath);

        Assert.NotNull(wordMap);
        Assert.NotNull(wordMap.WordMapTag);

        // Validate special tokens
        var (start, end, pad, unk) = wordMap.GetSpecialTokens();

        Assert.Equal(2, start);   // <start> should be ID 2
        Assert.Equal(3, end);     // <end> should be ID 3
        Assert.Equal(0, pad);     // <pad> should be ID 0
        Assert.Equal(1, unk);     // <unk> should be ID 1

        // Validate OTSL cell tokens exist
        Assert.True(wordMap.WordMapTag.ContainsKey("fcel"), "fcel token should exist");
        Assert.True(wordMap.WordMapTag.ContainsKey("ecel"), "ecel token should exist");
        Assert.True(wordMap.WordMapTag.ContainsKey("lcel"), "lcel token should exist");
        Assert.True(wordMap.WordMapTag.ContainsKey("xcel"), "xcel token should exist");
        Assert.True(wordMap.WordMapTag.ContainsKey("ucel"), "ucel token should exist");
        Assert.True(wordMap.WordMapTag.ContainsKey("nl"), "nl token should exist");

        // Validate token count (should have 13 tag tokens)
        Assert.Equal(13, wordMap.WordMapTag.Count);

        // Test reverse lookup
        var startToken = wordMap.GetTagToken(start);
        Assert.Equal("<start>", startToken);

        var endToken = wordMap.GetTagToken(end);
        Assert.Equal("<end>", endToken);
    }

    [Fact]
    public void OtslParser_ParsesSimpleTable()
    {
        // Simple 2x2 table: fcel fcel nl fcel fcel
        var tokens = new[] { "<start>", "fcel", "fcel", "nl", "fcel", "fcel", "<end>" };

        var tableStructure = OtslParser.ParseOtsl(tokens);

        Assert.Equal(2, tableStructure.RowCount);
        Assert.Equal(2, tableStructure.ColCount);

        // Verify first row
        Assert.Equal(2, tableStructure.Rows[0].Count);
        Assert.Equal("fcel", tableStructure.Rows[0][0].CellType);
        Assert.Equal("fcel", tableStructure.Rows[0][1].CellType);

        // Verify second row
        Assert.Equal(2, tableStructure.Rows[1].Count);
        Assert.Equal("fcel", tableStructure.Rows[1][0].CellType);
        Assert.Equal("fcel", tableStructure.Rows[1][1].CellType);
    }

    [Fact]
    public void OtslParser_ParsesHorizontalSpan()
    {
        // Table with horizontal span: fcel lcel fcel nl fcel fcel fcel
        // Row 1: [cell spanning 2 cols] [cell]
        // Row 2: [cell] [cell] [cell]
        var tokens = new[] { "<start>", "fcel", "lcel", "fcel", "nl", "fcel", "fcel", "fcel", "<end>" };

        var tableStructure = OtslParser.ParseOtsl(tokens);

        Assert.Equal(2, tableStructure.RowCount);

        // First row should have first cell with colspan=2
        Assert.Equal(2, tableStructure.Rows[0][0].ColSpan);
        Assert.Equal("linked", tableStructure.Rows[0][1].CellType); // Second cell is marked as linked

        // Second row should have 3 separate cells
        Assert.Equal(3, tableStructure.Rows[1].Count);
    }

    [Fact]
    public void OtslParser_ParsesVerticalSpan()
    {
        // Table with vertical span: fcel ucel nl fcel xcel nl fcel fcel
        // Row 1: [cell] [cell spanning down]
        // Row 2: [cell] [continuation]
        // Row 3: [cell] [cell]
        var tokens = new[] { "<start>", "fcel", "ucel", "nl", "fcel", "xcel", "nl", "fcel", "fcel", "<end>" };

        var tableStructure = OtslParser.ParseOtsl(tokens);

        Assert.Equal(3, tableStructure.RowCount);

        // First row, second cell should have rowspan=2
        Assert.Equal(2, tableStructure.Rows[0][1].RowSpan);
        Assert.Equal("ucel", tableStructure.Rows[0][1].CellType);

        // Second row, second cell should be marked as spanned
        Assert.Equal("spanned", tableStructure.Rows[1][1].CellType);

        // Third row should have 2 normal cells
        Assert.Equal(2, tableStructure.Rows[2].Count);
    }

    [Fact]
    public void OtslParser_ParsesHeaders()
    {
        // Table with headers: fcel ched fcel nl fcel fcel
        // Row 1: [header cell] [cell]
        // Row 2: [cell] [cell]
        var tokens = new[] { "<start>", "fcel", "ched", "fcel", "nl", "fcel", "fcel", "<end>" };

        var tableStructure = OtslParser.ParseOtsl(tokens);

        Assert.Equal(2, tableStructure.RowCount);

        // First cell should be marked as header
        Assert.True(tableStructure.Rows[0][0].IsHeader);
        Assert.Equal("ched", tableStructure.Rows[0][0].CellType);

        // Second cell should not be header
        Assert.False(tableStructure.Rows[0][1].IsHeader);
    }
}
