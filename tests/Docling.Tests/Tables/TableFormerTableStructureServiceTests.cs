using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using Docling.Models.Tables;
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;
using TableFormerSdk.Performance;
using Xunit;

namespace Docling.Tests.Tables;

public sealed class TableFormerTableStructureServiceTests : IDisposable
{
    private readonly string _workingDirectory;

    public TableFormerTableStructureServiceTests()
    {
        _workingDirectory = Path.Combine(Path.GetTempPath(), $"docling-tests-{Guid.NewGuid():N}");
    }

    [Fact]
    public async Task InferStructureAsyncMapsRegionsToCells()
    {
        var imageBytes = CreateImageBytes(width: 20, height: 10);
        var page = new PageReference(1, 300);
        var bounds = BoundingBox.FromSize(10, 20, 200, 100);
        var request = new TableStructureRequest(page, bounds, imageBytes);

        var regions = new List<TableRegion>
        {
            new(0f, 0f, 10f, 5f, "class_1"),
            new(10f, 5f, 10f, 5f, "class_1"),
            new(0f, 0f, 0f, 0f, "invalid") // ignored
        };

        var snapshot = new TableFormerPerformanceSnapshot(TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate, 1, 1, 10, 10, 10);
        var result = new TableStructureResult(regions, overlay: null, TableFormerLanguage.English, TableFormerRuntime.Onnx, TimeSpan.FromMilliseconds(25), snapshot);
        using var invoker = new RecordingInvoker(result);

        var options = new TableFormerStructureServiceOptions
        {
            Variant = TableFormerModelVariant.Accurate,
            Runtime = TableFormerRuntime.Onnx,
            WorkingDirectory = _workingDirectory,
        };

        using var service = new TableFormerTableStructureService(options, NullLogger<TableFormerTableStructureService>.Instance, invoker);
        var structure = await service.InferStructureAsync(request);

        Assert.Equal(page, structure.Page);
        Assert.Equal(2, structure.Cells.Count);
        Assert.Equal(2, structure.RowCount);
        Assert.Equal(2, structure.ColumnCount);

        var first = structure.Cells[0].BoundingBox;
        Assert.Equal(10, first.Left, 4);
        Assert.Equal(20, first.Top, 4);
        Assert.Equal(110, first.Right, 4);
        Assert.Equal(70, first.Bottom, 4);

        var second = structure.Cells[1].BoundingBox;
        Assert.Equal(110, second.Left, 4);
        Assert.Equal(70, second.Top, 4);
        Assert.Equal(210, second.Right, 4);
        Assert.Equal(120, second.Bottom, 4);

        Assert.Single(invoker.Invocations);
        var invocation = invoker.Invocations[0];
        Assert.False(invocation.Overlay);
        Assert.Equal(TableFormerModelVariant.Accurate, invocation.Variant);
        Assert.Equal(TableFormerRuntime.Onnx, invocation.Runtime);
        Assert.Null(invocation.Language);
    }

    [Fact]
    public async Task DisposeReleasesInvoker()
    {
        var imageBytes = CreateImageBytes(4, 4);
        var request = new TableStructureRequest(new PageReference(1, 300), BoundingBox.FromSize(0, 0, 4, 4), imageBytes);
        var snapshot = new TableFormerPerformanceSnapshot(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, 1, 1, 1, 1, 1);
        var result = new TableStructureResult(Array.Empty<TableRegion>(), null, TableFormerLanguage.English, TableFormerRuntime.Onnx, TimeSpan.Zero, snapshot);
        using var invoker = new RecordingInvoker(result);

        var options = new TableFormerStructureServiceOptions { WorkingDirectory = _workingDirectory };
        var service = new TableFormerTableStructureService(options, NullLogger<TableFormerTableStructureService>.Instance, invoker);
        await service.InferStructureAsync(request);

        Assert.False(invoker.Disposed);
        service.Dispose();
        Assert.True(invoker.Disposed);
    }

    [Fact]
    public async Task InferStructureAsyncReturnsEmptyStructureWhenNoRegions()
    {
        var imageBytes = CreateImageBytes(8, 8);
        var request = new TableStructureRequest(new PageReference(2, 200), BoundingBox.FromSize(5, 6, 40, 40), imageBytes);
        var snapshot = new TableFormerPerformanceSnapshot(TableFormerRuntime.Onnx, TableFormerModelVariant.Fast, 1, 1, 5, 5, 5);
        var result = new TableStructureResult(Array.Empty<TableRegion>(), null, TableFormerLanguage.English, TableFormerRuntime.Onnx, TimeSpan.Zero, snapshot);
        using var invoker = new RecordingInvoker(result);

        var options = new TableFormerStructureServiceOptions { WorkingDirectory = _workingDirectory };
        using var service = new TableFormerTableStructureService(options, NullLogger<TableFormerTableStructureService>.Instance, invoker);
        var structure = await service.InferStructureAsync(request);

        Assert.Empty(structure.Cells);
        Assert.Equal(0, structure.RowCount);
        Assert.Equal(0, structure.ColumnCount);
    }

    [Fact]
    public async Task InferStructureAsyncEmitsDebugArtifactWhenOverlayRequested()
    {
        var imageBytes = CreateImageBytes(16, 16);
        var request = new TableStructureRequest(new PageReference(3, 150), BoundingBox.FromSize(0, 0, 16, 16), imageBytes);

        SKBitmap? bitmap = null;
        try
        {
            bitmap = new SKBitmap(16, 16);
            using (var canvas = new SKCanvas(bitmap))
            {
                canvas.Clear(SKColors.Blue);
            }

            var regions = new List<TableRegion> { new(0f, 0f, 16f, 16f, "class_1") };
            var snapshot = new TableFormerPerformanceSnapshot(TableFormerRuntime.Onnx, TableFormerModelVariant.Accurate, 1, 1, 1, 1, 1);
            var result = new TableStructureResult(regions, bitmap, TableFormerLanguage.English, TableFormerRuntime.Onnx, TimeSpan.Zero, snapshot);
            using var invoker = new RecordingInvoker(result);

            var options = new TableFormerStructureServiceOptions
            {
                GenerateOverlay = true,
                WorkingDirectory = _workingDirectory,
            };

            using var service = new TableFormerTableStructureService(options, NullLogger<TableFormerTableStructureService>.Instance, invoker);
            var structure = await service.InferStructureAsync(request);

            Assert.NotNull(structure.DebugArtifact);
            var artifact = structure.DebugArtifact!;
            Assert.Equal(request.Page, artifact.Page);
            Assert.Equal("image/png", artifact.MediaType);
            Assert.True(artifact.ImageContent.Length > 0);

            Assert.Single(invoker.Invocations);
            Assert.True(invoker.Invocations[0].Overlay);
            Assert.NotNull(result.OverlayImage);
            Assert.Equal(IntPtr.Zero, result.OverlayImage!.Handle);

            bitmap = null;
        }
        finally
        {
            bitmap?.Dispose();
        }
    }

    private static ReadOnlyMemory<byte> CreateImageBytes(int width, int height)
    {
        using var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, quality: 90);
        return data.ToArray();
    }

    public void Dispose()
    {
        if (Directory.Exists(_workingDirectory))
        {
            Directory.Delete(_workingDirectory, recursive: true);
        }
    }

    private sealed class RecordingInvoker : ITableFormerInvoker
    {
        private readonly TableStructureResult _result;

        public RecordingInvoker(TableStructureResult result)
        {
            _result = result;
        }

        public List<(string Path, bool Overlay, TableFormerModelVariant Variant, TableFormerRuntime Runtime, TableFormerLanguage? Language)> Invocations { get; } = new();

        public bool Disposed { get; private set; }

        public TableStructureResult Process(string imagePath, bool overlay, TableFormerModelVariant variant, TableFormerRuntime runtime, TableFormerLanguage? language)
        {
            Invocations.Add((imagePath, overlay, variant, runtime, language));
            return _result;
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }
}
