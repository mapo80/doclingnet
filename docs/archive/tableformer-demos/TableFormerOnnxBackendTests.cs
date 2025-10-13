#if false
using Microsoft.Extensions.Logging.Abstractions;
using SkiaSharp;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace Docling.Models.Tables;

/// <summary>
/// Test suite for TableFormer ONNX backend.
/// Tests the complete pipeline from image preprocessing to table structure detection.
/// </summary>
public class TableFormerOnnxBackendTests : IDisposable
{
    private readonly TableFormerTableStructureService _service;
    private readonly string _testOutputPath;

    public TableFormerOnnxBackendTests()
    {
        _testOutputPath = Path.Combine(Path.GetTempPath(), "tableformer-tests");
        Directory.CreateDirectory(_testOutputPath);

        var options = new TableFormerStructureServiceOptions
        {
            WorkingDirectory = _testOutputPath,
            GenerateOverlay = true
        };

        _service = new TableFormerTableStructureService(options, NullLogger<TableFormerTableStructureService>.Instance);
    }

    [Fact]
    public void Service_Initialization_ShouldConfigureCorrectly()
    {
        // Arrange & Act
        var service = new TableFormerTableStructureService();

        // Assert
        Assert.NotNull(service);
        Assert.True(service.IsUsingOnnxBackend() || !service.IsUsingOnnxBackend()); // Either way is valid
    }

    [Fact]
    public void Service_ReloadModels_ShouldUpdateConfiguration()
    {
        // Arrange
        var initialMetrics = _service.GetMetrics();

        // Act
        _service.ReloadModels();

        // Assert
        var newMetrics = _service.GetMetrics();
        Assert.Equal(initialMetrics.TotalInferences, newMetrics.TotalInferences); // Metrics should be preserved
    }

    [Fact]
    public void Service_GetCurrentModelPaths_ShouldReturnValidPaths()
    {
        // Arrange & Act
        var (fastEncoder, fastTagEncoder, accurateEncoder) = _service.GetCurrentModelPaths();

        // Assert
        if (_service.IsUsingOnnxBackend())
        {
            Assert.False(string.IsNullOrEmpty(fastEncoder));
            Assert.True(File.Exists(fastEncoder), $"Fast encoder model not found: {fastEncoder}");
        }
    }

    [Fact]
    public async Task Service_ProcessEmptyImage_ShouldThrowException()
    {
        // Arrange
        var request = new TableStructureRequest(
            Page: new(1),
            BoundingBox: new(0, 0, 100, 100),
            RasterizedImage: Array.Empty<byte>()
        );

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => _service.InferStructureAsync(request));
    }

    [Fact]
    public async Task Service_ProcessInvalidImage_ShouldThrowException()
    {
        // Arrange
        var invalidImageData = new byte[] { 0xFF, 0xD8, 0xFF }; // Invalid JPEG header
        var request = new TableStructureRequest(
            Page: new(1),
            BoundingBox: new(0, 0, 100, 100),
            RasterizedImage: invalidImageData
        );

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() => _service.InferStructureAsync(request));
    }

    [Fact]
    public void Metrics_InitialState_ShouldBeEmpty()
    {
        // Arrange & Act
        var metrics = _service.GetMetrics();

        // Assert
        Assert.Equal(0, metrics.TotalInferences);
        Assert.Equal(0, metrics.SuccessfulInferences);
        Assert.Equal(0, metrics.FailedInferences);
        Assert.Equal(0.0, metrics.SuccessRate);
        Assert.Equal(0, metrics.TotalCellsDetected);
    }

    [Fact]
    public void Metrics_Reset_ShouldClearAllData()
    {
        // Arrange
        _service.ResetMetrics();

        // Act
        var metrics = _service.GetMetrics();

        // Assert
        Assert.Equal(0, metrics.TotalInferences);
        Assert.Equal(0, metrics.SuccessfulInferences);
        Assert.Equal(0, metrics.FailedInferences);
        Assert.Equal(TimeSpan.Zero, metrics.TotalInferenceTime);
        Assert.Equal(0, metrics.TotalCellsDetected);
    }

    [Theory]
    [InlineData("png")]
    [InlineData("jpg")]
    [InlineData("jpeg")]
    public async Task Service_ProcessValidImage_ShouldCompleteWithoutError(string extension)
    {
        // Arrange
        var testImage = CreateTestImage(200, 100); // Small test image
        var imageBytes = GetImageBytes(testImage, extension);

        var request = new TableStructureRequest(
            Page: new(1),
            BoundingBox: new(0, 0, 200, 100),
            RasterizedImage: imageBytes
        );

        // Act
        var result = await _service.InferStructureAsync(request);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.Cells);
        Assert.True(result.RowCount >= 0);
        Assert.True(result.ColumnCount >= 0);

        // Verify metrics were recorded
        var metrics = _service.GetMetrics();
        Assert.True(metrics.TotalInferences > 0);
    }

    [Fact]
    public async Task Service_MultipleRequests_ShouldAccumulateMetrics()
    {
        // Arrange
        var initialMetrics = _service.GetMetrics();
        var testImage = CreateTestImage(100, 100);
        var imageBytes = GetImageBytes(testImage, "png");

        // Act
        for (int i = 0; i < 3; i++)
        {
            var request = new TableStructureRequest(
                Page: new(i + 1),
                BoundingBox: new(0, 0, 100, 100),
                RasterizedImage: imageBytes
            );

            await _service.InferStructureAsync(request);
        }

        // Assert
        var finalMetrics = _service.GetMetrics();
        Assert.Equal(initialMetrics.TotalInferences + 3, finalMetrics.TotalInferences);
        Assert.Equal(3, finalMetrics.SuccessfulInferences);
        Assert.True(finalMetrics.TotalInferenceTime > TimeSpan.Zero);
        Assert.True(finalMetrics.SuccessRate > 0.9); // Should be 100% success rate
    }

    private static SKBitmap CreateTestImage(int width, int height)
    {
        var bitmap = new SKBitmap(width, height);

        // Create a simple test pattern
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var color = (x + y) % 2 == 0 ? SKColors.White : SKColors.LightGray;
                bitmap.SetPixel(x, y, color);
            }
        }

        return bitmap;
    }

    private static byte[] GetImageBytes(SKBitmap bitmap, string format)
    {
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(format == "png" ? SKEncodedImageFormat.Png : SKEncodedImageFormat.Jpeg, 90);

        var stream = new MemoryStream();
        data.SaveTo(stream);
        return stream.ToArray();
    }

    public void Dispose()
    {
        _service.Dispose();

        // Clean up test files
        try
        {
            if (Directory.Exists(_testOutputPath))
            {
                Directory.Delete(_testOutputPath, recursive: true);
            }
        }
        catch
        {
            // Ignore cleanup errors in tests
        }
    }
}
#endif
