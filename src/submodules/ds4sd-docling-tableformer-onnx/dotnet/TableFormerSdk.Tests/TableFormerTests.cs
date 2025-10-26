using SkiaSharp;

namespace TableFormerSdk.Tests;

public class TableFormerTests : IDisposable
{
    private readonly string _testModelsDir;

    public TableFormerTests()
    {
        _testModelsDir = FindModelsDirectory() ?? throw new InvalidOperationException("Models directory not found for tests");
    }

    [Fact]
    public void Constructor_WithValidDirectory_LoadsModels()
    {
        // Arrange & Act
        using var sdk = new TableFormer(_testModelsDir);

        // Assert
        Assert.NotNull(sdk);
        Assert.NotEmpty(sdk.LoadedVariants);
    }

    [Fact]
    public void Constructor_WithInvalidDirectory_ThrowsDirectoryNotFoundException()
    {
        // Arrange
        var invalidDir = "/nonexistent/directory";

        // Act & Assert
        Assert.Throws<DirectoryNotFoundException>(() => new TableFormer(invalidDir));
    }

    [Fact]
    public void Constructor_WithNullDirectory_ThrowsArgumentNullException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentNullException>(() => new TableFormer(null!));
    }

    [Fact]
    public void Constructor_WithEmptyDirectory_ThrowsFileNotFoundException()
    {
        // Arrange
        var emptyDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(emptyDir);

        try
        {
            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => new TableFormer(emptyDir));
        }
        finally
        {
            Directory.Delete(emptyDir);
        }
    }

    [Fact]
    public void LoadedVariants_ContainsFastModel()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);

        // Act
        var variants = sdk.LoadedVariants;

        // Assert
        Assert.Contains(TableFormerModelVariant.Fast, variants);
    }

    [Fact]
    public void IsModelLoaded_WithLoadedModel_ReturnsTrue()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);

        // Act
        var isLoaded = sdk.IsModelLoaded(TableFormerModelVariant.Fast);

        // Assert
        Assert.True(isLoaded);
    }

    [Fact]
    public void ExtractTableStructure_WithValidBitmap_ReturnsResult()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);
        using var bitmap = new SKBitmap(200, 150);

        // Act
        var result = sdk.ExtractTableStructure(bitmap, TableFormerModelVariant.Fast);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Regions);
        Assert.True(result.InferenceTime > TimeSpan.Zero);
    }

    [Fact]
    public void ExtractTableStructure_WithNullBitmap_ThrowsArgumentNullException()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            sdk.ExtractTableStructure((SKBitmap)null!, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void ExtractTableStructure_WithUnloadedVariant_ThrowsInvalidOperationException()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);
        using var bitmap = new SKBitmap(100, 100);

        // Remove accurate model if it exists (to test error case)
        if (!sdk.IsModelLoaded(TableFormerModelVariant.Accurate))
        {
            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                sdk.ExtractTableStructure(bitmap, TableFormerModelVariant.Accurate));
        }
    }

    [Fact]
    public void ExtractTableStructure_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var sdk = new TableFormer(_testModelsDir);
        using var bitmap = new SKBitmap(100, 100);
        sdk.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
            sdk.ExtractTableStructure(bitmap, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void ExtractTableStructure_WithImagePath_LoadsAndProcesses()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);
        var testImagePath = CreateTestImage();

        try
        {
            // Act
            var result = sdk.ExtractTableStructure(testImagePath, TableFormerModelVariant.Fast);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Regions);
        }
        finally
        {
            File.Delete(testImagePath);
        }
    }

    [Fact]
    public void ExtractTableStructure_WithNullPath_ThrowsArgumentNullException()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            sdk.ExtractTableStructure((string)null!, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void ExtractTableStructure_WithInvalidPath_ThrowsFileNotFoundException()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);
        var invalidPath = "nonexistent_image.png";

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            sdk.ExtractTableStructure(invalidPath, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void ExtractTableStructure_WithInvalidImage_ThrowsInvalidOperationException()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);
        var invalidImagePath = Path.GetTempFileName();
        File.WriteAllText(invalidImagePath, "This is not an image");

        try
        {
            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                sdk.ExtractTableStructure(invalidImagePath, TableFormerModelVariant.Fast));
        }
        finally
        {
            File.Delete(invalidImagePath);
        }
    }

    [Fact]
    public void Benchmark_WithValidParameters_ReturnsStatistics()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);

        // Act
        var result = sdk.Benchmark(TableFormerModelVariant.Fast, iterations: 10);

        // Assert
        Assert.True(result.MeanTimeMs > 0);
        Assert.True(result.ThroughputFps > 0);
    }

    [Fact]
    public void Benchmark_WithUnloadedVariant_ThrowsInvalidOperationException()
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);

        if (!sdk.IsModelLoaded(TableFormerModelVariant.Accurate))
        {
            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                sdk.Benchmark(TableFormerModelVariant.Accurate));
        }
    }

    [Fact]
    public void Benchmark_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var sdk = new TableFormer(_testModelsDir);
        sdk.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
            sdk.Benchmark(TableFormerModelVariant.Fast));
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var sdk = new TableFormer(_testModelsDir);

        // Act & Assert - should not throw
        sdk.Dispose();
        sdk.Dispose();
        sdk.Dispose();
    }

    [Theory]
    [InlineData(TableFormerModelVariant.Fast)]
    public void ExtractTableStructure_WithDifferentImageSizes_Works(TableFormerModelVariant variant)
    {
        // Arrange
        using var sdk = new TableFormer(_testModelsDir);
        var sizes = new[] { (50, 50), (100, 200), (300, 150), (1024, 768) };

        foreach (var (width, height) in sizes)
        {
            using var bitmap = new SKBitmap(width, height);

            // Act
            var result = sdk.ExtractTableStructure(bitmap, variant);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Regions);
        }
    }

    private static string? FindModelsDirectory()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "models")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "models")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "models")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "models")),
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate) &&
                (File.Exists(Path.Combine(candidate, "tableformer_fast.onnx")) ||
                 File.Exists(Path.Combine(candidate, "tableformer_accurate.onnx"))))
            {
                return candidate;
            }
        }

        return null;
    }

    private static string CreateTestImage()
    {
        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.png");
        using var bitmap = new SKBitmap(100, 100);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.OpenWrite(path);
        data.SaveTo(stream);
        return path;
    }

    public void Dispose()
    {
        // Cleanup if needed
    }
}
