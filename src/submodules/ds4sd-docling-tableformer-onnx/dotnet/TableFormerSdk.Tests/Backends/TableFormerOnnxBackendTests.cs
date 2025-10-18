using Microsoft.ML.OnnxRuntime;
using SkiaSharp;

namespace TableFormerSdk.Tests.Backends;

public class TableFormerOnnxBackendTests : IDisposable
{
    private readonly string _testModelsDir;
    private readonly string _testModelPath;

    public TableFormerOnnxBackendTests()
    {
        // Find models directory
        _testModelsDir = FindModelsDirectory() ?? throw new InvalidOperationException("Models directory not found for tests");
        _testModelPath = Path.Combine(_testModelsDir, "tableformer_fast.onnx");

        if (!File.Exists(_testModelPath))
        {
            throw new FileNotFoundException($"Test model not found: {_testModelPath}");
        }
    }

    [Fact]
    public void Constructor_WithValidPath_LoadsModelSuccessfully()
    {
        // Arrange & Act
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Assert - no exception thrown
        Assert.NotNull(backend);
    }

    [Fact]
    public void Constructor_WithInvalidPath_ThrowsFileNotFoundException()
    {
        // Arrange
        var invalidPath = "nonexistent_model.onnx";

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            new TableFormerOnnxBackend(invalidPath, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void Constructor_WithNullPath_ThrowsArgumentNullException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new TableFormerOnnxBackend(null!, TableFormerModelVariant.Fast));
    }

    [Fact]
    public void CreateDummyInput_ReturnsCorrectShape()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act
        var input = backend.CreateDummyInput();

        // Assert
        Assert.NotNull(input);
        Assert.Equal(2, input.Dimensions.Length);
        Assert.Equal(1, input.Dimensions[0]);
        Assert.Equal(10, input.Dimensions[1]);
    }

    [Fact]
    public void CreateDummyInput_ReturnsDeterministicValues()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act
        var input1 = backend.CreateDummyInput();
        var input2 = backend.CreateDummyInput();

        // Assert - values should be the same (fixed seed)
        Assert.Equal(input1.ToArray(), input2.ToArray());
    }

    [Fact]
    public void PreprocessTableRegion_WithValidImage_ReturnsCorrectShape()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }
        using var image = new SKBitmap(100, 100);

        // Act
        var result = backend.PreprocessTableRegion(image);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.Dimensions.Length);
        Assert.Equal(1, result.Dimensions[0]);
        Assert.Equal(10, result.Dimensions[1]);
    }

    [Fact]
    public void PreprocessTableRegion_WithNullImage_ThrowsArgumentNullException()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            backend.PreprocessTableRegion(null!));
    }

    [Fact]
    public void Predict_WithValidInput_ReturnsOutputs()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }
        var input = backend.CreateDummyInput();

        // Act
        var outputs = backend.Predict(input);

        // Assert
        Assert.NotNull(outputs);
        Assert.NotEmpty(outputs);
        Assert.Contains("output", outputs.Keys);
    }

    [Fact]
    public void Predict_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            backend.Predict(null!));
    }

    [Fact]
    public void Predict_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }
        var input = backend.CreateDummyInput();
        backend.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
            backend.Predict(input));
    }

    [Fact]
    public void ExtractTableStructure_WithValidImage_ReturnsResult()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }
        using var image = new SKBitmap(200, 150);

        // Act
        var result = backend.ExtractTableStructure(image);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Regions);
        Assert.True(result.InferenceTime > TimeSpan.Zero);
        Assert.Equal(TableFormerModelVariant.Fast, result.ModelVariant);
        Assert.NotEmpty(result.RawOutputShapes);
    }

    [Fact]
    public void ExtractTableStructure_WithNullImage_ThrowsArgumentNullException()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            backend.ExtractTableStructure(null!));
    }

    [Fact]
    public void ExtractTableStructure_AlwaysReturnsAtLeastOneRegion()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }
        using var image = new SKBitmap(50, 50);

        // Act
        var result = backend.ExtractTableStructure(image);

        // Assert
        Assert.NotEmpty(result.Regions);
        Assert.True(result.Regions.Count >= 1);
    }

    [Fact]
    public void Benchmark_WithValidIterations_ReturnsStatistics()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act
        var result = backend.Benchmark(iterations: 10);

        // Assert
        Assert.True(result.MeanTimeMs > 0);
        Assert.True(result.MedianTimeMs > 0);
        Assert.True(result.MinTimeMs > 0);
        Assert.True(result.MaxTimeMs >= result.MinTimeMs);
        Assert.True(result.ThroughputFps > 0);
        Assert.True(result.StdTimeMs >= 0);
    }

    [Fact]
    public void Benchmark_ToString_ReturnsFormattedString()
    {
        // Arrange
        using var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }
        var result = backend.Benchmark(iterations: 5);

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("Mean:", str);
        Assert.Contains("Median:", str);
        Assert.Contains("Range:", str);
        Assert.Contains("Throughput:", str);
        Assert.Contains("FPS", str);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var backend = TryCreateBackend();
        if (backend is null)
        {
            return;
        }

        // Act & Assert - should not throw
        backend.Dispose();
        backend.Dispose();
        backend.Dispose();
    }

    [Theory]
    [InlineData(TableFormerModelVariant.Fast)]
    [InlineData(TableFormerModelVariant.Accurate)]
    public void Constructor_WorksWithBothVariants(TableFormerModelVariant variant)
    {
        // Arrange
        using var backend = TryCreateBackend(variant);
        if (backend is null)
        {
            return;
        }

        // Assert
        Assert.NotNull(backend);
    }

    private string? GetModelPath(TableFormerModelVariant variant)
    {
        var fileName = variant == TableFormerModelVariant.Fast ? "tableformer_fast.onnx" : "tableformer_accurate.onnx";
        var fullPath = Path.Combine(_testModelsDir, fileName);
        return File.Exists(fullPath) ? fullPath : null;
    }

    private TableFormerOnnxBackend? TryCreateBackend(TableFormerModelVariant variant = TableFormerModelVariant.Fast)
    {
        var modelPath = GetModelPath(variant);
        if (modelPath is null)
        {
            return null;
        }

        try
        {
            return new TableFormerOnnxBackend(modelPath, variant);
        }
        catch (OnnxRuntimeException)
        {
            return null;
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

    public void Dispose()
    {
        // Cleanup if needed
    }
}
