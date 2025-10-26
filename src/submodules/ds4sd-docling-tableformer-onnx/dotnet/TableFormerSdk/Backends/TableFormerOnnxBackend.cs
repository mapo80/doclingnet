using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Diagnostics;
using TableFormerSdk.Enums;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

/// <summary>
/// ONNX backend for TableFormer models from HuggingFace asmud/ds4sd-docling-models-onnx
/// This implementation matches the Python example.py script exactly
/// </summary>
public sealed class TableFormerOnnxBackend : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int[] _inputShape;
    private readonly string[] _outputNames;
    private readonly TableFormerModelVariant _variant;
    private readonly object _lock = new();
    private bool _disposed;

    public TableFormerOnnxBackend(string modelPath, TableFormerModelVariant variant)
    {
        ArgumentNullException.ThrowIfNull(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model not found: {modelPath}", modelPath);
        }

        _variant = variant;

        // Create optimized session options
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            InterOpNumThreads = Environment.ProcessorCount,
            IntraOpNumThreads = Environment.ProcessorCount
        };

        try
        {
            _session = new InferenceSession(modelPath, options);

            // Get input metadata
            var inputMetadata = _session.InputMetadata.First();
            _inputName = inputMetadata.Key;
            _inputShape = inputMetadata.Value.Dimensions.ToArray();

            // Get output metadata
            _outputNames = _session.OutputMetadata.Keys.ToArray();

            Console.WriteLine($"✓ {variant} TableFormer model loaded successfully");
            Console.WriteLine($"  Input: {_inputName} {string.Join("x", _inputShape)} ({inputMetadata.Value.ElementDataType})");
            Console.WriteLine($"  Outputs: {_outputNames.Length} tensor(s)");
        }
        finally
        {
            options.Dispose();
        }
    }

    /// <summary>
    /// Create dummy input tensor for testing (matches Python create_dummy_input)
    /// Uses hardcoded values from Python np.random.seed(42) for exact reproducibility
    /// </summary>
    public DenseTensor<long> CreateDummyInput()
    {
        // These values match Python: np.random.seed(42); np.random.randint(0, 100, (1, 10))
        // Python output: [51 92 14 71 60 20 82 86 74 74]
        var pythonValues = new long[] { 51, 92, 14, 71, 60, 20, 82, 86, 74, 74 };

        var tensor = new DenseTensor<long>(_inputShape);
        for (int i = 0; i < Math.Min(pythonValues.Length, tensor.Length); i++)
        {
            tensor.SetValue(i, pythonValues[i]);
        }

        return tensor;
    }

    /// <summary>
    /// Preprocess table region image (matches Python preprocess_table_region)
    /// Note: For the JPQD quantized models, this creates dummy features matching input shape
    /// </summary>
    public DenseTensor<long> PreprocessTableRegion(SKBitmap image)
    {
        ArgumentNullException.ThrowIfNull(image);

        // For the JPQD quantized models, we create dummy features
        // matching the model's expected input (based on Python implementation)
        // Use the same hardcoded values as CreateDummyInput for reproducibility
        return CreateDummyInput();
    }

    /// <summary>
    /// Run inference on input tensor (matches Python predict)
    /// </summary>
    public Dictionary<string, Tensor<float>> Predict(DenseTensor<long> inputTensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(TableFormerOnnxBackend));
        ArgumentNullException.ThrowIfNull(inputTensor);

        // Validate input shape
        if (!inputTensor.Dimensions.SequenceEqual(_inputShape))
        {
            var inputShape = string.Join("x", inputTensor.Dimensions.ToArray());
            var expectedShape = string.Join("x", _inputShape);
            Console.WriteLine($"Warning: Input shape {inputShape} != expected {expectedShape}");
        }

        lock (_lock)
        {
            // Create input
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            // Run inference
            using var results = _session.Run(inputs);

            // Package results
            var outputs = new Dictionary<string, Tensor<float>>();
            foreach (var result in results)
            {
                if (result.Value is Tensor<float> tensor)
                {
                    // Clone tensor to detach from session
                    var clonedTensor = new DenseTensor<float>(
                        tensor.ToArray(),
                        tensor.Dimensions.ToArray()
                    );
                    outputs[result.Name] = clonedTensor;
                }
            }

            return outputs;
        }
    }

    /// <summary>
    /// Extract table structure from image (matches Python extract_table_structure)
    /// </summary>
    public TableStructureResult ExtractTableStructure(SKBitmap image)
    {
        ArgumentNullException.ThrowIfNull(image);

        var stopwatch = Stopwatch.StartNew();

        // Preprocess
        var inputTensor = PreprocessTableRegion(image);

        // Get raw predictions
        var rawOutputs = Predict(inputTensor);

        stopwatch.Stop();

        // Post-process to extract table structure
        // Note: For the JPQD demo models, we create dummy output matching Python behavior
        var regions = PostProcessOutputs(rawOutputs, image.Width, image.Height);

        var outputShapes = rawOutputs.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.Dimensions.ToArray()
        );

        return new TableStructureResult(
            regions,
            _variant,
            stopwatch.Elapsed,
            outputShapes
        );
    }

    /// <summary>
    /// Post-process model outputs into table regions
    /// This is a simplified implementation for the JPQD demo models
    /// </summary>
    private List<TableRegion> PostProcessOutputs(
        Dictionary<string, Tensor<float>> outputs,
        int imageWidth,
        int imageHeight)
    {
        var regions = new List<TableRegion>();

        // For the JPQD models, output is [1, 10] float32
        // This is a demonstration - real implementation would parse actual model outputs
        if (outputs.TryGetValue("output", out var outputTensor))
        {
            var outputArray = outputTensor.ToArray();

            // Split output: first half for rows, second half for columns (matching old logic)
            int halfLen = Math.Max(1, outputArray.Length / 2);
            var rowScores = outputArray.Take(halfLen).ToArray();
            var colScores = outputArray.Skip(halfLen).ToArray();

            // Normalize boundaries
            var rowBoundaries = NormalizeBoundaries(rowScores);
            var colBoundaries = NormalizeBoundaries(colScores);

            // Create cells from boundaries
            for (int row = 0; row < rowBoundaries.Count - 1; row++)
            {
                for (int col = 0; col < colBoundaries.Count - 1; col++)
                {
                    var x = (float)(colBoundaries[col] * imageWidth);
                    var y = (float)(rowBoundaries[row] * imageHeight);
                    var w = (float)Math.Max(1, (colBoundaries[col + 1] - colBoundaries[col]) * imageWidth);
                    var h = (float)Math.Max(1, (rowBoundaries[row + 1] - rowBoundaries[row]) * imageHeight);

                    regions.Add(new TableRegion(x, y, w, h, "table_cell"));
                }
            }
        }

        // Ensure at least one region
        if (regions.Count == 0)
        {
            regions.Add(new TableRegion(0, 0, imageWidth, imageHeight, "table_cell"));
        }

        return regions;
    }

    private static List<double> NormalizeBoundaries(float[] scores)
    {
        if (scores.Length == 0)
        {
            return new List<double> { 0.0, 1.0 };
        }

        var min = scores.Min();
        var max = scores.Max();
        var range = Math.Max(1e-6f, max - min);

        var normalized = scores
            .Select(score => Math.Clamp((score - min) / range, 0f, 1f))
            .OrderBy(value => value)
            .Distinct()
            .Select(x => (double)x)
            .ToList();

        if (normalized.Count == 0 || normalized[0] > 0f)
        {
            normalized.Insert(0, 0.0);
        }

        if (normalized[^1] < 1f)
        {
            normalized.Add(1.0);
        }

        // Remove extremely small segments
        var result = new List<double> { normalized[0] };
        foreach (var value in normalized.Skip(1))
        {
            if (value - result[^1] >= 0.01)
            {
                result.Add(value);
            }
        }

        if (result.Count < 2)
        {
            result.Add(1.0);
        }

        return result;
    }

    /// <summary>
    /// Run performance benchmark (matches Python benchmark)
    /// </summary>
    public BenchmarkResult Benchmark(int iterations = 100)
    {
        Console.WriteLine($"Running benchmark with {iterations} iterations...");

        var dummyInput = CreateDummyInput();

        // Warmup (5 iterations)
        for (int i = 0; i < 5; i++)
        {
            _ = Predict(dummyInput);
        }

        // Benchmark
        var times = new List<double>();
        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = Predict(dummyInput);
            sw.Stop();
            times.Add(sw.Elapsed.TotalMilliseconds);

            if ((i + 1) % 10 == 0)
            {
                Console.WriteLine($"  Progress: {i + 1}/{iterations}");
            }
        }

        return new BenchmarkResult
        {
            MeanTimeMs = times.Average(),
            StdTimeMs = CalculateStdDev(times),
            MinTimeMs = times.Min(),
            MaxTimeMs = times.Max(),
            MedianTimeMs = CalculateMedian(times),
            ThroughputFps = 1000.0 / times.Average()
        };
    }

    private static double CalculateStdDev(List<double> values)
    {
        var mean = values.Average();
        var sumOfSquares = values.Sum(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(sumOfSquares / values.Count);
    }

    private static double CalculateMedian(List<double> values)
    {
        var sorted = values.OrderBy(x => x).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid];
    }

    public void Dispose()
    {
        if (_disposed) return;

        lock (_lock)
        {
            if (_disposed) return;
            _session?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Benchmark results
/// </summary>
public sealed class BenchmarkResult
{
    public double MeanTimeMs { get; init; }
    public double StdTimeMs { get; init; }
    public double MinTimeMs { get; init; }
    public double MaxTimeMs { get; init; }
    public double MedianTimeMs { get; init; }
    public double ThroughputFps { get; init; }

    public override string ToString() =>
        $"Mean: {MeanTimeMs:F2}ms ± {StdTimeMs:F2}ms | " +
        $"Median: {MedianTimeMs:F2}ms | " +
        $"Range: [{MinTimeMs:F2}, {MaxTimeMs:F2}]ms | " +
        $"Throughput: {ThroughputFps:F1} FPS";
}
