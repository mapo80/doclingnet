using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TableFormerSdk.Configuration;
using TableFormerSdk.Models;

namespace TableFormerSdk.Backends;

internal sealed class TableFormerOnnxBackend : ITableFormerBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _featureLength;
    private readonly object _syncRoot = new();
    private bool _disposed;

    public TableFormerOnnxBackend(TableFormerVariantModelPaths modelPaths)
    {
        ArgumentNullException.ThrowIfNull(modelPaths);

        if (!File.Exists(modelPaths.ModelPath))
        {
            throw new FileNotFoundException("TableFormer model not found", modelPaths.ModelPath);
        }

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            InterOpNumThreads = 0,
            IntraOpNumThreads = 0
        };

        options.AppendExecutionProvider_CPU(0);

        try
        {
            _session = new InferenceSession(modelPaths.ModelPath, options);
            _inputName = _session.InputMetadata.Keys.First();
            _featureLength = DetermineFeatureLength(_session, _inputName);
        }
        finally
        {
            options.Dispose();
        }
    }

    public IReadOnlyList<TableRegion> Infer(SKBitmap image, string sourcePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(TableFormerOnnxBackend));
        ArgumentNullException.ThrowIfNull(image);

        var features = TableFormerOnnxFeatureExtractor.ExtractFeatures(image, _featureLength);
        var inputTensor = new DenseTensor<long>(features, new[] { 1, features.Length });

        using DisposableNamedOnnxValue input = (DisposableNamedOnnxValue)DisposableNamedOnnxValue.CreateFromTensor(_inputName, inputTensor);
        using var results = _session.Run(new[] { input });
        var output = results[0].AsTensor<float>().ToArray();

        return PostProcess(output, image.Width, image.Height);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_syncRoot)
        {
            if (_disposed)
            {
                return;
            }

            _session.Dispose();
            _disposed = true;
        }
    }

    private static int DetermineFeatureLength(InferenceSession session, string inputName)
    {
        var metadata = session.InputMetadata[inputName];
        if (metadata.Dimensions.Length == 0)
        {
            return 10; // fallback to default length
        }

        var lastDimension = metadata.Dimensions.Last();
        if (lastDimension > 0)
        {
            return lastDimension;
        }

        // Dynamic shape: use default length of 10
        return 10;
    }

    private static IReadOnlyList<TableRegion> PostProcess(float[] output, int width, int height)
    {
        if (output.Length == 0 || width <= 0 || height <= 0)
        {
            return Array.Empty<TableRegion>();
        }

        var half = Math.Max(1, output.Length / 2);
        var rowScores = output.Take(half).ToArray();
        var columnScores = output.Skip(half).ToArray();

        var rowBoundaries = NormalizeBoundaries(rowScores);
        var columnBoundaries = NormalizeBoundaries(columnScores);

        var regions = new List<TableRegion>();
        for (var row = 0; row < rowBoundaries.Count - 1; row++)
        {
            for (var col = 0; col < columnBoundaries.Count - 1; col++)
            {
                var x = (float)(columnBoundaries[col] * width);
                var y = (float)(rowBoundaries[row] * height);
                var w = (float)Math.Max(1, (columnBoundaries[col + 1] - columnBoundaries[col]) * width);
                var h = (float)Math.Max(1, (rowBoundaries[row + 1] - rowBoundaries[row]) * height);

                regions.Add(new TableRegion(x, y, w, h, "table_cell"));
            }
        }

        if (regions.Count == 0)
        {
            regions.Add(new TableRegion(0, 0, width, height, "table_cell"));
        }

        return regions;
    }

    private static List<double> NormalizeBoundaries(IReadOnlyList<float> scores)
    {
        if (scores.Count == 0)
        {
            return new List<double> { 0d, 1d };
        }

        var min = scores.Min();
        var max = scores.Max();
        var range = Math.Max(1e-6f, max - min);

        var normalized = scores
            .Select(score => Math.Clamp((score - min) / range, 0f, 1f))
            .OrderBy(value => value)
            .Distinct()
            .ToList();

        if (normalized.Count == 0 || normalized[0] > 0f)
        {
            normalized.Insert(0, 0f);
        }

        if (normalized[^1] < 1f)
        {
            normalized.Add(1f);
        }

        // Remove extremely small segments to avoid degenerate cells
        var result = new List<double> { normalized[0] };
        foreach (var value in normalized.Skip(1))
        {
            if (value - result[^1] >= 0.05)
            {
                result.Add(value);
            }
        }

        if (result.Count < 2)
        {
            result.Add(1d);
        }

        return result;
    }
}
