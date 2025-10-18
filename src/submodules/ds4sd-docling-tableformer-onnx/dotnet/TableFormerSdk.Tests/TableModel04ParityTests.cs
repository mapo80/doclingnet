//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using TableFormerSdk.Models;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace TableFormerSdk.Tests;

public sealed class TableModel04ParityTests : IDisposable
{
    private readonly string? _fastModelDir;
    private readonly string? _groundTruthDir;

    public TableModel04ParityTests()
    {
        _fastModelDir = ResolveFastModelDirectory();
        _groundTruthDir = ResolveGroundTruthDirectory();
    }

    [Fact]
    public void FastVariantMatchesPythonGroundTruth()
    {
        if (_fastModelDir is null || _groundTruthDir is null)
        {
            return;
        }

        var configPath = Path.Combine(_fastModelDir, "tm_config.json");
        var weightsPath = Path.Combine(_fastModelDir, "tableformer_fast.safetensors");
        var groundTruthPath = Path.Combine(_groundTruthDir, "tableformer_fast_prediction.json");
        var imagePath = Path.Combine(_groundTruthDir, "input_image.npy");

        if (!File.Exists(configPath) ||
            !File.Exists(weightsPath) ||
            !File.Exists(groundTruthPath) ||
            !File.Exists(imagePath))
        {
            return;
        }

        var config = TableModel04Config.FromJsonFile(configPath);
        using var model = TableModel04.FromSafeTensors(config, weightsPath);
        model.eval();

        using var imageTensor = NpyTensorLoader.LoadFloat32(imagePath, device: torch.CPU);

        TableModel04Result result;
        using (torch.no_grad())
        {
            result = model.forward(imageTensor);
        }

        try
        {
            var groundTruth = PythonGroundTruth.Load(groundTruthPath);

            Assert.Equal(groundTruth.Sequence.Count, result.Sequence.Count);
            for (var i = 0; i < groundTruth.Sequence.Count; i++)
            {
                Assert.Equal(groundTruth.Sequence[i], result.Sequence[i]);
            }

            using var expectedClasses = TensorFactory.FromJagged(groundTruth.BBoxClasses, result.BBoxClasses.device);
            using var expectedCoords = TensorFactory.FromJagged(groundTruth.BBoxCoords, result.BBoxCoords.device);

            using var classDiff = (result.BBoxClasses - expectedClasses).abs();
            var maxClassDiff = classDiff.max().item<float>();

            using var coordDiff = (result.BBoxCoords - expectedCoords).abs();
            var maxCoordDiff = coordDiff.max().item<float>();

            Assert.True(maxClassDiff < 1e-4f, $"bbox class diff {maxClassDiff}");
            Assert.True(maxCoordDiff < 1e-4f, $"bbox coord diff {maxCoordDiff}");
        }
        finally
        {
            result.BBoxClasses.Dispose();
            result.BBoxCoords.Dispose();
        }
    }

    public void Dispose()
    {
    }

    private static string? ResolveFastModelDirectory()
    {
        var envOverride = Environment.GetEnvironmentVariable("TABLEFORMER_FAST_DIR");
        if (!string.IsNullOrEmpty(envOverride) && Directory.Exists(envOverride))
        {
            return envOverride;
        }

        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "models", "model_artifacts", "tableformer", "fast")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "models", "model_artifacts", "tableformer", "fast")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "models", "model_artifacts", "tableformer", "fast"))
        };

        return candidates.FirstOrDefault(Directory.Exists);
    }

    private static string? ResolveGroundTruthDirectory()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "test-data-python-ground-truth")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "test-data-python-ground-truth")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "test-data-python-ground-truth"))
        };

        return candidates.FirstOrDefault(Directory.Exists);
    }

    private sealed record PythonGroundTruth(
        IReadOnlyList<long> Sequence,
        IReadOnlyList<IReadOnlyList<float>> BBoxClasses,
        IReadOnlyList<IReadOnlyList<float>> BBoxCoords)
    {
        public static PythonGroundTruth Load(string path)
        {
            using var stream = File.OpenRead(path);
            using var document = JsonDocument.Parse(stream);
            var root = document.RootElement;

            var sequence = root.GetProperty("sequence").EnumerateArray().Select(x => x.GetInt64()).ToList();

            static List<List<float>> ParseMatrix(JsonElement element)
            {
                var rows = new List<List<float>>();
                foreach (var row in element.EnumerateArray())
                {
                    rows.Add(row.EnumerateArray().Select(x => x.GetSingle()).ToList());
                }
                return rows;
            }

            var bboxClasses = ParseMatrix(root.GetProperty("bbox_classes"));
            var bboxCoords = ParseMatrix(root.GetProperty("bbox_coords"));

            return new PythonGroundTruth(sequence, bboxClasses, bboxCoords);
        }
    }

    private static class TensorFactory
    {
        public static Tensor FromJagged(IReadOnlyList<IReadOnlyList<float>> rows, Device device)
        {
            var columnCount = rows.Count > 0 ? rows[0].Count : 0;

            if (rows.Count == 0 || columnCount == 0)
            {
                return torch.zeros(new long[] { 0, columnCount }, dtype: ScalarType.Float32, device: device);
            }
            var flattened = new float[rows.Count * columnCount];
            var offset = 0;

            foreach (var row in rows)
            {
                if (row.Count != columnCount)
                {
                    throw new InvalidOperationException("Jagged array must have consistent column counts.");
                }

                for (var column = 0; column < columnCount; column++)
                {
                    flattened[offset + column] = row[column];
                }
                offset += columnCount;
            }

            var tensor = torch.tensor(flattened, dtype: ScalarType.Float32, device: device);
            return tensor.reshape(rows.Count, columnCount);
        }
    }

    private static class NpyTensorLoader
    {
        public static Tensor LoadFloat32(string path, Device device)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);

            var magic = reader.ReadBytes(6);
            var expectedMagic = new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };
            if (!magic.SequenceEqual(expectedMagic))
            {
                throw new InvalidDataException("Invalid NPY magic header.");
            }

            var major = reader.ReadByte();
            var minor = reader.ReadByte();
            int headerLength = major switch
            {
                1 => reader.ReadUInt16(),
                2 or 3 => (int)reader.ReadUInt32(),
                _ => throw new NotSupportedException($"Unsupported NPY version {major}.{minor}")
            };

            var headerBytes = reader.ReadBytes(headerLength);
            var header = Encoding.ASCII.GetString(headerBytes).Trim();

            var descr = ExtractDescr(header);
            if (descr != "<f4")
            {
                throw new NotSupportedException($"Only float32 NPY files are supported. Found '{descr}'.");
            }

            if (ExtractFortranOrder(header))
            {
                throw new NotSupportedException("Fortran-ordered NPY tensors are not supported.");
            }

            var shape = ExtractShape(header);
            var elementCount = checked((int)shape.Aggregate(1L, (acc, dim) => acc * dim));

            var data = new float[elementCount];
            for (var i = 0; i < elementCount; i++)
            {
                data[i] = reader.ReadSingle();
            }

            var tensor = torch.tensor(data, dtype: ScalarType.Float32, device: device);
            return tensor.reshape(shape.ToArray());
        }

        private static string ExtractDescr(string header)
        {
            const string key = "'descr':";
            var start = header.IndexOf(key, StringComparison.Ordinal);
            if (start < 0)
            {
                throw new InvalidDataException("NPY header missing 'descr'.");
            }

            start = header.IndexOf('\'', start + key.Length) + 1;
            var end = header.IndexOf('\'', start);
            return header.Substring(start, end - start);
        }

        private static bool ExtractFortranOrder(string header)
        {
            const string key = "'fortran_order':";
            var start = header.IndexOf(key, StringComparison.Ordinal);
            if (start < 0)
            {
                throw new InvalidDataException("NPY header missing 'fortran_order'.");
            }

            start += key.Length;
            var end = header.IndexOf(',', start);
            if (end < 0)
            {
                end = header.IndexOf('}', start);
            }

            var value = header.Substring(start, end - start).Trim();
            return string.Equals(value, "True", StringComparison.OrdinalIgnoreCase);
        }

        private static IReadOnlyList<long> ExtractShape(string header)
        {
            const string key = "'shape':";
            var start = header.IndexOf(key, StringComparison.Ordinal);
            if (start < 0)
            {
                throw new InvalidDataException("NPY header missing 'shape'.");
            }

            start = header.IndexOf('(', start + key.Length);
            var end = header.IndexOf(')', start + 1);
            var shapeContent = header.Substring(start + 1, end - start - 1);

            if (string.IsNullOrWhiteSpace(shapeContent))
            {
                return Array.Empty<long>();
            }

            var dims = shapeContent
                .Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(dim => long.Parse(dim.Trim(), System.Globalization.CultureInfo.InvariantCulture))
                .ToArray();

            return dims;
        }
    }
}
