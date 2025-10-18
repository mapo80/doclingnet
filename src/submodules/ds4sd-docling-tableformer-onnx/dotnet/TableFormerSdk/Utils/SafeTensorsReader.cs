//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using TorchSharp;

namespace TableFormerSdk.Utils;

/// <summary>
/// Reader for SafeTensors file format.
/// SafeTensors format: 8-byte header size (little endian) + JSON header + tensor data
/// </summary>
public sealed class SafeTensorsReader : IDisposable
{
    private readonly Dictionary<string, TensorInfo> _metadata;
    private readonly byte[] _data;
    private bool _disposed;

    public class TensorInfo
    {
        public string DType { get; set; } = "";
        public long[] Shape { get; set; } = Array.Empty<long>();
        public long DataOffset { get; set; }
        public long DataSize { get; set; }
    }

    public SafeTensorsReader(string filePath)
    {
        Console.WriteLine($"Loading SafeTensors file: {filePath}");

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // Read 8-byte header size (little endian)
        var headerSize = reader.ReadInt64();
        Console.WriteLine($"  Header size: {headerSize} bytes");

        // Read JSON header
        var headerBytes = reader.ReadBytes((int)headerSize);
        var headerJson = Encoding.UTF8.GetString(headerBytes);

        // Parse metadata
        _metadata = new Dictionary<string, TensorInfo>();
        using var jsonDoc = JsonDocument.Parse(headerJson);

        int tensorCount = 0;
        foreach (var prop in jsonDoc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;

            var dtype = prop.Value.GetProperty("dtype").GetString() ?? "float32";
            var shapeArray = prop.Value.GetProperty("shape").EnumerateArray();
            var shape = new List<long>();
            foreach (var dim in shapeArray)
            {
                shape.Add(dim.GetInt64());
            }

            var dataOffsets = prop.Value.GetProperty("data_offsets").EnumerateArray();
            var offsetsList = new List<long>();
            foreach (var offset in dataOffsets)
            {
                offsetsList.Add(offset.GetInt64());
            }

            var info = new TensorInfo
            {
                DType = dtype,
                Shape = shape.ToArray(),
                DataOffset = offsetsList[0],
                DataSize = offsetsList[1] - offsetsList[0]
            };

            _metadata[prop.Name] = info;
            tensorCount++;
        }

        Console.WriteLine($"  Found {tensorCount} tensors");

        // Read all tensor data
        var dataStart = 8 + headerSize;
        stream.Seek(dataStart, SeekOrigin.Begin);
        var totalDataSize = stream.Length - dataStart;
        _data = reader.ReadBytes((int)totalDataSize);

        Console.WriteLine($"  Loaded {_data.Length} bytes of tensor data");
    }

    public IEnumerable<string> TensorNames => _metadata.Keys;

    public torch.Tensor GetTensor(string name)
    {
        if (!_metadata.TryGetValue(name, out var info))
        {
            throw new KeyNotFoundException($"Tensor '{name}' not found in SafeTensors file");
        }

        // Extract data for this tensor
        var tensorData = new byte[info.DataSize];
        Array.Copy(_data, info.DataOffset, tensorData, 0, info.DataSize);

        // Convert to float array (assuming float32)
        if (info.DType != "F32" && info.DType != "float32")
        {
            throw new NotSupportedException($"DType {info.DType} not supported yet");
        }

        var floatCount = info.DataSize / sizeof(float);
        var floatData = new float[floatCount];
        Buffer.BlockCopy(tensorData, 0, floatData, 0, (int)info.DataSize);

        // Create TorchSharp tensor
        var tensor = torch.tensor(floatData).reshape(info.Shape);
        return tensor;
    }

    public TensorInfo? GetTensorInfo(string name)
    {
        return _metadata.TryGetValue(name, out var info) ? info : null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
