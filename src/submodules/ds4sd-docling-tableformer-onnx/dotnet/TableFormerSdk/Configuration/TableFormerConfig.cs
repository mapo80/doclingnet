using System.IO;

namespace TableFormerSdk.Configuration;

public sealed class TableFormerSdkOptions
{
    public TableFormerModelPaths Onnx { get; }

    public TableFormerSdkOptions(TableFormerModelPaths onnx)
    {
        Onnx = onnx;
    }
}

public sealed class TableFormerModelPaths
{
    public TableFormerVariantModelPaths Fast { get; }
    public TableFormerVariantModelPaths? Accurate { get; }

    public TableFormerModelPaths(TableFormerVariantModelPaths fast, TableFormerVariantModelPaths? accurate)
    {
        Fast = fast;
        Accurate = accurate;
    }
}

public sealed class TableFormerVariantModelPaths
{
    public string ModelPath { get; }
    public string? MetadataPath { get; }

    private TableFormerVariantModelPaths(string modelPath, string? metadataPath)
    {
        ModelPath = modelPath;
        MetadataPath = metadataPath;
    }

    public static TableFormerVariantModelPaths FromDirectory(string directory, string variant)
    {
        var modelPath = Path.Combine(directory, variant, "model.onnx");
        var metadataPath = Path.Combine(directory, variant, "metadata.json");

        if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found: {modelPath}");
        if (!File.Exists(metadataPath)) metadataPath = null;

        return new TableFormerVariantModelPaths(modelPath, metadataPath);
    }
}