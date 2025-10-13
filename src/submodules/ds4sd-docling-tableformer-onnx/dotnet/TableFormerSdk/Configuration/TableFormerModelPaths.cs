using System;
using System.Collections.Generic;
using System.IO;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;

namespace TableFormerSdk.Configuration;

public sealed class TableFormerModelPaths
{
    public TableFormerModelPaths(TableFormerVariantModelPaths fast, TableFormerVariantModelPaths? accurate = null)
    {
        Fast = fast ?? throw new ArgumentNullException(nameof(fast));
        Accurate = accurate;
    }

    public TableFormerVariantModelPaths Fast { get; }

    public TableFormerVariantModelPaths? Accurate { get; }

    public TableFormerVariantModelPaths GetModelPaths(TableFormerModelVariant variant) => variant switch
    {
        TableFormerModelVariant.Fast => Fast,
        TableFormerModelVariant.Accurate when Accurate is not null => Accurate,
        TableFormerModelVariant.Accurate => throw new InvalidOperationException(TableFormerConstants.AccurateModelNotConfiguredMessage),
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, TableFormerConstants.UnsupportedModelVariantMessage)
    };
}

public sealed class TableFormerVariantModelPaths
{
    public TableFormerVariantModelPaths(string modelPath, string? metadataPath = null)
    {
        ModelPath = ValidateModelPath(modelPath, nameof(modelPath));
        MetadataPath = ValidateOptionalPath(metadataPath);
    }

    public string ModelPath { get; }

    public string? MetadataPath { get; }

    public static TableFormerVariantModelPaths FromDirectory(string directory, string variantPrefix)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ArgumentException("Directory path is empty", nameof(directory));
        }

        directory = Path.GetFullPath(directory);
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Model directory not found: {directory}");
        }

        var modelFile = Path.Combine(directory, $"{variantPrefix}.onnx");
        var metadataFile = Path.Combine(directory, $"{variantPrefix}.yaml");

        return new TableFormerVariantModelPaths(
            ValidateModelPath(modelFile, nameof(modelFile)),
            File.Exists(metadataFile) ? metadataFile : null);
    }

    private static string ValidateModelPath(string path, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Model path is empty", argumentName);
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Model file not found: {path}", path);
        }

        return path;
    }

    private static string? ValidateOptionalPath(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Metadata file not found: {path}", path);
        }

        return path;
    }
}
