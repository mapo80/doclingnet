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
    private static readonly IReadOnlyDictionary<string, string> ExpectedFiles = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        ["encoder"] = "encoder.onnx",
        ["tag_encoder"] = "tag_transformer_encoder.onnx",
        ["decoder_step"] = "tag_transformer_decoder_step.onnx",
        ["bbox_decoder"] = "bbox_decoder.onnx",
        ["config"] = "config.json",
        ["wordmap"] = "wordmap.json",
    };

    public TableFormerVariantModelPaths(
        string encoderPath,
        string tagEncoderPath,
        string decoderStepPath,
        string bboxDecoderPath,
        string configPath,
        string wordMapPath)
    {
        EncoderPath = ValidateModelPath(encoderPath, nameof(encoderPath));
        TagEncoderPath = ValidateModelPath(tagEncoderPath, nameof(tagEncoderPath));
        DecoderStepPath = ValidateModelPath(decoderStepPath, nameof(decoderStepPath));
        BboxDecoderPath = ValidateModelPath(bboxDecoderPath, nameof(bboxDecoderPath));
        ConfigPath = ValidateModelPath(configPath, nameof(configPath));
        WordMapPath = ValidateModelPath(wordMapPath, nameof(wordMapPath));
    }

    public string EncoderPath { get; }

    public string TagEncoderPath { get; }

    public string DecoderStepPath { get; }

    public string BboxDecoderPath { get; }

    public string ConfigPath { get; }

    public string WordMapPath { get; }

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

        static string Resolve(string directory, string fileName)
        {
            var path = Path.Combine(directory, fileName);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Expected model artifact not found: {path}", path);
            }

            return path;
        }

        string WithPrefix(string suffix) => $"{variantPrefix}_{suffix}";

        return new TableFormerVariantModelPaths(
            Resolve(directory, WithPrefix(ExpectedFiles["encoder"])),
            Resolve(directory, WithPrefix(ExpectedFiles["tag_encoder"])),
            Resolve(directory, WithPrefix(ExpectedFiles["decoder_step"])),
            Resolve(directory, WithPrefix(ExpectedFiles["bbox_decoder"])),
            Resolve(directory, WithPrefix(ExpectedFiles["config"])),
            Resolve(directory, WithPrefix(ExpectedFiles["wordmap"])));
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
}
