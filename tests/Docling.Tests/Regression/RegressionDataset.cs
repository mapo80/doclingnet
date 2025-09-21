using System;
using System.IO;

namespace Docling.Tests.Regression;

internal static class RegressionDatasets
{
    public static RegressionDataset Load(
        RegressionDatasetId datasetId,
        string? assetsDirectory = null,
        string? groundTruthDirectory = null)
    {
        var descriptor = datasetId switch
        {
            RegressionDatasetId.AmtHandbookSample => RegressionDatasetDescriptor.Create(
                fixtureName: "amt_handbook_sample",
                assetFileName: "amt_handbook_sample.pdf"),
            RegressionDatasetId.Arxiv230503393Page9 => RegressionDatasetDescriptor.Create(
                fixtureName: "2305.03393v1-pg9",
                assetFileName: "2305.03393v1-pg9-img.png"),
            _ => throw new ArgumentOutOfRangeException(nameof(datasetId), datasetId, "Unsupported regression dataset identifier."),
        };

        return RegressionDataset.Load(descriptor, assetsDirectory, groundTruthDirectory);
    }
}

internal enum RegressionDatasetId
{
    AmtHandbookSample,
    Arxiv230503393Page9,
}

internal sealed record RegressionDatasetDescriptor
{
    private RegressionDatasetDescriptor(string fixtureName, string assetFileName)
    {
        FixtureName = fixtureName;
        AssetFileName = assetFileName;
    }

    public string FixtureName { get; }

    public string AssetFileName { get; }

    public static RegressionDatasetDescriptor Create(string fixtureName, string assetFileName)
    {
        if (string.IsNullOrWhiteSpace(fixtureName))
        {
            throw new ArgumentException("Fixture name must be provided.", nameof(fixtureName));
        }

        if (string.IsNullOrWhiteSpace(assetFileName))
        {
            throw new ArgumentException("Asset file name must be provided.", nameof(assetFileName));
        }

        return new RegressionDatasetDescriptor(fixtureName, assetFileName);
    }
}

internal sealed class RegressionDataset
{
    private RegressionDataset(string name, string assetPath, RegressionFixture fixture)
    {
        Name = name;
        AssetPath = assetPath;
        Fixture = fixture;
    }

    public string Name { get; }

    public string AssetPath { get; }

    public RegressionFixture Fixture { get; }

    internal static RegressionDataset Load(
        RegressionDatasetDescriptor descriptor,
        string? assetsDirectory,
        string? groundTruthDirectory)
    {
        assetsDirectory ??= Path.Combine(AppContext.BaseDirectory, "Assets");
        groundTruthDirectory ??= Path.Combine(AppContext.BaseDirectory, "GroundTruth");

        var assetPath = Path.Combine(assetsDirectory, descriptor.AssetFileName);
        if (!File.Exists(assetPath))
        {
            throw new FileNotFoundException(
                $"Unable to locate regression dataset asset '{descriptor.AssetFileName}' in '{assetsDirectory}'.",
                descriptor.AssetFileName);
        }

        var fixture = RegressionFixture.Load(descriptor.FixtureName, groundTruthDirectory);

        return new RegressionDataset(descriptor.FixtureName, assetPath, fixture);
    }
}
