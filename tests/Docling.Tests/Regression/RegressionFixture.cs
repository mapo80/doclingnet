using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using System.Text.Json.Nodes;

namespace Docling.Tests.Regression;

/// <summary>
/// Utility for loading ground truth artefacts produced by the Python pipeline.
/// </summary>
internal sealed class RegressionFixture
{
    private RegressionFixture(
        string name,
        string directory,
        string? markdownPath,
        string? documentJsonPath,
        string? pagesJsonPath,
        string? docTagsPath,
        string? markdownContent,
        JsonObject? documentJson,
        JsonArray? pagesJson,
        IReadOnlyList<string> docTags)
    {
        Name = name;
        Directory = directory;
        MarkdownPath = markdownPath;
        DocumentJsonPath = documentJsonPath;
        PagesJsonPath = pagesJsonPath;
        DocTagsPath = docTagsPath;
        Markdown = markdownContent;
        DocumentJson = documentJson;
        PagesJson = pagesJson;
        DocTags = docTags;
    }

    public string Name { get; }

    public string Directory { get; }

    public string? MarkdownPath { get; }

    public string? DocumentJsonPath { get; }

    public string? PagesJsonPath { get; }

    public string? DocTagsPath { get; }

    public string? Markdown { get; }

    public JsonObject? DocumentJson { get; }

    public JsonArray? PagesJson { get; }

    public IReadOnlyList<string> DocTags { get; }

    /// <summary>
    /// Loads a regression fixture by name from the ground truth folder copied next to the binaries.
    /// </summary>
    /// <param name="name">Fixture identifier (file stem).</param>
    /// <param name="baseDirectory">Optional override for the ground truth directory.</param>
    /// <returns>The loaded regression fixture.</returns>
    public static RegressionFixture Load(string name, string? baseDirectory = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Fixture name must be provided.", nameof(name));
        }

        baseDirectory ??= Path.Combine(AppContext.BaseDirectory, "GroundTruth");
        if (!System.IO.Directory.Exists(baseDirectory))
        {
            throw new DirectoryNotFoundException($"Ground truth directory '{baseDirectory}' was not found. Ensure dataset artefacts are copied to the test output directory.");
        }

        var searchPattern = name + ".*";
        var files = System.IO.Directory.GetFiles(baseDirectory, searchPattern, SearchOption.TopDirectoryOnly);
        if (files.Length == 0)
        {
            throw new FileNotFoundException($"Unable to locate regression fixture '{name}' in '{baseDirectory}'.");
        }

        string? markdownPath = null;
        string? documentJsonPath = null;
        string? pagesJsonPath = null;
        string? docTagsPath = null;

        foreach (var file in files)
        {
            if (file.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
            {
                markdownPath = file;
            }
            else if (file.EndsWith(".pages.json", StringComparison.OrdinalIgnoreCase))
            {
                pagesJsonPath = file;
            }
            else if (file.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                documentJsonPath = file;
            }
            else if (file.EndsWith(".doctags.txt", StringComparison.OrdinalIgnoreCase))
            {
                docTagsPath = file;
            }
        }

        var markdownContent = markdownPath is null ? null : File.ReadAllText(markdownPath, Encoding.UTF8);

        JsonObject? documentJson = null;
        if (documentJsonPath is not null)
        {
            var node = JsonNode.Parse(File.ReadAllText(documentJsonPath, Encoding.UTF8));
            if (node is JsonObject jsonObject)
            {
                documentJson = jsonObject;
            }
            else
            {
                throw new InvalidDataException($"Document JSON for fixture '{name}' was expected to be an object but was '{node?.GetType().Name ?? "null"}'.");
            }
        }

        JsonArray? pagesJson = null;
        if (pagesJsonPath is not null)
        {
            var node = JsonNode.Parse(File.ReadAllText(pagesJsonPath, Encoding.UTF8));
            if (node is JsonArray jsonArray)
            {
                pagesJson = jsonArray;
            }
            else
            {
                throw new InvalidDataException($"Pages JSON for fixture '{name}' was expected to be an array but was '{node?.GetType().Name ?? "null"}'.");
            }
        }
        IReadOnlyList<string> docTags;
        if (docTagsPath is null)
        {
            docTags = Array.Empty<string>();
        }
        else
        {
            docTags = Array.AsReadOnly(File.ReadAllLines(docTagsPath, Encoding.UTF8));
        }

        return new RegressionFixture(
            name,
            baseDirectory,
            markdownPath,
            documentJsonPath,
            pagesJsonPath,
            docTagsPath,
            markdownContent,
            documentJson,
            pagesJson,
            docTags);
    }

    /// <summary>
    /// Compares markdown content with the fixture baseline using the provided comparison options.
    /// </summary>
    public MarkdownDiffResult CompareMarkdown(string actualMarkdown, MarkdownComparisonOptions? options = null)
    {
        if (Markdown is null)
        {
            throw new InvalidOperationException($"Fixture '{Name}' does not expose Markdown ground truth.");
        }

        options ??= MarkdownComparisonOptions.Default;

        var expectedLines = NormalizeMarkdown(Markdown, options);
        var actualLines = NormalizeMarkdown(actualMarkdown ?? string.Empty, options);

        var differences = new List<MarkdownDifference>();
        var max = Math.Max(expectedLines.Count, actualLines.Count);
        for (var index = 0; index < max; index++)
        {
            var expected = index < expectedLines.Count ? expectedLines[index] : null;
            var actual = index < actualLines.Count ? actualLines[index] : null;
            if (!string.Equals(expected, actual, StringComparison.Ordinal))
            {
                differences.Add(new MarkdownDifference(index + 1, expected, actual));
            }
        }

        var expectedNormalized = string.Join("\n", expectedLines);
        var actualNormalized = string.Join("\n", actualLines);

        return new MarkdownDiffResult(differences.Count == 0, differences, expectedNormalized, actualNormalized);
    }

    private static IReadOnlyList<string> NormalizeMarkdown(string text, MarkdownComparisonOptions options)
    {
        if (text.Length == 0)
        {
            return Array.Empty<string>();
        }

        var normalized = text
            .Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n');

        var lines = normalized.Split('\n');
        var builder = new List<string>(lines.Length);

        foreach (var rawLine in lines)
        {
            var line = options.IgnoreTrailingWhitespace ? rawLine.TrimEnd() : rawLine;

            if (options.CollapseSequentialBlankLines && line.Length == 0)
            {
                if (builder.Count > 0 && builder[^1].Length == 0)
                {
                    continue;
                }
            }

            builder.Add(line);
        }

        if (options.TrimFinalBlankLines)
        {
            while (builder.Count > 0 && builder[^1].Length == 0)
            {
                builder.RemoveAt(builder.Count - 1);
            }
        }

        return new ReadOnlyCollection<string>(builder);
    }
}

internal sealed record MarkdownComparisonOptions
{
    public static MarkdownComparisonOptions Default { get; } = new MarkdownComparisonOptions
    {
        IgnoreTrailingWhitespace = true,
        CollapseSequentialBlankLines = true,
        TrimFinalBlankLines = true,
    };

    public bool IgnoreTrailingWhitespace { get; init; } = true;

    public bool CollapseSequentialBlankLines { get; init; } = true;

    public bool TrimFinalBlankLines { get; init; } = true;
}

internal sealed record MarkdownDiffResult(
    bool AreEquivalent,
    IReadOnlyList<MarkdownDifference> Differences,
    string ExpectedNormalized,
    string ActualNormalized);

internal sealed record MarkdownDifference(int LineNumber, string? Expected, string? Actual);
