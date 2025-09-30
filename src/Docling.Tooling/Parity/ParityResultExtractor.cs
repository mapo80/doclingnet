using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Export.Imaging;
using Docling.Export.Serialization;

namespace Docling.Tooling.Parity;

/// <summary>
/// Extracts normalised snapshots from pipeline results for parity comparisons.
/// </summary>
internal static class ParityResultExtractor
{
    /// <summary>
    /// Extracts a <see cref="ParityExtractionResult"/> from the supplied pipeline artefacts.
    /// </summary>
    /// <param name="markdownPath">Path where the markdown payload was written.</param>
    /// <param name="serializationResult">The serialization outcome returned by the markdown serializer.</param>
    /// <param name="document">The assembled <see cref="DoclingDocument"/>.</param>
    /// <param name="options">Optional normalisation options.</param>
    /// <returns>A normalised snapshot capturing document, page, and asset metadata.</returns>
    public static ParityExtractionResult Extract(
        string markdownPath,
        MarkdownSerializationResult serializationResult,
        DoclingDocument document,
        ParityExtractionOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(markdownPath);
        ArgumentNullException.ThrowIfNull(serializationResult);
        ArgumentNullException.ThrowIfNull(document);

        var effectiveOptions = options ?? ParityExtractionOptions.Default;

        var normalizedMarkdownPath = NormalizePath(markdownPath, effectiveOptions);
        var markdownSha = ComputeSha256(Encoding.UTF8.GetBytes(serializationResult.Markdown));

        var assets = serializationResult.Assets
            .Select(asset => CreateAssetSnapshot(asset, effectiveOptions))
            .OrderBy(asset => asset.RelativePath, StringComparer.Ordinal)
            .ToList();

        var pages = document.Pages
            .OrderBy(page => page.PageNumber)
            .Select(page => new ParityPageSnapshot(page.PageNumber, NormalizeDouble(page.Dpi, effectiveOptions)))
            .ToList();

        var documentProperties = document.Properties
            .OrderBy(kvp => kvp.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);

        var markdownMetadata = serializationResult.Metadata
            .OrderBy(kvp => kvp.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value, StringComparer.OrdinalIgnoreCase);

        return new ParityExtractionResult(
            document.Id,
            normalizedMarkdownPath,
            markdownSha,
            new ReadOnlyDictionary<string, string>(documentProperties),
            new ReadOnlyDictionary<string, string>(markdownMetadata),
            new ReadOnlyCollection<ParityPageSnapshot>(pages),
            new ReadOnlyCollection<ParityAssetSnapshot>(assets),
            DateTimeOffset.UtcNow);
    }

    private static ParityAssetSnapshot CreateAssetSnapshot(MarkdownAsset asset, ParityExtractionOptions options)
    {
        var image = asset.Image;
        var checksum = string.IsNullOrWhiteSpace(image.Checksum)
            ? ComputeSha256(image.Data.Span)
            : image.Checksum!;

        return new ParityAssetSnapshot(
            NormalizePath(asset.RelativePath, options),
            NormalizeKind(asset.Kind),
            asset.TargetItemId,
            image.Id,
            image.Page.PageNumber,
            NormalizeDouble(image.Dpi, options),
            image.Width,
            image.Height,
            image.MediaType,
            checksum,
            NormalizeBoundingBox(image.SourceRegion, options));
    }

    private static string NormalizeKind(ImageExportKind kind)
        => kind.ToString().ToUpperInvariant();

    private static string NormalizePath(string path, ParityExtractionOptions options)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        var normalized = path;
        if (Path.IsPathRooted(path))
        {
            var baseDirectory = !string.IsNullOrWhiteSpace(options.BaseDirectory)
                ? Path.GetFullPath(options.BaseDirectory)
                : Path.GetPathRoot(path) ?? string.Empty;

            var absolute = Path.GetFullPath(path);
            normalized = Path.GetRelativePath(baseDirectory, absolute);
        }

        normalized = normalized.Replace('\\', '/');
        if (normalized.StartsWith("./", StringComparison.Ordinal))
        {
            normalized = normalized[2..];
        }

        return normalized;
    }

    private static double NormalizeDouble(double value, ParityExtractionOptions options)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            return value;
        }

        var tolerance = options.CoordinateTolerance;
        var scaled = Math.Round(value / tolerance, MidpointRounding.AwayFromZero) * tolerance;
        return Math.Round(scaled, options.CoordinateDecimals, MidpointRounding.AwayFromZero);
    }

    private static ParityBoundingBox NormalizeBoundingBox(BoundingBox box, ParityExtractionOptions options)
    {
        return new ParityBoundingBox(
            NormalizeDouble(box.Left, options),
            NormalizeDouble(box.Top, options),
            NormalizeDouble(box.Right, options),
            NormalizeDouble(box.Bottom, options));
    }

    private static string ComputeSha256(ReadOnlySpan<byte> buffer)
    {
        if (buffer.IsEmpty)
        {
            return string.Empty;
        }

        var hash = SHA256.HashData(buffer);
        return Convert.ToHexString(hash);
    }
}
