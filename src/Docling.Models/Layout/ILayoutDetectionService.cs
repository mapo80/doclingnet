using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Primitives;

namespace Docling.Models.Layout;

/// <summary>
/// Abstraction that integrates with the layout detection machine learning model.
/// </summary>
public interface ILayoutDetectionService
{
    Task<IReadOnlyList<LayoutItem>> DetectAsync(LayoutRequest request, CancellationToken cancellationToken = default);
}

/// <summary>
/// Provides access to normalization metadata captured during layout inference.
/// </summary>
public interface ILayoutNormalizationMetadataSource
{
    IReadOnlyList<LayoutNormalizationTelemetry> ConsumeNormalizationMetadata();
}

/// <summary>
/// Exposes profiling telemetry captured while executing layout inference.
/// </summary>
public interface ILayoutProfilingTelemetrySource
{
    IReadOnlyList<LayoutInferenceProfilingTelemetry> ConsumeProfilingTelemetry();
}

/// <summary>
/// Represents the normalization parameters applied to a specific page before invoking the layout model.
/// </summary>
public sealed record LayoutNormalizationTelemetry(PageReference Page, LayoutSdkNormalisationMetadata Metadata);

/// <summary>
/// Timing breakdown collected while running layout inference for a single page.
/// </summary>
public readonly record struct LayoutSdkProfilingSnapshot(
    double PersistMilliseconds,
    double InferenceMilliseconds,
    double PostprocessMilliseconds,
    double TotalMilliseconds);

/// <summary>
/// Associates a profiling snapshot with the originating page.
/// </summary>
public sealed record LayoutInferenceProfilingTelemetry(PageReference Page, LayoutSdkProfilingSnapshot Snapshot);

/// <summary>
/// Defines the input payload for layout detection.
/// </summary>
public sealed class LayoutRequest
{
    public LayoutRequest(
        string documentId,
        string modelIdentifier,
        LayoutRequestOptions options,
        IReadOnlyList<LayoutPagePayload> pages)
    {
        if (string.IsNullOrWhiteSpace(documentId))
        {
            throw new ArgumentException("A document identifier is required.", nameof(documentId));
        }

        if (string.IsNullOrWhiteSpace(modelIdentifier))
        {
            throw new ArgumentException("A model identifier is required.", nameof(modelIdentifier));
        }

        var materializedPages = pages is null
            ? throw new ArgumentNullException(nameof(pages))
            : pages.ToList();

        DocumentId = documentId;
        ModelIdentifier = modelIdentifier;
        Options = options ?? throw new ArgumentNullException(nameof(options));
        Pages = new ReadOnlyCollection<LayoutPagePayload>(materializedPages);
    }

    public string DocumentId { get; }

    public string ModelIdentifier { get; }

    public LayoutRequestOptions Options { get; }

    public IReadOnlyList<LayoutPagePayload> Pages { get; }
}

/// <summary>
/// Options controlling the inference behaviour of the layout service.
/// </summary>
public sealed record LayoutRequestOptions(bool CreateOrphanClusters, bool KeepEmptyClusters, bool SkipCellAssignment);

/// <summary>
/// Describes an individual page image supplied to the layout model.
/// </summary>
public sealed class LayoutPagePayload
{
    public LayoutPagePayload(
        PageReference page,
        string artifactId,
        string mediaType,
        double dpi,
        int width,
        int height,
        IReadOnlyDictionary<string, string> metadata,
        ReadOnlyMemory<byte> imageContent)
    {
        Page = page;
        ArtifactId = string.IsNullOrWhiteSpace(artifactId)
            ? throw new ArgumentException("An artifact identifier is required.", nameof(artifactId))
            : artifactId;
        MediaType = string.IsNullOrWhiteSpace(mediaType)
            ? throw new ArgumentException("A media type is required.", nameof(mediaType))
            : mediaType;
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(dpi, 0, nameof(dpi));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(width, 0, nameof(width));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(height, 0, nameof(height));

        if (imageContent.IsEmpty)
        {
            throw new ArgumentException("Image content cannot be empty.", nameof(imageContent));
        }

        Page = page;
        ArtifactId = artifactId;
        MediaType = mediaType;
        Dpi = dpi;
        Width = width;
        Height = height;
        Metadata = metadata is null
            ? throw new ArgumentNullException(nameof(metadata))
            : new ReadOnlyDictionary<string, string>(new Dictionary<string, string>(metadata, StringComparer.OrdinalIgnoreCase));
        ImageContent = imageContent;
    }

    public PageReference Page { get; }

    public string ArtifactId { get; }

    public string MediaType { get; }

    public double Dpi { get; }

    public int Width { get; }

    public int Height { get; }

    public IReadOnlyDictionary<string, string> Metadata { get; }

    public ReadOnlyMemory<byte> ImageContent { get; }
}

/// <summary>
/// Represents a debug overlay produced from layout predictions.
/// </summary>
public sealed record LayoutDebugOverlay(PageReference Page, ReadOnlyMemory<byte> ImageContent, string MediaType = "image/png");
