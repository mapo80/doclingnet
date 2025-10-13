using System.Collections.Generic;
using LayoutSdk.Metrics;

namespace LayoutSdk.Processing;

internal sealed record LayoutPipelineResult(IReadOnlyList<BoundingBox> Boxes, LayoutExecutionMetrics Metrics);
