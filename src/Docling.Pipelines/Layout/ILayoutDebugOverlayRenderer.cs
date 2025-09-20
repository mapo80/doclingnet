using System.Collections.Generic;
using Docling.Backends.Pdf;
using Docling.Models.Layout;

namespace Docling.Pipelines.Layout;

/// <summary>
/// Generates visual overlays highlighting layout predictions for diagnostics.
/// </summary>
public interface ILayoutDebugOverlayRenderer
{
    LayoutDebugOverlay CreateOverlay(PageImage page, IReadOnlyList<LayoutItem> layoutItems);
}
