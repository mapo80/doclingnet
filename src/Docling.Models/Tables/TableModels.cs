using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Docling.Core.Geometry;
using Docling.Core.Primitives;

namespace Docling.Models.Tables;

public sealed record TableCell(BoundingBox BoundingBox, int RowSpan, int ColumnSpan, string? Text);

public sealed record TableStructure(PageReference Page, IReadOnlyList<TableCell> Cells, int RowCount, int ColumnCount);

public interface ITableStructureService
{
    Task<TableStructure> InferStructureAsync(TableStructureRequest request, CancellationToken cancellationToken = default);
}

public sealed record TableStructureRequest(PageReference Page, BoundingBox BoundingBox, ReadOnlyMemory<byte> RasterizedImage);
