using System;
using System.Collections.Generic;

namespace Docling.Core.Documents;

/// <summary>
/// Helper responsible for registering <see cref="DocItem"/> instances onto a <see cref="DoclingDocument"/>
/// while attaching provenance information that mirrors the Python implementation.
/// </summary>
public sealed class DoclingDocumentBuilder
{
    private readonly DoclingDocument _document;

    public DoclingDocumentBuilder(DoclingDocument document)
    {
        _document = document ?? throw new ArgumentNullException(nameof(document));
    }

    public T AddItem<T>(T item, IEnumerable<DocItemProvenance>? provenance = null)
        where T : DocItem
    {
        ArgumentNullException.ThrowIfNull(item);

        if (provenance is not null)
        {
            item.SetProvenance(provenance);
        }
        else
        {
            item.SetProvenance(Array.Empty<DocItemProvenance>());
        }

        _document.AddItem(item);
        return item;
    }

    public T AddItem<T>(T item, params DocItemProvenance[] provenance)
        where T : DocItem
    {
        ArgumentNullException.ThrowIfNull(item);
        item.SetProvenance(provenance);
        _document.AddItem(item);
        return item;
    }

    public DoclingDocument Build() => _document;
}
