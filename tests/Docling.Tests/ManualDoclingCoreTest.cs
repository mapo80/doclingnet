using Docling.Core.Documents;
using Docling.Core.Geometry;
using Docling.Core.Primitives;
using FluentAssertions;

namespace Docling.Tests;

/// <summary>
/// Test manuale semplificato per verificare il comportamento della libreria Docling.Core
/// con un documento di esempio (pagina accademica dal dataset).
/// </summary>
public sealed class ManualDoclingCoreTest
{
    [Fact]
    public void CanCreateDocumentWithPageFromDatasetImage()
    {
        // Arrange - Simula la pagina 9 del paper 2305.03393v1
        const string sourceId = "dataset/2305.03393v1-pg9-img.png";
        const int pageNumber = 9;
        const double pageHeight = 1650.0;

        var pageRef = new PageReference(pageNumber, pageHeight);
        var document = new DoclingDocument(
            sourceId: sourceId,
            pages: new[] { pageRef },
            documentId: "test-doc-2305.03393v1"
        );

        // Act - Aggiungi alcuni item che potrebbero essere rilevati nella pagina

        // 1. Titolo della sezione "5.1 Hyper Parameter Optimization"
        var titleBox = BoundingBox.FromSize(left: 100, top: 150, width: 400, height: 30);
        var titleItem = new ParagraphItem(
            page: pageRef,
            box: titleBox,
            text: "5.1 Hyper Parameter Optimization",
            id: "para-pg9-title"
        );
        titleItem.AddTag("heading");
        titleItem.AddTag("section-title");
        document.AddItem(titleItem);

        // 2. Primo paragrafo di testo
        var para1Box = BoundingBox.FromSize(left: 100, top: 200, width: 600, height: 80);
        var para1Item = new ParagraphItem(
            page: pageRef,
            box: para1Box,
            text: "We have chosen the PubTabNet data set to perform HPO, since it includes a highly diverse set of tables.",
            id: "para-pg9-001"
        );
        document.AddItem(para1Item);

        // 3. Una figura
        var figureBox = BoundingBox.FromSize(left: 100, top: 700, width: 300, height: 200);
        var figureItem = new PictureItem(
            page: pageRef,
            box: figureBox,
            id: "figure-pg9-001"
        );
        document.AddItem(figureItem);

        // Assert - Verifica la struttura del documento
        document.Items.Should().HaveCount(3, "abbiamo aggiunto 3 elementi: titolo, paragrafo, figura");

        // Verifica ordinamento corretto (per posizione Y)
        document.Items[0].Should().BeSameAs(titleItem, "il titolo è in alto");
        document.Items[1].Should().BeSameAs(para1Item, "il paragrafo segue il titolo");
        document.Items[2].Should().BeSameAs(figureItem, "la figura è in basso");

        // Verifica i tag
        titleItem.HasTag("heading").Should().BeTrue();
        titleItem.HasTag("section-title").Should().BeTrue();

        // Verifica le proprietà del documento
        document.SourceId.Should().Be(sourceId);
        document.Pages.Should().HaveCount(1);
        document.Pages[0].PageNumber.Should().Be(pageNumber);

        // Test clonazione del documento
        var clone = document.Clone();
        clone.Items.Should().HaveCount(3);
        clone.SourceId.Should().Be(document.SourceId);
    }

    [Fact]
    public void CanQueryDocumentItemsByBoundingBox()
    {
        // Arrange
        var pageRef = new PageReference(0, 1000);
        var document = new DoclingDocument("test-source", new[] { pageRef });

        var item1 = new ParagraphItem(pageRef, BoundingBox.FromSize(10, 10, 100, 50), "Text 1");
        var item2 = new ParagraphItem(pageRef, BoundingBox.FromSize(10, 100, 100, 50), "Text 2");
        var item3 = new ParagraphItem(pageRef, BoundingBox.FromSize(10, 200, 100, 50), "Text 3");

        document.AddItem(item1);
        document.AddItem(item2);
        document.AddItem(item3);

        // Act - Cerca il primo item nella parte superiore (Y < 150)
        var found = document.TryFindFirstBoundingBox(
            x => x.BoundingBox.Top < 150,
            out var bbox
        );

        // Assert
        found.Should().BeTrue();
        bbox.Should().Be(item1.BoundingBox);
    }

    [Fact]
    public void BoundingBoxOperationsWork()
    {
        // Test delle operazioni geometriche che sarebbero usate
        // durante l'analisi della pagina del dataset

        // Simula il bounding box del titolo
        var titleBox = BoundingBox.FromSize(100, 150, 400, 30);

        // Simula il bounding box di una tabella sotto
        var tableBox = BoundingBox.FromSize(100, 400, 600, 200);

        // Verifica che non si sovrappongano
        titleBox.Intersects(tableBox).Should().BeFalse();

        // Calcola l'unione (area che contiene entrambi)
        var union = titleBox.Union(tableBox);
        union.Top.Should().Be(150);
        union.Bottom.Should().Be(600);
        union.Left.Should().Be(100);
        union.Right.Should().Be(700);

        // Test di contenimento
        var point = new Point2D(300, 160);
        titleBox.Contains(point).Should().BeTrue();
        tableBox.Contains(point).Should().BeFalse();

        // Test di espansione
        var expanded = titleBox.Inflate(10, 5);
        expanded.Left.Should().Be(90);
        expanded.Top.Should().Be(145);
        expanded.Right.Should().Be(510);
        expanded.Bottom.Should().Be(185);
    }

    [Fact]
    public void DocumentSupportsMultipleItemTypes()
    {
        // Test che verifica il supporto di diversi tipi di item

        var pageRef = new PageReference(9, 1650);
        var document = new DoclingDocument("test-doc", new[] { pageRef });

        // Aggiungi vari tipi di item
        var para = new ParagraphItem(pageRef, BoundingBox.FromSize(0, 0, 100, 20), "Paragraph text");
        var picture = new PictureItem(pageRef, BoundingBox.FromSize(0, 30, 100, 100));
        var caption = new CaptionItem(pageRef, BoundingBox.FromSize(0, 140, 100, 20), "Figure caption");

        document.AddItem(para);
        document.AddItem(picture);
        document.AddItem(caption);

        // Verifica il conteggio per tipo
        var paragraphs = document.GetItemsOfKind(DocItemKind.Paragraph);
        paragraphs.Should().HaveCount(1);

        var pictures = document.GetItemsOfKind(DocItemKind.Picture);
        pictures.Should().HaveCount(1);

        var captions = document.GetItemsOfKind(DocItemKind.Caption);
        captions.Should().HaveCount(1);
    }
}
