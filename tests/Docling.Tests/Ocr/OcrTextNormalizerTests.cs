using Docling.Models.Ocr;
using FluentAssertions;

namespace Docling.Tests.Ocr;

public sealed class OcrTextNormalizerTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void NormalizeReturnsEmptyForWhitespace(string? input)
    {
        OcrTextNormalizer.Normalize(input).Should().BeEmpty();
    }

    [Fact]
    public void NormalizeExpandsLigaturesAndCollapsesWhitespace()
    {
        var text = "fiancée\t\ncoöperate  manœuvre";

        var result = OcrTextNormalizer.Normalize(text);

        result.Should().Be("fiancee cooperate manoeuvre");
    }

    [Fact]
    public void NormalizeDropsControlCharacters()
    {
        var text = "Hello\u0000World\u0007";

        var result = OcrTextNormalizer.Normalize(text);

        result.Should().Be("Hello World");
    }
}
