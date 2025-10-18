namespace TableFormerSdk.Tests.Models;

public class TableStructureResultTests
{
    [Fact]
    public void Constructor_SetsPropertiesCorrectly()
    {
        // Arrange
        var regions = new List<TableRegion>
        {
            new TableRegion(0, 0, 10, 10, "cell1"),
            new TableRegion(10, 0, 10, 10, "cell2")
        };
        var inferenceTime = TimeSpan.FromMilliseconds(5.5);
        var shapes = new Dictionary<string, int[]> { { "output", new[] { 1, 10 } } };

        // Act
        var result = new TableStructureResult(
            regions,
            TableFormerModelVariant.Fast,
            inferenceTime,
            shapes
        );

        // Assert
        Assert.Equal(2, result.Regions.Count);
        Assert.Equal(TableFormerModelVariant.Fast, result.ModelVariant);
        Assert.Equal(inferenceTime, result.InferenceTime);
        Assert.Single(result.RawOutputShapes);
        Assert.Equal(new[] { 1, 10 }, result.RawOutputShapes["output"]);
    }

    [Fact]
    public void Constructor_WithNullRegions_CreatesEmptyList()
    {
        // Arrange & Act
        var result = new TableStructureResult(
            null!,
            TableFormerModelVariant.Accurate,
            TimeSpan.Zero
        );

        // Assert
        Assert.Empty(result.Regions);
    }

    [Fact]
    public void Constructor_WithNullShapes_CreatesEmptyDictionary()
    {
        // Arrange & Act
        var result = new TableStructureResult(
            Array.Empty<TableRegion>(),
            TableFormerModelVariant.Fast,
            TimeSpan.Zero,
            null
        );

        // Assert
        Assert.Empty(result.RawOutputShapes);
    }

    [Fact]
    public void ToString_ContainsKeyInformation()
    {
        // Arrange
        var regions = new List<TableRegion>
        {
            new TableRegion(0, 0, 10, 10, "cell")
        };
        var result = new TableStructureResult(
            regions,
            TableFormerModelVariant.Accurate,
            TimeSpan.FromMilliseconds(3.14)
        );

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("1 regions", str);
        Assert.True(str.Contains("3.14ms") || str.Contains("3,14ms"), $"Expected '3.14ms' or '3,14ms' in: {str}");
        Assert.Contains("Accurate", str);
    }

    [Theory]
    [InlineData(TableFormerModelVariant.Fast)]
    [InlineData(TableFormerModelVariant.Accurate)]
    public void Constructor_AcceptsBothVariants(TableFormerModelVariant variant)
    {
        // Arrange & Act
        var result = new TableStructureResult(
            Array.Empty<TableRegion>(),
            variant,
            TimeSpan.Zero
        );

        // Assert
        Assert.Equal(variant, result.ModelVariant);
    }
}
