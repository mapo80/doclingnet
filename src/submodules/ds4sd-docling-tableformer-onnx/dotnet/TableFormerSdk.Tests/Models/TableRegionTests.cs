namespace TableFormerSdk.Tests.Models;

public class TableRegionTests
{
    [Fact]
    public void Constructor_SetsPropertiesCorrectly()
    {
        // Arrange & Act
        var region = new TableRegion(10.5f, 20.3f, 100.0f, 50.0f, "table_cell");

        // Assert
        Assert.Equal(10.5f, region.X);
        Assert.Equal(20.3f, region.Y);
        Assert.Equal(100.0f, region.Width);
        Assert.Equal(50.0f, region.Height);
        Assert.Equal("table_cell", region.CellType);
    }

    [Fact]
    public void Constructor_WithNullCellType_DefaultsToTableCell()
    {
        // Arrange & Act
        var region = new TableRegion(0f, 0f, 10f, 10f, null!);

        // Assert
        Assert.Equal("table_cell", region.CellType);
    }

    [Fact]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var region = new TableRegion(10.123f, 20.456f, 100.789f, 50.321f, "header");

        // Act
        var result = region.ToString();

        // Assert - Check for both dot and comma formats (culture-independent)
        Assert.True(result.Contains("10.1") || result.Contains("10,1"), $"Expected '10.1' or '10,1' in: {result}");
        Assert.True(result.Contains("20.5") || result.Contains("20,5"), $"Expected '20.5' or '20,5' in: {result}");
        Assert.True(result.Contains("100.8") || result.Contains("100,8"), $"Expected '100.8' or '100,8' in: {result}");
        Assert.True(result.Contains("50.3") || result.Contains("50,3"), $"Expected '50.3' or '50,3' in: {result}");
        Assert.Contains("header", result);
    }

    [Theory]
    [InlineData(0f, 0f, 0f, 0f)]
    [InlineData(100f, 200f, 300f, 400f)]
    [InlineData(-10f, -20f, 50f, 60f)]
    public void Constructor_AcceptsVariousCoordinates(float x, float y, float width, float height)
    {
        // Arrange & Act
        var region = new TableRegion(x, y, width, height, "test");

        // Assert
        Assert.Equal(x, region.X);
        Assert.Equal(y, region.Y);
        Assert.Equal(width, region.Width);
        Assert.Equal(height, region.Height);
    }
}
