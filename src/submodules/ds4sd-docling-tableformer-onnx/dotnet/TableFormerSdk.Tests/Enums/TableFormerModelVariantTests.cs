namespace TableFormerSdk.Tests.Enums;

public class TableFormerModelVariantTests
{
    [Fact]
    public void Enum_HasExpectedValues()
    {
        // Arrange & Act
        var values = Enum.GetValues<TableFormerModelVariant>();

        // Assert
        Assert.Equal(2, values.Length);
        Assert.Contains(TableFormerModelVariant.Fast, values);
        Assert.Contains(TableFormerModelVariant.Accurate, values);
    }

    [Fact]
    public void Fast_HasCorrectValue()
    {
        // Arrange & Act
        var fast = TableFormerModelVariant.Fast;

        // Assert
        Assert.Equal(0, (int)fast);
    }

    [Fact]
    public void Accurate_HasCorrectValue()
    {
        // Arrange & Act
        var accurate = TableFormerModelVariant.Accurate;

        // Assert
        Assert.Equal(1, (int)accurate);
    }

    [Theory]
    [InlineData(TableFormerModelVariant.Fast, "Fast")]
    [InlineData(TableFormerModelVariant.Accurate, "Accurate")]
    public void ToString_ReturnsExpectedName(TableFormerModelVariant variant, string expectedName)
    {
        // Arrange & Act
        var name = variant.ToString();

        // Assert
        Assert.Equal(expectedName, name);
    }

    [Fact]
    public void Parse_FromString_WorksCorrectly()
    {
        // Arrange & Act
        var fast = Enum.Parse<TableFormerModelVariant>("Fast");
        var accurate = Enum.Parse<TableFormerModelVariant>("Accurate");

        // Assert
        Assert.Equal(TableFormerModelVariant.Fast, fast);
        Assert.Equal(TableFormerModelVariant.Accurate, accurate);
    }
}
