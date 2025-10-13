using SkiaSharp;
using Xunit;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using EasyOcrNet;

namespace EasyOcrNet.Tests;

public class OcrTests
{
    [Fact]
    public void RecognizesSimpleText()
    {
        using var bmp = new SKBitmap(800, 608);
        using (var canvas = new SKCanvas(bmp))
        {
            canvas.Clear(SKColors.White);
            var paint = new SKPaint
            {
                Color = SKColors.Black,
                TextSize = 200,
                IsAntialias = true
            };
            canvas.DrawText("HELLO", 50, 300, paint);
        }

        var modelDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models", "cpu"));
        using var ocr = new EasyOcr(modelDir);
        var results = ocr.Read(bmp).ToList();
        Assert.NotEmpty(results.First().Text);
    }

    public static IEnumerable<object[]> ExampleImages()
    {
        var examplesDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "examples"));
        var extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };
        foreach (var file in Directory.GetFiles(examplesDir).Where(f => extensions.Contains(Path.GetExtension(f))))
            yield return new object[] { Path.GetFileName(file) };
    }

    [Theory]
    [MemberData(nameof(ExampleImages))]
    public void RecognizesExampleImages(string fileName)
    {
        var examplesDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "examples"));
        using var bmp = SKBitmap.Decode(Path.Combine(examplesDir, fileName));
        var modelDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models", "cpu"));
        var charset = DeriveCharset(fileName);
        using var ocr = new EasyOcr(modelDir, charset);
        var results = ocr.Read(bmp).ToList();
        var actual = string.Join(" ", results.Select(r => r.Text));

        var csharpPath = Path.Combine(examplesDir, Path.ChangeExtension(fileName, ".txt"));
        File.WriteAllText(csharpPath, actual);

        var pythonPath = Path.Combine(examplesDir, Path.ChangeExtension(fileName, ".python.txt"));
        var python = File.Exists(pythonPath) ? File.ReadAllText(pythonPath) : string.Empty;

        if (charset != Charset.ch_sim)
        {
            Assert.NotEmpty(actual);
        }
    }

    private static Charset DeriveCharset(string fileName)
    {
        var name = Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant();
        return name switch
        {
            "english" or "example" or "example2" or "example3" or "easyocr_framework" or "width_ths" => Charset.en,
            "french" => Charset.fr,
            "japanese" => Charset.ja,
            "korean" => Charset.ko,
            "chinese" => Charset.ch_sim,
            "thai" => Charset.th,
            _ => Charset.en
        };
    }
}
