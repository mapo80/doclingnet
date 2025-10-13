using EasyOcrNet;
using SkiaSharp;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

static Charset DeriveCharset(string fileName)
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

var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
var examplesDir = Path.Combine(baseDir, "examples");
var modelDir = Path.Combine(baseDir, "models");
var extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };
foreach (var file in Directory.GetFiles(examplesDir).Where(f => extensions.Contains(Path.GetExtension(f))))
{
    using var bmp = SKBitmap.Decode(file);
    var charset = DeriveCharset(Path.GetFileName(file));
    using var ocr = new EasyOcr(modelDir, charset);
    var text = string.Join(" ", ocr.Read(bmp).Select(r => r.Text));
    var outPath = Path.ChangeExtension(file, ".txt");
    File.WriteAllText(outPath, text);
    Console.WriteLine($"{Path.GetFileName(file)} -> '{text}'");
}
