using System;
using System.IO;
using System.Threading.Tasks;
using EasyOcrNet;
using EasyOcrNet.Assets;
using EasyOcrNet.Models;
using SkiaSharp;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("Testing EasyOCR auto-download...");

        var modelDirectory = Path.Combine(AppContext.BaseDirectory, "models");
        Directory.CreateDirectory(modelDirectory);

        Console.WriteLine($"Model directory: {modelDirectory}");

        var detectorPath = Path.Combine(modelDirectory, "detection.onnx");
        var recognizerPath = Path.Combine(modelDirectory, "english_g2_rec.onnx");

        // Test auto-download
        var options = new GithubReleaseOptions("mapo80/easyocrnet", "v1.0.0");

        Console.WriteLine("Downloading detection model...");
        await OcrReleaseDownloader.EnsureModelAsync(detectorPath, options, msg => Console.WriteLine($"  {msg}"));

        Console.WriteLine("Downloading recognition model...");
        await OcrReleaseDownloader.EnsureModelAsync(recognizerPath, options, msg => Console.WriteLine($"  {msg}"));

        Console.WriteLine("Models downloaded successfully!");
        Console.WriteLine($"Detection model exists: {File.Exists(detectorPath)}");
        Console.WriteLine($"Recognition model exists: {File.Exists(recognizerPath)}");
    }
}
