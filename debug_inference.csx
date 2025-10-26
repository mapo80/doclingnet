#!/usr/bin/env dotnet-script
// Debug script to investigate repetitive token generation
#r "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/bin/Debug/net9.0/TableFormerSdk.dll"
#r "nuget: TorchSharp-cpu, 0.105.1"
#r "nuget: SkiaSharp, 3.116.1"

using TableFormerSdk.Models;
using TorchSharp;
using static TorchSharp.torch;
using SkiaSharp;
using System.Linq;

Console.WriteLine("=== Debug Inference Investigation ===\n");

// Load model
var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json";
var weightsPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors";
var imagePath = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/2305.03393v1-pg9-img.png";

Console.WriteLine("Loading model...");
var config = TableModel04Config.FromJsonFile(configPath);
var model = TableModel04.FromSafeTensors(config, weightsPath);
model.eval();
Console.WriteLine("✓ Model loaded\n");

// Load image
Console.WriteLine($"Loading image: {imagePath}");
using var bitmap = SKBitmap.Decode(imagePath);
using var resized = bitmap.Resize(new SKImageInfo(896, 896), SKSamplingOptions.Default);

var imageData = new float[1 * 3 * 896 * 896];
int idx = 0;
float[] mean = { 0.485f, 0.456f, 0.406f };
float[] std = { 0.229f, 0.224f, 0.225f };

for (int c = 0; c < 3; c++)
{
    for (int h = 0; h < 896; h++)
    {
        for (int w = 0; w < 896; w++)
        {
            var pixel = resized.GetPixel(w, h);
            float value = c == 0 ? pixel.Red / 255.0f : c == 1 ? pixel.Green / 255.0f : pixel.Blue / 255.0f;
            imageData[idx++] = (value - mean[c]) / std[c];
        }
    }
}

using var imageTensor = torch.tensor(imageData, new long[] { 1, 3, 896, 896 });
Console.WriteLine($"✓ Image tensor shape: {imageTensor.shape[0]}x{imageTensor.shape[1]}x{imageTensor.shape[2]}x{imageTensor.shape[3]}\n");

// Run just 10 steps to debug
Console.WriteLine("Running first 10 inference steps with debugging...\n");

using var _ = torch.no_grad();

// Set environment variable for debug mode
Environment.SetEnvironmentVariable("TABLEFORMER_DEBUG", "1");

var result = model.forward(imageTensor);

var tokens = result.Sequence;
var tokenStrings = tokens.Select(t => config.WordMapTag.FirstOrDefault(kvp => kvp.Value == t).Key ?? $"idx{t}").ToList();

Console.WriteLine($"\n✓ Generated {tokens.Count} tokens");
Console.WriteLine($"  First 50: {string.Join(" ", tokenStrings.Take(50))}");
if (tokenStrings.Count > 50)
{
    Console.WriteLine($"  ... (showing first 50 of {tokenStrings.Count})");
}
Console.WriteLine($"  Unique tokens: {tokenStrings.Distinct().Count()}");
Console.WriteLine($"  Contains <end>: {tokenStrings.Contains("<end>")}");

// Check for repetition
bool hasRep = false;
if (tokenStrings.Count >= 5)
{
    for (int i = 0; i < Math.Min(tokenStrings.Count - 5, 100); i++)
    {
        if (tokenStrings[i] == tokenStrings[i+1] && tokenStrings[i] == tokenStrings[i+2] &&
            tokenStrings[i] == tokenStrings[i+3] && tokenStrings[i] == tokenStrings[i+4])
        {
            hasRep = true;
            Console.WriteLine($"  ⚠️  Repetition at position {i}: '{tokenStrings[i]}'");
            break;
        }
    }
}

if (!hasRep)
{
    Console.WriteLine("  ✓ No repetition");
}

Console.WriteLine("\n=== Analysis ===");
if (!hasRep && tokenStrings.Contains("<end>"))
{
    Console.WriteLine("✅ SUCCESS");
    return 0;
}
else if (hasRep)
{
    Console.WriteLine("❌ FAILURE: Repetition detected");
    return 1;
}
else
{
    Console.WriteLine("⚠️  PARTIAL: No <end> token");
    return 2;
}
