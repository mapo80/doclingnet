#!/usr/bin/env dotnet-script
// Test script to verify PositionalEncoding fix
#r "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/bin/Debug/net9.0/TableFormerSdk.dll"
#r "nuget: TorchSharp-cpu, 0.105.1"
#r "nuget: SkiaSharp, 3.116.1"

using TableFormerSdk.Models;
using TorchSharp;
using static TorchSharp.torch;
using SkiaSharp;

Console.WriteLine("=== Testing PositionalEncoding Fix ===\n");

// Load model
var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/config.json";
var weightsPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/model.safetensors";

Console.WriteLine("Loading model...");
var model = TableModel04Loader.Load(weightsPath, configPath);
model.eval();

// Verify PE buffer shape
var peModule = model.GetTagTransformer().GetPositionalEncoding();
var peBuffer = peModule.get_buffer("pe");
Console.WriteLine($"✓ PE buffer shape: [{peBuffer.shape[0]}, {peBuffer.shape[1]}, {peBuffer.shape[2]}]");
Console.WriteLine($"  Expected: [1024, 1, 512]");
Console.WriteLine($"  Match: {peBuffer.shape[0] == 1024 && peBuffer.shape[1] == 1 && peBuffer.shape[2] == 512}\n");

// Load test image
var imagePath = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/2305.03393v1-pg9-img.png";
Console.WriteLine($"Loading test image: {imagePath}");

using var bitmap = SKBitmap.Decode(imagePath);
using var resized = bitmap.Resize(new SKImageInfo(896, 896), SKFilterQuality.High);

// Convert to tensor: (1, 3, 896, 896)
var imageData = new float[1 * 3 * 896 * 896];
int idx = 0;

// Mean and std for normalization
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

// Run inference
Console.WriteLine("\nRunning inference (first 10 steps)...\n");

using var _ = torch.no_grad();
var result = model.forward(imageTensor);

// Analyze results
var tokens = result.DecodedTags;
var tagList = string.Join(" ", tokens.Take(Math.Min(20, tokens.Count)));

Console.WriteLine($"✓ Generated {tokens.Count} tokens");
Console.WriteLine($"  First 20 tokens: {tagList}");
Console.WriteLine($"  Contains <end>: {tokens.Contains("<end>")}");
Console.WriteLine($"  Unique tokens: {tokens.Distinct().Count()}");

// Check for repetitive pattern
bool hasRepetition = false;
if (tokens.Count >= 10)
{
    // Check if we have more than 5 consecutive identical tokens
    for (int i = 0; i < Math.Min(tokens.Count - 5, 100); i++)
    {
        if (tokens[i] == tokens[i+1] && tokens[i] == tokens[i+2] &&
            tokens[i] == tokens[i+3] && tokens[i] == tokens[i+4])
        {
            hasRepetition = true;
            Console.WriteLine($"  ⚠️ Repetitive pattern detected at position {i}: {tokens[i]}");
            break;
        }
    }
}

if (!hasRepetition && tokens.Contains("<end>"))
{
    Console.WriteLine("\n✅ SUCCESS: Model generates diverse tokens and terminates with <end>");
    return 0;
}
else if (hasRepetition)
{
    Console.WriteLine("\n❌ FAILURE: Repetitive pattern still present (PE fix may not be sufficient)");
    return 1;
}
else
{
    Console.WriteLine("\n⚠️  PARTIAL: No repetitive pattern but <end> not generated (check max_steps)");
    return 2;
}
