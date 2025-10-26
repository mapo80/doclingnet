#!/usr/bin/env dotnet-script
#r "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/bin/Debug/net9.0/TableFormerSdk.dll"
#r "nuget: TorchSharp-cpu, 0.105.1"
#r "nuget: SkiaSharp, 2.88.8"
#r "nuget: System.Text.Json, 8.0.3"

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TableFormerSdk.Models;
using TorchSharp;
using static TorchSharp.torch;
using SkiaSharp;

Console.WriteLine("=== C# TableFormer Inference Test (Refactored) ===\n");

// Paths
var weightsPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors";
var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json";
var imagePath = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark/HAL.2004.page_82.pdf_125315.png";

// Check files exist
if (!File.Exists(weightsPath)) { Console.WriteLine($"❌ Weights not found: {weightsPath}"); return 1; }
if (!File.Exists(configPath)) { Console.WriteLine($"❌ Config not found: {configPath}"); return 1; }
if (!File.Exists(imagePath)) { Console.WriteLine($"❌ Image not found: {imagePath}"); return 1; }

Console.WriteLine($"✓ Files found");
Console.WriteLine($"  Weights: {weightsPath}");
Console.WriteLine($"  Config: {configPath}");
Console.WriteLine($"  Image: {imagePath}\n");

// Load model
Console.WriteLine("Loading model...");
var modelConfig = TableModel04Config.FromJsonFile(configPath);
var model = TableModel04.FromSafeTensors(modelConfig, weightsPath);
model.eval();
Console.WriteLine("✓ Model loaded\n");

// --- Main processing ---
SKBitmap bitmap = null;
SKBitmap resized = null;
Tensor imageTensor = null;
int finalExitCode = -1;

try
{
    // Create reverse map for decoding tokens
    var reverseWordMap = model.GetConfig().WordMapTag.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

    // Verify PE shape
    var peModule = model.GetTagTransformer().GetPositionalEncoding();
    var peBuffer = peModule.get_buffer("pe");
    Console.WriteLine($"PositionalEncoding buffer shape: [{peBuffer.shape[0]}, {peBuffer.shape[1]}, {peBuffer.shape[2]}]");
    Console.WriteLine($"  Expected: [1024, 1, 512]");
    var shapeOk = peBuffer.shape[0] == 1024 && peBuffer.shape[1] == 1 && peBuffer.shape[2] == 512;
    Console.WriteLine($"  Match: {shapeOk}");
    if (!shapeOk)
    {
        Console.WriteLine("❌ PE shape mismatch! Fix not applied correctly.");
        return 1;
    }
    Console.WriteLine();

    // Load and preprocess image
    Console.WriteLine($"Loading image...");
    bitmap = SKBitmap.Decode(imagePath);
    resized = bitmap.Resize(new SKImageInfo(448, 448), SKFilterQuality.High);

    var imageData = new float[1 * 3 * 448 * 448];
    int idx = 0;

    float[] mean = { 0.485f, 0.456f, 0.406f };
    float[] std = { 0.229f, 0.224f, 0.225f };

    for (int c = 0; c < 3; c++)
    {
        for (int h = 0; h < 448; h++)
        {
            for (int w = 0; w < 448; w++)
            {
                var pixel = resized.GetPixel(w, h);
                float value = c == 0 ? pixel.Red / 255.0f : c == 1 ? pixel.Green / 255.0f : pixel.Blue / 255.0f;
                imageData[idx++] = (value - mean[c]) / std[c];
            }
        }
    }

    imageTensor = torch.tensor(imageData, new long[] { 1, 3, 448, 448 });
    Console.WriteLine($"✓ Image preprocessed: shape=[{imageTensor.shape[0]}, {imageTensor.shape[1]}, {imageTensor.shape[2]}, {imageTensor.shape[3]}]\n");

    // Run inference
    Console.WriteLine("Running inference (max 100 steps for test)...\n");

    TableModel04Result result;
    using (torch.no_grad())
    {
        // Set max_steps to 100 for testing
        model.SetMaxSteps(100);
        result = model.forward(imageTensor);
    }

    // Analyze results
    var tokenIndices = result.Sequence;
    var tokens = tokenIndices.Select(i => reverseWordMap.GetValueOrDefault(i, $"UNK_{i}")).ToList();

    Console.WriteLine($"✓ Generated {tokens.Count} tokens\n");
    Console.WriteLine($"First 50 tokens:");
    var first50 = string.Join(" ", tokens.Take(Math.Min(50, tokens.Count)));
    Console.WriteLine($"  {first50}");
    if (tokens.Count > 50)
    {
        Console.WriteLine($"  ... (showing first 50 of {tokens.Count} total)\n");
    }
    else
    {
        Console.WriteLine();
    }

    var hasEnd = tokens.Contains("<end>");
    var uniqueCount = tokens.Distinct().Count();

    Console.WriteLine($"Analysis:");
    Console.WriteLine($"  Total tokens: {tokens.Count}");
    Console.WriteLine($"  Unique tokens: {uniqueCount}");
    Console.WriteLine($"  Contains <end>: {hasEnd}");

    // Check for repetitive pattern (5+ consecutive identical tokens)
    bool hasRepetition = false;
    int repetitionStart = -1;
    string repetitiveToken = null;

    if (tokens.Count >= 5)
    {
        for (int i = 0; i < Math.Min(tokens.Count - 5, 100); i++)
        {
            if (tokens[i] == tokens[i+1] && tokens[i] == tokens[i+2] &&
                tokens[i] == tokens[i+3] && tokens[i] == tokens[i+4])
            {
                hasRepetition = true;
                repetitionStart = i;
                repetitiveToken = tokens[i];
                Console.WriteLine($"  ⚠️ Repetitive pattern at position {i}: '{repetitiveToken}'");

                // Count how many consecutive times
                int count = 1;
                for (int j = i + 1; j < tokens.Count && tokens[j] == repetitiveToken; j++)
                {
                    count++;
                }
                Console.WriteLine($"     Repeated {count} times consecutively");
                break;
            }
        }
    }

    if (!hasRepetition)
    {
        Console.WriteLine($"  ✓ No repetitive patterns detected");
    }

    // Save output
    var output = new
    {
        tokens = tokens.ToArray(),
        token_count = tokens.Count,
        has_end_token = hasEnd,
        unique_token_count = uniqueCount,
        has_repetition = hasRepetition,
        repetition_start = repetitionStart,
        repetitive_token = repetitiveToken,
        first_50_tokens = tokens.Take(50).ToArray(),
        pe_shape_correct = shapeOk
    };

    var outputPath = "/Users/politom/Documents/Workspace/personal/doclingnet/csharp_inference_output.json";
    File.WriteAllText(outputPath, JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"\n✓ Output saved to: {outputPath}");

    // Verdict
    Console.WriteLine("\n=== Verdict ===");
    if (!hasRepetition && hasEnd)
    {
        Console.WriteLine("✅ SUCCESS: Model generates diverse tokens and terminates with <end>");
        Console.WriteLine("   The PositionalEncoding and Orchestration fixes appear to have resolved the issue!");
        finalExitCode = 0;
    }
    else if (hasRepetition)
    {
        Console.WriteLine("❌ FAILURE: Repetitive pattern still detected");
        Console.WriteLine("   The issue may not be fully resolved or there are other problems.");
        finalExitCode = 1;
    }
    else if (!hasEnd)
    {
        Console.WriteLine("⚠️  PARTIAL: No repetitive pattern but <end> not generated");
        Console.WriteLine("   This could be due to max_steps limit (100). Try increasing it.");
        finalExitCode = 2;
    }
    else
    {
        Console.WriteLine("🤔 UNEXPECTED: Please review the output manually.");
        finalExitCode = 3;
    }
}
finally
{
    bitmap?.Dispose();
    resized?.Dispose();
    imageTensor?.Dispose();
    model?.Dispose();
}

return finalExitCode;
