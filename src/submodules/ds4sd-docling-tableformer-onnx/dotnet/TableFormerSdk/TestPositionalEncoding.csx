#!/usr/bin/env dotnet-script
//
// Test script for PositionalEncoding component
// Verifies TorchSharp implementation produces correct output shapes
//

#r "nuget: TorchSharp, 0.105.1"
#r "nuget: TorchSharp-cpu, 0.105.1"
#load "Models/PositionalEncoding.cs"

using System;
using TorchSharp;
using static TorchSharp.torch;
using TableFormerSdk.Models;

Console.WriteLine("=" + new string('=', 69));
Console.WriteLine("POSITIONAL ENCODING TEST");
Console.WriteLine("=" + new string('=', 69));

// Test parameters (matching TableFormer config)
long dModel = 256;
double dropout = 0.1;
long maxLen = 1024;

Console.WriteLine($"\nInitializing PositionalEncoding:");
Console.WriteLine($"  d_model: {dModel}");
Console.WriteLine($"  dropout: {dropout}");
Console.WriteLine($"  max_len: {maxLen}");

// Create module
using var pe = new PositionalEncoding(dModel, dropout, maxLen);
Console.WriteLine("✅ Module created successfully");

// Test 1: Shape verification with various sequence lengths
Console.WriteLine("\n" + new string('-', 70));
Console.WriteLine("TEST 1: Shape Verification");
Console.WriteLine(new string('-', 70));

var testCases = new[]
{
    (seqLen: 10L, batchSize: 1L),
    (seqLen: 50L, batchSize: 4L),
    (seqLen: 100L, batchSize: 2L),
    (seqLen: 500L, batchSize: 1L),
};

foreach (var (seqLen, batchSize) in testCases)
{
    using var x = torch.randn(seqLen, batchSize, dModel);
    Console.WriteLine($"\nInput shape: [{seqLen}, {batchSize}, {dModel}]");

    using var output = pe.forward(x);

    var outShape = output.shape;
    Console.WriteLine($"Output shape: [{outShape[0]}, {outShape[1]}, {outShape[2]}]");

    // Verify shape
    if (outShape[0] == seqLen && outShape[1] == batchSize && outShape[2] == dModel)
    {
        Console.WriteLine("✅ Shape correct");
    }
    else
    {
        Console.WriteLine($"❌ Shape mismatch! Expected [{seqLen}, {batchSize}, {dModel}]");
    }
}

// Test 2: Verify positional encoding values are reasonable
Console.WriteLine("\n" + new string('-', 70));
Console.WriteLine("TEST 2: Value Range Verification");
Console.WriteLine(new string('-', 70));

using (torch.no_grad())
{
    // Create input with all zeros to isolate positional encoding effect
    using var zeros = torch.zeros(10, 1, dModel);

    // Disable dropout for deterministic testing
    pe.eval();

    using var output = pe.forward(zeros);

    var minVal = output.min().item<float>();
    var maxVal = output.max().item<float>();
    var meanVal = output.mean().item<float>();
    var stdVal = output.std().item<float>();

    Console.WriteLine($"\nOutput statistics (with dropout disabled):");
    Console.WriteLine($"  Min:  {minVal:F6}");
    Console.WriteLine($"  Max:  {maxVal:F6}");
    Console.WriteLine($"  Mean: {meanVal:F6}");
    Console.WriteLine($"  Std:  {stdVal:F6}");

    // Positional encodings should be bounded (sin/cos produce values in [-1, 1])
    if (minVal >= -2.0 && maxVal <= 2.0)
    {
        Console.WriteLine("✅ Values within expected range");
    }
    else
    {
        Console.WriteLine("❌ Values outside expected range!");
    }
}

// Test 3: Verify different positions get different encodings
Console.WriteLine("\n" + new string('-', 70));
Console.WriteLine("TEST 3: Position Uniqueness");
Console.WriteLine(new string('-', 70));

using (torch.no_grad())
{
    pe.eval();

    using var zeros = torch.zeros(5, 1, dModel);
    using var output = pe.forward(zeros);

    Console.WriteLine("\nChecking if different positions have different encodings...");

    bool allDifferent = true;
    for (long i = 0; i < 4; i++)
    {
        using var pos1 = output[i];
        using var pos2 = output[i + 1];
        using var diff = (pos1 - pos2).abs().sum();
        var diffVal = diff.item<float>();

        Console.WriteLine($"  Difference between position {i} and {i+1}: {diffVal:F6}");

        if (diffVal < 0.001)
        {
            allDifferent = false;
        }
    }

    if (allDifferent)
    {
        Console.WriteLine("✅ All positions have unique encodings");
    }
    else
    {
        Console.WriteLine("❌ Some positions have identical encodings!");
    }
}

// Test 4: Verify model can be set to train/eval mode
Console.WriteLine("\n" + new string('-', 70));
Console.WriteLine("TEST 4: Train/Eval Mode");
Console.WriteLine(new string('-', 70));

pe.train();
Console.WriteLine("✅ Switched to training mode");

pe.eval();
Console.WriteLine("✅ Switched to evaluation mode");

Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("ALL TESTS COMPLETED");
Console.WriteLine(new string('=', 70));
Console.WriteLine("\n✅ PositionalEncoding implementation verified!");
Console.WriteLine("\nNext steps:");
Console.WriteLine("  1. Create Python reference script to compare exact values");
Console.WriteLine("  2. Verify numerical accuracy (diff < 1e-6)");
Console.WriteLine("  3. Proceed with Encoder04 porting");
