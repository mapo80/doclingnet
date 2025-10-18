#!/usr/bin/env dotnet-script
#r "nuget: SixLabors.ImageSharp, 3.1.5"
#r "nuget: Newtonsoft.Json, 13.0.3"

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Newtonsoft.Json;

var imagePath = "/Users/politom/Documents/Workspace/personal/doclingnet/src/submodules/ds4sd-docling-tableformer-onnx/dataset/FinTabNet/benchmark/HAL.2004.page_82.pdf_125317.png";

// Load and resize image
using var image = Image.Load<Rgb24>(imagePath);
Console.WriteLine($"Original size: {image.Width}x{image.Height}");

image.Mutate(x => x.Resize(448, 448));
Console.WriteLine($"Resized to: {image.Width}x{image.Height}");

// ImageNet normalization constants
var mean = new[] { 0.94247851f, 0.94254675f, 0.94292611f };
var std = new[] { 0.17910956f, 0.17940403f, 0.17931663f };

var height = image.Height;
var width = image.Width;

// Sample pixels before normalization
var pixel00 = image[0, 0];
var pixel224 = image[224, 224];

Console.WriteLine($"\nSample pixel at (0,0): R={pixel00.R}, G={pixel00.G}, B={pixel00.B}");
Console.WriteLine($"Sample pixel at (224,224): R={pixel224.R}, G={pixel224.G}, B={pixel224.B}");

// Create normalized tensor data
var channel0 = new List<float>();
var channel1 = new List<float>();
var channel2 = new List<float>();

// Channel R
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        var pixel = image[x, y];
        channel0.Add((pixel.R / 255.0f - mean[0]) / std[0]);
    }
}

// Channel G
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        var pixel = image[x, y];
        channel1.Add((pixel.G / 255.0f - mean[1]) / std[1]);
    }
}

// Channel B
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        var pixel = image[x, y];
        channel2.Add((pixel.B / 255.0f - mean[2]) / std[2]);
    }
}

Console.WriteLine($"\nNormalized sample at (0,0):");
Console.WriteLine($"  R: {channel0[0]:F6}");
Console.WriteLine($"  G: {channel1[0]:F6}");
Console.WriteLine($"  B: {channel2[0]:F6}");

var idx224 = 224 * width + 224;
Console.WriteLine($"\nNormalized sample at (224,224):");
Console.WriteLine($"  R: {channel0[idx224]:F6}");
Console.WriteLine($"  G: {channel1[idx224]:F6}");
Console.WriteLine($"  B: {channel2[idx224]:F6}");

// Save first 100 values
var output = new
{
    shape = new[] { 3, height, width },
    channel_0_first_100 = channel0.Take(100).ToArray(),
    channel_1_first_100 = channel1.Take(100).ToArray(),
    channel_2_first_100 = channel2.Take(100).ToArray(),
    mean = mean,
    std = std
};

var outputPath = "/Users/politom/Documents/Workspace/personal/doclingnet/debug/csharp_preprocessed_image.json";
File.WriteAllText(outputPath, JsonConvert.SerializeObject(output, Formatting.Indented));
Console.WriteLine($"\nSaved to: {outputPath}");
