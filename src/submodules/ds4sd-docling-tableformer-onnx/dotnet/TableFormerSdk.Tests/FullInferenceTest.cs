using System;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;
using Xunit.Abstractions;

namespace TableFormerSdk.Tests
{
    public class FullInferenceTest : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private TableModel04? _model;

        public FullInferenceTest(ITestOutputHelper output)
        {
            _output = output;
        }

        public void Dispose()
        {
            _model?.Dispose();
            GC.SuppressFinalize(this);
        }

        [Fact]
        public void FullInference_WithRealModel_GeneratesDiverseTokens()
        {
            // Paths to real model
            var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json";
            var weightsPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors";
            var imagePath = "/Users/politom/Documents/Workspace/personal/doclingnet/dataset/2305.03393v1-pg9-img.png";

            // Skip test if files not available
            if (!File.Exists(configPath) || !File.Exists(weightsPath) || !File.Exists(imagePath))
            {
                _output.WriteLine("⚠️  Skipping test: Required files not found");
                return;
            }

            _output.WriteLine("=== Full End-to-End Inference Test ===\n");

            // Load model
            _output.WriteLine("Loading model...");
            var config = TableModel04Config.FromJsonFile(configPath);
            _model = TableModel04.FromSafeTensors(config, weightsPath);
            _model.eval();
            _output.WriteLine("✓ Model loaded\n");

            // Load and preprocess image
            _output.WriteLine($"Loading image: {imagePath}");
            using var bitmap = SkiaSharp.SKBitmap.Decode(imagePath);
            using var resized = bitmap.Resize(new SkiaSharp.SKImageInfo(896, 896), SkiaSharp.SKSamplingOptions.Default);

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
            _output.WriteLine($"✓ Image preprocessed: {imageTensor.shape[0]}x{imageTensor.shape[1]}x{imageTensor.shape[2]}x{imageTensor.shape[3]}\n");

            // Run inference
            _output.WriteLine("Running inference...\n");
            using var _ = torch.no_grad();

            // Enable debug logging
            Environment.SetEnvironmentVariable("DEBUG_LOGITS", "1");

            var result = _model.forward(imageTensor);

            // Analyze results
            var tokens = result.Sequence;
            var tokenStrings = tokens.Select(t => config.WordMapTag.FirstOrDefault(kvp => kvp.Value == t).Key ?? $"idx{t}").ToList();

            _output.WriteLine($"✓ Generated {tokens.Count} tokens\n");

            // Print first 50 tokens
            var first50 = string.Join(" ", tokenStrings.Take(Math.Min(50, tokenStrings.Count)));
            _output.WriteLine($"First 50 tokens:");
            _output.WriteLine($"  {first50}");
            if (tokenStrings.Count > 50)
            {
                _output.WriteLine($"  ... (showing first 50 of {tokenStrings.Count} total)\n");
            }
            else
            {
                _output.WriteLine("");
            }

            // Analysis
            var hasEnd = tokenStrings.Contains("<end>");
            var uniqueCount = tokenStrings.Distinct().Count();

            _output.WriteLine($"Analysis:");
            _output.WriteLine($"  Total tokens: {tokenStrings.Count}");
            _output.WriteLine($"  Unique tokens: {uniqueCount}");
            _output.WriteLine($"  Contains <end>: {hasEnd}");

            // Check for repetitive pattern (5+ consecutive identical tokens)
            bool hasRepetition = false;
            int repetitionStart = -1;
            string? repetitiveToken = null;
            int consecutiveCount = 0;

            if (tokenStrings.Count >= 5)
            {
                for (int i = 0; i < Math.Min(tokenStrings.Count - 5, 100); i++)
                {
                    if (tokenStrings[i] == tokenStrings[i + 1] && tokenStrings[i] == tokenStrings[i + 2] &&
                        tokenStrings[i] == tokenStrings[i + 3] && tokenStrings[i] == tokenStrings[i + 4])
                    {
                        hasRepetition = true;
                        repetitionStart = i;
                        repetitiveToken = tokenStrings[i];

                        // Count consecutive repetitions
                        consecutiveCount = 1;
                        for (int j = i + 1; j < tokenStrings.Count && tokenStrings[j] == repetitiveToken; j++)
                        {
                            consecutiveCount++;
                        }

                        _output.WriteLine($"  ⚠️ Repetitive pattern at position {i}: '{repetitiveToken}'");
                        _output.WriteLine($"     Repeated {consecutiveCount} times consecutively");
                        break;
                    }
                }
            }

            if (!hasRepetition)
            {
                _output.WriteLine($"  ✓ No repetitive patterns detected");
            }

            _output.WriteLine("");

            // Verdict
            _output.WriteLine("=== Verdict ===");
            if (!hasRepetition && hasEnd)
            {
                _output.WriteLine("✅ SUCCESS: Model generates diverse tokens and terminates with <end>");
                _output.WriteLine("   The PositionalEncoding and BBoxDecoder fixes have resolved the issues!");
            }
            else if (hasRepetition)
            {
                _output.WriteLine("❌ FAILURE: Repetitive pattern still detected");
                _output.WriteLine("   Additional investigation needed.");
                Assert.Fail($"Repetitive token generation: '{repetitiveToken}' repeated {consecutiveCount} times");
            }
            else if (!hasEnd)
            {
                _output.WriteLine("⚠️  PARTIAL: No repetitive pattern but <end> not generated");
                _output.WriteLine($"   Max steps reached ({config.MaxSteps}). Try increasing MaxSteps.");
            }

            // Assertions
            Assert.True(uniqueCount > 1, "Should generate more than one unique token");
            Assert.False(hasRepetition, "Should not have repetitive patterns");

            // Dispose result tensors
            result.BBoxClasses.Dispose();
            result.BBoxCoords.Dispose();
        }
    }
}
