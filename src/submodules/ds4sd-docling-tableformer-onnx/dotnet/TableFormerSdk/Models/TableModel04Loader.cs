//
// Copyright IBM Corp. 2024 - 2024
// SPDX-License-Identifier: MIT
//

using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TableFormerSdk.Utils;

namespace TableFormerSdk.Models;

/// <summary>
/// Loader for TableModel04 weights from SafeTensors format.
/// </summary>
public static class TableModel04Loader
{
    /// <summary>
    /// Load weights from SafeTensors file into TableModel04.
    /// </summary>
    public static void LoadWeights(TableModel04 model, string safetensorsPath)
    {
        Console.WriteLine("\n" + new string('=', 70));
        Console.WriteLine("LOADING WEIGHTS FROM SAFETENSORS");
        Console.WriteLine(new string('=', 70) + "\n");

        using var reader = new SafeTensorsReader(safetensorsPath);

        // Get all named parameters from model
        var modelParams = model.named_parameters().ToList();
        Console.WriteLine($"\nModel has {modelParams.Count} parameters");

        // DEBUG: Save ALL names to files for analysis
        var debugDir = "/Users/politom/Documents/Workspace/personal/doclingnet/debug";
        System.IO.Directory.CreateDirectory(debugDir);

        System.IO.File.WriteAllLines(
            System.IO.Path.Combine(debugDir, "safetensors_names.txt"),
            reader.TensorNames);

        System.IO.File.WriteAllLines(
            System.IO.Path.Combine(debugDir, "torchsharp_names.txt"),
            modelParams.Select(p => p.name));

        Console.WriteLine($"\nSaved tensor names to {debugDir}/");
        Console.WriteLine($"  SafeTensors: {reader.TensorNames.Count()} tensors");
        Console.WriteLine($"  TorchSharp: {modelParams.Count} parameters");

        // Create mapping: SafeTensors name -> TorchSharp parameter name
        var nameMap = CreateNameMapping();

        int loaded = 0;
        int skipped = 0;
        var missingInFile = new List<string>();
        var missingInModel = new List<string>();
        var loadedParams = new HashSet<string>();

        // Load each tensor from SafeTensors
        foreach (var tensorName in reader.TensorNames)
        {
            // Try to find corresponding model parameter
            string? modelParamName = null;

            // Direct match from explicit map
            if (nameMap.TryGetValue(tensorName, out var mappedName))
            {
                modelParamName = mappedName;
            }
            // Try exact match
            else if (modelParams.Any(p => p.name == tensorName))
            {
                modelParamName = tensorName;
            }
            // Try transformed name
            else
            {
                modelParamName = TransformName(tensorName);
                if (modelParamName != null && !modelParams.Any(p => p.name == modelParamName))
                {
                    modelParamName = null;
                }
            }

            if (modelParamName == null)
            {
                // DEBUG: Show first few transformation failures
                if (missingInModel.Count < 10 && !tensorName.Contains("num_batches_tracked") &&
                    !tensorName.Contains("running_mean") && !tensorName.Contains("running_var"))
                {
                    var transformed = TransformName(tensorName);
                    Console.WriteLine($"  ✗ No match for: {tensorName}");
                    Console.WriteLine($"     Transformed to: {transformed}");
                }
                missingInModel.Add(tensorName);
                continue;
            }

            // Find the parameter in model
            var paramMatch = modelParams.FirstOrDefault(p => p.name == modelParamName);
            if (paramMatch.parameter is null)
            {
                missingInModel.Add(tensorName);
                continue;
            }
            var param = paramMatch.parameter;

            // Load tensor from file
            try
            {
                using var fileTensor = reader.GetTensor(tensorName);

                // Check shape compatibility
                if (!param.shape.SequenceEqual(fileTensor.shape))
                {
                    Console.WriteLine($"  ⚠️  Shape mismatch for {tensorName}:");
                    Console.WriteLine($"      Model: {string.Join("x", param.shape)}");
                    Console.WriteLine($"      File:  {string.Join("x", fileTensor.shape)}");
                    skipped++;
                    continue;
                }

                // Copy weights using no_grad() to avoid gradient tracking issues
                using (torch.no_grad())
                {
                    param.copy_(fileTensor);
                }
                loaded++;
                loadedParams.Add(modelParamName);

                if (loaded <= 5 || loaded % 50 == 0)
                {
                    Console.WriteLine($"  ✓ Loaded {tensorName} → {modelParamName}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ✗ Error loading {tensorName}: {ex.Message}");
                skipped++;
            }
        }

        // Check for parameters that weren't loaded
        foreach (var param in modelParams)
        {
            if (!loadedParams.Contains(param.name))
            {
                missingInFile.Add(param.name);
            }
        }

        // IMPORTANT: Load BatchNorm buffers (running_mean, running_var)
        // These are not in named_parameters() but are critical for eval mode!
        Console.WriteLine("\n" + new string('=', 70));
        Console.WriteLine("LOADING BATCHNORM BUFFERS (running_mean, running_var)");
        Console.WriteLine(new string('=', 70));

        int buffersLoaded = 0;
        var availableTensors = new HashSet<string>(reader.TensorNames);

        foreach (var (name, buffer) in model.named_buffers())
        {
            // Skip if not running_mean or running_var
            if (!name.EndsWith("running_mean") && !name.EndsWith("running_var"))
                continue;

            // Transform name to SafeTensors format
            var safetensorsName = name
                .Replace("_encoder.resnet.", "_encoder._resnet.")
                .Replace("_tagTransformer", "_tag_transformer")
                .Replace("_bboxDecoder", "_bbox_decoder");

            if (availableTensors.Contains(safetensorsName))
            {
                try
                {
                    using var fileTensor = reader.GetTensor(safetensorsName);
                    using (torch.no_grad())
                    {
                        buffer.copy_(fileTensor);
                    }
                    buffersLoaded++;
                    if (buffersLoaded <= 5)
                    {
                        Console.WriteLine($"  ✓ Loaded buffer {safetensorsName} → {name}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ✗ Error loading buffer {safetensorsName}: {ex.Message}");
                }
            }
        }
        Console.WriteLine($"✅ Loaded {buffersLoaded} BatchNorm buffers");

        // Summary
        Console.WriteLine("\n" + new string('=', 70));
        Console.WriteLine("WEIGHT LOADING SUMMARY");
        Console.WriteLine(new string('=', 70));
        Console.WriteLine($"✅ Loaded:  {loaded} parameters");
        Console.WriteLine($"⚠️  Skipped: {skipped} parameters (shape mismatch or errors)");
        Console.WriteLine($"❌ Missing in model: {missingInModel.Count} tensors from file");
        Console.WriteLine($"❌ Missing in file:  {missingInFile.Count} model parameters");

        if (missingInModel.Count > 0 && missingInModel.Count <= 10)
        {
            Console.WriteLine($"\nMissing in model:");
            foreach (var name in missingInModel.Take(10))
            {
                Console.WriteLine($"  - {name}");
            }
        }

        if (missingInFile.Count > 0)
        {
            Console.WriteLine($"\nMissing in file:");
            foreach (var name in missingInFile)
            {
                Console.WriteLine($"  - {name}");
            }
        }

        Console.WriteLine("\n" + new string('=', 70) + "\n");

        if (loaded == 0)
        {
            throw new InvalidOperationException("No weights were loaded! Check name mapping.");
        }
    }

    /// <summary>
    /// Create mapping from SafeTensors names to TorchSharp parameter names.
    /// SafeTensors uses Python naming convention, TorchSharp uses C# convention.
    /// </summary>
    private static Dictionary<string, string> CreateNameMapping()
    {
        var map = new Dictionary<string, string>();

        // Most mappings are handled by TransformName()
        // Only add explicit overrides here if needed

        return map;
    }

    /// <summary>
    /// Transform SafeTensors parameter name to TorchSharp convention.
    /// Returns null if the parameter should be skipped (e.g., BatchNorm tracking params).
    /// </summary>
    private static string? TransformName(string safetensorsName)
    {
        // Skip BatchNorm tracking parameters not exposed by TorchSharp
        if (safetensorsName.Contains("num_batches_tracked") ||
            safetensorsName.Contains("running_mean") ||
            safetensorsName.Contains("running_var"))
        {
            return null;
        }

        var name = safetensorsName;

        // Transform module names to TorchSharp conventions:
        // 1. _tag_transformer -> _tagTransformer (camelCase)
        name = name.Replace("_tag_transformer", "_tagTransformer");

        // 2. _bbox_decoder -> _bboxDecoder (camelCase)
        name = name.Replace("_bbox_decoder", "_bboxDecoder");

        // 3. Remove underscore prefix ONLY from attention sub-modules
        // ._encoder_att -> .encoder_att
        // ._tag_decoder_att -> .tag_decoder_att
        // ._language_att -> .language_att
        // ._full_att -> .full_att
        // But KEEP: ._embedding, ._encoder, ._decoder, ._encoderProjection
        name = name.Replace("._encoder_att", ".encoder_att");
        name = name.Replace("._tag_decoder_att", ".tag_decoder_att");
        name = name.Replace("._language_att", ".language_att");
        name = name.Replace("._full_att", ".full_att");

        // 4. _encoder._resnet -> _encoder.resnet (no underscore on nested)
        name = name.Replace("._resnet.", ".resnet.");

        // 5. layers.N. -> layers_N. ONLY for decoder (TorchSharp decoder uses underscore, encoder uses dot)
        name = System.Text.RegularExpressions.Regex.Replace(name, @"_decoder\.layers\.(\d+)\.", "_decoder.layers_$1.");

        // 6. MLP/embed layers: layers.N. -> layers_N. (for bbox_embed)
        name = System.Text.RegularExpressions.Regex.Replace(name, @"_embed\.layers\.(\d+)\.", "_embed.layers_$1.");


        // 7. Input filter: ._input_filter.N. -> ._input_filter_N. (TorchSharp does not allow dots in module names)
        name = System.Text.RegularExpressions.Regex.Replace(name, @"\._input_filter\.(\d+)\.", "._input_filter_$1.");

        // 8. Downsample: downsample.N -> downsample_N (TorchSharp does not allow dots in module names)
        name = System.Text.RegularExpressions.Regex.Replace(name, @"\.downsample\.(\d+)", ".downsample_$1");

        return name;
    }
}
