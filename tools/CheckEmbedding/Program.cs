using TableFormerSdk.Models;
using Newtonsoft.Json;

var configPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tm_config.json";
var safetensorsPath = "/Users/politom/Documents/Workspace/personal/doclingnet/models/model_artifacts/tableformer/fast/tableformer_fast.safetensors";

Console.WriteLine("Loading model...");
var config = TableModel04Config.FromJsonFile(configPath);
using var model = TableModel04.FromSafeTensors(config, safetensorsPath);
model.eval();

Console.WriteLine("\nSearching for FC layer weights...");
TorchSharp.torch.Tensor? fcWeight = null;
TorchSharp.torch.Tensor? fcBias = null;

foreach (var (name, param) in model.named_parameters())
{
    if (name.Contains("_fc", StringComparison.Ordinal))
    {
        Console.WriteLine($"Found: {name}");
        
        if (name.Contains("weight", StringComparison.Ordinal))
        {
            fcWeight = param;
        }
        else if (name.Contains("bias", StringComparison.Ordinal))
        {
            fcBias = param;
        }
    }
}

if (fcWeight is not null && fcBias is not null)
{
    Console.WriteLine("\n=== FC WEIGHT FOR <end> TOKEN (idx=10) ===");
    var endWeight = fcWeight[10];
    var endWeightData = endWeight.data<float>().ToArray();
    Console.WriteLine($"First 20 values: {string.Join(", ", endWeightData.Take(20).Select(x => x.ToString("F6")))}");

    Console.WriteLine("\n=== FC BIAS FOR ALL TOKENS ===");
    var biasData = fcBias.data<float>().ToArray();
    for (int i = 0; i < biasData.Length; i++)
    {
        Console.WriteLine($"Token {i}: bias={biasData[i]:F6}");
    }

    var output = new
    {
        fc_weight_shape = new[] { fcWeight.size(0), fcWeight.size(1) },
        fc_bias_shape = new[] { fcBias.size(0) },
        end_token_weight_first_20 = endWeightData.Take(20).ToArray(),
        end_token_bias = biasData[10],
        all_biases = biasData,
    };

    var json = JsonConvert.SerializeObject(output, Formatting.Indented);
    File.WriteAllText("/Users/politom/Documents/Workspace/personal/doclingnet/debug/csharp_fc_check.json", json);
    Console.WriteLine("\n✅ Saved");
}
