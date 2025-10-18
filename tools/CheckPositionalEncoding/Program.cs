using TableFormerSdk.Models;
using Newtonsoft.Json;
using TorchSharp;

// Create PositionalEncoding with same parameters as Python
Console.WriteLine("Creating PositionalEncoding with dModel=512, maxLen=1024...");
using var pe = new PositionalEncoding(dModel: 512, dropout: 0.1f, maxLen: 1024);
pe.eval();

// Get the PE buffer
Console.WriteLine("\nExtracting PE buffer...");
torch.Tensor? peBuffer = null;

foreach (var (name, buffer) in pe.named_buffers())
{
    if (name == "pe")
    {
        peBuffer = buffer;
        Console.WriteLine($"Found PE buffer: {name}");
        Console.WriteLine($"Shape: [{string.Join(", ", buffer.shape)}]");
        break;
    }
}

if (peBuffer is null)
{
    Console.WriteLine("ERROR: Could not find 'pe' buffer!");
    return;
}

// Extract position 0 and position 50
// PE buffer shape is (1, 1024, 512)
// But we need to verify it works when permuted to (1024, 1, 512) for transformer use
var pePos0 = peBuffer[0, 0];  // All 512 dimensions for position 0
var pePos50 = peBuffer[0, 50];  // All 512 dimensions for position 50

// Also test that forward() works with (seq, batch, dim) shape
Console.WriteLine("\n\nTesting forward() with transformer shape (seq, batch, dim)...");
var testInput = torch.randn(5, 2, 512);  // (seq_len=5, batch_size=2, d_model=512)
Console.WriteLine($"Test input shape: [{string.Join(", ", testInput.shape)}]");

using var testOutput = pe.forward(testInput);
Console.WriteLine($"Test output shape: [{string.Join(", ", testOutput.shape)}]");
Console.WriteLine($"Output matches input shape: {testOutput.shape[0] == 5 && testOutput.shape[1] == 2 && testOutput.shape[2] == 512}");

var pePos0Data = pePos0.data<float>().ToArray();
var pePos50Data = pePos50.data<float>().ToArray();

Console.WriteLine($"\nPE[0] shape: {pePos0Data.Length}");
Console.WriteLine($"PE[0] first 20 values: {string.Join(", ", pePos0Data.Take(20).Select(x => x.ToString("F6")))}");
Console.WriteLine($"PE[0] range: [{pePos0Data.Min():F6}, {pePos0Data.Max():F6}]");

Console.WriteLine($"\nPE[50] shape: {pePos50Data.Length}");
Console.WriteLine($"PE[50] first 20 values: {string.Join(", ", pePos50Data.Take(20).Select(x => x.ToString("F6")))}");
Console.WriteLine($"PE[50] range: [{pePos50Data.Min():F6}, {pePos50Data.Max():F6}]");

// Get the scale parameter
torch.Tensor? scaleParam = null;
foreach (var (name, param) in pe.named_parameters())
{
    if (name == "scale")
    {
        scaleParam = param;
        break;
    }
}

var output = new
{
    pe_shape = new[] { peBuffer.size(0), peBuffer.size(1), peBuffer.size(2) },
    pe_pos0_shape = new[] { pePos0Data.Length },
    pe_pos50_shape = new[] { pePos50Data.Length },
    pe_pos0_first_20 = pePos0Data.Take(20).ToArray(),
    pe_pos0_full = pePos0Data,
    pe_pos50_first_20 = pePos50Data.Take(20).ToArray(),
    pe_pos50_full = pePos50Data,
    pe_pos0_range = new[] { pePos0Data.Min(), pePos0Data.Max() },
    pe_pos50_range = new[] { pePos50Data.Min(), pePos50Data.Max() },
    scale = scaleParam?.item<float>() ?? 1.0f,
};

var json = JsonConvert.SerializeObject(output, Formatting.Indented);
var outputPath = "/Users/politom/Documents/Workspace/personal/doclingnet/debug/csharp_pe_check.json";
File.WriteAllText(outputPath, json);
Console.WriteLine($"\n✅ Saved to {outputPath}");
