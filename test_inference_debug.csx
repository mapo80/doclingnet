#r "nuget: TorchSharp-cpu, 0.105.1"

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// Simple test: check if argmax is working correctly
var logits = torch.tensor(new float[] { 0.1f, 0.5f, 0.3f, 0.9f, 0.2f });
var predicted = logits.argmax().item<long>();
Console.WriteLine($"Logits: {string.Join(", ", logits.data<float>().ToArray())}");
Console.WriteLine($"Argmax: {predicted}");
Console.WriteLine($"Expected: 3");

// Test with negative values
var logits2 = torch.tensor(new float[] { -1.5f, -0.5f, -2.0f, -0.1f });
var predicted2 = logits2.argmax().item<long>();
Console.WriteLine($"\nLogits2: {string.Join(", ", logits2.data<float>().ToArray())}");
Console.WriteLine($"Argmax: {predicted2}");
Console.WriteLine($"Expected: 3");

// Test softmax
var probs = functional.softmax(logits2, dim: 0);
Console.WriteLine($"\nSoftmax probs: {string.Join(", ", probs.data<float>().ToArray().Select(x => x.ToString("F4")))}");
