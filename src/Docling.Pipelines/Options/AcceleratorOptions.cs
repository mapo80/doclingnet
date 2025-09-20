using System;
using System.Collections.Generic;

namespace Docling.Pipelines.Options;

/// <summary>
/// Configuration controlling how model accelerators are selected and tuned.
/// Mirrors the semantics of Docling's <c>AcceleratorOptions</c> Pydantic model.
/// </summary>
public sealed class AcceleratorOptions
{
    private static readonly Dictionary<string, string> KnownDevices = new(StringComparer.OrdinalIgnoreCase)
    {
        [AcceleratorDevice.Auto] = AcceleratorDevice.Auto,
        [AcceleratorDevice.Cpu] = AcceleratorDevice.Cpu,
        [AcceleratorDevice.Cuda] = AcceleratorDevice.Cuda,
        [AcceleratorDevice.Mps] = AcceleratorDevice.Mps,
    };

    private int _numThreads = 4;
    private string _device = AcceleratorDevice.Auto;

    /// <summary>
    /// Number of threads the inference runtime should leverage. Defaults to four as in Python Docling.
    /// </summary>
    public int NumThreads
    {
        get => _numThreads;
        init => _numThreads = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "The number of threads must be positive.");
    }

    /// <summary>
    /// Preferred inference device. Accepts "auto", "cpu", "mps", "cuda" or a CUDA device index (e.g. "cuda:1").
    /// </summary>
    public string Device
    {
        get => _device;
        init => _device = NormalizeDevice(value);
    }

    /// <summary>
    /// Enables FlashAttention v2 optimisations for CUDA capable runtimes.
    /// </summary>
    public bool CudaUseFlashAttention2 { get; init; }

    private static string NormalizeDevice(string? candidate)
    {
        ArgumentNullException.ThrowIfNull(candidate);

        if (KnownDevices.TryGetValue(candidate, out var canonical))
        {
            return canonical;
        }

        if (candidate.Length >= 4 && candidate.StartsWith(AcceleratorDevice.Cuda, StringComparison.OrdinalIgnoreCase))
        {
            var suffix = candidate[4..];
            if (suffix.Length == 0)
            {
                return AcceleratorDevice.Cuda;
            }

            if (suffix[0] == ':' && suffix.Length > 1 && int.TryParse(suffix.AsSpan(1), out _))
            {
                return AcceleratorDevice.Cuda + suffix;
            }
        }

        throw new ArgumentException($"Unsupported accelerator device '{candidate}'.", nameof(candidate));
    }
}

/// <summary>
/// Identifier constants for known accelerator device types.
/// </summary>
public static class AcceleratorDevice
{
    public const string Auto = "auto";
    public const string Cpu = "cpu";
    public const string Cuda = "cuda";
    public const string Mps = "mps";
}
