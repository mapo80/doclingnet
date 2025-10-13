using System;
using System.Collections.Generic;
using System.IO;

namespace LayoutSdk.Configuration;

public static class LayoutSdkBundledModels
{
    public const string OptimizedOnnxFileName = "heron-optimized.onnx";
    public const string ConvertedOnnxFileName = "heron-converted.onnx";
    public const string OptimizedFp16OnnxFileName = "heron-optimized-fp16.onnx";
    public const string OptimizedOrtFileName = "heron-optimized.ort";
    public const string OptimizedRuntimeOrtFileName = "heron-optimized.with_runtime_opt.ort";

    public static string ModelsRoot => Path.Combine(AppContext.BaseDirectory, "models");


    public static string GetOptimizedOnnxPath() => Path.Combine(ModelsRoot, OptimizedOnnxFileName);

    public static string GetConvertedOnnxPath() => Path.Combine(ModelsRoot, ConvertedOnnxFileName);

    public static string GetOptimizedFp16OnnxPath() => Path.Combine(ModelsRoot, OptimizedFp16OnnxFileName);

    public static string GetOptimizedOrtPath() => Path.Combine(ModelsRoot, OptimizedOrtFileName);

    public static string GetOptimizedRuntimeOrtPath() => Path.Combine(ModelsRoot, OptimizedRuntimeOrtFileName);


    public static LayoutSdkOptions CreateOptions(
        DocumentLanguage defaultLanguage = DocumentLanguage.English,
        bool validateModelPaths = true)
        => new(
            onnxModelPath: GetOptimizedOnnxPath(),
            defaultLanguage: defaultLanguage,
            validateModelPaths: validateModelPaths);

    public static void EnsureAllFilesExist()
    {
        foreach (var path in EnumerateExpectedFiles())
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Bundled model file not found: {Path.GetFileName(path)}", path);
            }
        }
    }

    public static IEnumerable<string> EnumerateExpectedFiles()
    {
        yield return GetOptimizedOnnxPath();
        yield return GetConvertedOnnxPath();
        yield return GetOptimizedFp16OnnxPath();
        yield return GetOptimizedOrtPath();
        yield return GetOptimizedRuntimeOrtPath();
    }
}
