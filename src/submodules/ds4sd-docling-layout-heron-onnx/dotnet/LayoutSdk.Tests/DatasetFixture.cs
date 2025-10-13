using System;
using System.IO;

namespace LayoutSdk.Tests;

public sealed class DatasetFixture
{
    private const string ImageRelativePath = "dataset/2305.03393v1-pg9-img.png";
    private const string ModelsRelativePath = "src/submodules/ds4sd-docling-layout-heron-onnx/dotnet/LayoutSdk/PackagedModels/models";

    public DatasetFixture()
    {
        var root = LocateRepositoryRoot();
        RepositoryRoot = root;
        ImagePath = Path.Combine(root, ImageRelativePath);
        ModelsRoot = Path.Combine(root, ModelsRelativePath);

        if (!File.Exists(ImagePath))
        {
            throw new FileNotFoundException($"Sample layout image not found at {ImagePath}");
        }

        if (!Directory.Exists(ModelsRoot))
        {
            throw new DirectoryNotFoundException($"Layout models folder not found at {ModelsRoot}");
        }
    }

    public string RepositoryRoot { get; }

    public string ImagePath { get; }

    public string ModelsRoot { get; }

    private static string LocateRepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidateFile = Path.Combine(dir.FullName, ImageRelativePath);
            if (File.Exists(candidateFile))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        throw new InvalidOperationException($"Unable to locate repository root containing {ImageRelativePath}.");
    }
}
