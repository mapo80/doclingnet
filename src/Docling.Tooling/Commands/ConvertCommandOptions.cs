using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Docling.Export.Serialization;
using Docling.Pipelines.Options;

namespace Docling.Tooling.Commands;

internal enum DocumentInputKind
{
    Pdf,
    Image,
}

internal sealed class ConvertCommandOptions
{
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
    };

    private ConvertCommandOptions(
        string inputPath,
        DocumentInputKind inputKind,
        string outputDirectory,
        string markdownFileName,
        string assetsDirectoryName,
        string documentId,
        string sourceName,
        double preprocessingDpi,
        int renderDpi,
        bool generatePageImages,
        bool generatePictureImages,
        bool generateLayoutDebugArtifacts,
        bool generateImageDebugArtifacts,
        bool generateTableDebugArtifacts,
        bool forceFullPageOcr,
        IReadOnlyList<string> ocrLanguages,
        TableFormerMode tableMode,
        MarkdownImageMode imageMode)
    {
        InputPath = inputPath;
        InputKind = inputKind;
        OutputDirectory = outputDirectory;
        MarkdownFileName = markdownFileName;
        AssetsDirectoryName = assetsDirectoryName;
        DocumentId = documentId;
        SourceName = sourceName;
        PreprocessingDpi = preprocessingDpi;
        RenderDpi = renderDpi;
        GeneratePageImages = generatePageImages;
        GeneratePictureImages = generatePictureImages;
        GenerateLayoutDebugArtifacts = generateLayoutDebugArtifacts;
        GenerateImageDebugArtifacts = generateImageDebugArtifacts;
        GenerateTableDebugArtifacts = generateTableDebugArtifacts;
        ForceFullPageOcr = forceFullPageOcr;
        OcrLanguages = ocrLanguages;
        TableMode = tableMode;
        ImageMode = imageMode;
    }

    public string InputPath { get; }

    public DocumentInputKind InputKind { get; }

    public string OutputDirectory { get; }

    public string MarkdownFileName { get; }

    public string AssetsDirectoryName { get; }

    public string DocumentId { get; }

    public string SourceName { get; }

    public double PreprocessingDpi { get; }

    public int RenderDpi { get; }

    public bool GeneratePageImages { get; }

    public bool GeneratePictureImages { get; }

    public bool GenerateLayoutDebugArtifacts { get; }

    public bool GenerateImageDebugArtifacts { get; }

    public bool GenerateTableDebugArtifacts { get; }

    public bool ForceFullPageOcr { get; }

    public IReadOnlyList<string> OcrLanguages { get; }

    public TableFormerMode TableMode { get; }

    public MarkdownImageMode ImageMode { get; }

    public const string MetadataFileName = "docling.metadata.json";

    public static ParseResult Parse(ReadOnlySpan<string> args)
    {
        if (args.Length == 0)
        {
            return ParseResult.Failure("Missing arguments.");
        }

        string? inputPath = null;
        var outputDirectory = Path.Combine(Environment.CurrentDirectory, "docling-output");
        var markdownFileName = "docling.md";
        var assetsDirectoryName = "assets";
        var preprocessingDpi = 300d;
        var renderDpi = 300;
        var generatePageImages = true;
        var generatePictureImages = true;
        var generateLayoutDebugArtifacts = false;
        var generateImageDebugArtifacts = false;
        var generateTableDebugArtifacts = false;
        var forceFullPageOcr = false;
        var languages = new List<string>();
        var tableMode = TableFormerMode.Accurate;
        var imageMode = MarkdownImageMode.Referenced;

        for (var i = 0; i < args.Length; i++)
        {
            var token = args[i];
            if (IsHelpToken(token))
            {
                return ParseResult.Help();
            }

            if (!token.StartsWith("--", StringComparison.Ordinal))
            {
                return ParseResult.Failure($"Unrecognised argument '{token}'.");
            }

            var name = token[2..];
            switch (name)
            {
                case "input":
                    if (!TryReadValue(args, ref i, out inputPath))
                    {
                        return ParseResult.Failure("--input requires a value.");
                    }

                    break;

                case "output":
                    if (!TryReadValue(args, ref i, out outputDirectory))
                    {
                        return ParseResult.Failure("--output requires a value.");
                    }

                    break;

                case "markdown":
                    if (!TryReadValue(args, ref i, out markdownFileName))
                    {
                        return ParseResult.Failure("--markdown requires a value.");
                    }

                    break;

                case "assets":
                    if (!TryReadValue(args, ref i, out assetsDirectoryName))
                    {
                        return ParseResult.Failure("--assets requires a value.");
                    }

                    break;

                case "lang":
                case "languages":
                    if (!TryReadValue(args, ref i, out var langValue))
                    {
                        return ParseResult.Failure("--languages requires a comma-separated list of ISO codes.");
                    }

                    languages.Clear();
                    foreach (var segment in langValue.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        languages.Add(segment);
                    }

                    break;

                case "dpi":
                    if (!TryReadValue(args, ref i, out var dpiRaw) ||
                        !double.TryParse(dpiRaw, NumberStyles.Float, CultureInfo.InvariantCulture, out preprocessingDpi) ||
                        preprocessingDpi <= 0)
                    {
                        return ParseResult.Failure("--dpi expects a positive numeric value.");
                    }

                    break;

                case "render-dpi":
                    if (!TryReadValue(args, ref i, out var renderRaw) ||
                        !int.TryParse(renderRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out renderDpi) ||
                        renderDpi <= 0)
                    {
                        return ParseResult.Failure("--render-dpi expects a positive integer value.");
                    }

                    break;

                case "layout-debug":
                    generateLayoutDebugArtifacts = true;
                    break;

                case "image-debug":
                    generateImageDebugArtifacts = true;
                    break;

                case "table-debug":
                    generateTableDebugArtifacts = true;
                    break;

                case "no-page-images":
                    generatePageImages = false;
                    break;

                case "no-picture-images":
                    generatePictureImages = false;
                    break;

                case "full-page-ocr":
                    forceFullPageOcr = true;
                    break;

                case "table-mode":
                    if (!TryReadValue(args, ref i, out var tableModeRaw))
                    {
                        return ParseResult.Failure("--table-mode requires 'fast' or 'accurate'.");
                    }

                    if (tableModeRaw.Equals("fast", StringComparison.OrdinalIgnoreCase))
                    {
                        tableMode = TableFormerMode.Fast;
                    }
                    else if (tableModeRaw.Equals("accurate", StringComparison.OrdinalIgnoreCase))
                    {
                        tableMode = TableFormerMode.Accurate;
                    }
                    else
                    {
                        return ParseResult.Failure("--table-mode accepts 'fast' or 'accurate'.");
                    }

                    break;

                case "image-mode":
                    if (!TryReadValue(args, ref i, out var imageModeRaw))
                    {
                        return ParseResult.Failure("--image-mode requires 'referenced', 'embedded', or 'placeholder'.");
                    }

                    if (imageModeRaw.Equals("referenced", StringComparison.OrdinalIgnoreCase))
                    {
                        imageMode = MarkdownImageMode.Referenced;
                    }
                    else if (imageModeRaw.Equals("embedded", StringComparison.OrdinalIgnoreCase))
                    {
                        imageMode = MarkdownImageMode.Embedded;
                    }
                    else if (imageModeRaw.Equals("placeholder", StringComparison.OrdinalIgnoreCase))
                    {
                        imageMode = MarkdownImageMode.Placeholder;
                    }
                    else
                    {
                        return ParseResult.Failure("--image-mode accepts 'referenced', 'embedded', or 'placeholder'.");
                    }

                    break;

                default:
                    return ParseResult.Failure($"Unrecognised option '{token}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(inputPath))
        {
            return ParseResult.Failure("--input must be specified.");
        }

        inputPath = Path.GetFullPath(inputPath);
        if (!File.Exists(inputPath))
        {
            return ParseResult.Failure($"Input file '{inputPath}' was not found.");
        }

        var outputFullPath = Path.GetFullPath(outputDirectory);
        var normalizedRoot = EnsureTrailingSeparator(outputFullPath);
        var markdownName = markdownFileName.Trim();
        if (string.IsNullOrEmpty(markdownName))
        {
            return ParseResult.Failure("--markdown cannot be empty.");
        }

        if (Path.IsPathRooted(markdownName))
        {
            return ParseResult.Failure("--markdown must be a relative path inside the output directory.");
        }

        var markdownFullPath = Path.GetFullPath(Path.Combine(outputFullPath, markdownName));
        if (!IsWithinRoot(markdownFullPath, outputFullPath, normalizedRoot))
        {
            return ParseResult.Failure("--markdown path resolves outside the output directory.");
        }

        markdownName = GetRelativePathWithinRoot(markdownFullPath, outputFullPath);
        if (string.IsNullOrEmpty(markdownName))
        {
            return ParseResult.Failure("--markdown must resolve to a file inside the output directory.");
        }

        var assetsName = assetsDirectoryName.Trim();
        if (assetsName.Length == 0)
        {
            assetsName = "assets";
        }

        if (Path.IsPathRooted(assetsName))
        {
            return ParseResult.Failure("--assets must be a relative path inside the output directory.");
        }

        var assetsFullPath = Path.GetFullPath(Path.Combine(outputFullPath, assetsName));
        if (!IsWithinRoot(assetsFullPath, outputFullPath, normalizedRoot))
        {
            return ParseResult.Failure("--assets path resolves outside the output directory.");
        }

        if (languages.Count == 0)
        {
            languages.Add("en");
        }

        var extension = Path.GetExtension(inputPath);
        var inputKind = DetermineInputKind(extension);
        if (inputKind is null)
        {
            return ParseResult.Failure($"Unsupported input file extension '{extension}'.");
        }

        var documentId = Path.GetFileNameWithoutExtension(inputPath) ?? "docling";
        if (string.IsNullOrWhiteSpace(documentId))
        {
            documentId = "docling";
        }

        var sourceName = Path.GetFileName(inputPath) ?? documentId;

        return ParseResult.Successful(new ConvertCommandOptions(
            inputPath,
            inputKind.Value,
            outputFullPath,
            markdownName,
            GetRelativePathWithinRoot(assetsFullPath, outputFullPath),
            documentId,
            sourceName,
            preprocessingDpi,
            renderDpi,
            generatePageImages,
            generatePictureImages,
            generateLayoutDebugArtifacts,
            generateImageDebugArtifacts,
            generateTableDebugArtifacts,
            forceFullPageOcr,
            languages.AsReadOnly(),
            tableMode,
            imageMode));
    }

    private static bool TryReadValue(ReadOnlySpan<string> args, ref int index, out string value)
    {
        if (index + 1 >= args.Length)
        {
            value = string.Empty;
            return false;
        }

        index++;
        value = args[index];
        return true;
    }

    private static bool IsHelpToken(string token)
    {
        return token.Equals("--help", StringComparison.OrdinalIgnoreCase) ||
               token.Equals("-h", StringComparison.OrdinalIgnoreCase) ||
               token.Equals("/?", StringComparison.Ordinal);
    }

    private static DocumentInputKind? DetermineInputKind(string extension)
    {
        if (extension.Equals(".pdf", StringComparison.OrdinalIgnoreCase))
        {
            return DocumentInputKind.Pdf;
        }

        if (ImageExtensions.Contains(extension))
        {
            return DocumentInputKind.Image;
        }

        return null;
    }

    private static bool IsWithinRoot(string candidatePath, string root, string normalizedRoot)
    {
        var comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;

        if (candidatePath.Equals(root, comparison))
        {
            return true;
        }

        return candidatePath.StartsWith(normalizedRoot, comparison);
    }

    private static string EnsureTrailingSeparator(string path)
    {
        if (!path.EndsWith(Path.DirectorySeparatorChar) && !path.EndsWith(Path.AltDirectorySeparatorChar))
        {
            return path + Path.DirectorySeparatorChar;
        }

        return path;
    }

    private static string GetRelativePathWithinRoot(string candidatePath, string root)
    {
        var comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;

        if (candidatePath.Equals(root, comparison))
        {
            return string.Empty;
        }

        var relative = Path.GetRelativePath(root, candidatePath);
        return relative.Replace(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
    }

    internal readonly record struct ParseResult(bool Success, bool ShowHelp, string? Error, ConvertCommandOptions? Options)
    {
        public static ParseResult Successful(ConvertCommandOptions options)
            => new(true, false, null, options);

        public static ParseResult Failure(string error)
            => new(false, false, error, null);

        public static ParseResult Help()
            => new(false, true, null, null);
    }
}
