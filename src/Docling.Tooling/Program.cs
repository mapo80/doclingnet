using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Docling.Backends.Abstractions;
using Docling.Backends.Image;
using Docling.Backends.Pdf;
using Docling.Export.Abstractions;
using Docling.Export.Imaging;
using Docling.Export.Serialization;
using Docling.Models.Layout;
using Docling.Models.Ocr;
using Docling.Models.Tables;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Assembly;
using Docling.Pipelines.Export;
using Docling.Pipelines.Layout;
using Docling.Pipelines.Options;
using Docling.Pipelines.Preprocessing;
using Docling.Pipelines.Serialization;
using Docling.Pipelines.Tables;
using Docling.Pipelines.Ocr;
using Docling.Tooling.Commands;
using Docling.Tooling.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Events;
using TableFormerSdk.Enums;
using LayoutSdk;

namespace Docling.Tooling;

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .MinimumLevel.Override("Microsoft", LogEventLevel.Information)
            .Enrich.FromLogContext()
            .WriteTo.Console(formatProvider: CultureInfo.InvariantCulture)
            .CreateLogger();

        AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;

        try
        {
            return await RunAsync(args).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Log.Warning("Conversion cancelled by user.");
            return 2;
        }
        finally
        {
            await Log.CloseAndFlushAsync().ConfigureAwait(false);
        }
    }

    private static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 0;
        }

        if (IsRootHelp(args[0]))
        {
            PrintUsage();
            return 0;
        }

        if (!string.Equals(args[0], "convert", StringComparison.OrdinalIgnoreCase))
        {
            Log.Error("Unknown command '{Command}'.", args[0]);
            PrintUsage();
            return 1;
        }

        var commandArgs = args.Skip(1).ToArray();
        var parseResult = ConvertCommandOptions.Parse(commandArgs);
        if (parseResult.ShowHelp)
        {
            PrintConvertUsage();
            return 0;
        }

        if (!parseResult.Success)
        {
            Log.Error(parseResult.Error ?? "Failed to parse convert command arguments.");
            PrintConvertUsage();
            return 1;
        }

        var options = parseResult.Options!;
        using var host = BuildHost(options);
        var runner = host.Services.GetRequiredService<ConvertCommandRunner>();

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            Log.Warning("Cancellation requested, attempting to stop gracefully...");
            eventArgs.Cancel = true;
            cts.Cancel();
        };

        return await runner.ExecuteAsync(cts.Token).ConfigureAwait(false);
    }

    private static void OnUnhandledException(object? sender, UnhandledExceptionEventArgs args)
    {
        if (args.ExceptionObject is Exception exception)
        {
            Log.Fatal(exception, "Docling tooling terminated unexpectedly.");
        }
        else
        {
            Log.Fatal("Docling tooling terminated unexpectedly.");
        }

        if (!args.IsTerminating && Environment.ExitCode == 0)
        {
            Environment.ExitCode = 1;
        }
    }

    [SuppressMessage("Globalization", "CA1303:Do not pass literals as localized parameters", Justification = "CLI output is intentionally English-only.")]
    private static void PrintUsage()
    {
        Console.WriteLine("Docling .NET CLI");
        Console.WriteLine("Usage:");
        Console.WriteLine("  docling convert --input <file> [options]");
        Console.WriteLine();
        Console.WriteLine("Run 'docling convert --help' for available options.");
    }

    [SuppressMessage("Globalization", "CA1303:Do not pass literals as localized parameters", Justification = "CLI output is intentionally English-only.")]
    private static void PrintConvertUsage()
    {
        Console.WriteLine("Usage:");
        Console.WriteLine("  docling convert --input <file> [--output <directory>] [--markdown <file>] [--assets <relative-path>]");
        Console.WriteLine("                         [--languages <iso-codes>] [--dpi <value>] [--render-dpi <value>]");
        Console.WriteLine("                         [--table-mode fast|accurate] [--image-mode referenced|embedded|placeholder]");
        Console.WriteLine("                         [--no-page-images] [--no-picture-images] [--full-page-ocr]");
        Console.WriteLine("                         [--layout-debug] [--image-debug] [--table-debug] [--workflow-debug]");
    }

    private static bool IsRootHelp(string value)
        => value.Equals("--help", StringComparison.OrdinalIgnoreCase)
           || value.Equals("-h", StringComparison.OrdinalIgnoreCase)
           || value.Equals("/?", StringComparison.Ordinal);

    private static IHost BuildHost(ConvertCommandOptions options)
    {
        return Host.CreateDefaultBuilder()
            .ConfigureLogging(logging => logging.ClearProviders())
            .UseSerilog()
            .ConfigureServices(services => ConfigureServices(services, options))
            .Build();
    }

    private static void ConfigureServices(IServiceCollection services, ConvertCommandOptions options)
    {
        services.AddSingleton(options);

        services.AddSingleton(CreatePipelineOptions(options));
        services.AddSingleton(provider => provider.GetRequiredService<PdfPipelineOptions>().Layout);
        services.AddSingleton(provider => provider.GetRequiredService<PdfPipelineOptions>().TableStructure);

        services.AddSingleton(CreatePreprocessingOptions(options));
        services.AddSingleton<IPagePreprocessor>(provider => new DefaultPagePreprocessor(
            provider.GetRequiredService<PreprocessingOptions>(),
            provider.GetRequiredService<ILogger<DefaultPagePreprocessor>>()));
        services.AddSingleton<PagePreprocessingStage>();

        services.AddSingleton(new LayoutSdkDetectionOptions
        {
            ValidateModelFiles = true,
            MaxDegreeOfParallelism = 1,
            EnableAdvancedNonMaxSuppression = options.EnableAdvancedLayoutNms,
        });
        services.AddSingleton<ILayoutDetectionService>(provider => new LayoutSdkDetectionService(
            provider.GetRequiredService<LayoutSdkDetectionOptions>(),
            provider.GetRequiredService<ILogger<LayoutSdkDetectionService>>()));

        if (options.GenerateLayoutDebugArtifacts)
        {
            services.AddSingleton(new LayoutDebugOverlayOptions());
            services.AddSingleton<ILayoutDebugOverlayRenderer>(provider => new LayoutDebugOverlayRenderer(
                provider.GetRequiredService<LayoutDebugOverlayOptions>(),
                provider.GetRequiredService<ILogger<LayoutDebugOverlayRenderer>>()));
        }

        services.AddSingleton<LayoutAnalysisStage>(provider => new LayoutAnalysisStage(
            provider.GetRequiredService<ILayoutDetectionService>(),
            provider.GetRequiredService<LayoutOptions>(),
            provider.GetRequiredService<ILogger<LayoutAnalysisStage>>(),
            provider.GetService<ILayoutDebugOverlayRenderer>()));

        services.AddSingleton(provider => CreateTableStructureService(
            options,
            provider.GetRequiredService<ILogger<TableFormerTableStructureService>>()));
        services.AddSingleton<ITableStructureService>(provider => provider.GetRequiredService<TableFormerTableStructureService>());
        services.AddSingleton<TableStructureInferenceStage>(provider => new TableStructureInferenceStage(
            provider.GetRequiredService<ITableStructureService>(),
            provider.GetRequiredService<PdfPipelineOptions>(),
            provider.GetRequiredService<ILogger<TableStructureInferenceStage>>()));

        services.AddSingleton<OcrServiceFactory>();
        services.AddSingleton<OcrStage>(provider => new OcrStage(
            provider.GetRequiredService<OcrServiceFactory>(),
            provider.GetRequiredService<PdfPipelineOptions>(),
            provider.GetRequiredService<ILogger<OcrStage>>()));

        services.AddSingleton<PageAssemblyStage>();

        services.AddSingleton<IImageCropService>(_ => new ImageCropService());
        services.AddSingleton<ImageExportStage>(provider => new ImageExportStage(
            provider.GetRequiredService<IImageCropService>(),
            provider.GetRequiredService<PdfPipelineOptions>(),
            provider.GetRequiredService<ILogger<ImageExportStage>>()));

        services.AddSingleton(CreateSerializerOptions(options));
        services.AddSingleton(provider => new MarkdownDocSerializer(provider.GetRequiredService<MarkdownSerializerOptions>()));
        services.AddSingleton<MarkdownSerializationStage>(provider => new MarkdownSerializationStage(
            provider.GetRequiredService<MarkdownDocSerializer>()));

        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<PagePreprocessingStage>());
        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<LayoutAnalysisStage>());
        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<TableStructureInferenceStage>());
        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<OcrStage>());
        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<PageAssemblyStage>());
        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<ImageExportStage>());
        services.AddSingleton<IPipelineStage>(provider => provider.GetRequiredService<MarkdownSerializationStage>());

        services.AddSingleton<PipelineTelemetryObserver>();
        services.AddSingleton<IPipelineObserver, LoggingPipelineObserver>();

        if (options.GenerateWorkflowDebugArtifacts)
        {
            services.AddSingleton<WorkflowDebugObserver>();
            services.AddSingleton<IPipelineObserver>(provider => provider.GetRequiredService<WorkflowDebugObserver>());
        }

        if (options.InputKind == DocumentInputKind.Pdf)
        {
            services.AddSingleton(new PdfRenderSettings { Dpi = options.RenderDpi });
            services.AddSingleton(provider => new PdfBackendOptions
            {
                DocumentId = options.DocumentId,
                SourceName = options.SourceName,
                RenderSettings = provider.GetRequiredService<PdfRenderSettings>(),
                StreamFactory = _ => Task.FromResult<Stream>(OpenFile(options.InputPath)),
                Metadata = CreateSourceMetadata(options),
            });
            services.AddSingleton<IPdfPageRenderer, PdfToImageRenderer>();
            services.AddSingleton<IPdfBackend>(provider => new PdfBackend(
                provider.GetRequiredService<IPdfPageRenderer>(),
                provider.GetRequiredService<PdfBackendOptions>()));
        }
        else
        {
            services.AddSingleton(provider => new ImageBackendOptions
            {
                DocumentId = options.DocumentId,
                SourceName = options.SourceName,
                DefaultDpi = options.RenderDpi,
                Metadata = CreateSourceMetadata(options),
                Sources = new[]
                {
                    new ImageSourceDescriptor
                    {
                        Identifier = options.DocumentId,
                        FileName = Path.GetFileName(options.InputPath),
                        MediaType = GuessMediaType(options.InputPath),
                        Dpi = options.RenderDpi,
                        StreamFactory = _ => Task.FromResult<Stream>(OpenFile(options.InputPath)),
                    },
                },
            });
            services.AddSingleton<IImageBackend>(provider => new ImageBackend(
                provider.GetRequiredService<ImageBackendOptions>(),
                provider.GetRequiredService<ILogger<ImageBackend>>()));
        }

        services.AddSingleton<ConvertCommandRunner>();
    }

    private static PdfPipelineOptions CreatePipelineOptions(ConvertCommandOptions options)
    {
        return new PdfPipelineOptions
        {
            GeneratePageImages = options.GeneratePageImages,
            GeneratePictureImages = options.GeneratePictureImages,
            GenerateImageDebugArtifacts = options.GenerateImageDebugArtifacts,
            Layout = new LayoutOptions
            {
                Model = LayoutModelConfiguration.DoclingLayoutEgretMedium,
                CreateOrphanClusters = false,
                KeepEmptyClusters = true,
                GenerateDebugArtifacts = options.GenerateLayoutDebugArtifacts,
            },
            TableStructure = new TableStructureOptions
            {
                Mode = options.TableMode,
            },
            Ocr = new EasyOcrOptions
            {
                Languages = options.OcrLanguages.ToArray(),
                ForceFullPageOcr = options.ForceFullPageOcr,
                BitmapAreaThreshold = 0.0005,
                ModelStorageDirectory = ResolveEasyOcrModelDirectory(),
            },
        };
    }

    private static string? ResolveEasyOcrModelDirectory()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var defaultDirectory = Path.Combine(baseDirectory, "contentFiles", "any", "any", "models");
        var candidates = new[]
        {
            defaultDirectory,
            Path.GetFullPath(Path.Combine(baseDirectory, "..", "..", "..", "..", "packages", "custom", "EasyOcrNet.1.0.0", "contentFiles", "any", "any", "models")),
            Path.Combine(Environment.CurrentDirectory, "packages", "custom", "EasyOcrNet.1.0.0", "contentFiles", "any", "any", "models"),
        };

        foreach (var candidate in candidates)
        {
            if (!Directory.Exists(candidate))
            {
                continue;
            }

            if (File.Exists(Path.Combine(candidate, "detection.onnx")))
            {
                return candidate;
            }

            var onnxVariant = Path.Combine(candidate, "onnx");
            if (Directory.Exists(onnxVariant) && File.Exists(Path.Combine(onnxVariant, "detection.onnx")))
            {
                return onnxVariant;
            }
        }

        return null;
    }

    private static PreprocessingOptions CreatePreprocessingOptions(ConvertCommandOptions options)
    {
        var enableAdvancedPreprocessing = options.InputKind == DocumentInputKind.Pdf;

        return new PreprocessingOptions
        {
            TargetDpi = options.PreprocessingDpi,
            EnableDeskew = enableAdvancedPreprocessing,
            NormalizeContrast = enableAdvancedPreprocessing,
        };
    }

    private static TableFormerTableStructureService CreateTableStructureService(
        ConvertCommandOptions options,
        ILogger<TableFormerTableStructureService> logger)
    {
        var serviceOptions = new TableFormerStructureServiceOptions
        {
            Variant = options.TableMode == TableFormerMode.Fast
                ? TableFormerModelVariant.Fast
                : TableFormerModelVariant.Accurate,
            Runtime = TableFormerRuntime.Onnx,
            GenerateOverlay = options.GenerateTableDebugArtifacts,
            WorkingDirectory = Path.Combine(Path.GetTempPath(), $"docling-tableformer-{options.DocumentId}"),
        };

        return new TableFormerTableStructureService(serviceOptions, logger);
    }

    private static MarkdownSerializerOptions CreateSerializerOptions(ConvertCommandOptions options)
    {
        return new MarkdownSerializerOptions
        {
            AssetsPath = options.AssetsDirectoryName.Replace('\\', '/'),
            ImageMode = options.ImageMode,
        };
    }

    private static FileStream OpenFile(string path)
    {
        return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
    }

    private static Dictionary<string, string> CreateSourceMetadata(ConvertCommandOptions options)
    {
        return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["source_path"] = options.InputPath,
        };
    }

    private static string GuessMediaType(string path)
    {
        var extension = Path.GetExtension(path);
        if (string.Equals(extension, ".jpg", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".jpeg", StringComparison.OrdinalIgnoreCase))
        {
            return "image/jpeg";
        }

        if (string.Equals(extension, ".png", StringComparison.OrdinalIgnoreCase))
        {
            return "image/png";
        }

        if (string.Equals(extension, ".tif", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".tiff", StringComparison.OrdinalIgnoreCase))
        {
            return "image/tiff";
        }

        if (string.Equals(extension, ".bmp", StringComparison.OrdinalIgnoreCase))
        {
            return "image/bmp";
        }

        if (string.Equals(extension, ".gif", StringComparison.OrdinalIgnoreCase))
        {
            return "image/gif";
        }

        if (string.Equals(extension, ".webp", StringComparison.OrdinalIgnoreCase))
        {
            return "image/webp";
        }

        return "image/png";
    }
}
