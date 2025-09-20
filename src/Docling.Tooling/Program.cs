using Docling.Backends.Pdf;
using Docling.Export.Abstractions;
using Docling.Export.Imaging;
using Docling.Pipelines.Abstractions;
using Docling.Pipelines.Internal;
using Docling.Tooling.Runtime;
using Docling.Tooling.Stages;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Events;

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .MinimumLevel.Override("Microsoft", LogEventLevel.Information)
    .Enrich.FromLogContext()
    .WriteTo.Console()
    .CreateLogger();

try
{
    var builder = Host.CreateApplicationBuilder(args);
    builder.Logging.ClearProviders();
    builder.Host.UseSerilog();

    builder.Services.AddSingleton<IPdfPageRenderer, PdfToImageRenderer>();
    builder.Services.AddSingleton<IImageCropService, ImageCropService>();
    builder.Services.AddSingleton<PdfRenderSettings>();
    builder.Services.AddSingleton(provider =>
        new PdfBackendOptions
        {
            StreamFactory = _ => throw new NotSupportedException("No PDF source configured."),
            RenderSettings = provider.GetRequiredService<PdfRenderSettings>(),
        });

    builder.Services.AddSingleton<ConvertPipelineBuilder>(provider =>
    {
        var loggerFactory = provider.GetRequiredService<ILoggerFactory>();
        var diagnosticsStage = new DiagnosticsStage(loggerFactory.CreateLogger<DiagnosticsStage>());
        var observer = new LoggingPipelineObserver(loggerFactory.CreateLogger<LoggingPipelineObserver>());

        return new ConvertPipelineBuilder()
            .AddStage(diagnosticsStage)
            .AddObserver(observer);
    });

    builder.Services.AddSingleton(provider =>
        provider.GetRequiredService<ILoggerFactory>().CreateLogger<ConvertPipeline>());

    builder.Services.AddHostedService<PipelineHostedService>();

    using var host = builder.Build();
    await host.RunAsync().ConfigureAwait(false);
    return 0;
}
catch (Exception ex)
{
    Log.Fatal(ex, "Docling tooling terminated unexpectedly.");
    return 1;
}
finally
{
    Log.CloseAndFlush();
}
