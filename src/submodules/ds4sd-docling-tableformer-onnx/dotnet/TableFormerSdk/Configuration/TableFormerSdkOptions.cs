using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using TableFormerSdk.Constants;
using TableFormerSdk.Enums;
using TableFormerSdk.Performance;

namespace TableFormerSdk.Configuration;

public sealed class TableFormerSdkOptions
{
    private readonly IReadOnlyCollection<TableFormerLanguage> _supportedLanguages;

    public TableFormerSdkOptions(
        TableFormerModelPaths onnx,
        TableFormerLanguage defaultLanguage = TableFormerLanguage.English,
        IEnumerable<TableFormerLanguage>? supportedLanguages = null,
        TableVisualizationOptions? visualizationOptions = null,
        TableFormerPerformanceOptions? performanceOptions = null)
    {
        Onnx = onnx ?? throw new ArgumentNullException(nameof(onnx));

        _supportedLanguages = BuildSupportedLanguages(defaultLanguage, supportedLanguages);
        DefaultLanguage = defaultLanguage;
        Visualization = visualizationOptions ?? TableVisualizationOptions.CreateDefault();
        Performance = performanceOptions ?? TableFormerPerformanceOptions.CreateDefault();
        AvailableRuntimes = BuildAvailableRuntimes();
    }

    public TableFormerModelPaths Onnx { get; }

    public TableFormerLanguage DefaultLanguage { get; }

    public IReadOnlyCollection<TableFormerLanguage> SupportedLanguages => _supportedLanguages;

    public TableVisualizationOptions Visualization { get; }

    public TableFormerPerformanceOptions Performance { get; }

    public IReadOnlyList<TableFormerRuntime> AvailableRuntimes { get; }

    public void EnsureLanguageIsSupported(TableFormerLanguage language)
    {
        if (!_supportedLanguages.Contains(language))
        {
            throw new NotSupportedException($"Language '{language}' is not supported");
        }
    }

    private static IReadOnlyCollection<TableFormerLanguage> BuildSupportedLanguages(TableFormerLanguage defaultLanguage, IEnumerable<TableFormerLanguage>? supportedLanguages)
    {
        var languages = supportedLanguages?.Distinct().ToList() ?? new List<TableFormerLanguage> { defaultLanguage };
        if (!languages.Contains(defaultLanguage))
        {
            languages.Insert(0, defaultLanguage);
        }

        return new ReadOnlyCollection<TableFormerLanguage>(languages);
    }

    private static IReadOnlyList<TableFormerRuntime> BuildAvailableRuntimes()
        => new ReadOnlyCollection<TableFormerRuntime>(new[] { TableFormerRuntime.Onnx });
}
