using System;

namespace Docling.Tooling.Parity;

/// <summary>
/// Provides configuration knobs for normalising pipeline outputs prior to parity comparisons.
/// </summary>
internal sealed class ParityExtractionOptions
{
    private double _coordinateTolerance = 0.01d;
    private int _coordinateDecimals = 4;
    private string? _baseDirectory;

    /// <summary>
    /// Gets or sets the base directory used to relativise absolute paths.
    /// When set, any absolute path encountered during extraction is converted to a path relative to this directory.
    /// </summary>
    public string? BaseDirectory
    {
        get => _baseDirectory;
        init => _baseDirectory = string.IsNullOrWhiteSpace(value) ? null : value;
    }

    /// <summary>
    /// Gets or sets the coordinate tolerance applied when rounding floating point values.
    /// A tolerance greater than zero will round values to the nearest multiple of the tolerance before
    /// applying <see cref="CoordinateDecimals"/>.
    /// </summary>
    public double CoordinateTolerance
    {
        get => _coordinateTolerance;
        init
        {
            if (value <= 0 || double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Coordinate tolerance must be a positive finite value.");
            }

            _coordinateTolerance = value;
        }
    }

    /// <summary>
    /// Gets or sets the number of decimal digits preserved after tolerance normalisation.
    /// </summary>
    public int CoordinateDecimals
    {
        get => _coordinateDecimals;
        init
        {
            if (value < 0 || value > 6)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Coordinate decimals must be between 0 and 6.");
            }

            _coordinateDecimals = value;
        }
    }

    /// <summary>
    /// Gets a default set of options suitable for regression parity snapshots.
    /// </summary>
    public static ParityExtractionOptions Default { get; } = new();
}
