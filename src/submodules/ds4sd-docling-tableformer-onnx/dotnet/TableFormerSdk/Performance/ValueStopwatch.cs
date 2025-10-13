using System;
using System.Diagnostics;

namespace TableFormerSdk.Performance;

internal readonly struct ValueStopwatch
{
    private static readonly double TimestampToTicks = TimeSpan.TicksPerSecond / (double)Stopwatch.Frequency;
    private readonly long _startTimestamp;

    private ValueStopwatch(long startTimestamp)
    {
        _startTimestamp = startTimestamp;
    }

    public static ValueStopwatch StartNew() => new(Stopwatch.GetTimestamp());

    public TimeSpan GetElapsedTime()
    {
        var end = Stopwatch.GetTimestamp();
        var delta = end - _startTimestamp;
        var ticks = (long)(delta * TimestampToTicks);
        return TimeSpan.FromTicks(ticks);
    }
}
