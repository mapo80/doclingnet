using SkiaSharp;
using System;

namespace TableFormerSdk.Backends;

internal static class TableFormerOnnxFeatureExtractor
{
    public static long[] ExtractFeatures(SKBitmap image, int featureLength)
    {
        if (featureLength <= 0)
        {
            featureLength = 10;
        }

        var features = new long[featureLength];
        if (image.Width <= 0 || image.Height <= 0)
        {
            return features;
        }

        var stats = ComputeStatistics(image);

        features[0] = image.Width;
        if (featureLength > 1)
        {
            features[1] = image.Height;
        }

        if (featureLength > 2)
        {
            features[2] = (long)Math.Min((long)image.Width * image.Height, int.MaxValue);
        }

        if (featureLength > 3)
        {
            var aspect = image.Height == 0 ? 0d : (double)image.Width / image.Height;
            features[3] = (long)Math.Round(aspect * 1000d);
        }

        if (featureLength > 4)
        {
            features[4] = (long)Math.Round(stats.MeanLuminance * 1000d);
        }

        if (featureLength > 5)
        {
            features[5] = (long)Math.Round(stats.StandardDeviation * 1000d);
        }

        if (featureLength > 6)
        {
            features[6] = (long)Math.Round(stats.MinLuminance);
        }

        if (featureLength > 7)
        {
            features[7] = (long)Math.Round(stats.MaxLuminance);
        }

        if (featureLength > 8)
        {
            features[8] = (long)Math.Round(stats.NonWhiteRatio * 1000d);
        }

        if (featureLength > 9)
        {
            features[9] = (long)Math.Round(stats.EdgeDensity * 1000d);
        }

        for (var i = 10; i < featureLength; i++)
        {
            features[i] = features[i % 10];
        }

        return features;
    }

    private static ImageStatistics ComputeStatistics(SKBitmap image)
    {
        var width = image.Width;
        var height = image.Height;

        double sum = 0d;
        double sumSquares = 0d;
        double min = double.MaxValue;
        double max = double.MinValue;
        long nonWhite = 0;
        long total = (long)width * height;
        long edgeComparisons = 0;
        long strongEdges = 0;

        var previousRow = new double[width];

        for (var y = 0; y < height; y++)
        {
            double previousGray = 0d;
            for (var x = 0; x < width; x++)
            {
                var color = image.GetPixel(x, y);
                var gray = (0.299 * color.Red) + (0.587 * color.Green) + (0.114 * color.Blue);

                sum += gray;
                sumSquares += gray * gray;
                min = Math.Min(min, gray);
                max = Math.Max(max, gray);

                if (gray < 250d)
                {
                    nonWhite++;
                }

                if (x > 0)
                {
                    if (Math.Abs(gray - previousGray) > 25d)
                    {
                        strongEdges++;
                    }

                    edgeComparisons++;
                }

                if (y > 0)
                {
                    if (Math.Abs(gray - previousRow[x]) > 25d)
                    {
                        strongEdges++;
                    }

                    edgeComparisons++;
                }

                previousRow[x] = gray;
                previousGray = gray;
            }
        }

        var mean = total == 0 ? 0d : sum / total;
        var variance = total == 0 ? 0d : (sumSquares / total) - (mean * mean);
        if (variance < 0)
        {
            variance = 0;
        }

        var stdDev = Math.Sqrt(variance);
        var nonWhiteRatio = total == 0 ? 0d : (double)nonWhite / total;
        var edgeDensity = edgeComparisons == 0 ? 0d : (double)strongEdges / edgeComparisons;

        return new ImageStatistics(mean, stdDev, min == double.MaxValue ? 0d : min, max == double.MinValue ? 0d : max, nonWhiteRatio, edgeDensity);
    }

    private readonly record struct ImageStatistics(
        double MeanLuminance,
        double StandardDeviation,
        double MinLuminance,
        double MaxLuminance,
        double NonWhiteRatio,
        double EdgeDensity);
}
