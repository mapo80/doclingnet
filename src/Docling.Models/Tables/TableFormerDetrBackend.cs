using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Docling.Core.Geometry;

namespace Docling.Core.Models.Tables;

/// <summary>
/// New TableFormer backend using official Docling models with DETR single-model approach.
/// Replaces the component-based approach with a unified DETR architecture.
/// </summary>
internal sealed class TableFormerDetrBackend : IDisposable
{
    private readonly InferenceSession _session;
    private readonly ImagePreprocessor _preprocessor;
    private readonly string _modelPath;
    private readonly float _confidenceThreshold = 0.25f; // Lowered from 0.5 as per plan
    private bool _disposed;

    public TableFormerDetrBackend(string modelPath)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };

        _session = new InferenceSession(modelPath, sessionOptions);
        _preprocessor = new ImagePreprocessor();
    }

    /// <summary>
    /// Process table image using the official Docling DETR model.
    /// </summary>
    public IReadOnlyList<TableFormerDetection> Infer(SKBitmap image, BoundingBox tableBounds)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (tableBounds.IsEmpty)
        {
            throw new ArgumentException("Table bounds cannot be empty", nameof(tableBounds));
        }

        // Step 1: Preprocess image with letterboxing and ImageNet normalization
        var inputTensor = _preprocessor.PreprocessImage(image);

        // Step 2: Run ONNX inference
        var (logits, bboxes) = RunInference(inputTensor);

        // Step 3: Parse DETR output [batch, num_queries, num_classes] and [batch, num_queries, 4]
        var detections = ParseDetections(logits, bboxes);

        // Step 4: Filter by confidence threshold and apply NMS
        var filteredDetections = FilterDetections(detections);

        // Step 5: Transform coordinates back to original image space
        var transformedDetections = TransformCoordinates(
            filteredDetections, tableBounds, image.Width, image.Height);

        return transformedDetections;
    }

    private (DenseTensor<float> logits, DenseTensor<float> bboxes) RunInference(DenseTensor<float> input)
    {
        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, input) };

        using var results = _session.Run(inputs);


        var logits = results.First(x => x.Name == "logits").AsTensor<float>().ToDenseTensor();
        var bboxes = results.First(x => x.Name == "pred_boxes").AsTensor<float>().ToDenseTensor();

        return (logits, bboxes);
    }

    private List<RawDetection> ParseDetections(DenseTensor<float> logits, DenseTensor<float> bboxes)
    {
        var detections = new List<RawDetection>();

        var logitsArray = logits.ToArray();
        var bboxesArray = bboxes.ToArray();

        // Assuming shape [1, num_queries, num_classes] for logits
        // and [1, num_queries, 4] for bboxes
        var batchSize = logits.Dimensions[0];
        var numQueries = logits.Dimensions[1];
        var numClasses = logits.Dimensions[2];

        for (int query = 0; query < numQueries; query++)
        {
            // Find the class with highest probability for this query
            var queryStart = query * numClasses;
            var maxClassProb = 0f;
            var maxClassIndex = 0;

            for (int classIdx = 0; classIdx < numClasses; classIdx++)
            {
                var prob = logitsArray[queryStart + classIdx];
                if (prob > maxClassProb)
                {
                    maxClassProb = prob;
                    maxClassIndex = classIdx;
                }
            }

            // Get bounding box for this query
            var bboxStart = query * 4;
            var cx = bboxesArray[bboxStart];
            var cy = bboxesArray[bboxStart + 1];
            var w = bboxesArray[bboxStart + 2];
            var h = bboxesArray[bboxStart + 3];

            detections.Add(new RawDetection
            {
                ClassId = maxClassIndex,
                Confidence = maxClassProb,
                BoundingBox = (cx, cy, w, h)
            });
        }

        return detections;
    }

    private List<TableFormerDetection> FilterDetections(List<RawDetection> detections)
    {
        // Filter by confidence threshold
        var filtered = detections
            .Where(d => d.Confidence >= _confidenceThreshold)
            .OrderByDescending(d => d.Confidence)
            .ToList();

        // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        return ApplyNMS(filtered);
    }

    private List<TableFormerDetection> ApplyNMS(List<RawDetection> detections)
    {
        const float nmsThreshold = 0.5f;
        var result = new List<TableFormerDetection>();
        var remaining = detections.ToList();

        while (remaining.Count > 0)
        {
            // Take the detection with highest confidence
            var best = remaining.OrderByDescending(d => d.Confidence).First();
            result.Add(new TableFormerDetection
            {
                BoundingBox = best.BoundingBox,
                Confidence = best.Confidence,
                ClassId = best.ClassId
            });

            // Remove detections that overlap significantly with the best one
            remaining.RemoveAll(d => IoU(best.BoundingBox, d.BoundingBox) > nmsThreshold);
        }

        return result;
    }

    private static float IoU((float cx1, float cy1, float w1, float h1) box1,
                           (float cx2, float cy2, float w2, float h2) box2)
    {
        // Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        var x1_1 = box1.cx1 - box1.w1 / 2;
        var y1_1 = box1.cy1 - box1.h1 / 2;
        var x2_1 = box1.cx1 + box1.w1 / 2;
        var y2_1 = box1.cy1 + box1.h1 / 2;

        var x1_2 = box2.cx2 - box2.w2 / 2;
        var y1_2 = box2.cy2 - box2.h2 / 2;
        var x2_2 = box2.cx2 + box2.w2 / 2;
        var y2_2 = box2.cy2 + box2.h2 / 2;

        // Calculate intersection
        var interX1 = Math.Max(x1_1, x1_2);
        var interY1 = Math.Max(y1_1, y1_2);
        var interX2 = Math.Min(x2_1, x2_2);
        var interY2 = Math.Min(y2_1, y2_2);

        var interWidth = Math.Max(0, interX2 - interX1);
        var interHeight = Math.Max(0, interY2 - interY1);
        var interArea = interWidth * interHeight;

        // Calculate union
        var area1 = box1.w1 * box1.h1;
        var area2 = box2.w2 * box2.h2;
        var unionArea = area1 + area2 - interArea;

        return unionArea > 0 ? interArea / unionArea : 0;
    }

    private IReadOnlyList<TableFormerDetection> TransformCoordinates(
        List<TableFormerDetection> detections,
        BoundingBox tableBounds,
        int originalWidth,
        int originalHeight)
    {
        var transformed = new List<TableFormerDetection>();

        foreach (var detection in detections)
        {
            var (cx, cy, w, h) = detection.BoundingBox;

            // Transform coordinates using preprocessor
            var (left, top, right, bottom) = ImagePreprocessor.TransformCoordinates(
                cx, cy, w, h, originalWidth, originalHeight);

            // Convert to table-relative coordinates
            var tableX = (left - tableBounds.Left) / tableBounds.Width;
            var tableY = (top - tableBounds.Top) / tableBounds.Height;
            var tableWidth = (right - left) / tableBounds.Width;
            var tableHeight = (bottom - top) / tableBounds.Height;

            transformed.Add(new TableFormerDetection
            {
                BoundingBox = ((float)tableX, (float)tableY, (float)tableWidth, (float)tableHeight),
                Confidence = detection.Confidence,
                ClassId = detection.ClassId
            });
        }

        return transformed;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session.Dispose();
            _preprocessor.Dispose();
            _disposed = true;
        }
    }

    /// <summary>
    /// Raw detection result from DETR model.
    /// </summary>
    private sealed class RawDetection
    {
        public int ClassId { get; set; }
        public float Confidence { get; set; }
        public (float cx1, float cy1, float w1, float h1) BoundingBox { get; set; }
    }

    /// <summary>
    /// Final detection result with transformed coordinates.
    /// </summary>
    public sealed class TableFormerDetection
    {
        public (float cx, float cy, float w, float h) BoundingBox { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; }
    }
}