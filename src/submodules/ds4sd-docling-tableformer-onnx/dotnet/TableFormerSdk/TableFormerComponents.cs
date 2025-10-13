// File: TableFormerComponents.cs
// Contiene le classi necessarie per l'implementazione completa del backend TableFormer
// Copiate dal progetto principale per completare la FASE 4

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TableFormerSdk.Backends;

// BoundingBox copiata da Docling.Core.Geometry
public readonly record struct BoundingBox : IEquatable<BoundingBox>
{
    private const double Epsilon = 1e-9;

    public BoundingBox(double left, double top, double right, double bottom)
    {
        if (!IsValid(left, top, right, bottom))
        {
            throw new ArgumentOutOfRangeException(nameof(right), "Bounding box coordinates are invalid.");
        }

        Left = left;
        Top = top;
        Right = right;
        Bottom = bottom;
    }

    public double Left { get; }
    public double Top { get; }
    public double Right { get; }
    public double Bottom { get; }

    public bool IsEmpty => Width <= Epsilon || Height <= Epsilon;
    public double Width => Right - Left;
    public double Height => Bottom - Top;
    public double Area => Width * Height;

    public static BoundingBox FromSize(double left, double top, double width, double height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(width, nameof(width));
        ArgumentOutOfRangeException.ThrowIfNegative(height, nameof(height));
        return new BoundingBox(left, top, left + width, top + height);
    }

    public static bool TryCreate(double left, double top, double right, double bottom, out BoundingBox box)
    {
        if (!IsValid(left, top, right, bottom))
        {
            box = default;
            return false;
        }
        box = new BoundingBox(left, top, right, bottom);
        return true;
    }

    public BoundingBox Scale(double scaleX, double scaleY)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(scaleX, nameof(scaleX));
        ArgumentOutOfRangeException.ThrowIfNegative(scaleY, nameof(scaleY));
        return new BoundingBox(Left, Top, Left + (Width * scaleX), Top + (Height * scaleY));
    }

    public BoundingBox Translate(double deltaX, double deltaY) =>
        new(Left + deltaX, Top + deltaY, Right + deltaX, Bottom + deltaY);

    public bool Intersects(BoundingBox other)
    {
        if (IsEmpty || other.IsEmpty)
        {
            return false;
        }
        return !(other.Left >= Right || other.Right <= Left || other.Top >= Bottom || other.Bottom <= Top);
    }

    public BoundingBox Union(BoundingBox other)
    {
        var left = Math.Min(Left, other.Left);
        var top = Math.Min(Top, other.Top);
        var right = Math.Max(Right, other.Right);
        var bottom = Math.Max(Bottom, other.Bottom);
        return new BoundingBox(left, top, right, bottom);
    }

    public bool Contains(double x, double y) =>
        x >= Left && x <= Right && y >= Top && y <= Bottom;

    internal static bool IsValid(double left, double top, double right, double bottom) =>
        !double.IsNaN(left) && !double.IsNaN(top) && !double.IsNaN(right) && !double.IsNaN(bottom) &&
        !double.IsInfinity(left) && !double.IsInfinity(top) && !double.IsInfinity(right) && !double.IsInfinity(bottom) &&
        right >= left && bottom >= top;
}

/// <summary>
/// ONNX session wrappers for the 4-component TableFormer architecture.
/// This replaces the old single-model approach with a component-wise design.
/// </summary>
internal sealed class TableFormerOnnxComponents : IDisposable
{
    private readonly InferenceSession _encoderSession;
    private readonly InferenceSession _tagTransformerEncoderSession;
    private readonly InferenceSession _tagTransformerDecoderStepSession;
    private readonly InferenceSession _bboxDecoderSession;

    private readonly string _encoderInputName;
    private readonly string _tagTransformerEncoderInputName;
    private readonly string _tagTransformerDecoderStepInputNames;
    private readonly string _bboxDecoderInputNames;

    public TableFormerOnnxComponents(string modelsDirectory) : this(modelsDirectory, enableOptimizations: true)
    {
    }

    public TableFormerOnnxComponents(string modelsDirectory, bool enableOptimizations = true, bool useCUDA = false, bool enableQuantization = false)
    {
        if (string.IsNullOrWhiteSpace(modelsDirectory))
        {
            throw new ArgumentException("Models directory cannot be empty", nameof(modelsDirectory));
        }

        var sessionOptions = CreateOptimizedSessionOptions(useCUDA, enableQuantization, enableOptimizations);

        // Initialize encoder session - use new path in submodules
        var encoderPath = Path.Combine(modelsDirectory, "tableformer_fast_encoder.onnx");
        _encoderSession = new InferenceSession(encoderPath, sessionOptions);
        _encoderInputName = _encoderSession.InputMetadata.Keys.First();

        // Initialize tag transformer encoder session
        var tagTransformerEncoderPath = Path.Combine(modelsDirectory, "tableformer_fast_tag_transformer_encoder.onnx");
        _tagTransformerEncoderSession = new InferenceSession(tagTransformerEncoderPath, sessionOptions);
        _tagTransformerEncoderInputName = _tagTransformerEncoderSession.InputMetadata.Keys.First();

        // Initialize tag transformer decoder step session
        var tagTransformerDecoderStepPath = Path.Combine(modelsDirectory, "tableformer_fast_tag_transformer_decoder_step.onnx");
        _tagTransformerDecoderStepSession = new InferenceSession(tagTransformerDecoderStepPath, sessionOptions);
        _tagTransformerDecoderStepInputNames = string.Join(",",
            _tagTransformerDecoderStepSession.InputMetadata.Keys);

        // Initialize bbox decoder session
        var bboxDecoderPath = Path.Combine(modelsDirectory, "tableformer_fast_bbox_decoder.onnx");
        _bboxDecoderSession = new InferenceSession(bboxDecoderPath, sessionOptions);
        _bboxDecoderInputNames = string.Join(",",
            _bboxDecoderSession.InputMetadata.Keys);

        sessionOptions.Dispose();
    }

    /// <summary>
    /// Encoder: (1,3,448,448) → (1,28,28,256)
    /// </summary>
    public DenseTensor<float> RunEncoder(DenseTensor<float> input)
    {
        var inputNamed = NamedOnnxValue.CreateFromTensor(_encoderInputName, input);
        using var results = _encoderSession.Run(new[] { inputNamed });
        (inputNamed as IDisposable)?.Dispose();

        var outputName = _encoderSession.OutputMetadata.Keys.First();
        var outputTensor = results.First(x => x.Name == outputName).AsTensor<float>();
        return outputTensor.ToDenseTensor();
    }

    /// <summary>
    /// Tag Transformer Encoder: (1,28,28,256) → (784,1,512)
    /// </summary>
    public DenseTensor<float> RunTagTransformerEncoder(DenseTensor<float> encoderOutput)
    {
        var inputNamed = NamedOnnxValue.CreateFromTensor(_tagTransformerEncoderInputName, encoderOutput);
        using var results = _tagTransformerEncoderSession.Run(new[] { inputNamed });
        (inputNamed as IDisposable)?.Dispose();

        var outputName = _tagTransformerEncoderSession.OutputMetadata.Keys.First();
        var outputTensor = results.First(x => x.Name == outputName).AsTensor<float>();
        return outputTensor.ToDenseTensor();
    }

    /// <summary>
    /// Tag Transformer Decoder Step: tags + memory → logits + hidden state
    /// </summary>
    public (DenseTensor<float> logits, DenseTensor<float> hiddenState) RunTagTransformerDecoderStep(
        DenseTensor<long> decodedTags,
        DenseTensor<float> memory,
        DenseTensor<bool>? encoderMask = null)
    {
        // Note: encoder_mask is not used by the ONNX model, but kept as parameter for API compatibility
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("decoded_tags", decodedTags),
            NamedOnnxValue.CreateFromTensor("memory", memory)
        };

        using var results = _tagTransformerDecoderStepSession.Run(inputs);
        foreach (var input in inputs)
        {
            (input as IDisposable)?.Dispose();
        }

        var logitsTensor = results.First(x => x.Name == "logits").AsTensor<float>();
        var hiddenStateTensor = results.First(x => x.Name == "tag_hidden").AsTensor<float>();

        return (logitsTensor.ToDenseTensor(), hiddenStateTensor.ToDenseTensor());
    }

    /// <summary>
    /// BBox Decoder: encoder features + tag hidden states → bbox classes + coordinates
    /// </summary>
    public (DenseTensor<float> bboxClasses, DenseTensor<float> bboxCoords) RunBboxDecoder(
        DenseTensor<float> encoderOutput,
        DenseTensor<float> tagHiddenStates)
    {
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encoderOutput),
            NamedOnnxValue.CreateFromTensor("tag_hiddens", tagHiddenStates)
        };

        using var results = _bboxDecoderSession.Run(inputs);
        foreach (var input in inputs)
        {
            (input as IDisposable)?.Dispose();
        }

        var bboxClassesTensor = results.First(x => x.Name == "bbox_classes").AsTensor<float>();
        var bboxCoordsTensor = results.First(x => x.Name == "bbox_coords").AsTensor<float>();

        return (bboxClassesTensor.ToDenseTensor(), bboxCoordsTensor.ToDenseTensor());
    }

    public void Dispose()
    {
        _encoderSession.Dispose();
        _tagTransformerEncoderSession.Dispose();
        _tagTransformerDecoderStepSession.Dispose();
        _bboxDecoderSession.Dispose();
    }

    /// <summary>
    /// Create optimized ONNX Runtime session options for maximum performance.
    /// </summary>
    private static SessionOptions CreateOptimizedSessionOptions(bool useCUDA = false, bool enableQuantization = false, bool enableOptimizations = true)
    {
        var options = new SessionOptions();

        // Graph optimizations
        if (enableOptimizations)
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            // Enable memory optimizations
            options.EnableMemoryPattern = true;
            options.ExecutionMode = ExecutionMode.ORT_PARALLEL;

            // Optimize for inference
            options.AddSessionConfigEntry("session.intra_op_thread_affinity.enable", "1");
            options.AddSessionConfigEntry("session.inter_op_thread_affinity.enable", "1");
        }
        else
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        }

        // Thread configuration
        options.IntraOpNumThreads = 0; // Auto-detect optimal thread count
        options.InterOpNumThreads = 0; // Auto-detect optimal inter-op threads

        // Provider selection
        if (useCUDA && IsCUDAProviderAvailable())
        {
            // Configure CUDA provider if available
            options.AppendExecutionProvider_CUDA(0); // Use GPU 0
            options.AddSessionConfigEntry("session.execution_mode", "sequential");
        }
        else
        {
            // Use CPU with optimizations
            options.AppendExecutionProvider_CPU(0);
        }

        // Quantization settings
        if (enableQuantization)
        {
            // Enable quantization if supported
            options.AddSessionConfigEntry("session.qdq_enable", "1");
            options.AddSessionConfigEntry("session.qdq_matmul_enable", "1");
        }

        // Memory optimizations
        options.EnableCpuMemArena = true;
        options.EnableMemoryPattern = true;

        // Set memory limit to prevent excessive memory usage
        options.AddSessionConfigEntry("session.memory_optimizer", "1");

        return options;
    }

    /// <summary>
    /// Check if CUDA provider is available on the current system.
    /// </summary>
    private static bool IsCUDAProviderAvailable()
    {
        try
        {
            // Try to create a session with CUDA provider to test availability
            var testOptions = new SessionOptions();
            testOptions.AppendExecutionProvider_CUDA(0);
            testOptions.Dispose();
            return true;
        }
        catch
        {
            return false;
        }
    }
}

/// <summary>
/// Implements the autoregressive loop for TableFormer tag generation.
/// Handles OTSL (Ordered Table Structure Language) sequence generation with error correction.
/// </summary>
internal sealed class TableFormerAutoregressive
{
    private readonly TableFormerOnnxComponents _components;
    private readonly TableFormerSdk.Configuration.TableFormerWordMap _wordMap;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToToken;
    private readonly int _maxSteps = 1024;
    private readonly int _startTokenId;
    private readonly int _endTokenId;
    private readonly int _padTokenId;

    public TableFormerAutoregressive(TableFormerOnnxComponents components, TableFormerSdk.Configuration.TableFormerWordMap wordMap)
    {
        _components = components ?? throw new ArgumentNullException(nameof(components));
        _wordMap = wordMap ?? throw new ArgumentNullException(nameof(wordMap));

        // Load token mappings from word map
        _tokenToId = wordMap.WordMapTag ?? throw new InvalidOperationException("WordMapTag is null");

        // Build reverse mapping
        _idToToken = _tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        // Cache special token IDs
        var specialTokens = _wordMap.GetSpecialTokens();
        _startTokenId = specialTokens.start;
        _endTokenId = specialTokens.end;
        _padTokenId = specialTokens.pad;
    }

    /// <summary>
    /// Result from autoregressive generation.
    /// </summary>
    public sealed class AutoregressiveResult
    {
        public List<DenseTensor<float>> TagHiddenStates { get; init; } = new();
        public List<string> GeneratedTokens { get; init; } = new();
    }

    /// <summary>
    /// Generate OTSL tag sequence autoregressively.
    /// Returns the sequence of tag hidden states for bbox prediction and the generated tokens.
    /// </summary>
    public AutoregressiveResult GenerateTags(
        DenseTensor<float> memory,
        DenseTensor<bool> encoderMask)
    {
        var tagHiddenStates = new List<DenseTensor<float>>();
        var generatedTokenIds = new List<long> { _startTokenId };
        var generatedTokenStrings = new List<string> { "<start>" };

        // Initialize with start token: shape [seq_len=1, batch_size=1]
        var currentTags = new DenseTensor<long>(new[] { 1, 1 });
        currentTags[0, 0] = _startTokenId;

        var step = 0;

        while (step < _maxSteps)
        {
            // Run decoder step with correct shape [seq_len, batch_size]
            var (logits, hiddenState) = _components.RunTagTransformerDecoderStep(
                currentTags, memory, encoderMask);

            // Get next token (greedy decoding)
            var nextToken = GetNextToken(logits);

            // Early stopping on <end> token
            if (nextToken == _endTokenId)
            {
                generatedTokenStrings.Add("<end>");
                break;
            }

            // Apply structure error correction
            var correctedToken = ApplyStructureCorrection(generatedTokenIds, nextToken);

            // Add corrected token to sequence
            generatedTokenIds.Add(correctedToken);

            // Get token string
            var tokenStr = _idToToken.TryGetValue((int)correctedToken, out var token) ? token : "<unk>";
            generatedTokenStrings.Add(tokenStr);

            // Update current tags for next step: shape [seq_len, batch_size]
            var newCurrentTags = new DenseTensor<long>(new[] { generatedTokenIds.Count, 1 });
            for (int i = 0; i < generatedTokenIds.Count; i++)
            {
                newCurrentTags[i, 0] = generatedTokenIds[i];
            }

            currentTags = newCurrentTags;

            // Collect hidden state for bbox prediction (only for cell tokens)
            if (TableFormerSdk.Configuration.TableFormerWordMap.IsCellToken(tokenStr))
            {
                tagHiddenStates.Add(hiddenState);
            }

            step++;
        }

        // Return fallback if no cells generated
        if (tagHiddenStates.Count == 0)
        {
            // Suppress CA1303: Localization not required for debug message
            #pragma warning disable CA1303
            Console.WriteLine("Warning: No cell tokens generated in autoregressive loop");
            #pragma warning restore CA1303
        }

        return new AutoregressiveResult
        {
            TagHiddenStates = tagHiddenStates,
            GeneratedTokens = generatedTokenStrings
        };
    }

    private static long GetNextToken(DenseTensor<float> logits)
    {
        // Simple greedy decoding - take argmax
        var logitsArray = logits.ToArray();
        var maxIndex = 0;
        var maxValue = float.NegativeInfinity;

        for (int i = 0; i < logitsArray.Length; i++)
        {
            if (logitsArray[i] > maxValue)
            {
                maxValue = logitsArray[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private long ApplyStructureCorrection(List<long> generatedTokens, long nextToken)
    {
        if (!_idToToken.TryGetValue((int)nextToken, out var nextTokenStr))
        {
            return nextToken;
        }

        // Rule 1: First line should use lcel instead of xcel
        // In the first row, "xcel" (vertical continuation) doesn't make sense
        if (nextTokenStr == "xcel" && IsFirstLine(generatedTokens))
        {
            return _tokenToId["lcel"];
        }

        // Rule 2: After ucel (vertical span start), lcel should become fcel
        // This prevents incorrect horizontal linking after vertical spans
        if (nextTokenStr == "lcel" && generatedTokens.Count > 0)
        {
            var lastToken = generatedTokens[^1];
            if (_idToToken.TryGetValue((int)lastToken, out var lastTokenStr) && lastTokenStr == "ucel")
            {
                return _tokenToId["fcel"];
            }
        }

        return nextToken;
    }

    private bool IsFirstLine(List<long> tokens)
    {
        // Check if we're still in the first row (no "nl" token yet)
        var nlTokenId = _tokenToId["nl"];
        return !tokens.Contains(nlTokenId);
    }
}

/// <summary>
/// Parses OTSL (Ordered Table Structure Language) sequences into table structures.
/// Handles cell spans, headers, and table layout construction.
/// </summary>
internal sealed class OtslParser
{
    /// <summary>
    /// Represents a table cell with position and span information.
    /// </summary>
    public sealed class TableCell
    {
        public int Row { get; set; }
        public int Col { get; set; }
        public int RowSpan { get; set; } = 1;
        public int ColSpan { get; set; } = 1;
        public string CellType { get; set; } = "";
        public bool IsHeader { get; set; }
    }

    /// <summary>
    /// Represents a complete table structure.
    /// </summary>
    public sealed class TableStructure
    {
        public List<List<TableCell>> Rows { get; } = new();
        public int RowCount => Rows.Count;
        public int ColCount => Rows.Count > 0 ? Rows.Max(row => row.Count) : 0;
    }

    /// <summary>
    /// Parse OTSL token sequence into table structure.
    /// </summary>
    public static TableStructure ParseOtsl(IEnumerable<string> otslTokens)
    {
        var tokens = otslTokens.ToList();
        var table = new TableStructure();
        var currentRow = new List<TableCell>();
        var currentRowIndex = 0;
        var currentColIndex = 0;

        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];

            switch (token)
            {
                case "fcel": // First cell in row
                    // Only add previous row if it has cells (not for first fcel)
                    if (currentRow.Count > 0)
                    {
                        table.Rows.Add(currentRow);
                        currentRow = new List<TableCell>();
                        currentRowIndex++;
                    }
                    currentColIndex = 0;

                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex,
                        Col = currentColIndex,
                        CellType = "fcel"
                    });
                    currentColIndex++;
                    break;

                case "lcel": // Linked cell (horizontal span)
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex,
                        Col = currentColIndex,
                        CellType = "lcel"
                    });
                    currentColIndex++;
                    break;

                case "ecel": // Empty cell
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex,
                        Col = currentColIndex,
                        CellType = "ecel"
                    });
                    currentColIndex++;
                    break;

                case "ucel": // Up cell (vertical span start)
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex,
                        Col = currentColIndex,
                        CellType = "ucel"
                    });
                    currentColIndex++;
                    break;

                case "xcel": // Cross cell (vertical span continuation)
                    currentRow.Add(new TableCell
                    {
                        Row = currentRowIndex,
                        Col = currentColIndex,
                        CellType = "xcel"
                    });
                    currentColIndex++;
                    break;

                case "ched": // Column header
                    if (currentRow.Count > 0)
                    {
                        currentRow.Last().IsHeader = true;
                        currentRow.Last().CellType = "ched";
                    }
                    break;

                case "rhed": // Row header
                    if (currentRow.Count > 0)
                    {
                        currentRow.Last().IsHeader = true;
                        currentRow.Last().CellType = "rhed";
                    }
                    break;

                case "srow": // Spanning row
                    if (currentRow.Count > 0)
                    {
                        currentRow.Last().RowSpan = 2; // Default span, could be made configurable
                    }
                    break;

                case "nl": // New line (row separator)
                    if (currentRow.Count > 0)
                    {
                        table.Rows.Add(currentRow);
                        currentRow = new List<TableCell>();
                    }
                    currentRowIndex++;
                    currentColIndex = 0;
                    break;

                case "<end>":
                case "<pad>":
                    // End of sequence or padding, stop processing
                    goto end_of_sequence;

                default:
                    // Unknown token, skip
                    break;
            }
        }

        end_of_sequence:

        // Add the last row if it has cells
        if (currentRow.Count > 0)
        {
            table.Rows.Add(currentRow);
        }

        // Post-process to calculate spans
        CalculateSpans(table);

        return table;
    }

    private static void CalculateSpans(TableStructure table)
    {
        // Calculate horizontal spans (lcel)
        foreach (var row in table.Rows)
        {
            for (int col = 0; col < row.Count; col++)
            {
                var cell = row[col];

                // Count consecutive lcel cells for horizontal span
                if (cell.CellType == "fcel" || cell.CellType == "lcel")
                {
                    var span = 1;
                    var startCol = col;

                    // Look ahead for consecutive lcel
                    for (int nextCol = col + 1; nextCol < row.Count; nextCol++)
                    {
                        if (row[nextCol].CellType == "lcel")
                        {
                            span++;
                        }
                        else
                        {
                            break;
                        }
                    }

                    // Update the first cell in the span
                    if (span > 1)
                    {
                        row[startCol].ColSpan = span;
                        // Mark linked cells as not visible (they're part of the span)
                        for (int linkedCol = startCol + 1; linkedCol < startCol + span; linkedCol++)
                        {
                            row[linkedCol].CellType = "linked";
                        }
                    }
                }
            }
        }

        // Calculate vertical spans (ucel/xcel)
        for (int row = 0; row < table.Rows.Count; row++)
        {
            for (int col = 0; col < table.Rows[row].Count; col++)
            {
                var cell = table.Rows[row][col];

                if (cell.CellType == "ucel")
                {
                    var span = 1;

                    // Look down for xcel in the same column
                    for (int nextRow = row + 1; nextRow < table.Rows.Count; nextRow++)
                    {
                        if (nextRow < table.Rows.Count &&
                            col < table.Rows[nextRow].Count &&
                            table.Rows[nextRow][col].CellType == "xcel")
                        {
                            span++;
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (span > 1)
                    {
                        cell.RowSpan = span;
                        // Mark continuation cells as not visible
                        for (int spanRow = row + 1; spanRow < row + span; spanRow++)
                        {
                            if (col < table.Rows[spanRow].Count)
                            {
                                table.Rows[spanRow][col].CellType = "spanned";
                            }
                        }
                    }
                }
            }
        }
    }
}