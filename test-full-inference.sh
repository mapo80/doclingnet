#!/bin/bash
# Full end-to-end inference test script

set -e

echo "=== Full End-to-End Inference Test ==="
echo ""

# Paths
DATASET_DIR="/Users/politom/Documents/Workspace/personal/doclingnet/dataset"
OUTPUT_DIR="/Users/politom/Documents/Workspace/personal/doclingnet/test-inference-results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test image
TEST_IMAGE="$DATASET_DIR/2305.03393v1-pg9-img.png"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Test image not found: $TEST_IMAGE"
    exit 1
fi

echo "✓ Test image found: $TEST_IMAGE"
echo ""

# Run inference with debug enabled
echo "Running inference with full model..."
echo ""

cd /Users/politom/Documents/Workspace/personal/doclingnet

export DEBUG_LOGITS=1
export DEBUG_ENCODER=0

dotnet run --project src/Docling.Tooling/Docling.Tooling.csproj /p:TreatWarningsAsErrors=false -- \
    convert \
    --input "$TEST_IMAGE" \
    --output "$OUTPUT_DIR" \
    --table-mode fast \
    --table-debug 2>&1 | tee "$OUTPUT_DIR/inference_log.txt"

echo ""
echo "=== Test Complete ==="
echo "Output saved to: $OUTPUT_DIR"
