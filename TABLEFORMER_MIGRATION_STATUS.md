# TableFormer Migration Status

**Last Updated**: 2025-10-12
**Current Phase**: Phase 2 - COMPLETED ✅

## Overall Progress

| Phase | Status | Duration | Completion Date |
|-------|--------|----------|----------------|
| **Phase 1**: Analysis and Preparation | ✅ COMPLETED | 4 hours | 2025-10-12 |
| **Phase 2**: ONNX Conversion | ✅ COMPLETED | 2 hours | 2025-10-12 |
| **Phase 3**: Cleanup | ⏳ PENDING | 2-3 hours (est) | - |
| **Phase 4**: .NET Implementation | ⏳ PENDING | 8-12 hours (est) | - |
| **Phase 5**: Integration | ⏳ PENDING | 3-4 hours (est) | - |
| **Phase 6**: Optimization | ⏳ PENDING | 4-6 hours (est) | - |
| **Phase 7**: Documentation | ⏳ PENDING | 2-3 hours (est) | - |
| **Phase 8**: Validation | ⏳ PENDING | 3-4 hours (est) | - |

**Total Progress**: 2/8 phases (25% complete)
**Time Spent**: 6 hours
**Time Remaining**: 22-32 hours (estimated)

## Phase Details

### ✅ Phase 1: Analysis and Preparation (COMPLETED)

**Achievements**:
- ✅ Cloned docling-ibm-models repository
- ✅ Analyzed TableModel04_rs architecture
- ✅ Downloaded models from HuggingFace (fast + accurate variants)
- ✅ Documented OTSL language and preprocessing requirements
- ✅ Identified key differences from old implementation
- ✅ Created comprehensive architecture analysis document

**Key Discoveries**:
- Uses PubTabNet normalization (mean ~0.94, std ~0.18) NOT ImageNet
- Autoregressive model generating OTSL tags
- Output format: tag sequence + bbox classes + bbox coordinates
- ResNet-18 backbone outputting (1, 28, 28, 256)
- Complex transformer encoder-decoder with multi-head attention

**Deliverables**:
- [docs/TABLEFORMER_ARCHITECTURE_ANALYSIS.md](docs/TABLEFORMER_ARCHITECTURE_ANALYSIS.md)
- [FASE1_COMPLETED_SUMMARY.md](FASE1_COMPLETED_SUMMARY.md)
- Downloaded models in `models/tableformer/fast/` and `models/tableformer/accurate/`

### ✅ Phase 2: ONNX Conversion (COMPLETED)

**Achievements**:
- ✅ Created component-wise conversion strategy
- ✅ Exported 4 ONNX models successfully:
  - Encoder (11 MB)
  - Tag Transformer Encoder (64 MB)
  - Tag Transformer Decoder Step (26 MB)
  - BBox Decoder (38 MB)
- ✅ Validated all models with ONNX checker
- ✅ Exported config and word map

**Challenges Overcome**:
1. **Bbox Classes Dimension Mismatch**: Config specified 2 classes, model outputs 3 (num_classes+1)
   - Solution: Dynamic dimension detection
2. **Autoregressive Loop Export**: Monolithic export timed out after 5+ minutes
   - Solution: Split into components, implement loop in C# .NET

**Deliverables**:
- [tools/convert_tableformer_components_to_onnx.py](tools/convert_tableformer_components_to_onnx.py) ✅ Working
- [FASE2_COMPLETED_SUMMARY.md](FASE2_COMPLETED_SUMMARY.md)
- ONNX models in `models/tableformer-onnx/`:
  - `tableformer_fast_encoder.onnx`
  - `tableformer_fast_tag_transformer_encoder.onnx`
  - `tableformer_fast_tag_transformer_decoder_step.onnx`
  - `tableformer_fast_bbox_decoder.onnx`
  - `tableformer_fast_config.json`
  - `tableformer_fast_wordmap.json`

### ⏳ Phase 3: Cleanup (PENDING)

**Objectives**:
- Remove old ONNX models from `src/submodules/ds4sd-docling-tableformer-onnx/models/`
- Remove obsolete backend implementations
- Clean up old SDK code

**Estimated Duration**: 2-3 hours

### ⏳ Phase 4: New .NET Implementation (PENDING)

**Objectives**:
- Create new backend structure for component-based inference
- Implement autoregressive loop in C#
- OTSL parser for table structure
- BBox merging and coordinate transformation

**Key Tasks**:
1. Load 4 ONNX models (2 hours)
2. Autoregressive loop implementation (3-4 hours)
3. BBox prediction and merging (2 hours)
4. OTSL parser (2-3 hours)
5. Coordinate transformation (1 hour)
6. Integration (1-2 hours)

**Estimated Duration**: 8-12 hours

### ⏳ Phase 5: Integration (PENDING)

**Objectives**:
- Update `TableFormerTableStructureService.cs`
- Wire up new backend
- Update configuration and options

**Estimated Duration**: 3-4 hours

### ⏳ Phase 6-8: Optimization, Documentation, Validation (PENDING)

**Estimated Duration**: 9-13 hours combined

## Technical Architecture

### Current State (After Phase 2)

```
Python Side (Conversion):
  SafeTensors Models
    ↓
  Component Export Script
    ↓
  4 ONNX Models + Config + WordMap

.NET Side (To Be Implemented):
  4 ONNX InferenceSessions
    ↓
  Autoregressive Loop (C#)
    ↓
  OTSL Parser (C#)
    ↓
  Table Structure Output
```

### Component Architecture

```
Input Image (1, 3, 448, 448)
    ↓
[ONNX 1] Encoder
    ↓
Encoder Features (1, 28, 28, 256)
    ↓
[ONNX 2] Tag Transformer Encoder
    ↓
Memory (784, 1, 512)
    ↓
    ┌─────────────────────┐
    │ C# Autoregressive   │
    │ Loop                │
    │ (max 1024 steps)    │
    └─────────────────────┘
            ↓
    [ONNX 3] Decoder Step (called N times)
            ↓
    Tag Sequence + Hidden States
            ↓
    [ONNX 4] BBox Decoder
            ↓
    BBox Classes + Coordinates
            ↓
    C# OTSL Parser + BBox Merger
            ↓
    Final Table Structure
```

## Key Files

### Documentation
- `PIANO_INTERVENTO_TABLEFORMER_NEW.md` - Full intervention plan
- `docs/TABLEFORMER_ARCHITECTURE_ANALYSIS.md` - Architecture analysis
- `FASE1_COMPLETED_SUMMARY.md` - Phase 1 summary
- `FASE2_COMPLETED_SUMMARY.md` - Phase 2 summary
- `TABLEFORMER_MIGRATION_STATUS.md` - This file (overall status)

### Python Tools
- `tools/convert_tableformer_components_to_onnx.py` - Component converter (✅ working)
- `tools/convert_tableformer_to_onnx.py` - Monolithic converter (deprecated)

### Models
- `models/tableformer/fast/` - Original SafeTensors models (139 MB)
- `models/tableformer/accurate/` - Original SafeTensors models (203 MB)
- `models/tableformer-onnx/` - Exported ONNX components (139 MB fast variant)

### .NET Code (To Be Modified)
- `src/Docling.Models/Tables/TableFormerTableStructureService.cs` - Main service
- `src/submodules/ds4sd-docling-tableformer-onnx/dotnet/TableFormerSdk/` - SDK to be rewritten

## Next Actions

**Immediate**: Await user approval to proceed with Phase 3 (Cleanup)

**Phase 3 Tasks**:
1. Backup old models and code
2. Remove obsolete ONNX models
3. Remove obsolete backend implementations
4. Clean up SDK code structure

**Questions for User**:
- Should we also convert the "accurate" variant now, or wait until fast variant is fully integrated?
- Do you want to review the ONNX models before proceeding with cleanup?
- Any specific performance requirements or constraints for the .NET implementation?

## Risk Assessment

**LOW RISK**:
- ✅ ONNX models validated and working
- ✅ Clear component architecture
- ✅ Python implementation as reference

**MEDIUM RISK**:
- ⚠️ Autoregressive loop complexity in C# (mitigated: have Python reference)
- ⚠️ OTSL parsing correctness (mitigated: comprehensive test cases)
- ⚠️ Performance optimization needed (planned in Phase 6)

**MITIGATION STRATEGIES**:
- Keep old code until new implementation is validated
- Comprehensive unit tests for each component
- Regression testing against Python golden outputs
- Incremental integration with validation at each step

## Success Criteria

Phase 3-4 will be considered successful when:
- ✅ Old models and code removed
- ✅ New backend loads all 4 ONNX models
- ✅ Autoregressive loop generates valid OTSL sequences
- ✅ OTSL parser produces correct table structure
- ✅ BBox coordinates correctly transformed
- ✅ Integration tests pass with test images

Final success (Phase 8):
- ✅ TableFormer returns non-zero cell counts
- ✅ Cell coordinates match golden Python output
- ✅ Row and column counts correct
- ✅ Performance acceptable (<5s per table on CPU)
- ✅ All regression tests pass
