#!/usr/bin/env python3
"""
Test script to compare Python vs .NET TableFormer pipeline
"""
import time
from pathlib import Path
from PIL import Image
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from docling_ibm_models.tableformer.tableformer_predictor import TableFormerPredictor

def main():
    image_path = Path("dataset/2305.03393v1-pg9-img.png")

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    print("=" * 80)
    print("PYTHON PIPELINE TEST")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Size: {image_path.stat().st_size / 1024:.1f} KB")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image dimensions: {image.size[0]}x{image.size[1]}")

    # Step 1: Layout detection
    print("\n" + "-" * 80)
    print("STEP 1: Layout Detection")
    print("-" * 80)

    layout_start = time.time()
    layout_predictor = LayoutPredictor()
    layout_result = layout_predictor.predict(image)
    layout_time = time.time() - layout_start

    print(f"Layout detection time: {layout_time:.3f}s")
    print(f"Total cells detected: {len(layout_result.cells)}")

    # Find table cells
    table_cells = [cell for cell in layout_result.cells if cell.label == "table"]
    print(f"Table cells found: {len(table_cells)}")

    if not table_cells:
        print("No tables found in the image!")
        return

    # Step 2: TableFormer on each table
    print("\n" + "-" * 80)
    print("STEP 2: TableFormer Structure Recognition")
    print("-" * 80)

    tableformer_predictor = TableFormerPredictor()

    for idx, table_cell in enumerate(table_cells, 1):
        print(f"\nTable {idx}:")
        print(f"  BBox: ({table_cell.bbox.l:.1f}, {table_cell.bbox.t:.1f}, "
              f"{table_cell.bbox.r:.1f}, {table_cell.bbox.b:.1f})")
        print(f"  Size: {table_cell.bbox.r - table_cell.bbox.l:.1f}x"
              f"{table_cell.bbox.b - table_cell.bbox.t:.1f}")

        # Crop table region
        crop_box = (
            int(table_cell.bbox.l),
            int(table_cell.bbox.t),
            int(table_cell.bbox.r),
            int(table_cell.bbox.b)
        )
        table_image = image.crop(crop_box)

        # Run TableFormer
        tf_start = time.time()
        table_structure = tableformer_predictor.predict(table_image)
        tf_time = time.time() - tf_start

        print(f"  TableFormer time: {tf_time:.3f}s")
        print(f"  Rows detected: {table_structure.num_rows}")
        print(f"  Columns detected: {table_structure.num_cols}")
        print(f"  Cells detected: {len(table_structure.table_cells)}")

        # Show first few cells
        if table_structure.table_cells:
            print(f"  First 3 cells:")
            for cell_idx, cell in enumerate(table_structure.table_cells[:3], 1):
                print(f"    Cell {cell_idx}: row={cell.row_idx}, col={cell.col_idx}, "
                      f"rowspan={cell.row_span}, colspan={cell.col_span}")

    # Summary
    total_time = layout_time + sum(time.time() - tf_start for _ in table_cells)
    print("\n" + "=" * 80)
    print("PYTHON SUMMARY")
    print("=" * 80)
    print(f"Total layout time: {layout_time:.3f}s")
    print(f"Total tables processed: {len(table_cells)}")
    print(f"Total time: {layout_time + tf_time:.3f}s")

if __name__ == "__main__":
    main()
