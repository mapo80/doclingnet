#!/usr/bin/env python3
"""
Extract text from image using layout detection results.
Orders text by Y position (top to bottom) to maintain reading order.
"""
import sys
from PIL import Image
import pytesseract

# Layout detection results (from CLI output)
layout_elements = [
    {"type": "Page-header", "x": 404.76, "y": 194.26, "w": 526.47, "h": 18.61},
    {"type": "Page-header", "x": 990.57, "y": 195.01, "w": 10.10, "h": 14.51},
    {"type": "Text", "x": 279.81, "y": 246.79, "w": 721.73, "h": 64.76},
    {"type": "Section-header", "x": 280.19, "y": 350.81, "w": 382.18, "h": 20.20},
    {"type": "Text", "x": 279.08, "y": 384.86, "w": 723.04, "h": 144.97},
    {"type": "Caption", "x": 279.17, "y": 571.08, "w": 721.71, "h": 111.55},
    {"type": "Table", "x": 291.11, "y": 703.14, "w": 698.20, "h": 275.40},
    {"type": "Section-header", "x": 280.30, "y": 1056.53, "w": 269.76, "h": 20.13},
    {"type": "Text", "x": 278.99, "y": 1091.87, "w": 723.02, "h": 194.31},
    {"type": "Text", "x": 278.77, "y": 1290.62, "w": 723.05, "h": 95.86},
]

def extract_text_from_box(image, box):
    """Extract text from a bounding box using pytesseract."""
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    crop = image.crop((x, y, x + w, y + h))
    text = pytesseract.image_to_string(crop, lang='eng').strip()
    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_text_from_layout.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = image_path.replace('.png', '_extracted_text.txt')

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)

    # Sort layout elements by Y position (top to bottom)
    sorted_elements = sorted(layout_elements, key=lambda e: e["y"])

    print(f"Extracting text from {len(sorted_elements)} layout elements...")
    results = []

    for idx, element in enumerate(sorted_elements, 1):
        elem_type = element["type"]
        print(f"  [{idx}/{len(sorted_elements)}] Processing {elem_type}...", end=" ")

        text = extract_text_from_box(image, element)

        if text:
            print(f"✓ ({len(text)} chars)")
            results.append({
                "type": elem_type,
                "text": text,
                "y": element["y"]
            })
        else:
            print("✗ (no text)")

    # Write to output file
    print(f"\nWriting results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"=== {result['type'].upper()} ===\n")
            f.write(f"{result['text']}\n")
            f.write("\n")

    print(f"Done! Extracted text from {len(results)} elements.")

if __name__ == "__main__":
    main()
