"""
generate_pixel_sizes_csv.py

Auto-generates a pixel_sizes.csv file from all TIFF images in a folder,
using a single pixel size value for all of them.

Usage:
    python generate_pixel_sizes_csv.py \
        --images path/to/images/ \
        --um_per_px 0.1730 \
        --output pixel_sizes_ctx.csv

For cortex images:
    Scale bar: 289 pixels = 50um -> 50/289 = 0.1730 um/px
"""

import os
import glob
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate pixel_sizes.csv from a folder of images")
    parser.add_argument("--images", required=True,
                        help="Folder containing image files")
    parser.add_argument("--um_per_px", required=True, type=float,
                        help="Pixel size in micrometers (e.g. 0.1730)")
    parser.add_argument("--output", default="pixel_sizes.csv",
                        help="Output CSV filename (default: pixel_sizes.csv)")
    args = parser.parse_args()

    extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.images, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(args.images, f"*{ext.upper()}")))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {args.images}")
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "um_per_px"])
        for path in image_files:
            fname = os.path.basename(path)
            writer.writerow([fname, args.um_per_px])

    print(f"Written {len(image_files)} entries to {args.output}")
    print(f"Pixel size: {args.um_per_px} um/px (289 px = 50um)")
    print()
    print("First few entries:")
    with open(args.output) as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
