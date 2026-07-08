"""
check_pixel_sizes.py

Extracts pixel size / resolution metadata from every image in data/images/,
printing a summary table so you can confirm which images share a pixel size
and which need separate calibration values.

Run from your vessel_seg directory:
    python3 check_pixel_sizes.py

Checks multiple metadata sources in order of reliability:
  1. TIFF XResolution/YResolution tags (most reliable if present)
  2. ImageJ metadata embedded in the ImageDescription tag
  3. PIL dpi field (often unreliable -- placeholder values common)
  4. Filename hints (e.g. "10x" in the filename)
  5. Image dimensions (useful context for cross-checking)
"""

import os
import glob
import re
import struct

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("[warning] tifffile not installed -- some metadata may be missed")
    print("  Install with: pip install tifffile")

from PIL import Image
import numpy as np


IMAGE_DIR = "data/images"


def extract_pixel_size(image_path):
    """
    Try to extract pixel size (um/px) from a TIFF image using multiple methods.
    Returns a dict with all available information.
    """
    result = {
        "path":          image_path,
        "stem":          os.path.splitext(os.path.basename(image_path))[0],
        "dimensions":    None,
        "pixel_size_um": None,
        "source":        None,    # where the pixel size came from
        "raw_dpi":       None,
        "magnification": None,    # extracted from filename if present
        "notes":         [],
    }

    # --- PIL: basic info ---
    try:
        img = Image.open(image_path)
        result["dimensions"] = img.size  # (W, H)
        result["raw_dpi"] = img.info.get("dpi")
    except Exception as e:
        result["notes"].append(f"PIL error: {e}")
        return result

    # --- Method 1: tifffile XResolution/YResolution tags ---
    if HAS_TIFFFILE:
        try:
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]

                # XResolution tag (tag 282): stored as (numerator, denominator)
                # ResolutionUnit tag (tag 296): 1=no unit, 2=inch, 3=cm
                x_res = page.tags.get("XResolution")
                res_unit = page.tags.get("ResolutionUnit")

                if x_res is not None:
                    val = x_res.value
                    # value can be a tuple (num, denom) or a float
                    if isinstance(val, tuple):
                        num, denom = val
                        res_px_per_unit = num / denom if denom != 0 else None
                    else:
                        res_px_per_unit = float(val)

                    unit_val = res_unit.value if res_unit else 2  # default inches

                    if res_px_per_unit and res_px_per_unit > 1:
                        if unit_val == 2:    # pixels per inch
                            um_per_px = 25400.0 / res_px_per_unit
                        elif unit_val == 3:  # pixels per cm
                            um_per_px = 10000.0 / res_px_per_unit
                        else:
                            um_per_px = None

                        if um_per_px and 0.01 < um_per_px < 100:
                            result["pixel_size_um"] = round(um_per_px, 4)
                            result["source"] = "TIFF XResolution tag"

                # ImageJ metadata -- often has calibration
                ij_meta = getattr(tif, "imagej_metadata", None) or {}
                if ij_meta:
                    unit = ij_meta.get("unit", "")
                    spacing = ij_meta.get("spacing") or ij_meta.get("finterval")
                    x_um = ij_meta.get("xresolution") or ij_meta.get("PhysicalSizeX")
                    if x_um and 0.01 < float(x_um) < 100:
                        if result["pixel_size_um"] is None:
                            result["pixel_size_um"] = round(float(x_um), 4)
                            result["source"] = f"ImageJ metadata (unit={unit})"

                # ZEN / other software: sometimes in ImageDescription as XML/text
                img_desc = page.tags.get("ImageDescription")
                if img_desc:
                    desc_text = str(img_desc.value)
                    # Look for pixel size patterns like "PhysicalSizeX="0.325""
                    for pattern in [
                        r'PhysicalSizeX[=:\"\']\s*([\d.]+)',
                        r'pixel.size[=:\"\']\s*([\d.]+)',
                        r'Scale\s*=\s*([\d.]+)\s*um',
                        r'VoxelSizeX[=:\"\']\s*([\d.]+)',
                    ]:
                        match = re.search(pattern, desc_text, re.IGNORECASE)
                        if match:
                            val = float(match.group(1))
                            if 0.01 < val < 100:
                                if result["pixel_size_um"] is None:
                                    result["pixel_size_um"] = round(val, 4)
                                    result["source"] = f"ImageDescription ({pattern})"

        except Exception as e:
            result["notes"].append(f"tifffile error: {e}")

    # --- Method 2: PIL DPI (often unreliable but worth noting) ---
    if result["raw_dpi"] and result["pixel_size_um"] is None:
        dpi = result["raw_dpi"]
        if isinstance(dpi, tuple):
            dpi_val = dpi[0]
        else:
            dpi_val = dpi
        try:
            dpi_float = float(dpi_val)
            if dpi_float > 1:  # filter out placeholder (1, 1) DPI
                um_per_px = 25400.0 / dpi_float
                if 0.01 < um_per_px < 100:
                    result["pixel_size_um"] = round(um_per_px, 4)
                    result["source"] = f"PIL DPI ({dpi_float:.1f} dpi)"
                    result["notes"].append("PIL DPI may be unreliable -- verify")
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    # --- Method 3: Magnification hint in filename ---
    stem = result["stem"]
    mag_match = re.search(r'(\d+)[xX]', stem)
    if mag_match:
        result["magnification"] = f"{mag_match.group(1)}x"
        result["notes"].append(f"Filename suggests {result['magnification']} objective")

    # --- Flag if nothing found ---
    if result["pixel_size_um"] is None:
        result["notes"].append("No pixel size found in metadata -- manual check needed")

    return result


def print_summary(results):
    """Print a formatted summary table."""
    print()
    print("=" * 100)
    print(f"{'Stem':<20} {'Dimensions (WxH)':<22} {'Pixel size (um/px)':<22} "
          f"{'Source':<30} {'Notes'}")
    print("=" * 100)

    pixel_sizes_seen = {}

    for r in results:
        dims = f"{r['dimensions'][0]}x{r['dimensions'][1]}" \
               if r["dimensions"] else "unknown"
        px = f"{r['pixel_size_um']:.4f}" if r["pixel_size_um"] else "UNKNOWN"
        src = r["source"] or "-"
        notes = "; ".join(r["notes"]) if r["notes"] else ""

        print(f"{r['stem']:<20} {dims:<22} {px:<22} {src:<30} {notes}")

        if r["pixel_size_um"]:
            key = round(r["pixel_size_um"], 3)
            pixel_sizes_seen.setdefault(key, []).append(r["stem"])

    print()
    print("=" * 100)
    print("PIXEL SIZE GROUPS:")
    for px_size, stems in sorted(pixel_sizes_seen.items()):
        print(f"  {px_size:.4f} um/px ({len(stems)} images): {', '.join(stems)}")

    unknown = [r["stem"] for r in results if r["pixel_size_um"] is None]
    if unknown:
        print(f"  UNKNOWN ({len(unknown)} images): {', '.join(unknown)}")
        print()
        print("  --> For unknown images: check acquisition settings, or measure")
        print("      a known structure (e.g. scale bar) in the image manually.")

    print()
    print(f"Total images checked: {len(results)}")

    # Sanity check against known original cohort value
    original_cohort_px = round(50.0 / 298.0, 4)  # 0.1678
    print()
    print(f"Reference: original cohort pixel size = {original_cohort_px:.4f} um/px "
          f"(from scale bar: 298px = 50um)")


if __name__ == "__main__":
    extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*{ext.upper()}")))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {IMAGE_DIR}/")
        print("Make sure you're running from the vessel_seg directory.")
        exit(1)

    print(f"Checking pixel size metadata for {len(image_paths)} images...")

    results = []
    for path in image_paths:
        results.append(extract_pixel_size(path))

    print_summary(results)
