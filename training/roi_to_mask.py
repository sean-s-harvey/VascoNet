"""
Convert ImageJ/Fiji RoiSet.zip files (manually traced vessel ROIs) into
binary segmentation masks aligned to the original RGB IHC image.

Usage:
    mask = rois_to_mask("RoiSet.zip", image_shape=(H, W))
"""

import numpy as np
import roifile
from PIL import Image, ImageDraw


def rois_to_mask(roi_path, image_shape):
    """
    Rasterize all ROIs in a .roi or RoiSet.zip file into a single binary mask.

    Parameters
    ----------
    roi_path : str
        Path to a .roi file or a RoiSet.zip containing multiple ROIs.
    image_shape : tuple (H, W)
        Shape of the corresponding image, so the mask is pixel-aligned.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype uint8
        Binary mask: 1 = inside a traced vessel ROI, 0 = background.
        Overlapping ROIs are simply unioned (still 1, not double-counted).
    """
    H, W = image_shape
    mask_img = Image.new("L", (W, H), 0)  # PIL uses (width, height)
    draw = ImageDraw.Draw(mask_img)

    rois = roifile.roiread(roi_path)
    # roiread returns a single ImagejRoi if there's only one, else a list
    if not isinstance(rois, list):
        rois = [rois]

    n_filled = 0
    for roi in rois:
        coords = roi.coordinates()
        if coords is None or len(coords) < 3:
            # Not a fillable polygon (e.g., a point or line ROI) — skip,
            # but warn so you notice if this happens unexpectedly.
            print(f"  [warning] skipped non-polygon ROI (type={roi.roitype}, "
                  f"name={getattr(roi, 'name', '?')})")
            continue
        polygon = [(float(x), float(y)) for x, y in coords]
        draw.polygon(polygon, outline=1, fill=1)
        n_filled += 1

    mask = np.array(mask_img, dtype=np.uint8)
    if n_filled == 0:
        print(f"  [warning] no fillable ROIs found in {roi_path} — mask is all zeros")
    return mask


def mask_area_fraction(mask):
    """Vascular area fraction = fraction of pixels labeled vessel."""
    return float(mask.sum()) / float(mask.size)


if __name__ == "__main__":
    # Self-test using the synthetic 10x10 square ROI created earlier.
    # Expected area ~ 100 px^2 (polygon fill of a 10x10 square, edge-inclusive
    # rasterization may give slightly more, e.g. 121 for an 11x11 inclusive fill —
    # that's expected and fine; the point is it should be close, not exactly 100).
    mask = rois_to_mask("test_roi.roi", image_shape=(40, 40))
    area = mask.sum()
    frac = mask_area_fraction(mask)
    print(f"Test mask area: {area} px (expected ~100-121 px for a 10x10 square)")
    print(f"Test mask area fraction: {frac:.4f} (expected ~{100/1600:.4f}-{121/1600:.4f})")
    assert 90 <= area <= 130, f"Sanity check failed: area={area}, expected ~100-121"
    print("PASS: rasterization sanity check OK")
