"""
Tile full-resolution images + masks into fixed-size tiles for training,
and splits data at the sample level (not tile level) to avoid leakage.

Directory convention expected:
    data/
        images/   img001.tif, img002.tif, ...
        rois/     img001_RoiSet.zip, img002_RoiSet.zip, ...
    (one ROI file per image, matched by filename stem)
"""

import os
import glob
import numpy as np
from PIL import Image

from roi_to_mask import rois_to_mask


def build_dataset_index(image_dir, roi_dir, image_ext=".tif", roi_suffix="_RoiSet.zip"):
    """
    Match each image file to its corresponding ROI file by filename stem.
    Returns a list of dicts: [{"image_path":..., "roi_path":..., "stem":...}, ...]

    Raises if any image is missing a matching ROI file, or vice versa --
    silent mismatches here are a common, hard-to-debug source of bugs.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, f"*{image_ext}")))
    if not image_paths:
        raise FileNotFoundError(f"No images with extension {image_ext} found in {image_dir}")

    records = []
    missing = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        roi_path = os.path.join(roi_dir, f"{stem}{roi_suffix}")
        if not os.path.exists(roi_path):
            missing.append((stem, roi_path))
            continue
        records.append({"image_path": img_path, "roi_path": roi_path, "stem": stem})

    if missing:
        msg = "\n".join(f"  {stem}: expected {p}" for stem, p in missing)
        raise FileNotFoundError(
            f"{len(missing)} image(s) have no matching ROI file:\n{msg}\n"
            f"Fix naming/matching before proceeding -- do not silently drop these."
        )

    print(f"Matched {len(records)} image/ROI pairs.")
    return records


def load_image_and_mask(record):
    """Load one full-resolution RGB image and its rasterized binary mask."""
    img = np.array(Image.open(record["image_path"]).convert("RGB"))
    mask = rois_to_mask(record["roi_path"], image_shape=img.shape[:2])
    return img, mask


def auto_crop_to_tissue(img, mask, black_threshold=10, padding=64):
    """
    Crop image and mask to the bounding box of non-black tissue pixels,
    removing artificial black borders from ROI-masked images (e.g. striatum
    images where everything outside the region of interest is zeroed out).

    Works universally: for images with no black border (full tissue sections),
    the detected bounding box covers essentially the whole image and the crop
    is a no-op. For cropped images, such as our striatum samples, it trims the 
    black padding and returns only the tissue-containing region.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, 3), uint8
    mask : np.ndarray, shape (H, W), uint8
    black_threshold : int
        Pixels with mean gray value <= this are considered black padding.
        Same threshold used in the tile filter (default 10).
    padding : int
        Number of pixels to add around the detected tissue bounding box,
        so the crop doesn't cut right at the tissue edge. Clamped to image
        boundaries automatically. Default 64px (1/4 of a tile).

    Returns
    -------
    img_cropped : np.ndarray, same dtype as input
    mask_cropped : np.ndarray, same dtype as input
    crop_box : tuple (y0, y1, x0, x1) -- the crop coordinates applied,
        useful for debugging or converting ROI coordinates if needed.
    """
    H, W = img.shape[:2]
    gray = img.mean(axis=-1)
    tissue = gray > black_threshold

    # Find rows and columns that contain at least one tissue pixel
    rows_with_tissue = np.any(tissue, axis=1)
    cols_with_tissue = np.any(tissue, axis=0)

    if not rows_with_tissue.any():
        # Pathological case: entire image is black -- return unchanged
        print("  [warning] auto_crop_to_tissue: no tissue pixels found, "
              "returning image unchanged")
        return img, mask, (0, H, 0, W)

    y0 = max(0, int(np.argmax(rows_with_tissue)) - padding)
    y1 = min(H, int(len(rows_with_tissue) - np.argmax(rows_with_tissue[::-1])) + padding)
    x0 = max(0, int(np.argmax(cols_with_tissue)) - padding)
    x1 = min(W, int(len(cols_with_tissue) - np.argmax(cols_with_tissue[::-1])) + padding)

    img_cropped = img[y0:y1, x0:x1]
    mask_cropped = mask[y0:y1, x0:x1]

    return img_cropped, mask_cropped, (y0, y1, x0, x1)


def load_image_and_mask_cropped(record, black_threshold=10, padding=64,
                                 crop_black_border=True):
    """
    Like load_image_and_mask, but optionally auto-crops black borders first.

    Use this as a drop-in replacement for load_image_and_mask when your
    dataset contains a mix of full-tissue images (no cropping needed) and
    ROI-masked images (black border should be cropped). The auto-detection
    is conservative -- full-tissue images will not be meaningfully cropped.

    Parameters
    ----------
    crop_black_border : bool
        If True, apply auto_crop_to_tissue before returning.
        If False, behaves identically to load_image_and_mask.
    """
    img = np.array(Image.open(record["image_path"]).convert("RGB"))
    mask = rois_to_mask(record["roi_path"], image_shape=img.shape[:2])

    if not crop_black_border:
        return img, mask

    img_orig_shape = img.shape
    img, mask, (y0, y1, x0, x1) = auto_crop_to_tissue(
        img, mask, black_threshold=black_threshold, padding=padding)

    if img.shape != img_orig_shape:
        frac_retained = (img.shape[0] * img.shape[1]) / (img_orig_shape[0] * img_orig_shape[1])
        print(f"  [{record['stem']}] cropped {img_orig_shape[:2]} -> {img.shape[:2]} "
              f"({frac_retained:.0%} of original area retained)")

    return img, mask


def tile_image_and_mask(img, mask, tile_size=256, stride=None, min_tissue_frac=0.05):
    """
    Split a full-resolution image+mask pair into fixed-size tiles.

    Parameters
    ----------
    tile_size : int
        Height/width of square tiles.
    stride : int or None
        Step between tile origins. Defaults to tile_size (no overlap).
        Use stride < tile_size for overlapping tiles (more patches, more
        redundancy -- helpful when you have very few source images).
    min_tissue_frac : float
        Skip tiles where less than this fraction of pixels are "tissue"
        (non-background, approximated here as non-near-white/black).
        Prevents wasting training time/capacity on empty slide background.
        Set to 0 to disable this filter.

    Returns
    -------
    img_tiles : list of np.ndarray, shape (tile_size, tile_size, 3)
    mask_tiles : list of np.ndarray, shape (tile_size, tile_size)
    """
    if stride is None:
        stride = tile_size

    H, W = mask.shape
    img_tiles, mask_tiles = [], []

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            img_tile = img[y:y + tile_size, x:x + tile_size]
            mask_tile = mask[y:y + tile_size, x:x + tile_size]

            if min_tissue_frac > 0:
                # Tissue heuristic: exclude tiles that are mostly empty background.
                # Two types of background need filtering:
                #   1. Near-white (bright field background): gray > 240
                #   2. Near-black (artificial ROI masking, e.g. striatum images
                #      where everything outside the region of interest is zeroed
                #      out, producing pure black padding). Without this check,
                #      black-padded tiles are incorrectly counted as "tissue"
                #      since black pixels (gray~0) are well below the 240 threshold.
                # Tissue = pixels that are NEITHER near-white NOR near-black.
                gray = img_tile.mean(axis=-1)
                tissue_frac = ((gray > 10) & (gray < 240)).mean()
                if tissue_frac < min_tissue_frac:
                    continue

            img_tiles.append(img_tile)
            mask_tiles.append(mask_tile)

    return img_tiles, mask_tiles


def split_records_by_slide(records, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split at the sample level so patches from the same source image never
    appear in more than one of train/val/test. This is the split that
    matters for getting an honest performance estimate.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(records))
    n = len(records)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    print(f"Slide-level split: {len(train)} train / {len(val)} val / {len(test)} test "
          f"(out of {n} total slides)")
    return train, val, test


def build_tile_arrays(records, tile_size=256, stride=None, min_tissue_frac=0.05,
                       crop_black_border=True):
    """
    Load + tile every record in a list, concatenate into single arrays.
    Use this per-split (train/val/test) after split_records_by_slide.

    crop_black_border : bool
        If True (default), auto-crops black borders before tiling using
        auto_crop_to_tissue. Safe to leave on for all images -- full tissue
        images without black borders are unaffected (crop is a no-op).
        This is the main fix for striatum-style images with ROI masking.
    """
    all_img_tiles, all_mask_tiles = [], []
    for rec in records:
        img, mask = load_image_and_mask_cropped(rec, crop_black_border=crop_black_border)
        # Area fraction after crop -- for striatum images this now reflects
        # vessels / tissue area (matching manual measurement), not vessels / full image.
        frac = mask.sum() / mask.size
        print(f"  {rec['stem']}: image {img.shape}, mask area fraction = {frac:.4f}")

        img_tiles, mask_tiles = tile_image_and_mask(
            img, mask, tile_size=tile_size, stride=stride, min_tissue_frac=min_tissue_frac
        )
        all_img_tiles.extend(img_tiles)
        all_mask_tiles.extend(mask_tiles)

    X = np.stack(all_img_tiles).astype(np.float32) / 255.0
    Y = np.stack(all_mask_tiles).astype(np.float32)[..., np.newaxis]  # add channel dim
    print(f"Built {X.shape[0]} tiles total, shape {X.shape[1:3]}")
    return X, Y


if __name__ == "__main__":
    # --- Self-test with synthetic data (no real images needed) ---
    # Simulate a 600x600 "image" and a mask with a few blobs, confirm tiling
    # produces the expected number of non-empty tiles and correct shapes.
    rng = np.random.RandomState(0)
    fake_img = (rng.rand(600, 600, 3) * 255).astype(np.uint8)
    fake_mask = np.zeros((600, 600), dtype=np.uint8)
    fake_mask[100:150, 100:150] = 1
    fake_mask[400:450, 400:470] = 1

    # make sure "tissue" heuristic doesn't filter everything out -- darken image
    fake_img = (fake_img * 0.5).astype(np.uint8)

    img_tiles, mask_tiles = tile_image_and_mask(fake_img, fake_mask, tile_size=200, stride=200,
                                                  min_tissue_frac=0.0)
    print(f"Generated {len(img_tiles)} tiles from a 600x600 image at tile_size=200, stride=200")
    assert len(img_tiles) == 9, f"Expected 9 tiles (3x3 grid), got {len(img_tiles)}"
    assert img_tiles[0].shape == (200, 200, 3)
    assert mask_tiles[0].shape == (200, 200)

    # Check that the tile containing [100:150,100:150] actually has the vessel pixels
    total_mask_pixels = sum(t.sum() for t in mask_tiles)
    assert total_mask_pixels == fake_mask.sum(), \
        f"Pixel count mismatch: tiles sum to {total_mask_pixels}, mask sum is {fake_mask.sum()}"

    print("PASS: tiling self-test OK (correct tile count, shapes, and pixel conservation)")
