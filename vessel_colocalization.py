"""
vessel_colocalization.py

Co-localization analysis between VascoNet vessel predictions and a
second fluorescence marker channel.

Given:
  - A predicted binary vessel mask (from VascoNet)
  - An RGB image containing a second marker of interest in a known channel

Computes:
  - % of vessel area that is positive for the second marker
    ("what fraction of vasculature expresses this protein?")
  - % of second marker signal that co-localizes with vessels
    ("how vascular is this marker's expression?")
  - A co-localization overlay image for visual QC

The second channel is thresholded using Otsu's method by default
(automatically separates signal from background without manual input),
with an option to supply a fixed threshold instead.

Usage example:
    from vessel_colocalization import run_colocalization

    results = run_colocalization(
        image_path   = "my_image.tif",
        pred_mask    = vasconet_pred_mask,    # binary np.ndarray (H, W)
        marker_channel = "red",               # channel containing marker of interest
        vessel_channel   = "green",             # channel VascoNet used for vessels
        output_dir   = "results/",
        pixel_size_um = 0.1730,               # optional, for reporting in um2
    )

    print(results["pct_vessel_positive_for_marker"])
    print(results["pct_marker_on_vessel"])
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ── Channel helpers ───────────────────────────────────────────────────────────

CHANNEL_MAP = {"red": 0, "green": 1, "blue": 2}


def extract_channel(img_rgb, channel_name):
    """
    Extract a single channel from an RGB image array.

    Parameters
    ----------
    img_rgb : np.ndarray, shape (H, W, 3), dtype uint8
    channel_name : str -- "red", "green", or "blue"

    Returns
    -------
    np.ndarray, shape (H, W), dtype uint8
    """
    channel_name = channel_name.lower().strip()
    if channel_name not in CHANNEL_MAP:
        raise ValueError(
            f"channel_name must be 'red', 'green', or 'blue', got '{channel_name}'"
        )
    return img_rgb[:, :, CHANNEL_MAP[channel_name]]


# ── Thresholding ──────────────────────────────────────────────────────────────

def otsu_threshold(channel_array):
    """
    Compute Otsu's optimal threshold for a single-channel image.

    Otsu's method finds the intensity value that minimizes within-class
    variance (equivalently, maximizes between-class variance), automatically
    separating signal from background without requiring manual input.

    This is the standard approach in fluorescence microscopy for unbiased
    thresholding and is equivalent to what ImageJ's "Auto Threshold > Otsu"
    does.

    Parameters
    ----------
    channel_array : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    int -- the optimal threshold value (pixels > threshold are "positive")
    """
    # Build intensity histogram (256 bins for uint8)
    hist, bin_edges = np.histogram(channel_array.flatten(), bins=256,
                                   range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()

    # Compute cumulative sums and means
    weight_bg = np.cumsum(hist) / total
    weight_fg = 1.0 - weight_bg

    cumsum = np.cumsum(hist * np.arange(256))
    mean_bg = np.divide(cumsum, np.cumsum(hist),
                        out=np.zeros(256), where=np.cumsum(hist) > 0)
    mean_total = cumsum[-1] / total
    mean_fg = np.divide(
        (mean_total * total - cumsum),
        (total - np.cumsum(hist)),
        out=np.zeros(256),
        where=(total - np.cumsum(hist)) > 0
    )

    # Between-class variance
    between_class_var = (weight_bg * weight_fg *
                         (mean_bg - mean_fg) ** 2)

    return int(np.argmax(between_class_var))


def threshold_channel(channel_array, threshold=None):
    """
    Apply a threshold to a single channel to produce a binary mask.

    Parameters
    ----------
    channel_array : np.ndarray, shape (H, W), dtype uint8
    threshold : int or None
        If None, Otsu's method is used automatically.
        If int, pixels > threshold are considered positive.

    Returns
    -------
    binary_mask : np.ndarray, shape (H, W), dtype bool
    threshold_used : int
    """
    if threshold is None:
        threshold = otsu_threshold(channel_array)
    return channel_array > threshold, threshold


# ── Core co-localization computation ─────────────────────────────────────────

def compute_colocalization(vessel_mask, marker_mask):
    """
    Compute co-localization metrics between a vessel mask and a marker mask.

    Parameters
    ----------
    vessel_mask : np.ndarray, shape (H, W), dtype bool or uint8
        Binary mask where 1 = vessel (from VascoNet prediction)
    marker_mask : np.ndarray, shape (H, W), dtype bool or uint8
        Binary mask where 1 = marker positive (from thresholding)

    Returns
    -------
    dict with keys:
        vessel_area_px          : int   -- total vessel pixels
        marker_area_px          : int   -- total marker-positive pixels
        overlap_px              : int   -- pixels positive for both
        pct_vessel_positive     : float -- % of vessel area co-localizing
                                          with marker ("vessel+marker / vessel")
        pct_marker_on_vessel    : float -- % of marker signal on vessels
                                          ("vessel+marker / marker")
        jaccard_index           : float -- overlap / union (symmetric measure)
    """
    vessel_mask  = vessel_mask.astype(bool)
    marker_mask  = marker_mask.astype(bool)

    vessel_area  = int(vessel_mask.sum())
    marker_area  = int(marker_mask.sum())
    overlap      = int((vessel_mask & marker_mask).sum())
    union        = int((vessel_mask | marker_mask).sum())

    pct_vessel_positive  = (overlap / vessel_area  * 100) if vessel_area  > 0 else 0.0
    pct_marker_on_vessel = (overlap / marker_area  * 100) if marker_area  > 0 else 0.0
    jaccard              = (overlap / union         )      if union        > 0 else 0.0

    return {
        "vessel_area_px":       vessel_area,
        "marker_area_px":       marker_area,
        "overlap_px":           overlap,
        "pct_vessel_positive":  round(pct_vessel_positive,  3),
        "pct_marker_on_vessel": round(pct_marker_on_vessel, 3),
        "jaccard_index":        round(jaccard,              4),
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def save_colocalization_overlay(img_rgb, vessel_mask, marker_mask,
                                 output_path, stem,
                                 vessel_channel, marker_channel):
    """
    Save a 4-panel co-localization QC image:
      1. Original image
      2. VascoNet vessel mask
      3. Second marker binary mask (after thresholding)
      4. Co-localization overlay:
           - Vessels only        → blue
           - Marker only         → red
           - Co-localization     → yellow (vessels + marker)
    """
    vessel_mask = vessel_mask.astype(bool)
    marker_mask = marker_mask.astype(bool)

    # Build co-localization colour image
    coloc_img = np.zeros((*vessel_mask.shape, 3), dtype=np.uint8)
    coloc_img[vessel_mask & ~marker_mask] = [0,   100, 255]  # blue  = vessel only
    coloc_img[marker_mask & ~vessel_mask] = [255, 50,  50]   # red   = marker only
    coloc_img[vessel_mask & marker_mask]  = [255, 230, 0]    # yellow = co-localizing

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"{stem} — co-localization: {vessel_channel} (vessel) vs "
                 f"{marker_channel} (marker)", fontsize=11, y=1.01)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original image"); axes[0].axis("off")

    axes[1].imshow(vessel_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Vessel mask\n({vessel_channel} channel, VascoNet)")
    axes[1].axis("off")

    axes[2].imshow(marker_mask, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title(f"Marker mask\n({marker_channel} channel, Otsu threshold)")
    axes[2].axis("off")

    axes[3].imshow(coloc_img)
    axes[3].set_title("Co-localization\n■ Blue=vessel  ■ Red=marker  ■ Yellow=both")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_masks(vessel_mask, marker_mask, overlap_mask,
               output_dir, stem):
    """
    Export binary masks as PNG files for use in other software
    (e.g. ImageJ, CellProfiler, custom analysis pipelines).

    Saves:
        <stem>_vessel_mask.png      -- VascoNet vessel prediction (binary)
        <stem>_marker_mask.png      -- thresholded second marker (binary)
        <stem>_colocalization_mask.png -- pixels positive for both (binary)
    """
    os.makedirs(output_dir, exist_ok=True)

    def to_png(mask, path):
        Image.fromarray((mask.astype(np.uint8) * 255)).save(path)

    to_png(vessel_mask,  os.path.join(output_dir, f"{stem}_vessel_mask.png"))
    to_png(marker_mask,  os.path.join(output_dir, f"{stem}_marker_mask.png"))
    to_png(overlap_mask, os.path.join(output_dir, f"{stem}_colocalization_mask.png"))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_colocalization(img_rgb, pred_mask, marker_channel,
                       vessel_channel="green",
                       stem="image",
                       output_dir=None,
                       marker_threshold=None,
                       pixel_size_um=None):
    """
    Run co-localization analysis between VascoNet vessel predictions and a
    second fluorescence marker channel.

    Parameters
    ----------
    img_rgb : np.ndarray, shape (H, W, 3), dtype uint8
        The original RGB image (after channel normalization, pre-crop).
    pred_mask : np.ndarray, shape (H, W), dtype float32 or bool
        Binary vessel mask output from VascoNet (thresholded at 0.5).
    marker_channel : str
        Which RGB channel contains the second marker of interest.
        Must be "red", "green", or "blue", and must differ from vessel_channel.
    vessel_channel : str
        Which channel was used for vessel detection (default "green").
        Used only for labeling outputs.
    stem : str
        Image name/stem used for output filenames.
    output_dir : str or None
        If provided, saves overlay PNG and binary mask PNGs here.
        If None, no files are saved (metrics only).
    marker_threshold : int or None
        Pixel intensity threshold for the second channel.
        If None (default), Otsu's method is used automatically.
        Values 0–255. Pixels > threshold are "positive".
    pixel_size_um : float or None
        Pixel size in micrometers. If provided, areas are also reported in um².

    Returns
    -------
    dict with co-localization metrics and (if output_dir provided) output paths.
    """
    if marker_channel.lower() == vessel_channel.lower():
        raise ValueError(
            f"marker_channel ('{marker_channel}') and vessel_channel "
            f"('{vessel_channel}') must be different channels."
        )

    vessel_mask = (pred_mask > 0.5).astype(bool)

    # Extract and threshold second marker channel
    marker_ch = extract_channel(img_rgb, marker_channel)
    marker_mask, threshold_used = threshold_channel(marker_ch, marker_threshold)

    overlap_mask = vessel_mask & marker_mask

    # Compute metrics
    metrics = compute_colocalization(vessel_mask, marker_mask)
    metrics["marker_threshold_used"] = threshold_used
    metrics["threshold_method"] = "otsu" if marker_threshold is None else "user-defined"

    # Add area in um² if pixel size known
    if pixel_size_um is not None:
        px_area = pixel_size_um ** 2
        metrics["vessel_area_um2"]  = round(metrics["vessel_area_px"] * px_area, 2)
        metrics["marker_area_um2"]  = round(metrics["marker_area_px"] * px_area, 2)
        metrics["overlap_area_um2"] = round(metrics["overlap_px"]     * px_area, 2)

    # Print summary
    print(f"\n  Co-localization results for {stem}:")
    print(f"    Vessel area:                  {metrics['vessel_area_px']:>8,} px")
    print(f"    Marker area ({marker_channel}):        "
          f"{metrics['marker_area_px']:>8,} px")
    print(f"    Overlap (both):               {metrics['overlap_px']:>8,} px")
    print(f"    % vessel co-localizing:       {metrics['pct_vessel_positive']:>7.2f}%  "
          f"← % of vessel area positive for {marker_channel} marker")
    print(f"    % marker on vessel:           {metrics['pct_marker_on_vessel']:>7.2f}%  "
          f"← % of {marker_channel} marker signal on vessels")
    print(f"    Jaccard index:                {metrics['jaccard_index']:>7.4f}")
    print(f"    Marker threshold (Otsu):      {threshold_used}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save overlay QC image
        overlay_path = os.path.join(output_dir, f"{stem}_colocalization.png")
        save_colocalization_overlay(
            img_rgb, vessel_mask, marker_mask,
            overlay_path, stem, vessel_channel, marker_channel
        )
        print(f"    Overlay saved:  {overlay_path}")

        # Save binary masks for use in other software
        masks_dir = os.path.join(output_dir, "masks")
        save_masks(vessel_mask, marker_mask, overlap_mask, masks_dir, stem)
        print(f"    Masks saved:    {masks_dir}/{stem}_*.png")

        metrics["overlay_path"] = overlay_path
        metrics["masks_dir"]    = masks_dir

    return metrics


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== Test 1: Otsu threshold on known distribution ===")
    rng = np.random.RandomState(42)
    # Simulate a channel with background (~40) and signal (~180) populations
    background = rng.normal(40,  10, 9000).clip(0, 255).astype(np.uint8)
    signal     = rng.normal(180, 15, 1000).clip(0, 255).astype(np.uint8)
    channel = np.concatenate([background, signal]).reshape(100, 100)
    threshold = otsu_threshold(channel)
    print(f"  Otsu threshold: {threshold}  (expected between 80-140)")
    assert 70 < threshold < 150, f"Otsu threshold out of range: {threshold}"
    print("  PASS")

    print()
    print("=== Test 2: Co-localization metrics with known overlap ===")
    H, W = 100, 100
    vessel_mask = np.zeros((H, W), dtype=bool)
    vessel_mask[20:60, 20:60] = True   # 40x40 = 1600 vessel pixels

    marker_mask = np.zeros((H, W), dtype=bool)
    marker_mask[40:80, 40:80] = True   # 40x40 = 1600 marker pixels
    # Overlap: rows 40-60, cols 40-60 = 20x20 = 400 pixels

    results = compute_colocalization(vessel_mask, marker_mask)
    print(f"  Vessel area:   {results['vessel_area_px']} (expect 1600)")
    print(f"  Marker area:   {results['marker_area_px']} (expect 1600)")
    print(f"  Overlap:       {results['overlap_px']} (expect 400)")
    print(f"  % vessel colocal: {results['pct_vessel_positive']:.1f}%  (expect 25.0%)")
    print(f"  % marker on vessel: {results['pct_marker_on_vessel']:.1f}%  (expect 25.0%)")
    assert results['vessel_area_px'] == 1600
    assert results['overlap_px'] == 400
    assert abs(results['pct_vessel_positive'] - 25.0) < 0.1
    assert abs(results['pct_marker_on_vessel'] - 25.0) < 0.1
    print("  PASS")

    print()
    print("=== Test 3: Full run_colocalization pipeline ===")
    rng = np.random.RandomState(0)
    img = (rng.rand(200, 200, 3) * 60 + 30).astype(np.uint8)
    # Put bright vessels in green channel
    img[50:100, 50:100, 1] = 200
    # Put bright marker in red channel, partially overlapping vessels
    img[70:130, 70:130, 0] = 210

    pred_mask = np.zeros((200, 200), dtype=np.float32)
    pred_mask[50:100, 50:100] = 1.0   # vessel region

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_colocalization(
            img_rgb        = img,
            pred_mask      = pred_mask,
            marker_channel = "red",
            vessel_channel   = "green",
            stem           = "test_image",
            output_dir     = tmpdir,
            pixel_size_um  = 0.168,
        )

        assert "pct_vessel_positive"  in results
        assert "pct_marker_on_vessel" in results
        assert "vessel_area_um2"      in results
        assert os.path.exists(results["overlay_path"])

        masks_dir = results["masks_dir"]
        assert os.path.exists(os.path.join(masks_dir, "test_image_vessel_mask.png"))
        assert os.path.exists(os.path.join(masks_dir, "test_image_marker_mask.png"))
        assert os.path.exists(os.path.join(masks_dir, "test_image_colocalization_mask.png"))

    print("  All files generated correctly")
    print("  PASS")

    print()
    print("=== Test 4: Same-channel error ===")
    try:
        run_colocalization(img, pred_mask, marker_channel="green", vessel_channel="green")
        print("  FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        print("  PASS")

    print()
    print("All tests PASSED")
