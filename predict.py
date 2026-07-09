"""
predict.py — Automated vessel quantification from IHC images

Runs a trained U-Net model on a folder of IHC images and outputs:
  - measurements.csv : area fraction + morphology metrics for every image
  - overlays/        : side-by-side QC images (original | mask | overlay)
  - prediction_log.txt : processing log with any warnings

Usage:
    python predict.py \\
        --images  path/to/your/images/ \\
        --model   final_model.keras \\
        --output  results/ \\
        --cd31_channel green \\
        [--pixel_sizes pixel_sizes.csv] \\
        [--tile_size 256] \\
        [--threshold 0.5]

Supported image formats: .tif, .tiff, .png, .jpg, .jpeg

CD31 channel specification:
    Replace green with red or blue depending on which RGB channel contains your CD31/vessel signal.

Pixel sizes (optional):
    If you provide a pixel_sizes.csv file (two columns: filename, um_per_px),
    morphology metrics (vessel diameter etc.) will be reported in micrometers.
    Otherwise they are reported in pixels. Example pixel_sizes.csv:
        filename,um_per_px
        image001.tif,0.173
        image002.tif,0.336

Requirements:
    pip install tensorflow numpy pillow scipy scikit-image matplotlib tqdm
"""

import os
import sys
import csv
import glob
import argparse
import warnings
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Suppress TF startup noise -- users don't need to see CUDA/oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf

# Optional: scipy/skimage for morphology metrics
try:
    from scipy import ndimage
    from skimage import measure
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False


# ── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
TILE_SIZE = 256
BLACK_THRESHOLD = 10    # pixels with mean gray <= this are considered padding
MIN_VESSEL_AREA_PX = 10 # minimum vessel area in pixels (filters noise)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "prediction_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ── Image loading and preprocessing ──────────────────────────────────────────

def load_image(path):
    """Load image as RGB numpy array, handling various formats."""
    img = np.array(Image.open(path).convert("RGB"))
    return img


def detect_cd31_channel(img):
    """Removed -- auto-detection is unreliable when multiple channels have signal.
    Channel must be specified explicitly by the user via --cd31_channel."""
    raise NotImplementedError(
        "Auto-detection removed. Please specify --cd31_channel red, green, or blue."
    )


def normalize_channel_to_green(img, cd31_channel):
    """
    Rearrange channels so CD31 signal is always in channel index 1 (green).
    The model was trained with CD31 in the green channel.

    Parameters
    ----------
    cd31_channel : str
        "red", "green", or "blue" -- which channel contains the CD31 signal.
        Specified explicitly by the user; never auto-detected.

    This is a channel permutation -- no pixel values are changed,
    just which position in the RGB array they occupy.
    """
    cd31_channel = cd31_channel.lower().strip()

    if cd31_channel == "green":
        return img  # already correct, no change needed

    img_norm = img.copy()
    if cd31_channel == "red":
        # Swap R and G
        img_norm[:, :, 0] = img[:, :, 1]  # R slot <- original G
        img_norm[:, :, 1] = img[:, :, 0]  # G slot <- original R
    elif cd31_channel == "blue":
        # Swap B and G
        img_norm[:, :, 2] = img[:, :, 1]  # B slot <- original G
        img_norm[:, :, 1] = img[:, :, 2]  # G slot <- original B
    else:
        raise ValueError(
            f"Unrecognised cd31_channel '{cd31_channel}'. "
            f"Must be 'red', 'green', or 'blue'."
        )
    return img_norm


def crop_black_border(img, padding=64):
    """
    Remove black border padding (from ROI-masked images like striatum sections).
    Returns cropped image and the crop box (y0, y1, x0, x1).
    For full-tissue images with no black border, this is a near no-op.
    """
    H, W = img.shape[:2]
    gray = img.mean(axis=-1)
    tissue = gray > BLACK_THRESHOLD

    rows = np.any(tissue, axis=1)
    cols = np.any(tissue, axis=0)

    if not rows.any():
        return img, (0, H, 0, W)

    y0 = max(0, int(np.argmax(rows)) - padding)
    y1 = min(H, int(len(rows) - np.argmax(rows[::-1])) + padding)
    x0 = max(0, int(np.argmax(cols)) - padding)
    x1 = min(W, int(len(cols) - np.argmax(cols[::-1])) + padding)

    return img[y0:y1, x0:x1], (y0, y1, x0, x1)


# ── Tiling and prediction ─────────────────────────────────────────────────────

def predict_full_image(model, img, tile_size=TILE_SIZE, threshold=0.5):
    """
    Tile a full image, predict each tile, stitch predictions back together.
    Handles images of any size by padding to a multiple of tile_size.

    Returns: pred_mask (H, W), float32, values in [0, 1] before thresholding.
    """
    H, W = img.shape[:2]

    # Pad to multiple of tile_size
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    pred_padded = np.zeros(img_padded.shape[:2], dtype=np.float32)

    for y in range(0, img_padded.shape[0], tile_size):
        for x in range(0, img_padded.shape[1], tile_size):
            tile = img_padded[y:y+tile_size, x:x+tile_size].astype(np.float32) / 255.0
            pred = model.predict(tile[np.newaxis, ...], verbose=0)[0, ..., 0]
            pred_padded[y:y+tile_size, x:x+tile_size] = pred

    # Remove padding and threshold
    pred_mask = (pred_padded[:H, :W] > threshold).astype(np.float32)
    return pred_mask


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_area_fraction(pred_mask):
    """Vessel area / tissue area. Tissue = non-black pixels."""
    return float(pred_mask.sum()) / float(pred_mask.size)


def compute_morphology(pred_mask, pixel_size_um=None):
    """
    Compute vessel morphology metrics from binary mask.
    Returns dict of metrics, or None if scipy/skimage not available.
    """
    if not HAS_MORPHOLOGY:
        return None

    mask = (pred_mask > 0.5).astype(np.uint8)
    scale = pixel_size_um if pixel_size_um is not None else 1.0
    unit = "um" if pixel_size_um is not None else "px"

    if mask.sum() == 0:
        return {
            "vessel_count": 0, "unit": unit,
            "mean_diameter": 0.0, "median_diameter": 0.0,
            "max_diameter": 0.0, "std_diameter": 0.0,
            "mean_circularity": 0.0, "mean_elongation": 0.0,
        }

    dist_transform = ndimage.distance_transform_edt(mask)
    labeled_mask, _ = ndimage.label(mask)
    regions = measure.regionprops(labeled_mask)

    diameters, circularities, elongations = [], [], []
    for region in regions:
        if region.area < MIN_VESSEL_AREA_PX:
            continue
        vessel_px = dist_transform[labeled_mask == region.label]
        diameters.append(2.0 * float(vessel_px.max()) * scale)
        if region.perimeter > 0:
            circ = (4 * np.pi * region.area) / (region.perimeter ** 2)
            circularities.append(min(circ, 1.0))
        try:
            minor = region.axis_minor_length
            major = region.axis_major_length
        except AttributeError:
            minor = region.minor_axis_length
            major = region.major_axis_length
        if minor > 0:
            elongations.append(major / minor)

    if not diameters:
        return {"vessel_count": 0, "unit": unit,
                "mean_diameter": 0.0, "median_diameter": 0.0,
                "max_diameter": 0.0, "std_diameter": 0.0,
                "mean_circularity": 0.0, "mean_elongation": 0.0}

    d = np.array(diameters)
    return {
        "vessel_count":      len(diameters),
        "unit":              unit,
        "mean_diameter":     float(d.mean()),
        "median_diameter":   float(np.median(d)),
        "max_diameter":      float(d.max()),
        "std_diameter":      float(d.std()),
        "mean_circularity":  float(np.mean(circularities)) if circularities else 0.0,
        "mean_elongation":   float(np.mean(elongations)) if elongations else 0.0,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def save_overlay(img, pred_mask, output_path, stem):
    """
    Save a 3-panel QC image: original | binary mask | overlay.
    The overlay shows predicted vessels in red on the original image
    so users can visually verify predictions are correct.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{stem}", fontsize=13, y=1.01)

    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicted vessel mask")
    axes[1].axis("off")

    axes[2].imshow(img)
    # Overlay predicted vessels in semi-transparent red
    overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    overlay[pred_mask > 0.5] = [1.0, 0.0, 0.0, 0.45]  # red, 45% opacity
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (red = predicted vessel)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Pixel size loading ────────────────────────────────────────────────────────

def load_pixel_sizes(csv_path):
    """
    Load pixel size mapping from a CSV file.
    Expected format:
        filename,um_per_px
        image001.tif,0.168
        image002.tif,0.336

    Returns dict mapping filename stem (without extension) -> float um/px.
    If csv_path is None or file doesn't exist, returns empty dict
    (all measurements will be in pixels).
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    pixel_sizes = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "").strip()
            px_size = row.get("um_per_px", "").strip()
            if fname and px_size:
                # Accept both full filename and stem (without extension)
                stem = os.path.splitext(fname)[0]
                try:
                    pixel_sizes[stem] = float(px_size)
                    pixel_sizes[fname] = float(px_size)
                except ValueError:
                    pass
    return pixel_sizes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated vessel quantification from IHC images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--images", required=True,
                        help="Folder containing IHC images to process")
    parser.add_argument("--model", required=True,
                        help="Path to trained model file (final_model.keras)")
    parser.add_argument("--output", required=True,
                        help="Folder where results will be saved")
    parser.add_argument("--pixel_sizes", default=None,
                        help="CSV file mapping image filenames to um/px "
                             "(optional; if omitted, metrics reported in pixels)")
    parser.add_argument("--cd31_channel", required=True,
                        choices=["red", "green", "blue"],
                        help="Which RGB channel contains the CD31/vessel signal. "
                             "Must be specified explicitly -- do not assume green. "
                             "Check your image in ImageJ: the vessel channel should "
                             "light up brightly when viewed as a single channel. "
                             "Example: --cd31_channel green")
    parser.add_argument("--tile_size", type=int, default=256,
                        help="Tile size for prediction (default: 256)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for vessel/background "
                             "classification (default: 0.5)")
    parser.add_argument("--no_overlays", action="store_true",
                        help="Skip saving overlay images (faster, less disk)")
    args = parser.parse_args()

    # Setup
    log = setup_logging(args.output)
    overlay_dir = os.path.join(args.output, "overlays")
    if not args.no_overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Vessel Segmentation Tool")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Find images
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.images, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.images, f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        log.error(f"No images found in {args.images}")
        log.error(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    log.info(f"Found {len(image_paths)} images to process")

    # Load model
    log.info(f"Loading model from {args.model}...")
    if not os.path.exists(args.model):
        log.error(f"Model file not found: {args.model}")
        sys.exit(1)

    # Suppress TF model loading messages
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={
            "dice_bce_loss": lambda y_true, y_pred: y_pred,  # placeholder
            "dice_coefficient": lambda y_true, y_pred: y_pred,
            "iou_metric": lambda y_true, y_pred: y_pred,
        },
        compile=False,  # don't need to compile for inference
    )
    log.info("Model loaded successfully")

    # Load pixel sizes
    pixel_sizes = load_pixel_sizes(args.pixel_sizes)
    if pixel_sizes:
        log.info(f"Loaded pixel sizes for {len(pixel_sizes)//2} images")
    else:
        log.info("No pixel size file provided -- morphology metrics will be in pixels")

    if not HAS_MORPHOLOGY:
        log.warning("scipy/skimage not installed -- morphology metrics unavailable")
        log.warning("Install with: pip install scipy scikit-image")

    # Process images
    results = []
    log.info(f"CD31 channel: {args.cd31_channel} (applied to all images)")

    for i, path in enumerate(image_paths):
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        log.info(f"[{i+1}/{len(image_paths)}] Processing {fname}...")

        try:
            # Load
            img_orig = load_image(path)
            orig_shape = img_orig.shape[:2]

            # Normalize channel order (CD31 -> green position)
            # Channel is user-specified -- never auto-detected
            img = normalize_channel_to_green(img_orig, args.cd31_channel)
            log.info(f"  CD31 channel: {args.cd31_channel} (user-specified)")

            # Crop black border (handles striatum-style ROI-masked images)
            img_cropped, crop_box = crop_black_border(img)
            if img_cropped.shape != img.shape:
                retained = (img_cropped.shape[0]*img_cropped.shape[1]) / \
                           (img.shape[0]*img.shape[1])
                log.info(f"  Cropped black border: {img.shape[:2]} -> "
                         f"{img_cropped.shape[:2]} ({retained:.0%} retained)")

            # Predict
            pred_mask = predict_full_image(
                model, img_cropped, args.tile_size, args.threshold)

            # Pixel size for this image
            px_size = pixel_sizes.get(stem) or pixel_sizes.get(fname)
            if px_size:
                log.info(f"  Pixel size: {px_size} um/px")
            else:
                log.info(f"  Pixel size: not specified (reporting in pixels)")

            # Metrics
            area_frac = compute_area_fraction(pred_mask)
            morph = compute_morphology(pred_mask, px_size) if HAS_MORPHOLOGY else {}

            log.info(f"  Area fraction: {area_frac:.4f}")
            if morph:
                u = morph.get("unit", "px")
                log.info(f"  Vessels detected: {morph.get('vessel_count', 0)}")
                log.info(f"  Mean diameter: {morph.get('mean_diameter', 0):.2f} {u}")
                log.info(f"  Max diameter:  {morph.get('max_diameter', 0):.2f} {u} "
                         f"(vasodilation marker)")

            # Build result row
            row = {
                "filename":          fname,
                "original_size_HxW": f"{orig_shape[0]}x{orig_shape[1]}",
                "cd31_channel":      args.cd31_channel,
                "pixel_size_um":     px_size if px_size else "unknown",
                "area_fraction":     round(area_frac, 6),
            }
            if morph:
                unit = morph.get("unit", "px")
                row.update({
                    "vessel_count":                morph.get("vessel_count", 0),
                    f"mean_diameter_{unit}":        round(morph.get("mean_diameter", 0), 3),
                    f"median_diameter_{unit}":      round(morph.get("median_diameter", 0), 3),
                    f"max_diameter_{unit}":         round(morph.get("max_diameter", 0), 3),
                    f"std_diameter_{unit}":         round(morph.get("std_diameter", 0), 3),
                    "mean_circularity":             round(morph.get("mean_circularity", 0), 4),
                    "mean_elongation":              round(morph.get("mean_elongation", 0), 4),
                })
            row["notes"] = ""
            results.append(row)

            # Save overlay
            if not args.no_overlays:
                overlay_path = os.path.join(overlay_dir, f"{stem}_overlay.png")
                save_overlay(img_cropped, pred_mask, overlay_path, stem)
                log.info(f"  Overlay saved: {overlay_path}")

        except Exception as e:
            log.error(f"  ERROR processing {fname}: {e}")
            results.append({
                "filename": fname,
                "notes": f"ERROR: {e}",
                "area_fraction": "",
            })

    # Write CSV
    csv_path = os.path.join(args.output, "measurements.csv")
    if results:
        # Collect all keys across all result rows (some may have morph, some not)
        all_keys = []
        seen = set()
        for row in results:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, "") for k in all_keys})

    log.info("")
    log.info("=" * 60)
    log.info(f"Done. Processed {len(results)} images.")
    log.info(f"Results saved to: {csv_path}")
    if not args.no_overlays:
        log.info(f"Overlay images: {overlay_dir}/")
    log.info(f"Full log: {os.path.join(args.output, 'prediction_log.txt')}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
