"""
vessel_morphology.py

Computes morphological metrics from a binary vessel segmentation mask:
  - Vessel area fraction (density)
  - Individual vessel diameters (mean, median, max, std, distribution)
  - Vessel count
  - Vessel elongation / circularity (shape descriptors)

All morphology measurements reported in real-world units (micrometers) when
pixel_size_um is provided, or in pixels if not.

Core method: distance transform-based diameter estimation.
For each connected vessel region, the distance transform gives the distance
from each vessel pixel to the nearest non-vessel pixel (i.e., the nearest
edge). The maximum distance transform value within a vessel = the radius of
the largest inscribed circle at that vessel's widest point. Doubling this
gives the vessel diameter at that point. This is robust to irregular vessel
shapes and tortuous paths.

Dependencies: scipy, scikit-image, numpy
"""

import numpy as np
from scipy import ndimage
from skimage import measure


# Pixel size for original cohort: 298 pixels = 50um -> 50/289 um/pixel
PIXEL_SIZE_ORIGINAL_COHORT_UM = 50.0 / 289.0  # ~0.173 um/pixel


def calculate_vessel_morphology(
    pred_mask,
    pixel_size_um=None,
    min_vessel_area_px=10,
):
    """
    Calculate vessel morphology metrics from a binary segmentation mask.

    Parameters
    ----------
    pred_mask : np.ndarray, shape (H, W), dtype float32 or uint8
        Binary vessel mask (1 = vessel, 0 = background).
        Should already be thresholded (e.g. > 0.5 from model sigmoid output).
    pixel_size_um : float or None
        Physical size of one pixel in micrometers.
        For original cohort: use PIXEL_SIZE_ORIGINAL_COHORT_UM (~0.168).
        If None, all spatial measurements are reported in pixels.
    min_vessel_area_px : int
        Minimum vessel area in pixels to be considered a real vessel.
        Filters out single-pixel noise and tiny artifacts.
        Default 10px -- at 0.168 um/px this corresponds to ~0.28 um²,
        well below the smallest real capillary (~3-5 um diameter).

    Returns
    -------
    dict with the following keys:
        area_fraction       : float  -- vessel pixels / tissue pixels
        vessel_count        : int    -- number of individual vessel segments
        pixel_size_um       : float or None -- pixel size used
        unit                : str    -- "um" or "px"

        -- Diameter metrics (in um or px depending on pixel_size_um) --
        mean_diameter       : float  -- mean vessel diameter across all vessels
        median_diameter     : float  -- median vessel diameter
        max_diameter        : float  -- maximum vessel diameter (vasodilation marker)
        std_diameter        : float  -- standard deviation of vessel diameters
        min_diameter        : float  -- minimum vessel diameter

        -- Per-vessel data (for plotting distributions) --
        per_vessel_diameters : list of float -- one diameter value per vessel

        -- Shape descriptors --
        mean_circularity    : float  -- 1.0 = perfect circle, <1 = elongated
                                        (4π × area / perimeter²)
        mean_elongation     : float  -- major_axis / minor_axis
                                        (1.0 = circle, >1 = elongated)
    """
    # Ensure binary uint8
    mask = (pred_mask > 0.5).astype(np.uint8)

    unit = "um" if pixel_size_um is not None else "px"
    scale = pixel_size_um if pixel_size_um is not None else 1.0

    # --- Area fraction ---
    area_fraction = float(mask.sum()) / float(mask.size)

    if mask.sum() == 0:
        # No vessels detected -- return zeros rather than crashing
        return {
            "area_fraction": 0.0, "vessel_count": 0,
            "pixel_size_um": pixel_size_um, "unit": unit,
            "mean_diameter": 0.0, "median_diameter": 0.0,
            "max_diameter": 0.0, "std_diameter": 0.0, "min_diameter": 0.0,
            "per_vessel_diameters": [],
            "mean_circularity": 0.0, "mean_elongation": 0.0,
        }

    # --- Distance transform: gives radius of largest inscribed circle
    #     at each vessel pixel ---
    dist_transform = ndimage.distance_transform_edt(mask)

    # --- Label connected vessel regions ---
    labeled_mask, n_vessels_raw = ndimage.label(mask)

    # --- Per-vessel metrics ---
    per_vessel_diameters = []
    per_vessel_circularity = []
    per_vessel_elongation = []

    regions = measure.regionprops(labeled_mask)

    for region in regions:
        # Skip tiny noise artifacts
        if region.area < min_vessel_area_px:
            continue

        # Diameter: 2 × max distance transform value within this vessel
        # (= diameter of largest circle that fits inside the vessel)
        vessel_pixels = dist_transform[labeled_mask == region.label]
        diameter_px = 2.0 * float(vessel_pixels.max())
        per_vessel_diameters.append(diameter_px * scale)

        # Circularity: 4π × area / perimeter²
        # = 1.0 for a perfect circle, approaches 0 for very elongated shapes
        # skimage perimeter can be 0 for tiny regions -- guard against div/0
        if region.perimeter > 0:
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            per_vessel_circularity.append(min(circularity, 1.0))  # cap at 1.0

        # Elongation: major axis / minor axis
        # skimage gives these in pixels; ratio is dimensionless
        # Use axis_minor/major_length (skimage >= 0.26) with fallback
        # to minor/major_axis_length for older versions
        try:
            minor = region.axis_minor_length
            major = region.axis_major_length
        except AttributeError:
            minor = region.minor_axis_length
            major = region.major_axis_length
        if minor > 0:
            elongation = major / minor
            per_vessel_elongation.append(elongation)

    vessel_count = len(per_vessel_diameters)

    if vessel_count == 0:
        # All detected regions were below min_vessel_area_px
        return {
            "area_fraction": area_fraction, "vessel_count": 0,
            "pixel_size_um": pixel_size_um, "unit": unit,
            "mean_diameter": 0.0, "median_diameter": 0.0,
            "max_diameter": 0.0, "std_diameter": 0.0, "min_diameter": 0.0,
            "per_vessel_diameters": [],
            "mean_circularity": 0.0, "mean_elongation": 0.0,
        }

    diameters = np.array(per_vessel_diameters)

    return {
        "area_fraction":         area_fraction,
        "vessel_count":          vessel_count,
        "pixel_size_um":         pixel_size_um,
        "unit":                  unit,

        "mean_diameter":         float(diameters.mean()),
        "median_diameter":       float(np.median(diameters)),
        "max_diameter":          float(diameters.max()),
        "std_diameter":          float(diameters.std()),
        "min_diameter":          float(diameters.min()),
        "per_vessel_diameters":  diameters.tolist(),

        "mean_circularity":      float(np.mean(per_vessel_circularity))
                                 if per_vessel_circularity else 0.0,
        "mean_elongation":       float(np.mean(per_vessel_elongation))
                                 if per_vessel_elongation else 0.0,
    }


def print_morphology_summary(stem, metrics):
    """Pretty-print morphology results for one image."""
    u = metrics["unit"]
    print(f"\n  {stem}:")
    print(f"    Vessel count:      {metrics['vessel_count']}")
    print(f"    Area fraction:     {metrics['area_fraction']:.4f}")
    print(f"    Mean diameter:     {metrics['mean_diameter']:.2f} {u}")
    print(f"    Median diameter:   {metrics['median_diameter']:.2f} {u}")
    print(f"    Max diameter:      {metrics['max_diameter']:.2f} {u}")
    print(f"    Std diameter:      {metrics['std_diameter']:.2f} {u}")
    print(f"    Mean circularity:  {metrics['mean_circularity']:.3f}  "
          f"(1.0=circle, <1=elongated)")
    print(f"    Mean elongation:   {metrics['mean_elongation']:.2f}  "
          f"(1.0=circle, >1=elongated)")


if __name__ == "__main__":
    import numpy as np

    print("=== Test 1: Known circular vessel ===")
    # A 40px diameter circle: expected diameter ~40px, circularity ~1.0
    mask = np.zeros((100, 100), dtype=np.uint8)
    cy, cx, r = 50, 50, 20  # center y,x, radius 20px -> diameter 40px
    y, x = np.ogrid[:100, :100]
    mask[(y - cy)**2 + (x - cx)**2 <= r**2] = 1

    metrics = calculate_vessel_morphology(mask, pixel_size_um=None)
    print(f"  Expected diameter: ~40px")
    print(f"  Got diameter:       {metrics['mean_diameter']:.1f}px")
    print(f"  Circularity:        {metrics['mean_circularity']:.3f} (expect ~1.0)")
    assert abs(metrics['mean_diameter'] - 40.0) < 3.0, \
        f"Diameter off: {metrics['mean_diameter']}"
    assert metrics['mean_circularity'] > 0.9, \
        f"Circularity too low: {metrics['mean_circularity']}"
    print("  PASS")

    print()
    print("=== Test 2: Known circular vessel with pixel size conversion ===")
    PIXEL_SIZE = 50.0 / 298.0  # original cohort pixel size
    metrics_um = calculate_vessel_morphology(mask, pixel_size_um=PIXEL_SIZE)
    expected_um = 40.0 * PIXEL_SIZE
    print(f"  Expected diameter: ~{expected_um:.2f} um")
    print(f"  Got diameter:       {metrics_um['mean_diameter']:.2f} um")
    print(f"  Unit reported:      {metrics_um['unit']}")
    assert abs(metrics_um['mean_diameter'] - expected_um) < 1.0
    assert metrics_um['unit'] == "um"
    print("  PASS")

    print()
    print("=== Test 3: Two vessels of different sizes ===")
    mask2 = np.zeros((200, 200), dtype=np.uint8)
    # Vessel 1: radius 10px (diameter 20px)
    y, x = np.ogrid[:200, :200]
    mask2[(y - 50)**2 + (x - 50)**2 <= 10**2] = 1
    # Vessel 2: radius 30px (diameter 60px) -- the "dilated" vessel
    mask2[(y - 150)**2 + (x - 150)**2 <= 30**2] = 1

    metrics2 = calculate_vessel_morphology(mask2, pixel_size_um=None)
    print(f"  Vessel count:     {metrics2['vessel_count']} (expect 2)")
    print(f"  Min diameter:     {metrics2['min_diameter']:.1f}px (expect ~20)")
    print(f"  Max diameter:     {metrics2['max_diameter']:.1f}px (expect ~60)")
    print(f"  Mean diameter:    {metrics2['mean_diameter']:.1f}px (expect ~40)")
    assert metrics2['vessel_count'] == 2
    assert abs(metrics2['min_diameter'] - 20.0) < 3.0
    assert abs(metrics2['max_diameter'] - 60.0) < 3.0
    print("  PASS")

    print()
    print("=== Test 4: Empty mask (no vessels) ===")
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    metrics_empty = calculate_vessel_morphology(empty_mask, pixel_size_um=PIXEL_SIZE)
    assert metrics_empty['vessel_count'] == 0
    assert metrics_empty['mean_diameter'] == 0.0
    assert metrics_empty['area_fraction'] == 0.0
    print("  PASS: empty mask handled gracefully")

    print()
    print("=== Test 5: Biological plausibility check ===")
    print(f"  Pixel size (original cohort): {PIXEL_SIZE:.4f} um/px")
    print(f"  Typical brain capillary: 5-10 um diameter")
    print(f"  = {5/PIXEL_SIZE:.0f}-{10/PIXEL_SIZE:.0f} pixels at this magnification")
    print(f"  Your model's 256x256 tile covers: "
          f"{256*PIXEL_SIZE:.1f} x {256*PIXEL_SIZE:.1f} um of tissue")
    print()
    print("All tests PASSED")
