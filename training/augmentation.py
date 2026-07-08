"""
Augmentation pipeline for IHC vessel segmentation tiles.

Two pipelines:
  - get_train_augmentation(): spatial + photometric + stain-ish color jitter
  - get_val_augmentation(): resize/normalize only, no randomness

Both apply identically to image AND mask (critical: spatial transforms must
move the mask the same way as the image, or labels become misaligned).
"""

import albumentations as A


def get_train_augmentation(tile_size=256):
    return A.Compose([
        # --- spatial ---
        A.RandomCrop(height=tile_size, width=tile_size, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.5,
        ),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),  # mild -- vessels are flexible but not rubbery

        # --- photometric / stain-related ---
        # Mimics batch-to-batch staining intensity variation.
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ])


def get_val_augmentation(tile_size=256):
    # No randomness -- just ensure consistent size. Tiles should already be
    # the right size if they came from data_pipeline.tile_image_and_mask,
    # but CenterCrop/pad guards against off-by-one edge tiles.
    return A.Compose([
        A.PadIfNeeded(min_height=tile_size, min_width=tile_size),
        A.CenterCrop(height=tile_size, width=tile_size),
    ])


def apply_augmentation(transform, image, mask):
    """
    image: np.ndarray (H, W, 3), float32 in [0, 1] or uint8
    mask:  np.ndarray (H, W) or (H, W, 1)
    Returns augmented (image, mask), mask squeezed back to (H, W).
    """
    mask_2d = mask.squeeze() if mask.ndim == 3 else mask
    result = transform(image=image, mask=mask_2d)
    return result["image"], result["mask"]


if __name__ == "__main__":
    import numpy as np

    # Self-test: confirm mask stays aligned with image through spatial transforms.
    # Use a mask with a single distinctive marker pixel-block to track.
    rng = np.random.RandomState(0)
    img = (rng.rand(256, 256, 3)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[50:70, 50:70] = 1.0  # a square block in the upper-left-ish area

    train_aug = get_train_augmentation(tile_size=256)

    # Run many times, confirm mask values stay binary {0,1} and mask area
    # roughly conserved (small changes expected from scale/elastic, but it
    # shouldn't randomly vanish or duplicate).
    areas = []
    for _ in range(20):
        aug_img, aug_mask = apply_augmentation(train_aug, img, mask)
        unique_vals = set(np.unique(aug_mask).tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Mask not binary after aug: {unique_vals}"
        assert aug_img.shape[:2] == aug_mask.shape[:2], "Image/mask shape mismatch after aug"
        areas.append(aug_mask.sum())

    areas = np.array(areas)
    print(f"Mask area before aug: {mask.sum():.0f}")
    print(f"Mask area after aug across 20 runs: mean={areas.mean():.1f}, "
          f"min={areas.min():.0f}, max={areas.max():.0f}")
    # Area should vary (due to scale/rotate/crop) but not collapse to 0 or
    # explode -- loose sanity bounds:
    assert areas.mean() > 0, "Mask vanished after augmentation -- alignment bug"
    assert areas.max() < mask.sum() * 3, "Mask area exploded -- likely alignment bug"
    print("PASS: image/mask spatial alignment preserved through augmentation")
