"""
Main training script: vessel segmentation U-Net with transfer learning,
slide-level k-fold cross-validation, and area-fraction evaluation.

Expected directory layout:
    data/
        images/   <stem>.tif  (RGB IHC images)
        rois/     <stem>_RoiSet.zip  (ImageJ ROI sets, one per image)

Run:
    python train.py

Outputs written to results_v2/:
    fold{N}_best.keras          -- best checkpoint per fold (5 files)
    fold{N}_training_curves.png -- loss/Dice/IoU curves per fold (5 files)
    fold{N}_history.json        -- raw epoch values per fold (5 files)
    crossfold_summary.png       -- all folds' val curves overlaid
    cv_results.json             -- full numeric summary, machine-readable
    final_model.keras           -- model trained on ALL annotated slides (keeper)
    final_model_metadata.json   -- provenance record for the final model
"""

import os
import json
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- essential for batch jobs
                        # that have no display; without this matplotlib tries
                        # to open a GUI window and crashes immediately
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_pipeline import build_dataset_index, split_records_by_slide, build_tile_arrays
from augmentation import get_train_augmentation, get_val_augmentation, apply_augmentation
from model import build_unet
from losses_metrics import dice_bce_loss, dice_coefficient, iou_metric, area_fraction_error

# ---------------- config ----------------
IMAGE_DIR = "data/images"
ROI_DIR = "data/rois"
TILE_SIZE = 256
TILE_STRIDE = 192
N_FOLDS = 5
BATCH_SIZE = 8
EPOCHS_FROZEN = 15
EPOCHS_FINETUNE = 30
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
RESULTS_DIR = "results_v2"
# ----------------------------------------


def make_tf_dataset(X, Y, augment, batch_size, tile_size, shuffle):
    def aug_fn(img, mask):
        transform = get_train_augmentation(tile_size) if augment else get_val_augmentation(tile_size)
        img_aug, mask_aug = apply_augmentation(transform, img.numpy(), mask.numpy())
        return img_aug.astype(np.float32), mask_aug.astype(np.float32)

    def tf_aug_wrapper(img, mask):
        img_aug, mask_aug = tf.py_function(aug_fn, [img, mask], [tf.float32, tf.float32])
        img_aug.set_shape([tile_size, tile_size, 3])
        mask_aug.set_shape([tile_size, tile_size])
        return img_aug, mask_aug[..., tf.newaxis]

    ds = tf.data.Dataset.from_tensor_slices((X, Y[..., 0]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.map(tf_aug_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def plot_training_history(history_phase1, history_phase2, fold_idx, save_dir):
    """
    Save loss/Dice/IoU curves for one fold as a PNG, combining Phase 1 and
    Phase 2 into one continuous plot with a vertical dashed line at the phase
    switch -- same as the combined-history plot the notebook showed you live,
    just saved to a file instead of displayed interactively.

    Also saves raw epoch values as JSON so you can re-plot or compare later.
    """
    os.makedirs(save_dir, exist_ok=True)

    phase1_len = len(history_phase1.history["loss"])
    combined = {}
    for key in ["loss", "dice_coefficient", "iou_metric"]:
        combined[key] = history_phase1.history[key] + history_phase2.history[key]
        combined[f"val_{key}"] = (history_phase1.history[f"val_{key}"] +
                                   history_phase2.history[f"val_{key}"])

    epochs = list(range(1, len(combined["loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Fold {fold_idx} — Training History", fontsize=13)

    plot_specs = [
        ("loss",             "Loss (Dice + BCE)",  "lower right"),
        ("dice_coefficient", "Dice coefficient",    "upper left"),
        ("iou_metric",       "IoU",                 "upper left"),
    ]

    for ax, (key, title, legend_loc) in zip(axes, plot_specs):
        ax.plot(epochs, combined[key],          label="train", linewidth=1.5)
        ax.plot(epochs, combined[f"val_{key}"], label="val",   linewidth=1.5)
        ax.axvline(phase1_len + 0.5, color="gray", linestyle="--",
                   linewidth=1, label="phase 1 → 2")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(loc=legend_loc, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"fold{fold_idx}_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {plot_path}")

    # Save raw values so you can re-plot or further analyse without rerunning
    history_json_path = os.path.join(save_dir, f"fold{fold_idx}_history.json")
    with open(history_json_path, "w") as f:
        json.dump(
            {"phase1": {k: [float(v) for v in vs]
                        for k, vs in history_phase1.history.items()},
             "phase2": {k: [float(v) for v in vs]
                        for k, vs in history_phase2.history.items()}},
            f, indent=2)


def plot_crossfold_summary(all_histories, save_dir):
    """
    After all folds finish, overlay all folds' val curves on the same axes.
    This lets you see at a glance how consistent training was across different
    slide groupings -- and whether any one fold was a systematic outlier
    (e.g., whichever fold happened to contain an artifact-heavy slide).
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cross-validation summary — all folds", fontsize=13)

    plot_specs = [
        ("val_loss",             "Val loss (Dice + BCE)"),
        ("val_dice_coefficient", "Val Dice coefficient"),
        ("val_iou_metric",       "Val IoU"),
    ]

    for ax, (key, title) in zip(axes, plot_specs):
        for fold_idx, (h1, h2) in enumerate(all_histories):
            series = h1.history.get(key, []) + h2.history.get(key, [])
            ax.plot(series, label=f"Fold {fold_idx}", linewidth=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch (combined phases)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "crossfold_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cross-fold summary saved: {plot_path}")


def train_one_fold(train_records, val_records, fold_idx):
    print(f"\n{'='*60}\nFOLD {fold_idx}\n{'='*60}")

    X_train, Y_train = build_tile_arrays(train_records, tile_size=TILE_SIZE, stride=TILE_STRIDE,
                                          crop_black_border=True)
    X_val, Y_val = build_tile_arrays(val_records, tile_size=TILE_SIZE, stride=TILE_SIZE,
                                      crop_black_border=True)

    train_ds = make_tf_dataset(X_train, Y_train, augment=True, batch_size=BATCH_SIZE,
                                tile_size=TILE_SIZE, shuffle=True)
    val_ds = make_tf_dataset(X_val, Y_val, augment=False, batch_size=BATCH_SIZE,
                              tile_size=TILE_SIZE, shuffle=False)

    model = build_unet(input_shape=(TILE_SIZE, TILE_SIZE, 3), encoder_weights="imagenet",
                        freeze_encoder=True)

    # --- Phase 1: decoder only, encoder frozen ---
    model.compile(optimizer=keras.optimizers.Adam(LR_FROZEN),
                   loss=dice_bce_loss, metrics=[dice_coefficient, iou_metric])
    print("\n--- Phase 1: training decoder, encoder frozen ---")
    history_phase1 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2)

    # --- Phase 2: unfreeze encoder, fine-tune end-to-end at low LR ---
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE),
                   loss=dice_bce_loss, metrics=[dice_coefficient, iou_metric])
    print("\n--- Phase 2: fine-tuning full model, encoder unfrozen ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f"fold{fold_idx}_best.keras")
    history_phase2 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                             monitor="val_loss"),
        ],
        verbose=2)

    # --- Save training curves for this fold ---
    plot_training_history(history_phase1, history_phase2, fold_idx, RESULTS_DIR)

    # --- Evaluate ---
    val_results = model.evaluate(val_ds, return_dict=True)
    print(f"Fold {fold_idx} val results: {val_results}")
    per_image_area_errors = evaluate_area_fraction_per_image(model, val_records)

    return model, val_results, per_image_area_errors, (history_phase1, history_phase2)


def evaluate_area_fraction_per_image(model, records):
    """
    For each held-out whole image, compute predicted vs. manual vascular area
    fraction by tiling, predicting, and stitching.

    Uses load_image_and_mask_cropped so black borders are removed before
    evaluation -- denominator is always tissue area, matching how manual
    area fractions were calculated (vessels / ROI area for striatum images).
    """
    from data_pipeline import load_image_and_mask_cropped

    results = []
    for rec in records:
        img, true_mask = load_image_and_mask_cropped(rec, crop_black_border=True)
        H, W = true_mask.shape

        pad_h = (TILE_SIZE - H % TILE_SIZE) % TILE_SIZE
        pad_w = (TILE_SIZE - W % TILE_SIZE) % TILE_SIZE
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        pred_mask_padded = np.zeros(img_padded.shape[:2], dtype=np.float32)
        for y in range(0, img_padded.shape[0], TILE_SIZE):
            for x in range(0, img_padded.shape[1], TILE_SIZE):
                tile = img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE].astype(np.float32) / 255.0
                pred = model.predict(tile[np.newaxis, ...], verbose=0)[0, ..., 0]
                pred_mask_padded[y:y+TILE_SIZE, x:x+TILE_SIZE] = pred

        pred_mask = (pred_mask_padded[:H, :W] > 0.5).astype(np.float32)

        # After cropping, denominator is simply mask.size -- no black padding left.
        true_frac = float(true_mask.sum()) / true_mask.size
        pred_frac = float(pred_mask.sum()) / pred_mask.size

        results.append({
            "stem": rec["stem"],
            "true_area_fraction": true_frac,
            "pred_area_fraction": pred_frac,
            "abs_error": abs(true_frac - pred_frac),
        })
        print(f'  {rec["stem"]}: true={true_frac:.4f}, pred={pred_frac:.4f}, '
              f'abs_error={abs(true_frac - pred_frac):.4f}')
    return results



def train_final_model(all_records):
    """
    Train one final model on ALL annotated slides (no held-out fold).
    This is the model you keep and build on going forward -- the 5 fold
    models exist only to estimate how well the recipe generalises, not
    to be used directly.
    """
    print(f"\n{'='*60}\nFINAL MODEL: training on all {len(all_records)} slides\n{'='*60}")

    X_train, Y_train = build_tile_arrays(all_records, tile_size=TILE_SIZE, stride=TILE_STRIDE,
                                          crop_black_border=True)
    train_ds = make_tf_dataset(X_train, Y_train, augment=True, batch_size=BATCH_SIZE,
                                tile_size=TILE_SIZE, shuffle=True)

    model = build_unet(input_shape=(TILE_SIZE, TILE_SIZE, 3), encoder_weights="imagenet",
                        freeze_encoder=True)

    model.compile(optimizer=keras.optimizers.Adam(LR_FROZEN),
                   loss=dice_bce_loss, metrics=[dice_coefficient, iou_metric])
    print("\n--- Final model, Phase 1: decoder only ---")
    # No validation_data -- we're deliberately using every slide.
    # Fixed epoch count informed by what cross-validation folds typically needed.
    model.fit(train_ds, epochs=EPOCHS_FROZEN, verbose=2)

    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE),
                   loss=dice_bce_loss, metrics=[dice_coefficient, iou_metric])
    print("\n--- Final model, Phase 2: fine-tuning ---")
    model.fit(train_ds, epochs=EPOCHS_FINETUNE, verbose=2)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    final_path = os.path.join(RESULTS_DIR, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")
    return model, final_path


def to_native(obj):
    """Recursively convert numpy scalar types to native Python types.
    json.dump cannot serialize np.float32/np.float64 directly, and
    evaluate_area_fraction_per_image produces exactly these types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    records = build_dataset_index(IMAGE_DIR, ROI_DIR)

    rng = np.random.RandomState(42)
    shuffled = [records[i] for i in rng.permutation(len(records))]
    folds = np.array_split(shuffled, N_FOLDS)

    all_fold_results = []
    all_area_errors = []
    all_histories = []   # collect for the cross-fold summary plot

    for fold_idx in range(N_FOLDS):
        val_records = list(folds[fold_idx])
        train_records = [r for i, f in enumerate(folds) if i != fold_idx for r in f]

        model, val_results, area_errors, histories = train_one_fold(
            train_records, val_records, fold_idx)
        all_fold_results.append(val_results)
        all_area_errors.extend(area_errors)
        all_histories.append(histories)

    # Cross-fold summary plot -- all folds' val curves overlaid
    plot_crossfold_summary(all_histories, RESULTS_DIR)

    # Numeric summary
    print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY ({N_FOLDS} folds)\n{'='*60}")
    cv_summary = {}
    for metric in all_fold_results[0].keys():
        values = [r[metric] for r in all_fold_results]
        cv_summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    abs_errors = [r["abs_error"] for r in all_area_errors]
    area_summary = {
        "mean_abs_error":   float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "max_abs_error":    float(np.max(abs_errors)),
    }
    print(f"\nPer-image area-fraction MAE (pooled across folds): "
          f"mean={area_summary['mean_abs_error']:.4f}, "
          f"median={area_summary['median_abs_error']:.4f}, "
          f"max={area_summary['max_abs_error']:.4f}")

    # Save full results as JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_payload = {
        "cv_summary": cv_summary,
        "area_fraction_summary": area_summary,
        "per_image_area_results": all_area_errors,
        "config": {
            "n_folds": N_FOLDS, "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE,
            "batch_size": BATCH_SIZE, "epochs_frozen": EPOCHS_FROZEN,
            "epochs_finetune": EPOCHS_FINETUNE, "lr_frozen": LR_FROZEN,
            "lr_finetune": LR_FINETUNE, "n_total_slides": len(records),
        },
    }
    results_json_path = os.path.join(RESULTS_DIR, "cv_results.json")
    with open(results_json_path, "w") as f:
        json.dump(to_native(results_payload), f, indent=2)
    print(f"\nFull CV results saved: {results_json_path}")

    # Train and save the one model you'll actually keep
    final_model, final_path = train_final_model(records)

    # Provenance record for the final model
    metadata = {
        "trained_on": datetime.now().isoformat(),
        "n_training_slides": len(records),
        "training_slide_stems": [r["stem"] for r in records],
        "config": results_payload["config"],
        "cross_validation_summary": cv_summary,
        "cross_validation_area_fraction_summary": area_summary,
        "notes": (
            "Trained on ALL annotated slides (no held-out fold). "
            "Cross-validation numbers above describe expected generalisation "
            "for this recipe, estimated from a separate 5-fold run -- NOT "
            "this exact model (which saw every slide). When retraining with "
            "more data, compare new CV numbers against these as a baseline."
        ),
    }
    metadata_path = os.path.join(RESULTS_DIR, "final_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(to_native(metadata), f, indent=2)
    print(f"Final model metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
