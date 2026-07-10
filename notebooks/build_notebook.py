"""
Builds vessel_segmentation_walkthrough.ipynb programmatically using nbformat.
Run once to regenerate the notebook after any changes to this file.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ============================================================================
# SECTION 0: Introduction
# ============================================================================
md("""
# Vessel Segmentation and Building VascoNet:

This notebook builds and trains a U-Net to automate vessel tracing in mouse 
brain immunofluorescence images, going step by step through how we developed
VascoNet.

**Assumed background:** Familiarity with cost functions, gradient descent, neural 
networks, CNNs, and forward/backward propagation.

**Additional concepts:**
- **Transfer learning** — reusing a network pretrained on ImageNet as a
  starting point for vessel segmentation
- **Semantic segmentation** — predicting a label for *every pixel*
- **U-Net architecture** — encoder (compresses image) + decoder
  (rebuilds to full resolution) + skip connections (preserve fine detail)
- **Dice loss** — a cost function robust to class imbalance (vessels are
  a small minority of total pixels)

Structure: data → architecture → forward pass → cost function → training
→ evaluation → morphology metrics.
""")

# ============================================================================
# SECTION 1: Setup
# ============================================================================
md("## 1. Setup\n\nImport everything and confirm GPU visibility.")

code("""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Make the training/ modules importable regardless of where Jupyter was
# launched from -- this notebook's kernel cwd is its own directory
# (notebooks/), while the pipeline modules live in ../training.
sys.path.insert(0, os.path.abspath(os.path.join("..", "training")))

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for g in gpus:
    print(" -", g)
""")

code("""
from roi_to_mask import rois_to_mask, mask_area_fraction
from data_pipeline import (
    build_dataset_index, load_image_and_mask, load_image_and_mask_cropped,
    tile_image_and_mask, split_records_by_slide, build_tile_arrays
)
from augmentation import get_train_augmentation, get_val_augmentation, apply_augmentation
from model import build_unet
from losses_metrics import (
    dice_bce_loss, dice_coefficient, iou_metric, area_fraction_error,
    precision_metric, recall_metric, specificity_metric
)
""")

# ============================================================================
# SECTION 2: Data loading
# ============================================================================
md("""
## 2. Load and inspect your data
""")

code("""
# Paths are relative to this notebook's directory (notebooks/); the data/
# folder lives at the repo root alongside training/.
IMAGE_DIR = "../data/images"
ROI_DIR   = "../data/rois"

records = build_dataset_index(IMAGE_DIR, ROI_DIR)
print(f"Found {len(records)} image/ROI pairs")
for r in records[:5]:
    print(" ", r["stem"])
""")

md("""
### Sanity check: rasterized mask area fraction vs. manual measurement

The rasterized mask area fraction should match your manually recorded value closely.
Large differences indicate a coordinate-space mismatch between image and ROI.

**Note:** for striatum images with black-padded borders, `load_image_and_mask_cropped`
automatically removes the padding so the area fraction reflects vessels / tissue
area rather than vessels / full image including black border.
""")

code("""
example_record = records[0]
print("Checking:", example_record["stem"])

img, mask = load_image_and_mask_cropped(example_record, crop_black_border=True)
auto_frac = mask.sum() / mask.size

print(f"Image shape after crop: {img.shape}")
print(f"Auto area fraction:     {auto_frac:.4f}")
print(f"--> Compare to your manually recorded value for {example_record['stem']}")
""")

md("### Visualize image, mask, and overlay")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title("Original IHC image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title(f"Rasterized mask (area frac={auto_frac:.4f})")
axes[1].axis("off")

axes[2].imshow(img)
axes[2].imshow(mask, cmap="Reds", alpha=0.4)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
""")

# ============================================================================
# SECTION 3: Tiling
# ============================================================================
md("""
## 3. Tiling

Full-resolution images are too large to feed into the network at once (GPU
memory limits) and the network architecture requires input dimensions
divisible by 32. So we cut each image into 256x256 tiles.

The effective training set size is much larger than the number of slides:
a 4096x3008 image at stride=192 produces hundreds of tiles, each its own
training example.

**Black border cropping:** striatum images with ROI-masked borders are
automatically cropped to the tissue bounding box before tiling, so tiles
contain only real tissue rather than mostly-black padding.
""")

code("""
TILE_SIZE = 256

img_tiles, mask_tiles = tile_image_and_mask(img, mask, tile_size=TILE_SIZE, stride=TILE_SIZE)
print(f"Got {len(img_tiles)} tiles from this image")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(min(4, len(img_tiles))):
    axes[0, i].imshow(img_tiles[i])
    axes[0, i].set_title(f"Tile {i} image")
    axes[0, i].axis("off")
    axes[1, i].imshow(mask_tiles[i], cmap="gray")
    axes[1, i].set_title(f"Tile {i} mask")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
""")

# ============================================================================
# SECTION 4: Train/val/test split
# ============================================================================
md("""
## 4. Slide-level train / val / test split

Data was always split at the slide level, not the tile level. If tiles
from the same slide appear in both training and validation, the model can
memorize slide-specific staining quirks rather than learning vessel morphology, 
artificially inflated validation performance.

With a slide-level split:
- Gradient descent only sees training tiles
- Validation tiles measure generalization but also guide early stopping and
  checkpoint selection (slightly optimistic)
- Test tiles are untouched until final evaluation
""")

code("""
train_records, val_records, test_records = split_records_by_slide(
    records, val_frac=0.15, test_frac=0.15, seed=42
)
""")

code("""
TILE_STRIDE_TRAIN = 192  # overlapping tiles for more training examples

X_train, Y_train = build_tile_arrays(
    train_records, tile_size=TILE_SIZE, stride=TILE_STRIDE_TRAIN,
    crop_black_border=True)
X_val, Y_val = build_tile_arrays(
    val_records, tile_size=TILE_SIZE, stride=TILE_SIZE,
    crop_black_border=True)

print(f"Training tiles:   {X_train.shape[0]}")
print(f"Validation tiles: {X_val.shape[0]}")
""")

# ============================================================================
# SECTION 5: Augmentation
# ============================================================================
md("""
## 5. Augmentation

With a limited number of source slides, the model could memorize
slide-specific details rather than learning general vessel features
(overfitting). Augmentation manufactures additional variety by applying
random spatial and photometric transforms to each tile on-the-fly each epoch.

**Key requirement:** spatial transforms must apply identically to image AND
mask -- if we flip the image but not the mask, labels become misaligned.
The albumentations library handles this automatically.
""")

code("""
train_aug = get_train_augmentation(tile_size=TILE_SIZE)

idx_with_vessel = np.argmax(Y_train[..., 0].sum(axis=(1, 2)))
example_img  = X_train[idx_with_vessel]
example_mask = Y_train[idx_with_vessel]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    aug_img, aug_mask = apply_augmentation(train_aug, example_img, example_mask)
    axes[0, i].imshow(aug_img)
    axes[0, i].set_title(f"Augmented image {i}")
    axes[0, i].axis("off")
    axes[1, i].imshow(aug_mask, cmap="gray")
    axes[1, i].set_title(f"Augmented mask {i}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
""")

# ============================================================================
# SECTION 6: Model
# ============================================================================
md("""
## 6. U-Net with EfficientNetB0 encoder

**Architecture overview:**
- **Encoder (EfficientNetB0, ImageNet-pretrained):** compresses the image
  through 5 downsampling stages (256→128→64→32→16→8 pixels), building
  increasingly abstract feature representations. Frozen in Phase 1.
- **Skip connections:** at each encoder stage, feature maps are saved and
  handed to the matching decoder stage, preserving fine spatial detail that
  the bottleneck would otherwise lose.
- **Decoder (trained from scratch):** 5 upsampling stages expand the
  compressed representation back to full resolution, using skip connections
  to recover precise vessel boundary locations.
- **Output:** 256×256×1 sigmoid activation -- per-pixel vessel probability.

**Transfer learning:** the encoder starts with weights already trained on
1.2M natural images (ImageNet), giving it general visual feature detectors
(edges, textures, shapes) without needing to learn them from our 49 images.
""")

code("""
model = build_unet(
    input_shape=(TILE_SIZE, TILE_SIZE, 3),
    encoder_weights="imagenet",
    freeze_encoder=True
)
model.summary()
print(f"Total parameters: {model.count_params():,}")
""")

# ============================================================================
# SECTION 7: Forward pass sanity check
# ============================================================================
md("""
## 7. Sanity check: untrained forward pass

Before training, push a batch through the model and verify output shape and
value range. Predictions should look like noise -- the model hasn't learned
anything yet.
""")

code("""
sample_batch = X_train[:4]
predictions  = model.predict(sample_batch, verbose=0)
print(f"Input shape:  {sample_batch.shape}")
print(f"Output shape: {predictions.shape}")
print(f"Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axes[0, i].imshow(sample_batch[i])
    axes[0, i].set_title(f"Input {i}")
    axes[0, i].axis("off")
    axes[1, i].imshow(predictions[i, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title(f"Untrained pred {i}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()
print("Output should look like noise -- model hasn't learned anything yet.")
""")

# ============================================================================
# SECTION 8: Loss function
# ============================================================================
md("""
## 8. Loss function: Dice + Binary Cross-Entropy

**Why not plain BCE?** Vessel pixels are a small minority (~5-20%) of total
pixels. A model predicting 'no vessel' everywhere would get low BCE despite
being useless. BCE is dominated by the easy background majority.

**Dice loss** measures overlap directly:
$$Dice = \\frac{2 \\times |pred \\cap truth|}{|pred| + |truth|}$$

It ranges 0 (no overlap) to 1 (perfect), and is not fooled by class
imbalance -- predicting all background gives Dice≈0, correctly flagging
failure.

**Combined:** BCE gives stable gradients early in training when predictions
and ground truth don't overlap at all (Dice loss alone is uninformative
then). Dice keeps optimization focused on the metric that matters.
""")

code("""
y_true_demo = np.zeros((1, 10, 10, 1), dtype=np.float32)
y_true_demo[0, 2:5, 2:5, 0] = 1.0

y_pred_perfect = y_true_demo.copy()
y_pred_wrong   = np.zeros((1, 10, 10, 1), dtype=np.float32)
y_pred_wrong[0, 7:9, 7:9, 0] = 1.0
y_pred_partial = np.zeros((1, 10, 10, 1), dtype=np.float32)
y_pred_partial[0, 3:6, 3:6, 0] = 1.0

for name, pred in [("Perfect", y_pred_perfect),
                   ("No overlap", y_pred_wrong),
                   ("Partial", y_pred_partial)]:
    d = dice_coefficient(y_true_demo, pred).numpy()
    p = precision_metric(y_true_demo, pred).numpy()
    r = recall_metric(y_true_demo, pred).numpy()
    print(f"{name:12s}: Dice={d:.4f}  Precision={p:.4f}  Recall={r:.4f}")
""")

# ============================================================================
# SECTION 9: Training Phase 1
# ============================================================================
md("""
## 9. Training: Phase 1 (decoder only, encoder frozen)

**Why two phases?**

The decoder starts with random weights. If we immediately allow gradients
to flow into the pretrained encoder, large noisy gradients from the
untrained decoder will degrade the useful ImageNet features before the
decoder has learned anything sensible.

**Phase 1 fix:** freeze the encoder (`encoder.trainable = False`), train
only the decoder at a higher learning rate (1e-3). The decoder learns
"given what EfficientNetB0 sees, where are the vessels?" without touching
the pretrained encoder weights.
""")

code("""
LR_FROZEN  = 1e-3
EPOCHS_FROZEN = 15
BATCH_SIZE = 8

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FROZEN),
    loss=dice_bce_loss,
    metrics=[dice_coefficient, iou_metric, precision_metric,
             recall_metric, specificity_metric]
)
""")

code("""
def make_tf_dataset(X, Y, augment, batch_size, tile_size, shuffle):
    def aug_fn(img, mask):
        transform = get_train_augmentation(tile_size) if augment \
                    else get_val_augmentation(tile_size)
        img_aug, mask_aug = apply_augmentation(transform, img.numpy(), mask.numpy())
        return img_aug.astype(np.float32), mask_aug.astype(np.float32)

    def tf_aug_wrapper(img, mask):
        img_aug, mask_aug = tf.py_function(
            aug_fn, [img, mask], [tf.float32, tf.float32])
        img_aug.set_shape([tile_size, tile_size, 3])
        mask_aug.set_shape([tile_size, tile_size])
        return img_aug, mask_aug[..., tf.newaxis]

    ds = tf.data.Dataset.from_tensor_slices((X, Y[..., 0]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.map(tf_aug_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_tf_dataset(X_train, Y_train, augment=True,
                            batch_size=BATCH_SIZE, tile_size=TILE_SIZE, shuffle=True)
val_ds   = make_tf_dataset(X_val,   Y_val,   augment=False,
                            batch_size=BATCH_SIZE, tile_size=TILE_SIZE, shuffle=False)
""")

md("""
**Note on val metrics:** val always looks better than train on Dice/IoU
because training tiles are augmented (harder) while validation tiles are
clean. This is expected and not a sign of data leakage.
""")

code("""
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=[keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)],
    verbose=1,
)
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Phase 1: decoder only (encoder frozen)", fontsize=13)

for ax, key, title in zip(axes,
    ["loss", "dice_coefficient", "precision_metric"],
    ["Loss (Dice+BCE)", "Dice coefficient", "Precision"]):
    ax.plot(history_phase1.history[key],        label="train")
    ax.plot(history_phase1.history[f"val_{key}"], label="val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ============================================================================
# SECTION 10: Phase 2
# ============================================================================
md("""
## 10. Training: Phase 2 (full fine-tuning, encoder unfrozen)

Now that the decoder has learned something sensible, we unfreeze the encoder
and fine-tune the entire network at a learning rate 100× smaller (1e-5).

A smaller learning rate is used because the encoder's pretrained weights are 
already good. A large learning rate would erase them; a small one lets them 
specialize gently towards vessel images (adapting to your specific stain colors 
and vessel morphology) without forgetting general visual features.
""")

code("""
for layer in model.layers:
    layer.trainable = True

LR_FINETUNE    = 1e-5
EPOCHS_FINETUNE = 30

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FINETUNE),
    loss=dice_bce_loss,
    metrics=[dice_coefficient, iou_metric, precision_metric,
             recall_metric, specificity_metric]
)
print(f"Trainable parameters: {sum(np.prod(w.shape) for w in model.trainable_weights):,}")
""")

code("""
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            "vessel_model_best.keras", save_best_only=True, monitor="val_loss"),
    ],
    verbose=1,
)
""")

code("""
# Combined Phase 1 + Phase 2 loss curves
phase1_len = len(history_phase1.history["loss"])

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Training history: Phase 1 + Phase 2", fontsize=13)

for ax, key, title in zip(axes,
    ["loss", "dice_coefficient", "recall_metric"],
    ["Loss (Dice+BCE)", "Dice coefficient", "Recall"]):
    full_train = history_phase1.history[key] + history_phase2.history[key]
    full_val   = (history_phase1.history[f"val_{key}"] +
                  history_phase2.history[f"val_{key}"])
    epochs = range(1, len(full_train) + 1)
    ax.plot(epochs, full_train, label="train")
    ax.plot(epochs, full_val,   label="val")
    ax.axvline(phase1_len + 0.5, color="gray", linestyle="--",
               linewidth=1, label="phase 1→2")
    ax.set_title(title); ax.set_xlabel("Epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
""")

# ============================================================================
# SECTION 11: Visual inspection -- val and test
# ============================================================================
md("""
## 11. Visual inspection: original image | true mask | predicted mask


For each image we show:
- **Original IHC image**
- **True mask** — manual/ImageJ vessel tracings (ground truth)
- **Predicted mask** — automated detection by model
""")

code("""
def visualize_predictions(records, model, tile_size, split_name="val"):
    \"\"\"Show original | true mask | predicted mask for each image.\"\"\"
    for rec in records:
        img, true_mask = load_image_and_mask_cropped(rec, crop_black_border=True)
        H, W = true_mask.shape

        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        img_padded = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="constant")
        pred_padded = np.zeros(img_padded.shape[:2], dtype=np.float32)
        for y in range(0, img_padded.shape[0], tile_size):
            for x in range(0, img_padded.shape[1], tile_size):
                tile = img_padded[y:y+tile_size, x:x+tile_size].astype(np.float32)/255.0
                pred = model.predict(tile[np.newaxis,...], verbose=0)[0,...,0]
                pred_padded[y:y+tile_size, x:x+tile_size] = pred
        pred_mask = (pred_padded[:H,:W] > 0.5).astype(np.float32)

        true_frac = float(true_mask.sum()) / true_mask.size
        pred_frac = float(pred_mask.sum()) / pred_mask.size
        err = abs(true_frac - pred_frac)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"{rec['stem']}  |  true={true_frac:.4f}  pred={pred_frac:.4f}  "
            f"abs_error={err:.4f}", fontsize=11, y=1.01)
        axes[0].imshow(img);       axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(true_mask,  cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("True mask (manual)"); axes[1].axis("off")
        axes[2].imshow(pred_mask,  cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Predicted mask"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{rec['stem']}_{split_name}_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved {rec['stem']}_{split_name}_comparison.png")
""")

code("""
print("=== Validation set ===")
visualize_predictions(val_records, model, TILE_SIZE, split_name="val")
""")

code("""
print("=== Test set ===")
visualize_predictions(test_records, model, TILE_SIZE, split_name="test")
""")

# ============================================================================
# SECTION 12: Area fraction evaluation
# ============================================================================
md("""
## 12. Per-image vascular area fraction

The primary biological metric: how closely does the model's automated area
fraction match your manual measurement, per whole image?

This is evaluated on the test set, slides the model and training process
never touched in any way, giving an honest, unbiased performance estimate.

**Bland-Altman plot:** the standard method validation plot in biomedical
research. Plots mean of automated+manual (x-axis) vs. their difference
(y-axis). Shows both systematic bias and spread of agreement.
""")

code("""
def evaluate_area_fraction_per_image(model, records, tile_size):
    results = []
    for rec in records:
        img, true_mask = load_image_and_mask_cropped(rec, crop_black_border=True)
        H, W = true_mask.shape
        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        img_padded = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="constant")
        pred_padded = np.zeros(img_padded.shape[:2], dtype=np.float32)
        for y in range(0, img_padded.shape[0], tile_size):
            for x in range(0, img_padded.shape[1], tile_size):
                tile = img_padded[y:y+tile_size, x:x+tile_size].astype(np.float32)/255.0
                pred = model.predict(tile[np.newaxis,...], verbose=0)[0,...,0]
                pred_padded[y:y+tile_size, x:x+tile_size] = pred
        pred_mask = (pred_padded[:H,:W] > 0.5).astype(np.float32)
        true_frac = float(true_mask.sum()) / true_mask.size
        pred_frac = float(pred_mask.sum()) / pred_mask.size
        results.append({
            "stem": rec["stem"],
            "true_area_fraction": true_frac,
            "pred_area_fraction": pred_frac,
            "abs_error": abs(true_frac - pred_frac),
        })
        print(f"  {rec['stem']}: true={true_frac:.4f}, pred={pred_frac:.4f}, "
              f"abs_error={abs(true_frac-pred_frac):.4f}")
    return results

print("=== Test set: area fraction evaluation ===")
test_area_results = evaluate_area_fraction_per_image(model, test_records, TILE_SIZE)
""")

code("""
# Bland-Altman plot
true_fracs = np.array([r["true_area_fraction"] for r in test_area_results])
pred_fracs = np.array([r["pred_area_fraction"] for r in test_area_results])
means = (true_fracs + pred_fracs) / 2
diffs = pred_fracs - true_fracs
mean_diff = diffs.mean()
std_diff  = diffs.std()

plt.figure(figsize=(8, 6))
plt.scatter(means, diffs, alpha=0.7, zorder=3)
for stem, m, d in zip([r["stem"] for r in test_area_results], means, diffs):
    plt.annotate(stem, (m, d), fontsize=7, xytext=(4, 4),
                 textcoords="offset points", alpha=0.7)
plt.axhline(mean_diff,              color="red", linestyle="-",
            label=f"Mean bias = {mean_diff:.4f}")
plt.axhline(mean_diff + 1.96*std_diff, color="red", linestyle="--",
            label="±1.96 SD")
plt.axhline(mean_diff - 1.96*std_diff, color="red", linestyle="--")
plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("Mean of predicted & true area fraction")
plt.ylabel("Predicted − True area fraction")
plt.title("Bland-Altman: automated vs. manual vascular area fraction (test set)")
plt.legend(); plt.tight_layout(); plt.show()

abs_errors = [r["abs_error"] for r in test_area_results]
print(f"Mean absolute error: {np.mean(abs_errors):.4f}")
print(f"Mean bias:           {mean_diff:.4f} "
      f"({'over-predicts' if mean_diff > 0 else 'under-predicts'})")
""")

# ============================================================================
# SECTION 13: Full pixel-level metrics (back-calculated)
# ============================================================================
md("""
## 13. Full pixel-level metrics on test images

Back-calculated on whole stitched images after training -- not from the
epoch-averaged training log. This gives one clean value per metric per image.

**Metrics reported:**
- **Dice:** overlap between predicted and true vessel regions (primary metric)
- **IoU:** stricter overlap (penalises both over and under prediction)
- **Precision:** of pixels called vessel, what fraction actually were?
  Low precision = over-prediction 
- **Recall:** of actual vessel pixels, what fraction were found?
  Low recall = under-prediction 
- **Specificity:** of background pixels, what fraction correctly unlabeled?
  Low specificity directly inflates area fraction measurements
""")

code("""
def compute_full_pixel_metrics(model, records, tile_size):
    \"\"\"
    Compute all pixel-level metrics on full stitched images.
    No retraining needed -- uses the trained model's predictions.
    \"\"\"
    all_metrics = []
    for rec in records:
        img, true_mask = load_image_and_mask_cropped(rec, crop_black_border=True)
        H, W = true_mask.shape
        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        img_padded = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="constant")
        pred_padded = np.zeros(img_padded.shape[:2], dtype=np.float32)
        for y in range(0, img_padded.shape[0], tile_size):
            for x in range(0, img_padded.shape[1], tile_size):
                tile = img_padded[y:y+tile_size,x:x+tile_size].astype(np.float32)/255.0
                pred = model.predict(tile[np.newaxis,...], verbose=0)[0,...,0]
                pred_padded[y:y+tile_size, x:x+tile_size] = pred
        pred_mask = (pred_padded[:H,:W] > 0.5).astype(np.float32)

        # Reshape for metric functions (need batch + channel dimensions)
        y_true = true_mask[np.newaxis,...,np.newaxis].astype(np.float32)
        y_pred = pred_mask[np.newaxis,...,np.newaxis].astype(np.float32)

        m = {
            "stem":        rec["stem"],
            "dice":        float(dice_coefficient(y_true, y_pred).numpy()),
            "iou":         float(iou_metric(y_true, y_pred).numpy()),
            "precision":   float(precision_metric(y_true, y_pred).numpy()),
            "recall":      float(recall_metric(y_true, y_pred).numpy()),
            "specificity": float(specificity_metric(y_true, y_pred).numpy()),
        }
        all_metrics.append(m)
        print(f"  {rec['stem']:12s}  Dice={m['dice']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"Spec={m['specificity']:.4f}")

    print()
    for metric in ["dice", "iou", "precision", "recall", "specificity"]:
        vals = [m[metric] for m in all_metrics]
        print(f"  Mean {metric:12s}: {np.mean(vals):.4f}  (std={np.std(vals):.4f})")
    return all_metrics

print("=== Test set: full pixel-level metrics ===")
test_pixel_metrics = compute_full_pixel_metrics(model, test_records, TILE_SIZE)
""")

code("""
# Bar chart: all five metrics per image
import pandas as pd
df_metrics = pd.DataFrame(test_pixel_metrics)
metrics_to_plot = ["dice", "iou", "precision", "recall", "specificity"]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(22, 5))
fig.suptitle("Pixel-level metrics per test image", fontsize=13)
colors = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7F77DD"]

for ax, metric, color in zip(axes, metrics_to_plot, colors):
    ax.bar(range(len(df_metrics)), df_metrics[metric], color=color, alpha=0.85)
    ax.axhline(df_metrics[metric].mean(), color="black", linestyle="--",
               linewidth=1, label=f"mean={df_metrics[metric].mean():.3f}")
    ax.set_xticks(range(len(df_metrics)))
    ax.set_xticklabels(df_metrics["stem"], rotation=45, ha="right", fontsize=8)
    ax.set_title(metric.capitalize())
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("test_pixel_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: test_pixel_metrics.png")
""")

# ============================================================================
# SECTION 14: Vessel morphology
# ============================================================================
md("""
## 14. Vessel morphology metrics

Beyond vascular density (area fraction), the predicted mask gives access to
morphological measurements for each individual vessel:

- **Vessel count** — number of discrete vessel segments
- **Mean / median diameter** — typical vessel caliber
- **Max diameter** — widest vessel in a given image
- **Circularity** — 1.0 = perfect circle; lower = elongated
- **Elongation** — major / minor axis ratio

**Method:** distance transform. For each vessel pixel, its distance to the
nearest non-vessel pixel = the radius of the largest inscribed circle at
that point. Maximum distance × 2 = vessel diameter at its widest. This is
robust to irregular shapes and tortuous paths.

**Pixel size:** update `PIXEL_SIZES` below for morphology in micrometers.
Original cohort: 289px = 50μm → 0.173 μm/px. MA images: confirm from
acquisition metadata or measure a scale bar in ImageJ.
""")

code("""
from vessel_morphology import (
    calculate_vessel_morphology,
    print_morphology_summary,
    PIXEL_SIZE_ORIGINAL_COHORT_UM,
)

# ── Update these with your actual pixel sizes ─────────────────────────────────
PIXEL_SIZES = {
    # Original cohort (all 4096x3008 images): 0.1678 um/px
    # MA/striatum: update once confirmed from acquisition metadata
    # "MA16": 0.336,
}
MA_PIXEL_SIZE = None  # e.g. 0.336 -- update once known

def get_pixel_size(stem):
    if stem in PIXEL_SIZES:
        return PIXEL_SIZES[stem]
    if stem.startswith("MA"):
        if MA_PIXEL_SIZE is not None:
            return MA_PIXEL_SIZE
        print(f"  [{stem}] pixel size unknown -- reporting in pixels")
        return None
    return PIXEL_SIZE_ORIGINAL_COHORT_UM
# ─────────────────────────────────────────────────────────────────────────────
""")

code("""
def predict_full_mask(model, rec, tile_size):
    \"\"\"Tile, predict, and stitch -- returns full-resolution binary mask.\"\"\"
    img, true_mask = load_image_and_mask_cropped(rec, crop_black_border=True)
    H, W = true_mask.shape
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    img_padded = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="constant")
    pred_padded = np.zeros(img_padded.shape[:2], dtype=np.float32)
    for y in range(0, img_padded.shape[0], tile_size):
        for x in range(0, img_padded.shape[1], tile_size):
            tile = img_padded[y:y+tile_size,x:x+tile_size].astype(np.float32)/255.0
            pred = model.predict(tile[np.newaxis,...], verbose=0)[0,...,0]
            pred_padded[y:y+tile_size, x:x+tile_size] = pred
    return img, true_mask, (pred_padded[:H,:W] > 0.5).astype(np.float32)

print("=== Test set: vessel morphology ===")
morphology_results = []
for rec in test_records:
    img, true_mask, pred_mask = predict_full_mask(model, rec, TILE_SIZE)
    pixel_size = get_pixel_size(rec["stem"])
    metrics = calculate_vessel_morphology(pred_mask, pixel_size_um=pixel_size)
    metrics["stem"] = rec["stem"]
    morphology_results.append(metrics)
    print_morphology_summary(rec["stem"], metrics)
""")

code("""
# Summary table
df_morph = pd.DataFrame([{
    "stem":         r["stem"],
    "vessel_count": r["vessel_count"],
    "mean_diam":    r["mean_diameter"],
    "median_diam":  r["median_diameter"],
    "max_diam":     r["max_diameter"],
    "std_diam":     r["std_diameter"],
    "circularity":  r["mean_circularity"],
    "elongation":   r["mean_elongation"],
    "unit":         r["unit"],
} for r in morphology_results])

unit = morphology_results[0]["unit"] if morphology_results else "px"
print(df_morph.to_string(index=False, float_format="{:.3f}".format))
print()
print(f"Summary across {len(df_morph)} test images:")
for col in ["vessel_count","mean_diam","median_diam","max_diam","circularity"]:
    print(f"  {col:15s}: mean={df_morph[col].mean():.3f}  std={df_morph[col].std():.3f}")
""")

code("""
# Four-panel morphology bar chart
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Vessel morphology metrics -- test set ({unit})", fontsize=13)

stems = df_morph["stem"].tolist()
x = range(len(stems))

axes[0,0].bar(x, df_morph["mean_diam"], color="#2a78d6")
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(stems, rotation=45, ha="right")
axes[0,0].set_title(f"Mean diameter ({unit})"); axes[0,0].set_ylabel(f"({unit})")

axes[0,1].bar(x, df_morph["max_diam"], color="#e34948")
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(stems, rotation=45, ha="right")
axes[0,1].set_title(f"Max diameter ({unit}) -- vasodilation marker")
axes[0,1].set_ylabel(f"({unit})")

axes[1,0].bar(x, df_morph["vessel_count"], color="#1baf7a")
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(stems, rotation=45, ha="right")
axes[1,0].set_title("Vessel count"); axes[1,0].set_ylabel("Count")

axes[1,1].bar(x, df_morph["circularity"], color="#eda100")
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(stems, rotation=45, ha="right")
axes[1,1].set_title("Mean circularity (1.0 = circle)")
axes[1,1].set_ylabel("Circularity"); axes[1,1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("test_morphology_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: test_morphology_summary.png")
""")

code("""
# Save both result tables to CSV
df_metrics.drop(columns=["stem"], errors="ignore").to_csv(
    "test_pixel_metrics.csv", index=False)
df_morph.to_csv("test_morphology_results.csv", index=False)
print("Saved: test_pixel_metrics.csv")
print("Saved: test_morphology_results.csv")
""")

# ============================================================================
# SECTION 15: Next steps
# ============================================================================
md("""
## 15. Next steps

**If val/test Dice or area fraction look poor:**
- Check the visual predictions (Section 11) to understand where errors are
- Inspect the precision/recall split: low precision = over-prediction
  (artifacts/bright images); low recall = under-prediction (faint signal)
- Consider whether problematic images warrant exclusion with documented reasons

**Overall performance estimate:**
- 5-fold cross-validation via `train.py` / `submit_cv_training.sh`
- The CV summary averages over all possible slide groupings, more stable than
  any single train/val/test split

**For deployment:**
- Use `predict.py` with the trained `final_model.keras` -- handles arbitrary
  image sizes, vessel marker in any channel, and black-bordered ROI images automatically
- `vessel_morphology.py` provides the same diameter/circularity metrics as
  Section 14 for deployment predictions
""")

# ============================================================================
# Build notebook
# ============================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (vessel-seg)",
        "language": "python",
        "name": "vessel-seg"
    },
    "language_info": {"name": "python", "version": "3.11"}
}

with open("vessel_segmentation_walkthrough_v2.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Notebook written: vessel_segmentation_walkthrough_v2.ipynb")
print(f"Total cells: {len(cells)}")
