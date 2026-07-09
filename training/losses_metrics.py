"""
Loss functions and metrics for binary vessel segmentation.

Why Dice+BCE: vessel pixels are typically a small minority of total tissue
pixels, and plain BCE alone tends to be dominated by the easy majority (background) 
class. Dice loss directly optimizes the overlap metric; 
combining with BCE keeps gradients stable early in training when Dice loss alone 
can be noisy/uninformative on mostly-empty masks.
"""

import tensorflow as tf
from tensorflow import keras


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def dice_bce_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, threshold=0.5, smooth=1.0):
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def area_fraction_error(y_true, y_pred, threshold=0.5):
    """
    Per-batch mean absolute error between predicted and true vessel area
    fraction -- the metric that maps directly onto your actual biological
    readout, as distinct from pixel-wise Dice/IoU.
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    true_frac = tf.reduce_mean(y_true, axis=[1, 2, 3])
    pred_frac = tf.reduce_mean(y_pred_bin, axis=[1, 2, 3])
    return tf.reduce_mean(tf.abs(true_frac - pred_frac))


def precision_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Precision = TP / (TP + FP)
    "Of the pixels the model called vessel, what fraction actually were?"
    Low precision = model is over-predicting (false positives).
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return (tp + smooth) / (tp + fp + smooth)


def recall_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Recall (Sensitivity) = TP / (TP + FN)
    "Of the actual vessel pixels, what fraction did the model find?"
    Low recall = model is under-predicting (false negatives, missing vessels).
    Note: Dice = 2 * (precision * recall) / (precision + recall),
    so Dice alone can't tell you which direction errors are going.
    Reporting precision and recall separately reveals that asymmetry.
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    return (tp + smooth) / (tp + fn + smooth)


def specificity_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Specificity = TN / (TN + FP)
    "Of the actual background pixels, what fraction did the model
    correctly leave unlabeled?"
    Low specificity = high false positive rate = model over-calls vessels,
    directly inflating area fraction measurements.
    Particularly important to report for vessel density quantification
    since false positives inflate your primary biological readout.
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    tn = tf.reduce_sum((1.0 - y_true_f) * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return (tn + smooth) / (tn + fp + smooth)



    import numpy as np

    # Self-test with known cases:
    # 1. Perfect prediction -> Dice = 1, loss ~ 0, IoU = 1, area error = 0
    y_true = np.zeros((1, 10, 10, 1), dtype=np.float32)
    y_true[0, 2:5, 2:5, 0] = 1.0
    y_pred_perfect = y_true.copy()

    d = dice_coefficient(y_true, y_pred_perfect).numpy()
    iou = iou_metric(y_true, y_pred_perfect).numpy()
    err = area_fraction_error(y_true, y_pred_perfect).numpy()
    print(f"Perfect match -> Dice={d:.4f} (expect ~1.0), IoU={iou:.4f} (expect ~1.0), "
          f"area_err={err:.4f} (expect ~0.0)")
    assert d > 0.99 and iou > 0.99 and err < 0.01

    # 2. No overlap at all -> Dice ~ 0, IoU ~ 0
    y_pred_wrong = np.zeros((1, 10, 10, 1), dtype=np.float32)
    y_pred_wrong[0, 7:9, 7:9, 0] = 1.0  # disjoint region
    d2 = dice_coefficient(y_true, y_pred_wrong).numpy()
    iou2 = iou_metric(y_true, y_pred_wrong).numpy()
    print(f"No overlap -> Dice={d2:.4f} (expect ~0.0), IoU={iou2:.4f} (expect ~0.0)")
    assert d2 < 0.1 and iou2 < 0.1

    # 3. Same area, no overlap -> area_fraction_error should be ~0 (areas match)
    #    even though Dice/IoU are near 0 (location is wrong). This demonstrates
    #    WHY you need both metrics -- area fraction alone can hide bad localization.
    # (must construct a disjoint region of EXACTLY the same pixel count as
    #  y_true's 3x3=9 pixel region to isolate "same area" from "same shape")
    y_pred_same_area_wrong_loc = np.zeros((1, 10, 10, 1), dtype=np.float32)
    y_pred_same_area_wrong_loc[0, 7, 0:9, 0] = 1.0  # a 1x9 strip = 9 pixels, disjoint from y_true
    d3 = dice_coefficient(y_true, y_pred_same_area_wrong_loc).numpy()
    err3 = area_fraction_error(y_true, y_pred_same_area_wrong_loc).numpy()
    print(f"Same area (9px), wrong location/shape -> area_err={err3:.4f} (expect ~0.0) "
          f"despite Dice={d3:.4f} (expect ~0.0)")
    assert err3 < 0.01
    assert d3 < 0.1

    print("\nPASS: loss/metric self-tests OK")
    print("Note: case 3 demonstrates why you must report BOTH pixel-level Dice/IoU")
    print("AND area-fraction agreement -- a model could get area fraction right by")
    print("luck/bias while still localizing vessels incorrectly.")

    # 4. Precision / recall / specificity with known values
    # True mask: 3x3 block at [2:5, 2:5] = 9 positive pixels, 91 negative
    # Perfect prediction: precision=1, recall=1, specificity=1
    p = precision_metric(y_true, y_pred_perfect).numpy()
    r = recall_metric(y_true, y_pred_perfect).numpy()
    s = specificity_metric(y_true, y_pred_perfect).numpy()
    print(f"\nPerfect match -> precision={p:.4f}, recall={r:.4f}, specificity={s:.4f} "
          f"(all expect ~1.0)")
    assert p > 0.99 and r > 0.99 and s > 0.99

    # All-zero prediction: precision=undefined(~0), recall=0, specificity=1
    y_pred_zeros = np.zeros((1, 10, 10, 1), dtype=np.float32)
    p0 = precision_metric(y_true, y_pred_zeros).numpy()
    r0 = recall_metric(y_true, y_pred_zeros).numpy()
    s0 = specificity_metric(y_true, y_pred_zeros).numpy()
    print(f"All-zero pred -> precision={p0:.4f} (~0), recall={r0:.4f} (~0), "
          f"specificity={s0:.4f} (~1.0)")
    assert r0 < 0.05 and s0 > 0.99

    # All-ones prediction: precision=low, recall=1, specificity=0
    y_pred_ones = np.ones((1, 10, 10, 1), dtype=np.float32)
    p1 = precision_metric(y_true, y_pred_ones).numpy()
    r1 = recall_metric(y_true, y_pred_ones).numpy()
    s1 = specificity_metric(y_true, y_pred_ones).numpy()
    print(f"All-ones pred -> precision={p1:.4f} (~0.09), recall={r1:.4f} (~1.0), "
          f"specificity={s1:.4f} (~0.0)")
    assert p1 < 0.15 and r1 > 0.99 and s1 < 0.05

    print("\nPASS: precision/recall/specificity self-tests OK")
    print("Key insight: a model that predicts ALL pixels as vessel gets recall=1.0")
    print("but precision~0 and specificity~0 -- Dice and these three metrics together")
    print("catch what any single metric alone would miss.")
