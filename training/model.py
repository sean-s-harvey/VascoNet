"""
U-Net for binary vessel segmentation, using an ImageNet-pretrained
EfficientNetB0 as the encoder (transfer learning).

We build this directly on tf.keras.applications rather than the
`segmentation_models` package, because segmentation_models is currently
broken on Keras 3 / TF 2.21+ (it calls a removed Keras 2 internal API).
This implementation has no dependency on unmaintained third-party glue code.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def decoder_block(x, skip, filters, name):
    x = layers.UpSampling2D(size=2, interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=name)
    return x


def build_unet(input_shape=(256, 256, 3), encoder_weights="imagenet", freeze_encoder=False):
    """
    Build a U-Net with EfficientNetB0 encoder.

    Parameters
    ----------
    input_shape : (H, W, C). H and W should be divisible by 32.
    encoder_weights : "imagenet" or None. Use None if you have a non-RGB
        channel count (e.g., 4+ channels) -- imagenet weights require 3
        input channels.
    freeze_encoder : bool. If True, encoder weights are frozen (useful for
        an initial warm-up phase before unfreezing for fine-tuning -- see
        train.py for the two-phase training schedule).

    Returns
    -------
    keras.Model with output shape (H, W, 1), sigmoid activation
    (per-pixel probability of "vessel").
    """
    H, W, C = input_shape
    if H % 32 != 0 or W % 32 != 0:
        raise ValueError(
            f"Input H,W must be divisible by 32 for this encoder's 5 downsampling "
            f"stages; got {H}x{W}. Use a tile_size like 224, 256, 320, 384, etc."
        )
    if encoder_weights == "imagenet" and C != 3:
        raise ValueError(
            f"imagenet pretrained weights require 3 input channels, got {C}. "
            f"Either set encoder_weights=None (train encoder from scratch) or "
            f"reduce your input to 3 channels."
        )

    inputs = keras.Input(shape=input_shape)

    # EfficientNetB0 expects inputs in [0, 255] (it has its own internal
    # rescaling/normalization). Our pipeline produces [0, 1] floats
    # (see data_pipeline.build_tile_arrays), so rescale back up here.
    x = layers.Rescaling(255.0)(inputs)

    encoder = keras.applications.EfficientNetB0(
        include_top=False,
        weights=encoder_weights,
        input_tensor=x,
    )
    encoder.trainable = not freeze_encoder

    # Skip-connection layer names for EfficientNetB0 at each downsampling
    # stage (verified against the model's layer graph below).
    skip_names = [
        "block2a_expand_activation",  # stride 4   (after stem + block1)
        "block3a_expand_activation",  # stride 8
        "block4a_expand_activation",  # stride 16
        "block6a_expand_activation",  # stride 32 input to bottleneck
    ]
    skips = [encoder.get_layer(name).output for name in skip_names]
    bottleneck = encoder.output  # stride 32

    # Decoder: go back up through stride 16 -> 8 -> 4 -> 2 -> 1
    d = decoder_block(bottleneck, skips[3], 256, name="decoder1")  # -> stride 16
    d = decoder_block(d, skips[2], 128, name="decoder2")           # -> stride 8
    d = decoder_block(d, skips[1], 64, name="decoder3")            # -> stride 4
    d = decoder_block(d, skips[0], 32, name="decoder4")            # -> stride 2

    # Final upsample to stride 1 (full resolution). No skip available here
    # (would need the raw input/stem), so just a conv block after upsampling.
    d = layers.UpSampling2D(size=2, interpolation="bilinear", name="decoder5_up")(d)
    d = conv_block(d, 16, name="decoder5")

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask_output")(d)

    model = keras.Model(inputs, outputs, name="unet_efficientnetb0")
    return model


if __name__ == "__main__":
    # Self-test: build the model, confirm output shape matches input H,W,
    # confirm parameter count is reasonable, confirm forward pass works on
    # a dummy batch (catches shape-mismatch bugs in the skip connections
    # immediately, rather than after a long training run).
    import numpy as np

    model = build_unet(input_shape=(256, 256, 3))
    model.summary(line_length=100)

    n_params = model.count_params()
    print(f"\nTotal parameters: {n_params:,}")

    dummy_input = np.random.rand(2, 256, 256, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (2, 256, 256, 1), f"Unexpected output shape: {output.shape}"
    assert output.min() >= 0.0 and output.max() <= 1.0, "Sigmoid output out of [0,1] range"
    print("\nPASS: model builds, forward pass produces correctly-shaped sigmoid output")
