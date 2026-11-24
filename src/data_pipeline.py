"""Phase 1: tf.data input pipeline and per-backbone in-model preprocessing layers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import keras
import pandas as pd
import tensorflow as tf

from src.config import BATCH_SIZE, IMG_SIZE, PROCESSED_DIR, SEED

BackbonePreproc = Literal["scratch", "mobilenetv2", "resnet50"]


def load_splits(csv_path: Path = PROCESSED_DIR / "splits.csv") -> pd.DataFrame:
    """Load stratified split table and resolve image paths portably across OSes."""
    df = pd.read_csv(csv_path)
    images_dir = PROCESSED_DIR / "images"
    df["filepath"] = df["filepath"].apply(
        lambda fp: fp if Path(fp).exists() else str(images_dir / Path(fp).name)
    )
    return df


def build_augmentation_layer(name: str = "train_augmentation") -> keras.Sequential:
    """Conservative spatial augmentation that preserves subtle AI generation textures."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=SEED),
            keras.layers.RandomRotation(0.05, seed=SEED),
            keras.layers.RandomZoom(0.1, seed=SEED),
        ],
        name=name,
    )


@keras.saving.register_keras_serializable(package="art_classifier")
class ResNet50PreprocessLayer(keras.layers.Layer):
    """Apply Caffe-style RGB->BGR conversion and ImageNet channel mean subtraction."""

    IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.cast(inputs, tf.float32)
        x = x[..., ::-1]  # RGB -> BGR
        mean = tf.constant(self.IMAGENET_BGR_MEAN, dtype=tf.float32)
        return x - mean

    def get_config(self) -> dict[str, object]:
        return super().get_config()


def get_preprocessing_layer(backbone: BackbonePreproc) -> keras.layers.Layer:
    """Return architecture-matched preprocessing layer expecting raw [0, 255] RGB inputs."""
    key = backbone.lower()
    if key == "scratch":
        return keras.layers.Rescaling(1.0 / 255.0, name="scratch_rescale_0_1")
    if key == "mobilenetv2":
        return keras.layers.Rescaling(
            scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess"
        )
    if key == "resnet50":
        return ResNet50PreprocessLayer(name="resnet50_preprocess")
    raise ValueError(
        f"Unsupported backbone '{backbone}'. Expected 'scratch', 'mobilenetv2', or 'resnet50'."
    )


def _decode_and_resize(
    filepath: tf.Tensor,
    label: tf.Tensor,
    img_size: tuple[int, int],
    center_crop: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_png(raw, channels=3)
    img = tf.cast(img, tf.float32)

    if center_crop:
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        side = tf.minimum(h, w)
        offset_h = (h - side) // 2
        offset_w = (w - side) // 2
        img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, side, side)

    img = tf.image.resize(img, img_size, method="bilinear")
    img.set_shape([img_size[0], img_size[1], 3])
    return img, tf.cast(label, tf.float32)


def build_tf_dataset(
    df: pd.DataFrame,
    split_name: str,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    center_crop: bool = True,
    cache: bool = True,
    shuffle: bool | None = None,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset emitting float32 RGB tensors in [0.0, 255.0].

    Note: center_crop defaults to True to mitigate the strong 1:1 aspect-ratio
    bias detected in Phase 1 forensics (79.01% AI Art vs 1.55% Real Art square).
    """
    sub = df[df["split"] == split_name].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows found for split '{split_name}'.")

    if shuffle is None:
        shuffle = split_name == "train"

    filepaths = sub["filepath"].astype(str).tolist()
    labels = sub["label"].astype("float32").tolist()

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(
        lambda fp, lbl: _decode_and_resize(fp, lbl, img_size, center_crop),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=len(sub), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if augment and split_name == "train":
        aug_layer = build_augmentation_layer()
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_keras_model_compat(
    filepath: str | Path, compile: bool = False
) -> keras.Model:
    """Load a .keras archive across Keras 3.15->3.13 minor-version config diffs."""
    import inspect

    from keras.src.initializers.initializer import Initializer
    from keras.src.layers.normalization.batch_normalization import (
        BatchNormalization,
    )

    prev_init = Initializer.from_config
    prev_bn = BatchNormalization.from_config

    @classmethod
    def _compat_init_from_config(cls, config):
        cfg = dict(config)
        sig = inspect.signature(cls.__init__).parameters
        if "input_axes" not in sig or cfg.get("input_axes") is None:
            cfg.pop("input_axes", None)
        if "output_axes" not in sig or cfg.get("output_axes") is None:
            cfg.pop("output_axes", None)
        return prev_init.__func__(cls, cfg)

    @classmethod
    def _compat_bn_from_config(cls, config):
        cfg = dict(config)
        sig = inspect.signature(cls.__init__).parameters
        if "renorm" not in sig or cfg.get("renorm") is False:
            cfg.pop("renorm", None)
            cfg.pop("renorm_clipping", None)
            cfg.pop("renorm_momentum", None)
        return prev_bn.__func__(cls, cfg)

    Initializer.from_config = _compat_init_from_config
    BatchNormalization.from_config = _compat_bn_from_config
    try:
        return keras.models.load_model(filepath, compile=compile)
    finally:
        Initializer.from_config = prev_init
        BatchNormalization.from_config = prev_bn


if __name__ == "__main__":
    splits_df = load_splits()
    for sp in ("train", "val", "test"):
        ds = build_tf_dataset(
            splits_df,
            split_name=sp,
            augment=(sp == "train"),
            center_crop=True,
            cache=False,
        )
        xb, yb = next(iter(ds))
        print(
            f"[{sp}] batch shape={xb.shape}, dtype={xb.dtype}, "
            f"range=[{float(tf.reduce_min(xb)):.2f}, {float(tf.reduce_max(xb)):.2f}], "
            f"labels shape={yb.shape}"
        )

    sample_ds = build_tf_dataset(splits_df, split_name="val", augment=False, cache=False)
    sample_x, _ = next(iter(sample_ds))
    for name in ("scratch", "mobilenetv2", "resnet50"):
        layer = get_preprocessing_layer(name)
        out = layer(sample_x)
        print(
            f"[preproc:{name}] shape={out.shape}, "
            f"range=[{float(tf.reduce_min(out)):.3f}, {float(tf.reduce_max(out)):.3f}]"
        )
