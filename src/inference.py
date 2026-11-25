"""Phase 5: Inference engine & Pre-Sigmoid Logit Grad-CAM for exported art classifiers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Quiet TensorFlow INFO logs and prevent Matplotlib GUI backend issues
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.config import IMG_SIZE, LABEL_MAP, MODELS_DIR  # noqa: E402
from src.data_pipeline import load_keras_model_compat  # noqa: E402

EXPORT_DIR = MODELS_DIR / "export"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "resnet50_ft": {
        "key": "resnet50_ft",
        "run_id": "Exp-3B",
        "label": "ResNet50 Fine-Tuned (Exp-3B — Primary Champion)",
        "exported_filename": "art_classifier_resnet50_ft.keras",
        "fallback_filename": "exp3b_resnet50_finetuned.keras",
        "gradcam_layer": "resnet50",
        "prob_layer": "prob_ai",
        "test_acc": "98.58%",
        "test_roc_auc": "0.9985",
        "ood_acc": "90.00% (100% Modern AI / 80% WikiArt)",
        "size_note": "91.61 MB inference export (224.75 MB checkpoint)",
    },
    "mobilenetv2_ft": {
        "key": "mobilenetv2_ft",
        "run_id": "Exp-2B",
        "label": "MobileNetV2 Fine-Tuned (Exp-2B — Pareto Lightweight)",
        "exported_filename": "art_classifier_mobilenetv2_ft.keras",
        "fallback_filename": "exp2b_mobilenetv2_finetuned.keras",
        "gradcam_layer": "mobilenetv2_1.00_224",
        "prob_layer": "prob_ai",
        "test_acc": "97.87%",
        "test_roc_auc": "0.9948",
        "ood_acc": "83.33% (80% MJ v6 / 100% FLUX & DALL-E 3)",
        "size_note": "9.80 MB inference export (24.65 MB checkpoint)",
    },
}


def resolve_model_path(model_key: str) -> Path:
    """Resolve exported inference-only .keras model first, falling back to checkpoint."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Valid keys: {list(MODEL_REGISTRY)}"
        )
    spec = MODEL_REGISTRY[model_key]
    exported_path = EXPORT_DIR / spec["exported_filename"]
    if exported_path.exists():
        return exported_path
    fallback_path = MODELS_DIR / spec["fallback_filename"]
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Neither exported model ({exported_path}) nor fallback checkpoint ({fallback_path}) exists."
    )


def preprocess_pil_image(
    img: Image.Image,
    center_crop: bool = True,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert PIL image to 224x224 uint8 and float32 [0..255] RGB arrays."""
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    if center_crop:
        w, h = rgb.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        rgb = rgb.crop((left, top, left + side, top + side))
    resized = rgb.resize(img_size, Image.Resampling.BILINEAR)
    img_uint8 = np.asarray(resized, dtype=np.uint8)
    img_float32 = img_uint8.astype(np.float32)
    return img_uint8, img_float32


def overlay_gradcam(
    img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42
) -> np.ndarray:
    """Overlay normalized [0, 1] Grad-CAM heatmap onto uint8 RGB image using jet colormap."""
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heatmap)[..., :3] * 255.0).astype(np.float32)
    blended = (1.0 - alpha) * img_uint8.astype(np.float32) + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_gradcam_heatmap(
    model: Any,
    img_rgb_224: np.ndarray,
    spatial_layer_name: str,
    target_class: int | None = None,
    threshold: float = 0.50,
    prob_layer_name: str = "prob_ai",
) -> tuple[np.ndarray, float, int]:
    """Compute Pre-Sigmoid Logit Grad-CAM heatmap and return (heatmap, prob_ai, target_class).

    Gradients are taken wrt pre-sigmoid logits (W @ x + b) rather than post-sigmoid
    probabilities to prevent gradient vanishing when confidence approaches 0.0 or 1.0.
    """
    layer_names = [lyr.name for lyr in model.layers]
    sp_idx = layer_names.index(spatial_layer_name)
    prob_idx = layer_names.index(prob_layer_name)

    pre_layers = model.layers[1 : sp_idx + 1]
    post_layers = model.layers[sp_idx + 1 : prob_idx]
    prob_layer = model.layers[prob_idx]

    img_batch = tf.convert_to_tensor(img_rgb_224[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        x = img_batch
        for lyr in pre_layers:
            x = lyr(x, training=False)
        conv_out = x
        tape.watch(conv_out)
        for lyr in post_layers:
            x = lyr(x, training=False)
        logit = tf.matmul(x, prob_layer.kernel) + prob_layer.bias
        logit_scalar = logit[:, 0]

    prob_ai = float(tf.math.sigmoid(logit_scalar[0]).numpy())
    resolved_class = (1 if prob_ai >= threshold else 0) if target_class is None else target_class

    # Negating score is equivalent to negating gradients due to derivative linearity
    grads = tape.gradient(logit_scalar, conv_out)
    if resolved_class == 0:
        grads = -grads

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam)
    cam_max = tf.reduce_max(cam)
    if float(cam_max) > 1e-8:
        cam = cam / cam_max
    cam_resized = tf.image.resize(cam[..., None], IMG_SIZE, method="bilinear")[..., 0]
    heatmap = np.clip(cam_resized.numpy(), 0.0, 1.0)
    return heatmap, prob_ai, resolved_class


class ArtClassifierEngine:
    """Lazy-loading inference and Grad-CAM engine for deployed art classifiers."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._model_paths: dict[str, Path] = {}

    def get_model(self, model_key: str) -> Any:
        """Load and cache the requested Keras model."""
        if model_key not in self._models:
            path = resolve_model_path(model_key)
            model = load_keras_model_compat(path, compile=False)
            self._models[model_key] = model
            self._model_paths[model_key] = path
        return self._models[model_key]

    def predict_with_gradcam(
        self,
        img: Image.Image,
        model_key: str = "resnet50_ft",
        threshold: float = 0.50,
        center_crop: bool = True,
        alpha: float = 0.42,
    ) -> dict[str, Any]:
        """Run end-to-end preprocessing, prediction, and Pre-Sigmoid Logit Grad-CAM."""
        if img is None:
            raise ValueError("Input image cannot be None.")

        spec = MODEL_REGISTRY[model_key]
        model = self.get_model(model_key)

        t0 = time.perf_counter()
        img_uint8, img_float32 = preprocess_pil_image(img, center_crop=center_crop)
        heatmap, prob_ai, pred_cls = compute_gradcam_heatmap(
            model=model,
            img_rgb_224=img_float32,
            spatial_layer_name=spec["gradcam_layer"],
            target_class=None,
            threshold=threshold,
            prob_layer_name=spec["prob_layer"],
        )

        prob_real = float(1.0 - prob_ai)
        predicted_label = LABEL_MAP[pred_cls]
        gradcam_overlay = overlay_gradcam(img_uint8, heatmap, alpha=alpha)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "cropped_rgb": img_uint8,
            "gradcam_overlay": gradcam_overlay,
            "heatmap": heatmap,
            "probs": {"Real Art": prob_real, "AI-Generated Art": prob_ai},
            "pred_class": pred_cls,
            "predicted_label": predicted_label,
            "prob_ai": prob_ai,
            "prob_real": prob_real,
            "latency_ms": latency_ms,
            "model_info": {
                **spec,
                "loaded_path": str(self._model_paths[model_key]),
            },
        }
