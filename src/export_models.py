"""Phase 5: Export self-contained inference-only .keras models and verify numerical parity."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.config import (
    EXPERIMENTS_DIR,
    IMG_SIZE,
    LABEL_MAP,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from src.data_pipeline import load_keras_model_compat, load_splits

EXPORT_DIR = MODELS_DIR / "export"
MANIFEST_PATH = EXPERIMENTS_DIR / "exported_models_manifest.json"
WIB_TZ = timezone(timedelta(hours=7))

EXPORT_SPECS: list[dict[str, Any]] = [
    {
        "key": "resnet50_ft",
        "run_id": "Exp-3B",
        "display_name": "ResNet50 Fine-Tuned (Primary Champion)",
        "src_filename": "exp3b_resnet50_finetuned.keras",
        "dst_filename": "art_classifier_resnet50_ft.keras",
        "gradcam_layer": "resnet50",
        "preprocessing": "In-model ResNet50PreprocessLayer (RGB->BGR + ImageNet mean subtraction on [0..255] inputs)",
        "recommended_threshold": 0.50,
        "val_tuned_threshold": 0.47,
    },
    {
        "key": "mobilenetv2_ft",
        "run_id": "Exp-2B",
        "display_name": "MobileNetV2 Fine-Tuned (Pareto Lightweight)",
        "src_filename": "exp2b_mobilenetv2_finetuned.keras",
        "dst_filename": "art_classifier_mobilenetv2_ft.keras",
        "gradcam_layer": "mobilenetv2_1.00_224",
        "preprocessing": "In-model Rescaling(scale=1/127.5, offset=-1.0) on [0..255] inputs",
        "recommended_threshold": 0.50,
        "val_tuned_threshold": 0.55,
    },
]


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample_batch(n_samples: int = 5) -> np.ndarray:
    """Load up to n_samples test images preprocessed identically to the inference contract."""
    splits_df = load_splits(PROCESSED_DIR / "splits.csv")
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    # Deterministic sample mix across both classes
    real_rows = test_df[test_df["label"] == 0].head(3)
    ai_rows = test_df[test_df["label"] == 1].head(max(1, n_samples - len(real_rows)))
    chosen = list(real_rows["filepath"]) + list(ai_rows["filepath"])

    batch: list[np.ndarray] = []
    for fp in chosen[:n_samples]:
        with Image.open(fp) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize(IMG_SIZE, Image.Resampling.BILINEAR)
            batch.append(np.asarray(im, dtype=np.float32))
    return np.stack(batch, axis=0)


def load_benchmark_metrics() -> dict[str, dict[str, Any]]:
    """Extract verified Phase 4 benchmark metrics from experiments/phase4_summary.json."""
    summary_path = EXPERIMENTS_DIR / "phase4_summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for run_id in ("Exp-2B", "Exp-3B"):
        test_m = data.get("test_set_evaluation", {}).get("test_default_050", {}).get(run_id, {})
        ood_m = data.get("ood_stress_test", {}).get("ood_default_050", {}).get(run_id, {})
        eff_m = data.get("inference_efficiency", {}).get(run_id, {})
        out[run_id] = {
            "test_accuracy": test_m.get("accuracy"),
            "test_loss": test_m.get("loss"),
            "test_f1": test_m.get("f1"),
            "test_roc_auc": test_m.get("roc_auc"),
            "test_pr_auc": test_m.get("pr_auc"),
            "ood_accuracy": ood_m.get("accuracy"),
            "ood_roc_auc": ood_m.get("roc_auc"),
            "gpu_t4_single_ms_median": eff_m.get("single_ms_median"),
            "gpu_t4_throughput_img_s": eff_m.get("throughput_img_per_sec"),
            "params_total": eff_m.get("params_total"),
            "params_trainable_ft": eff_m.get("params_trainable"),
        }
    return out


def export_all_models() -> dict[str, Any]:
    """Export production .keras models without optimizer state and verify output parity."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sample_batch = load_sample_batch(n_samples=5)
    benchmarks = load_benchmark_metrics()

    exported_entries: dict[str, Any] = {}
    for spec in EXPORT_SPECS:
        src_path = MODELS_DIR / spec["src_filename"]
        dst_path = EXPORT_DIR / spec["dst_filename"]
        if not src_path.exists():
            raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

        src_model = load_keras_model_compat(src_path, compile=False)
        src_preds = src_model.predict(sample_batch, verbose=0)

        # Save inference-only archive (model was loaded with compile=False, so optimizer is None)
        src_model.save(dst_path)

        reloaded_model = load_keras_model_compat(dst_path, compile=False)
        dst_preds = reloaded_model.predict(sample_batch, verbose=0)

        max_abs_diff = float(np.max(np.abs(src_preds - dst_preds)))
        if max_abs_diff >= 1e-6:
            raise RuntimeError(
                f"Numerical parity check failed for {spec['key']}: max_abs_diff={max_abs_diff:.3e}"
            )

        orig_bytes = src_path.stat().st_size
        exp_bytes = dst_path.stat().st_size
        orig_mb = round(orig_bytes / (1024 * 1024), 2)
        exp_mb = round(exp_bytes / (1024 * 1024), 2)
        reduction_pct = round((1.0 - (exp_bytes / orig_bytes)) * 100.0, 2)

        run_id = spec["run_id"]
        entry = {
            "key": spec["key"],
            "run_id": run_id,
            "display_name": spec["display_name"],
            "source_checkpoint": str(src_path.relative_to(PROJECT_ROOT)),
            "exported_model": str(dst_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_sha256(dst_path),
            "original_size_bytes": orig_bytes,
            "original_size_mb": orig_mb,
            "exported_size_bytes": exp_bytes,
            "exported_size_mb": exp_mb,
            "size_reduction_pct": reduction_pct,
            "input_shape": [None, IMG_SIZE[0], IMG_SIZE[1], 3],
            "input_dtype": "float32",
            "input_range": [0.0, 255.0],
            "preprocessing": spec["preprocessing"],
            "gradcam_layer": spec["gradcam_layer"],
            "prob_output_layer": "prob_ai",
            "label_map": {str(k): v for k, v in LABEL_MAP.items()},
            "recommended_threshold": spec["recommended_threshold"],
            "val_tuned_threshold": spec["val_tuned_threshold"],
            "parity_check_samples": int(len(sample_batch)),
            "parity_max_abs_diff": max_abs_diff,
            "parity_passed": True,
            "drive_file_id": None,
            "benchmarks": benchmarks.get(run_id, {}),
        }
        exported_entries[spec["key"]] = entry
        print(
            f"[OK] {run_id} ({spec['key']}): {orig_mb:.2f} MB -> {exp_mb:.2f} MB "
            f"(-{reduction_pct:.1f}%), max_abs_diff={max_abs_diff:.2e}, sha256={entry['sha256'][:12]}..."
        )

    manifest = {
        "generated_at_wib": datetime.now(WIB_TZ).isoformat(),
        "inference_contract": {
            "expected_input": "RGB float32 tensor in [0.0, 255.0] with shape (N, 224, 224, 3)",
            "center_crop_recommended": True,
            "center_crop_rationale": (
                "Mitigates Phase 1 aspect-ratio bias (79.01% AI Art vs 1.55% Real Art are 1:1 square)"
            ),
            "output": "Sigmoid scalar P(AI-Generated Art) in [0.0, 1.0]; class 0 = Real Art, class 1 = AI-Generated Art",
        },
        "models": exported_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    export_all_models()
