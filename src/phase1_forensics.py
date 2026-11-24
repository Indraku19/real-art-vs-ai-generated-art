"""Phase 1: Data Forensics, Perceptual-Hash Deduplication, and Stratified Splitting."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import imagehash
import matplotlib
import numpy as np
import pandas as pd
from datasets import ClassLabel, Image as HFImage, load_dataset
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    FIGURES_DIR,
    HF_DATASET_ID,
    LABEL_MAP,
    PROCESSED_DIR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    set_global_seed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
NEAR_DUP_MAX_DIST = 4


def _detect_format_from_bytes(raw_bytes: bytes | None, path: str | None) -> str:
    if raw_bytes:
        if raw_bytes.startswith(b"\xff\xd8\xff"):
            return "JPEG"
        if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "PNG"
        if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
            return "WEBP"
        if raw_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "GIF"
        if raw_bytes.startswith(b"BM"):
            return "BMP"
    if path:
        ext = Path(path).suffix.lstrip(".").upper()
        if ext:
            return "JPEG" if ext == "JPG" else ext
    return "UNKNOWN"


def _infer_label_remap(features: Any) -> tuple[str, str, dict[Any, int], dict[str, Any]]:
    """Identify image/label columns and build verified canonical mapping (0=Real, 1=AI)."""
    img_col = "image" if "image" in features else None
    lbl_col = "label" if "label" in features else None

    for col, feat in features.items():
        if img_col is None and isinstance(feat, HFImage):
            img_col = col
        if lbl_col is None and isinstance(feat, ClassLabel):
            lbl_col = col

    if img_col is None or lbl_col is None:
        raise ValueError(f"Unexpected dataset schema: {features}")

    lbl_feat = features[lbl_col]
    meta_info: dict[str, Any] = {
        "image_column": img_col,
        "label_column": lbl_col,
        "feature_repr": repr(lbl_feat),
    }

    remap: dict[Any, int] = {}
    if isinstance(lbl_feat, ClassLabel):
        names = list(lbl_feat.names)
        meta_info["hf_class_names"] = names
        for idx, name in enumerate(names):
            low = name.lower()
            if any(tok in low for tok in ("ai", "fake", "generated", "synth")):
                remap[idx] = 1
            elif any(tok in low for tok in ("real", "human", "original", "art")):
                remap[idx] = 0
            else:
                raise ValueError(f"Cannot infer canonical class for HF ClassLabel '{name}'")
        if set(remap.values()) != {0, 1}:
            raise ValueError(f"Incomplete label mapping from HF ClassLabel {names}: {remap}")
    else:
        raise ValueError(f"Expected ClassLabel for '{lbl_col}', got {lbl_feat}")

    return img_col, lbl_col, remap, meta_info


def _union_find_clusters(hashes_u64: np.ndarray, max_dist: int) -> tuple[list[list[int]], int, int]:
    """Cluster indices whose 64-bit pHash Hamming distance <= max_dist."""
    n = len(hashes_u64)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    exact_pairs = 0
    near_pairs = 0
    for i in range(n - 1):
        xor_arr = hashes_u64[i] ^ hashes_u64[i + 1 :]
        dists = POPCOUNT_LUT[xor_arr.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        matches = np.where(dists <= max_dist)[0]
        for offset in matches:
            d = int(dists[offset])
            if d == 0:
                exact_pairs += 1
            else:
                near_pairs += 1
            union(i, i + 1 + int(offset))

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    return list(clusters.values()), exact_pairs, near_pairs


def _plot_forensics(raw_df: pd.DataFrame, clean_df: pd.DataFrame, splits_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    colors = {"Real Art": "#2563eb", "AI-Generated Art": "#dc2626"}

    # Panel A: Class counts (Raw -> Clean -> Splits)
    ax = axes[0, 0]
    stages = ["Raw", "Clean (Dedup)", "Train (80%)", "Val (10%)", "Test (10%)"]
    real_counts = [
        int((raw_df["label"] == 0).sum()),
        int((clean_df["label"] == 0).sum()),
        int(((splits_df["split"] == "train") & (splits_df["label"] == 0)).sum()),
        int(((splits_df["split"] == "val") & (splits_df["label"] == 0)).sum()),
        int(((splits_df["split"] == "test") & (splits_df["label"] == 0)).sum()),
    ]
    ai_counts = [
        int((raw_df["label"] == 1).sum()),
        int((clean_df["label"] == 1).sum()),
        int(((splits_df["split"] == "train") & (splits_df["label"] == 1)).sum()),
        int(((splits_df["split"] == "val") & (splits_df["label"] == 1)).sum()),
        int(((splits_df["split"] == "test") & (splits_df["label"] == 1)).sum()),
    ]
    x = np.arange(len(stages))
    width = 0.36
    b1 = ax.bar(x - width / 2, real_counts, width, label="Real Art (0)", color=colors["Real Art"], alpha=0.88)
    b2 = ax.bar(x + width / 2, ai_counts, width, label="AI-Generated Art (1)", color=colors["AI-Generated Art"], alpha=0.88)
    ax.bar_label(b1, padding=2, fontsize=8)
    ax.bar_label(b2, padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=12)
    ax.set_ylabel("Image Count")
    ax.set_title("A. Class Balance: Raw vs Clean vs Stratified Splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Resolution scatter (Width vs Height)
    ax = axes[0, 1]
    for lbl, name in LABEL_MAP.items():
        sub = raw_df[raw_df["label"] == lbl]
        ax.scatter(
            sub["width"],
            sub["height"],
            s=12,
            alpha=0.45,
            label=f"{name} (med: {int(sub['width'].median())}x{int(sub['height'].median())})",
            color=colors[name],
        )
    max_dim = max(raw_df["width"].max(), raw_df["height"].max())
    ax.plot([0, max_dim], [0, max_dim], "--", color="#64748b", linewidth=1, label="1:1 Square")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("B. Resolution Distribution by Class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel C: Aspect ratio distribution
    ax = axes[1, 0]
    clipped_ar = raw_df["aspect_ratio"].clip(0.25, 3.0)
    bins = np.linspace(0.25, 3.0, 28)
    for lbl, name in LABEL_MAP.items():
        mask = raw_df["label"] == lbl
        ax.hist(
            clipped_ar[mask],
            bins=bins,
            alpha=0.55,
            label=f"{name} (mean AR={raw_df.loc[mask, 'aspect_ratio'].mean():.2f})",
            color=colors[name],
        )
    ax.axvline(1.0, color="#0f172a", linestyle="--", linewidth=1, label="1:1 Aspect Ratio")
    ax.set_xlabel("Aspect Ratio (Width / Height, clipped to [0.25, 3.0])")
    ax.set_ylabel("Image Count")
    ax.set_title("C. Aspect Ratio Bias Check")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel D: Original format & color mode breakdown
    ax = axes[1, 1]
    raw_df = raw_df.copy()
    raw_df["fmt_mode"] = raw_df["orig_format"] + " / " + raw_df["orig_mode"]
    top_combos = raw_df["fmt_mode"].value_counts().head(8).index.tolist()
    combo_df = (
        raw_df[raw_df["fmt_mode"].isin(top_combos)]
        .groupby(["fmt_mode", "label_name"])
        .size()
        .unstack(fill_value=0)
    )
    combo_df.plot(
        kind="barh",
        ax=ax,
        color=[colors.get(c, "#64748b") for c in combo_df.columns],
        alpha=0.88,
    )
    ax.set_xlabel("Image Count")
    ax.set_ylabel("Format / Color Mode")
    ax.set_title("D. Original Format & Color Mode Breakdown")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Phase 1 Data Forensics — DataScienceProject/Art_Images_Ai_And_Real_",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIGURES_DIR / "phase1_data_forensics.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_forensics() -> dict[str, Any]:
    set_global_seed(SEED)
    images_dir = PROCESSED_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 1] Loading Hugging Face dataset: {HF_DATASET_ID}")
    ds_dict = load_dataset(HF_DATASET_ID)
    print(f"[Phase 1] HF Splits found: {list(ds_dict.keys())}")

    first_split = next(iter(ds_dict.keys()))
    print(f"[Phase 1] Features schema ({first_split}): {ds_dict[first_split].features}")
    img_col, lbl_col, label_remap, schema_meta = _infer_label_remap(ds_dict[first_split].features)
    print(f"[Phase 1] Verified label remap (HF idx -> Canonical 0=Real,1=AI): {label_remap}")

    records: list[dict[str, Any]] = []
    corrupt_records: list[dict[str, Any]] = []
    sample_idx = 0

    for hf_split in ds_dict.keys():
        ds_raw_bytes = ds_dict[hf_split].cast_column(img_col, HFImage(decode=False))
        for i in tqdm(range(len(ds_raw_bytes)), desc=f"Auditing {hf_split}"):
            row = ds_raw_bytes[i]
            raw_lbl = row[lbl_col]
            canon_lbl = label_remap[raw_lbl]
            canon_name = LABEL_MAP[canon_lbl]

            img_entry = row[img_col]
            raw_bytes = img_entry.get("bytes")
            orig_path = img_entry.get("path")
            orig_format = _detect_format_from_bytes(raw_bytes, orig_path)
            byte_size = len(raw_bytes) if raw_bytes else (
                os.path.getsize(orig_path) if orig_path and os.path.exists(orig_path) else -1
            )

            sample_id = f"{hf_split}_{i:05d}_{sample_idx:05d}"
            sample_idx += 1

            try:
                if raw_bytes:
                    img = Image.open(BytesIO(raw_bytes))
                elif orig_path:
                    img = Image.open(orig_path)
                else:
                    raise ValueError("Neither image bytes nor path available")

                img.load()
                if orig_format == "UNKNOWN" and img.format:
                    orig_format = img.format.upper()
                orig_mode = img.mode
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                aspect_ratio = round(width / height, 4) if height else 0.0

                if img.mode != "RGB":
                    if img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    ):
                        rgba = img.convert("RGBA")
                        bg = Image.new("RGB", rgba.size, (255, 255, 255))
                        bg.paste(rgba, mask=rgba.split()[3])
                        img_rgb = bg
                    else:
                        img_rgb = img.convert("RGB")
                else:
                    img_rgb = img

                phash_hex = str(imagehash.phash(img_rgb))
                out_img_path = images_dir / f"{sample_id}.png"
                img_rgb.save(out_img_path, format="PNG")

                records.append(
                    {
                        "sample_id": sample_id,
                        "hf_split": hf_split,
                        "raw_label": int(raw_lbl),
                        "label": int(canon_lbl),
                        "label_name": canon_name,
                        "width": int(width),
                        "height": int(height),
                        "aspect_ratio": float(aspect_ratio),
                        "orig_mode": orig_mode,
                        "orig_format": orig_format,
                        "byte_size": int(byte_size),
                        "phash": phash_hex,
                        "filepath": str(out_img_path),
                    }
                )
            except Exception as exc:
                corrupt_records.append(
                    {
                        "sample_id": sample_id,
                        "hf_split": hf_split,
                        "raw_label": raw_lbl,
                        "error": str(exc),
                    }
                )

    raw_df = pd.DataFrame(records)
    metadata_csv = PROCESSED_DIR / "dataset_metadata.csv"
    raw_df.to_csv(metadata_csv, index=False)

    # Deduplication using 64-bit pHash
    hashes_u64 = np.array([int(h, 16) for h in raw_df["phash"]], dtype=np.uint64)
    clusters, exact_pairs, near_pairs = _union_find_clusters(hashes_u64, NEAR_DUP_MAX_DIST)

    keep_indices: list[int] = []
    dropped_intra_class = 0
    dropped_cross_class_clusters = 0
    dropped_cross_class_samples = 0

    for cluster in clusters:
        if len(cluster) == 1:
            keep_indices.append(cluster[0])
            continue
        labels_in_cluster = set(raw_df.iloc[cluster]["label"].tolist())
        if len(labels_in_cluster) > 1:
            dropped_cross_class_clusters += 1
            dropped_cross_class_samples += len(cluster)
        else:
            keep_indices.append(cluster[0])
            dropped_intra_class += len(cluster) - 1

    clean_df = raw_df.iloc[sorted(keep_indices)].reset_index(drop=True)

    # Stratified 80 / 10 / 10 split
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=clean_df["label"],
        random_state=SEED,
    )
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_cols = [
        "sample_id",
        "filepath",
        "label",
        "label_name",
        "split",
        "width",
        "height",
        "aspect_ratio",
        "phash",
    ]
    splits_df = splits_df[split_cols]
    splits_csv = PROCESSED_DIR / "splits.csv"
    splits_df.to_csv(splits_csv, index=False)

    # Anti-leakage verification across splits
    train_ph = set(train_df["phash"])
    val_ph = set(val_df["phash"])
    test_ph = set(test_df["phash"])
    cross_split_exact_overlap = (
        len(train_ph & val_ph) + len(train_ph & test_ph) + len(val_ph & test_ph)
    )

    fig_path = _plot_forensics(raw_df, clean_df, splits_df)

    # Compute class-wise forensic stats
    class_forensics: dict[str, Any] = {}
    for lbl, name in LABEL_MAP.items():
        sub = raw_df[raw_df["label"] == lbl]
        sq_pct = float(((sub["aspect_ratio"] >= 0.95) & (sub["aspect_ratio"] <= 1.05)).mean() * 100.0)
        class_forensics[name] = {
            "raw_count": int(len(sub)),
            "clean_count": int((clean_df["label"] == lbl).sum()),
            "width_min_med_max": [int(sub["width"].min()), float(sub["width"].median()), int(sub["width"].max())],
            "height_min_med_max": [int(sub["height"].min()), float(sub["height"].median()), int(sub["height"].max())],
            "aspect_ratio_mean_std": [round(float(sub["aspect_ratio"].mean()), 4), round(float(sub["aspect_ratio"].std()), 4)],
            "square_ratio_pct": round(sq_pct, 2),
            "formats": sub["orig_format"].value_counts().to_dict(),
            "modes": sub["orig_mode"].value_counts().to_dict(),
            "median_byte_size": float(sub["byte_size"].median()),
        }

    summary: dict[str, Any] = {
        "dataset_id": HF_DATASET_ID,
        "schema_meta": schema_meta,
        "label_remap": {str(k): v for k, v in label_remap.items()},
        "raw_total_readable": int(len(raw_df)),
        "corrupt_count": int(len(corrupt_records)),
        "corrupt_samples": corrupt_records[:10],
        "deduplication": {
            "near_dup_max_hamming": NEAR_DUP_MAX_DIST,
            "exact_duplicate_pairs": int(exact_pairs),
            "near_duplicate_pairs": int(near_pairs),
            "dropped_intra_class": int(dropped_intra_class),
            "dropped_cross_class_clusters": int(dropped_cross_class_clusters),
            "dropped_cross_class_samples": int(dropped_cross_class_samples),
            "clean_total": int(len(clean_df)),
        },
        "splits": {
            "train": {
                "total": int(len(train_df)),
                "real": int((train_df["label"] == 0).sum()),
                "ai": int((train_df["label"] == 1).sum()),
            },
            "val": {
                "total": int(len(val_df)),
                "real": int((val_df["label"] == 0).sum()),
                "ai": int((val_df["label"] == 1).sum()),
            },
            "test": {
                "total": int(len(test_df)),
                "real": int((test_df["label"] == 0).sum()),
                "ai": int((test_df["label"] == 1).sum()),
            },
            "cross_split_exact_phash_overlap": int(cross_split_exact_overlap),
        },
        "class_forensics": class_forensics,
        "artifacts": {
            "metadata_csv": str(metadata_csv),
            "splits_csv": str(splits_csv),
            "figure_path": str(fig_path),
        },
    }

    summary_path = PROCESSED_DIR / "forensics_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n=== PHASE 1 FORENSICS SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_forensics()
