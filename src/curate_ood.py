"""Phase 4: Curate, deduplicate (pHash), and package a 30-image OOD Stress Test benchmark."""

from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

from src.config import FIGURES_DIR, LABEL_MAP, OOD_DIR, PROCESSED_DIR

HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
MIN_SIDE_PX = 500
MAX_SAVE_DIM = 1024
MIN_HAMMING_DIST = 5  # strictly > 4

ART_PREFER_KEYWORDS = (
    "art",
    "painting",
    "illustration",
    "portrait",
    "landscape",
    "surreal",
    "watercolor",
    "oil",
    "cinematic",
    "canvas",
    "brush",
    "fantasy",
    "sketch",
    "digital art",
    "impressionist",
    "abstract",
    "renaissance",
    "drawing",
    "symbolism",
    "stylized",
)
AVOID_KEYWORDS = (
    "text",
    "logo",
    "ui",
    "screenshot",
    "diagram",
    "chart",
    "infographic",
    "typography",
    "banner",
    "poster",
    "webpage",
    "website",
    "app interface",
    "slide",
    "comic convention",
    "shrimp",
)


def _fetch_json(url: str, timeout: int = 20, retries: int = 3) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "real-vs-ai-art-ood-curator/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed fetching JSON from {url}: {last_err}") from last_err


def _fetch_hf_rows(
    dataset_id: str,
    total_rows: int,
    start_offset: int = 0,
    page_size: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    label_names: dict[str, list[str]] = {}
    offset = start_offset
    end_offset = start_offset + total_rows
    while offset < end_offset:
        length = min(page_size, end_offset - offset)
        params = urllib.parse.urlencode(
            {"dataset": dataset_id, "config": "default", "split": "train", "offset": offset, "length": length}
        )
        data = _fetch_json(f"{HF_ROWS_ENDPOINT}?{params}")
        if not label_names:
            for feat in data.get("features", []):
                ftype = feat.get("type")
                if isinstance(ftype, dict) and ftype.get("_type") == "ClassLabel":
                    label_names[feat["name"]] = ftype.get("names", [])
        batch = [entry["row"] for entry in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows, label_names


def _download_rgb_image(url: str, timeout: int = 20) -> Image.Image | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "real-vs-ai-art-ood-curator/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception:
        return None


def _text_score(text: str) -> tuple[bool, int]:
    low = text.strip().lower()
    if len(low) < 15 or any(bad in low for bad in AVOID_KEYWORDS):
        return False, -1
    score = sum(1 for kw in ART_PREFER_KEYWORDS if kw in low)
    return True, score


def _center_crop_square_224(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((224, 224), Image.Resampling.BILINEAR)


def curate_ood_dataset(skip_urls: set[str] | None = None) -> pd.DataFrame:
    """Curate 30 high-res OOD images (15 WikiArt + 15 Modern AI), verify pHash, save PNGs & zip."""
    skip_urls = skip_urls or set()
    splits_df = pd.read_csv(PROCESSED_DIR / "splits.csv")
    main_phashes = [imagehash.hex_to_hash(h) for h in splits_df["phash"].astype(str)]

    images_dir = OOD_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old_png in images_dir.glob("*.png"):
        old_png.unlink()

    selected_phashes: list[imagehash.ImageHash] = []
    records: list[dict[str, Any]] = []

    def try_accept(
        img: Image.Image,
        ood_id: str,
        filename: str,
        label: int,
        source_dataset: str,
        source_group: str,
        details: str,
    ) -> bool:
        w, h = img.size
        if min(w, h) < MIN_SIDE_PX:
            return False

        phash = imagehash.phash(img)
        min_main = min(int(phash - mh) for mh in main_phashes)
        if min_main < MIN_HAMMING_DIST:
            return False
        if selected_phashes:
            min_self = min(int(phash - sh) for sh in selected_phashes)
            if min_self < MIN_HAMMING_DIST:
                return False

        img_save = img.copy()
        img_save.thumbnail((MAX_SAVE_DIM, MAX_SAVE_DIM), Image.Resampling.LANCZOS)
        out_path = images_dir / filename
        img_save.save(out_path, format="PNG")

        selected_phashes.append(phash)
        records.append(
            {
                "ood_id": ood_id,
                "filename": filename,
                "filepath": f"data/ood/images/{filename}",
                "label": label,
                "label_name": LABEL_MAP[label],
                "source_dataset": source_dataset,
                "source_group": source_group,
                "orig_width": w,
                "orig_height": h,
                "aspect_ratio": round(w / h, 4),
                "phash": str(phash),
                "min_hamming_vs_main": min_main,
                "details": details[:140].replace("\n", " "),
            }
        )
        return True

    # 1. 15 High-Resolution Real Art from huggan/wikiart (diverse styles & genres)
    print("Fetching candidates from huggan/wikiart...")
    wikiart_rows, label_names = _fetch_hf_rows("huggan/wikiart", total_rows=80, start_offset=0, page_size=40)
    style_names = label_names.get("style", [])
    genre_names = label_names.get("genre", [])
    style_counts: dict[str, int] = {}
    genre_counts: dict[str, int] = {}
    real_count = 0

    for style_cap in (1, 2):
        if real_count >= 15:
            break
        for row in wikiart_rows:
            if real_count >= 15:
                break
            img_info = row.get("image") or {}
            src = img_info.get("src", "")
            if not src or src in skip_urls:
                continue
            st_idx = row.get("style")
            gn_idx = row.get("genre")
            style = style_names[st_idx] if isinstance(st_idx, int) and st_idx < len(style_names) else str(st_idx)
            genre = genre_names[gn_idx] if isinstance(gn_idx, int) and gn_idx < len(genre_names) else str(gn_idx)
            if style_counts.get(style, 0) >= style_cap or genre_counts.get(genre, 0) >= 3:
                continue
            img = _download_rgb_image(src)
            if img is None:
                continue
            idx = real_count + 1
            details = f"WikiArt {style.replace('_', ' ')} ({genre.replace('_', ' ')})"
            if try_accept(
                img=img,
                ood_id=f"OOD-R{idx:02d}",
                filename=f"ood_real_{idx:02d}.png",
                label=0,
                source_dataset="huggan/wikiart",
                source_group="WikiArt_Museum",
                details=details,
            ):
                skip_urls.add(src)
                style_counts[style] = style_counts.get(style, 0) + 1
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                real_count += 1
    if real_count < 15:
        raise RuntimeError(f"Expected 15 WikiArt images, got {real_count}")

    # Helper for AI datasets with prompt-level deduplication
    def curate_ai_group(
        dataset_id: str,
        source_group: str,
        id_prefix: str,
        file_prefix: str,
        fetch_offsets: list[tuple[int, int]],
        target_count: int,
        text_field: str | None,
        min_text_score: int = 0,
    ) -> None:
        print(f"Fetching candidates from {dataset_id} ({source_group})...")
        rows: list[dict[str, Any]] = []
        for off, length in fetch_offsets:
            try:
                batch, _ = _fetch_hf_rows(dataset_id, total_rows=length, start_offset=off, page_size=30)
                rows.extend(batch)
            except Exception as exc:
                print(f"  [warn] skipping {dataset_id} offset={off} len={length}: {exc}")

        scored: list[tuple[int, dict[str, Any]]] = []
        seen_prompts: set[str] = set()
        for row in rows:
            if text_field:
                txt = str(row.get(text_field) or "")
                ok, sc = _text_score(txt)
                if not ok or sc < min_text_score:
                    continue
                prompt_key = " ".join(txt.lower().split())[:70]
                if prompt_key in seen_prompts:
                    continue
                seen_prompts.add(prompt_key)
                scored.append((sc, row))
            else:
                scored.append((0, row))

        if text_field:
            scored.sort(key=lambda item: item[0], reverse=True)

        accepted = 0
        for _, row in scored:
            if accepted >= target_count:
                break
            img_info = row.get("image") or {}
            src = img_info.get("src", "")
            if not src or src in skip_urls:
                continue
            img = _download_rgb_image(src)
            if img is None:
                continue
            idx = accepted + 1
            details = str(row.get(text_field) or f"{source_group} sample #{idx}") if text_field else f"{source_group} sample #{idx}"
            if try_accept(
                img=img,
                ood_id=f"{id_prefix}{idx:02d}",
                filename=f"{file_prefix}_{idx:02d}.png",
                label=1,
                source_dataset=dataset_id,
                source_group=source_group,
                details=details,
            ):
                skip_urls.add(src)
                accepted += 1
        if accepted < target_count:
            raise RuntimeError(f"Expected {target_count} images from {dataset_id}, got {accepted}")

    # 2. 5 Midjourney v6 (distinct prompts across wide offsets)
    curate_ai_group(
        dataset_id="brivangl/midjourney-v6-llava",
        source_group="Midjourney_v6",
        id_prefix="OOD-MJ",
        file_prefix="ood_ai_mj",
        fetch_offsets=[(0, 40), (60, 40), (140, 40)],
        target_count=5,
        text_field="prompt",
        min_text_score=0,
    )

    # 3. 5 FLUX.1-Dev (diverse samples across offsets)
    curate_ai_group(
        dataset_id="LukasT9/Flux-1-Dev-Images-1k",
        source_group="FLUX_1_Dev",
        id_prefix="OOD-FX",
        file_prefix="ood_ai_flux",
        fetch_offsets=[(0, 8), (25, 8), (60, 8), (110, 8), (170, 8)],
        target_count=5,
        text_field=None,
    )

    # 4. 5 DALL-E 3 (distinct painterly/illustrative captions)
    curate_ai_group(
        dataset_id="OpenDatasets/dalle-3-dataset",
        source_group="DALLE_3",
        id_prefix="OOD-DL",
        file_prefix="ood_ai_dalle3",
        fetch_offsets=[(0, 30), (30, 30), (100, 20), (120, 20)],
        target_count=5,
        text_field="caption",
        min_text_score=1,
    )

    ood_df = pd.DataFrame(records)
    meta_path = OOD_DIR / "ood_metadata.csv"
    ood_df.to_csv(meta_path, index=False)

    # Generate 5x6 sanity contact sheet
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 6, figsize=(16, 14), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, (_, row) in zip(axes.flat, ood_df.iterrows()):
        with Image.open(images_dir / row["filename"]) as im:
            thumb = _center_crop_square_224(im)
        ax.imshow(thumb)
        color = "#15803d" if row["label"] == 0 else "#1d4ed8"
        ax.set_title(
            f"{row['ood_id']} | {row['label_name']}\n{row['source_group']} ({row['orig_width']}×{row['orig_height']})",
            fontsize=8.2,
            fontweight="bold",
            color=color,
            pad=4,
        )
        ax.axis("off")

    fig.suptitle(
        "Phase 4 OOD Stress Test — 30 Curated External Images (15 WikiArt Real + 15 Modern AI: Midjourney v6, FLUX.1-Dev, DALL-E 3)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    grid_path = FIGURES_DIR / "phase4_ood_sanity_grid.png"
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Package zip for Colab 1-click bootstrap
    zip_path = OOD_DIR / "ood_stress_test_30.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(meta_path, arcname="ood_metadata.csv")
        for fp in sorted(images_dir.glob("*.png")):
            zf.write(fp, arcname=f"images/{fp.name}")

    print(f"Saved {len(ood_df)} OOD images -> {meta_path}")
    print(f"Min pHash Hamming distance vs main dataset: {ood_df['min_hamming_vs_main'].min()}")
    print(f"Sanity grid: {grid_path}")
    print(f"Zip bundle: {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return ood_df


if __name__ == "__main__":
    curate_ood_dataset()
