"""Interactive Gradio demo for Real Art vs AI-Generated Art classification + Grad-CAM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from src.config import PROCESSED_DIR
from src.inference import ArtClassifierEngine, MODEL_REGISTRY

ENGINE = ArtClassifierEngine()

CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.4rem;
}
.diag-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 12px 14px;
    background: #fafafa;
}
"""


def _collect_local_examples() -> list[list[Any]]:
    """Collect up to 4 test-split sample rows if local processed images exist."""
    splits_path = PROCESSED_DIR / "splits.csv"
    images_dir = PROCESSED_DIR / "images"
    if not splits_path.exists() or not images_dir.exists():
        return []

    import pandas as pd

    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"].copy()
    examples: list[list[Any]] = []
    for lbl in (0, 1):
        sub = test_df[test_df["label"] == lbl].head(2)
        for _, row in sub.iterrows():
            candidate = Path(str(row["filepath"]))
            if not candidate.exists():
                candidate = images_dir / candidate.name
            if candidate.exists():
                examples.append([str(candidate), "resnet50_ft", 0.50, True, 0.42])
    return examples


def run_inference(
    img: Image.Image | None,
    model_key: str = "resnet50_ft",
    threshold: float = 0.50,
    center_crop: bool = True,
    alpha: float = 0.42,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float], str]:
    """Run image classification and Grad-CAM explanation for Gradio UI."""
    if img is None:
        return (
            None,
            None,
            {},
            "Unggah gambar karya seni terlebih dahulu untuk memulai analisis.",
        )

    result = ENGINE.predict_with_gradcam(
        img=img,
        model_key=model_key,
        threshold=float(threshold),
        center_crop=bool(center_crop),
        alpha=float(alpha),
    )
    info = result["model_info"]
    prob_ai = result["prob_ai"]
    prob_real = result["prob_real"]
    badge = "AI-GENERATED ART" if result["pred_class"] == 1 else "REAL ART"

    diag_md = (
        f"### Prediksi: **{badge}** (`{result['predicted_label']}`)\n"
        f"- **Probabilitas AI `P(AI)`:** `{prob_ai:.4f}` (`{prob_ai * 100:.2f}%`) | "
        f"**Real `P(Real)`:** `{prob_real:.4f}` (`{prob_real * 100:.2f}%`)\n"
        f"- **Decision Threshold:** `{threshold:.2f}` | "
        f"**Preprocessing:** `{'Center-Crop 1:1 -> 224x224' if center_crop else 'Direct Resize -> 224x224'}`\n"
        f"- **Model Aktif:** `{info['run_id']}` — {info['label']} (`{info['size_note']}`)\n"
        f"- **Benchmark Terverifikasi:** Test Acc `{info['test_acc']}`, ROC-AUC `{info['test_roc_auc']}`, "
        f"OOD Acc `{info['ood_acc']}`\n"
        f"- **Waktu Inferensi + Grad-CAM:** `{result['latency_ms']:.1f} ms` (`gradcam_layer={info['gradcam_layer']}`)"
    )

    return (
        result["cropped_rgb"],
        result["gradcam_overlay"],
        result["probs"],
        diag_md,
    )


def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks application."""
    model_choices = [
        (spec["label"], key) for key, spec in MODEL_REGISTRY.items()
    ]

    with gr.Blocks(title="Real Art vs AI-Generated Art Classifier + Grad-CAM") as demo:
        gr.Markdown(
            "# Real Art vs AI-Generated Art — CNN & Transfer Learning Demo\n"
            "Klasifikasi citra lukisan asli (*Real Art*) vs karya generatif AI (*AI-Generated Art*) "
            "dilengkapi **Pre-Sigmoid Logit Grad-CAM** untuk memverifikasi fokus fitur interior vs *shortcut* bingkai.",
            elem_classes=["main-title"],
        )

        with gr.Row():
            with gr.Column(scale=5):
                input_img = gr.Image(
                    type="pil",
                    label="1. Upload Citra Seni (PNG / JPG / WebP)",
                    height=340,
                )
                model_selector = gr.Radio(
                    choices=model_choices,
                    value="resnet50_ft",
                    label="2. Pilih Model Produksi",
                    info=(
                        "Exp-3B (ResNet50 FT) = Akurasi & ketahanan OOD tertinggi; "
                        "Exp-2B (MobileNetV2 FT) = 9.3x lebih ringan (9.80 MB) & lebih cepat."
                    ),
                )
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.50,
                        step=0.01,
                        label="Decision Threshold P(AI)",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.20,
                        maximum=0.70,
                        value=0.42,
                        step=0.02,
                        label="Opasitas Heatmap Grad-CAM",
                    )
                center_crop_cb = gr.Checkbox(
                    value=True,
                    label="Aktifkan Center-Crop 1:1 (Disarankan)",
                    info=(
                        "Mitigasi bias aspek rasio Phase 1 (79.01% dataset AI berbentuk 1:1 vs 1.55% Real Art)."
                    ),
                )
                analyze_btn = gr.Button("Analisis & Render Grad-CAM", variant="primary")

            with gr.Column(scale=7):
                with gr.Row():
                    out_cropped = gr.Image(
                        type="numpy",
                        label="Input Terproses (224x224 RGB)",
                        height=300,
                    )
                    out_gradcam = gr.Image(
                        type="numpy",
                        label="Pre-Sigmoid Logit Grad-CAM Overlay",
                        height=300,
                    )
                out_probs = gr.Label(
                    num_top_classes=2,
                    label="Skor Keyakinan Model",
                )
                out_diag = gr.Markdown(
                    value="Unggah gambar lalu klik **Analisis & Render Grad-CAM**.",
                    elem_classes=["diag-card"],
                )

        examples = _collect_local_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[
                    input_img,
                    model_selector,
                    threshold_slider,
                    center_crop_cb,
                    alpha_slider,
                ],
                label="Contoh Sampel Test Set (2 Real Art + 2 AI-Generated Art)",
            )

        analyze_btn.click(
            fn=run_inference,
            inputs=[
                input_img,
                model_selector,
                threshold_slider,
                center_crop_cb,
                alpha_slider,
            ],
            outputs=[out_cropped, out_gradcam, out_probs, out_diag],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo for Real vs AI Art classification."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (recommended for Google Colab).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local server port (default: 7860).",
    )
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0" if args.share else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
