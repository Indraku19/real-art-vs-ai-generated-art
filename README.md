# Klasifikasi Citra Real Art vs AI-Generated Art

Proyek riset dan eksperimen komparatif (*Deep Learning Side Project*) untuk membandingkan **CNN Baseline (From Scratch)** dan **Transfer Learning (MobileNetV2 & ResNet50)** pada dataset [`DataScienceProject/Art_Images_Ai_And_Real_`](https://huggingface.co/datasets/DataScienceProject/Art_Images_Ai_And_Real_).

## Tautan Dokumentasi & Artefak Cloud
- **Folder Google Drive:** [Real-vs-AI-Art-Classification](https://drive.google.com/drive/folders/1-H_FJGoVT5PmSaYm0xVSzQhkO9Bn9pWu)
- **Living Research Log (Google Docs):** [Research Log & Progress Tracker — Real vs AI Art Classification](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)
- **Laporan Progress Phase 3 (Google Docs):** [Report 04 — Phase 3: Transfer Learning & Ablation Study (MobileNetV2 & ResNet50)](https://docs.google.com/document/d/11YJE47lC4AV0nDrFhVB0ug2vMSyKHh1c2GjKs7E_9TM/edit) · [Salinan Lokal Markdown](reports/phase3_progress_report.md)
- **Laporan Progress Phase 4 (Google Docs):** [Report 05 — Phase 4: Deep Evaluation, Error Analysis, Grad-CAM & OOD Stress Test](https://docs.google.com/document/d/1x9Pa3yWlTYrP42CiYI_1Bq65I3_sp8uWmbFdYv8H5xk/edit) · [Salinan Lokal Markdown](reports/phase4_progress_report.md)
- **Dataset Cache Colab:** [GitHub Release v1.0-dataset (`processed_images_224.zip`)](https://github.com/Indraku19/real-art-vs-ai-generated-art/releases/tag/v1.0-dataset)

## Status Progres & Ringkasan Hasil Final (Test Set `N=282` & OOD `N=30`)
- **Tahap Selesai:** Step -1, Phase 0, Phase 1 (Data Forensics & pHash), Phase 2 (Controlled Baseline CNN), Phase 3 (Transfer Learning & Ablation), **Phase 4 (Deep Evaluation, Test Set Unlock, Error Analysis, Grad-CAM & OOD Stress Test — selesai di Colab T4 GPU)**.
- **Tahap Selanjutnya:** **Phase 5** (Packaging Model `.keras`, Demo Interaktif Gradio + Grad-CAM, & Final Technical Write-Up).
- **Model Juara:** **`Exp-3B` (ResNet50 Fine-Tuned)** meraih **`98.58%` Test Accuracy**, **`0.9985` ROC-AUC**, dan **`90.00%` OOD Accuracy** (`100%` pada Midjourney v6, FLUX.1-Dev, DALL-E 3), sementara **`Exp-2B` (MobileNetV2 Fine-Tuned)** unggul di efisiensi Pareto (**`24.65 MB`**, **`97.87%` Test Accuracy**, McNemar `p = 0.625`).

| Run ID | Model & Strategi | Size | Test Acc | Test Loss | Test F1 | Test ROC-AUC | GPU T4 Latency | OOD Acc (`N=30`) | OOD ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ref-0** | DataScienceProject/CNN *(HF)* | N/A | 77.00% | 0.4900 | 0.7700 | N/A | N/A | N/A | N/A |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) | 273.64 MB | 69.50% | 0.5893 | 0.6161 | 0.8085 | 8.72 ms | 73.33% | 0.7333 |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) | 4.92 MB | 80.14% | 0.4658 | 0.7705 | 0.9198 | 16.97 ms | 73.33% | 0.9156 |
| **Exp-2A** | MobileNetV2 — Frozen | 11.06 MB | 93.26% | 0.1607 | 0.9309 | 0.9845 | 162.24 ms | 83.33% | 0.9378 |
| **Exp-2B** | MobileNetV2 — Selective FT | **24.65 MB** | **97.87%** | 0.0855 | **0.9786** | **0.9948** | **160.99 ms** | 83.33% | 0.9600 |
| **Exp-3A** | ResNet50 — Frozen | 93.62 MB | 98.23% | 0.0588 | 0.9822 | 0.9981 | 232.86 ms | **90.00%** | **0.9956** |
| **Exp-3B** | ResNet50 — Selective FT | 224.75 MB | **98.58%** | **0.0414** | **0.9858** | **0.9985** | 232.69 ms | **90.00%** | **0.9956** |

## Struktur Direktori Lokal
```text
deep-learning/
├── data/
│   ├── raw/                              # Unduhan mentah dataset (diabaikan git)
│   ├── processed/                        # splits.csv, metadata audit, forensics_summary.json, images/ (diabaikan git)
│   └── ood/                              # ood_metadata.csv (30 citra OOD terverifikasi), images/ & .zip (diabaikan git)
├── experiments/
│   ├── experiment_log.csv                # Log global lengkap Ref-0 hingga Exp-3B (Test set unlocked)
│   ├── final_master_benchmark_table.csv  # Tabel master 31 kolom Phase 2-4 siap untuk Final Report
│   ├── phase2_phase3_ablation_table.csv  # Tabel ablasi validasi Phase 2-3
│   ├── phase{2,3,4}_summary.json         # Ringkasan JSON terstruktur tiap fase
│   ├── phase4_{test,ood}_predictions.csv # Prediksi per sampel pada Test Set & OOD Stress Test
│   └── exp{1a,1b,2a,2b,3a,3b}_history.json
├── models/                               # 6 checkpoint terlatih .keras (diabaikan git; tersinkron di Drive)
├── notebooks/                            # 01_phase1_data_forensics .. 04_phase4_deep_evaluation
├── reports/
│   ├── phase3_progress_report.md         # Laporan detail Phase 3 (cerminan Report 04)
│   ├── phase4_progress_report.md         # Laporan detail Phase 4 (cerminan Report 05)
│   └── figures/                          # Grafik forensik, learning curves, ROC/PR, CM, Error, Grad-CAM & OOD
├── src/
│   ├── __init__.py
│   ├── config.py                         # Konfigurasi path, seed locking, konvensi label
│   ├── data_pipeline.py                  # Pipeline tf.data, preprocessing layer & compat loader
│   ├── phase1_forensics.py               # Audit integritas, pHash deduplication, stratified split
│   └── curate_ood.py                     # Kurasi & pHash deduplication 30 citra OOD eksternal
├── tracker/                              # Progress tracker lokal (progress.json & UI D3.js)
├── requirements.txt
└── README.md
```

## Setup Environment (Python 3.12 via `uv`)
TensorFlow membutuhkan Python ≤ 3.12 pada macOS `arm64`, sehingga environment proyek dikunci ke **Python 3.12**:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Konvensi Label
- `0`: `Real Art`
- `1`: `AI-Generated Art`
