# Klasifikasi Citra Real Art vs AI-Generated Art

Proyek riset dan eksperimen komparatif (*Deep Learning Side Project*) untuk membandingkan **CNN Baseline (From Scratch)** dan **Transfer Learning (MobileNetV2 & ResNet50)** pada dataset [`DataScienceProject/Art_Images_Ai_And_Real_`](https://huggingface.co/datasets/DataScienceProject/Art_Images_Ai_And_Real_).

## Tautan Dokumentasi & Artefak Cloud
- **Folder Google Drive:** [Real-vs-AI-Art-Classification](https://drive.google.com/drive/folders/1-H_FJGoVT5PmSaYm0xVSzQhkO9Bn9pWu)
- **Living Research Log (Google Docs):** [Research Log & Progress Tracker — Real vs AI Art Classification](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)
- **Laporan Progress Phase 3 (Google Docs):** [Report 04 — Phase 3: Transfer Learning & Ablation Study (MobileNetV2 & ResNet50)](https://docs.google.com/document/d/11YJE47lC4AV0nDrFhVB0ug2vMSyKHh1c2GjKs7E_9TM/edit) · [Salinan Lokal Markdown](reports/phase3_progress_report.md)
- **Dataset Cache Colab:** [GitHub Release v1.0-dataset (`processed_images_224.zip`)](https://github.com/Indraku19/real-art-vs-ai-generated-art/releases/tag/v1.0-dataset)

## Status Progres & Ringkasan Hasil Validasi (Phase 2 & Phase 3)
- **Tahap Selesai:** Step -1, Phase 0, Phase 1 (Data Forensics & pHash Deduplication), Phase 2 (Controlled Baseline CNN), **Phase 3 (Transfer Learning & Ablation — selesai di Colab T4 GPU)**.
- **Tahap Selanjutnya:** **Phase 4** (Threshold Tuning, Evaluasi Final Test Set, Error Analysis, Grad-CAM & OOD Stress Test).
- **Catatan Disiplin Evaluasi:** **Test Set (`282` citra) tetap `LOCKED` hingga Phase 4**; seluruh metrik berikut diukur pada **Validation Set (`282` citra)** kecuali `Ref-0` (publikasi Hugging Face).

| Run ID | Model & Strategi | Params (Trainable) | Size | Best Ep | Val Loss | Val Acc | Val F1 | Val ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ref-0** | DataScienceProject/CNN (Scratch HF) | N/A | N/A | N/A | 0.4900 *(Test)* | 77.00% *(Test)* | 0.7700 *(Test)* | N/A |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) | 23,907,521 | 273.64 MB | 2 / 8 | 0.4869 | 79.43% | 0.7820 | 0.8554 |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) | 422,881 | 4.92 MB | 9 / 15 | 0.3547 | 86.17% | 0.8582 | 0.9306 |
| **Exp-2A** | MobileNetV2 — Frozen | 2,422,081 (164,097) | 11.06 MB | 12 / 17 | 0.1277 | 94.33% | 0.9441 | 0.9909 |
| **Exp-2B** | MobileNetV2 — Selective FT (top 30%, BN frozen) | 2,422,081 (1,943,233) | 24.65 MB | 8 / 13 | 0.0643 | **97.52%** | **0.9754** | **0.9975** |
| **Exp-3A** | ResNet50 — Frozen | 23,850,113 (262,401) | 93.62 MB | 6 / 11 | 0.0612 | 97.87% | 0.9790 | 0.9983 |
| **Exp-3B** | ResNet50 — Selective FT (top 30%, BN frozen) | 23,850,113 (17,447,169) | 224.75 MB | 2 / 7 | **0.0519** | **98.23%** | **0.9825** | **0.9989** |

## Struktur Direktori Lokal
```text
deep-learning/
├── data/
│   ├── raw/                             # Unduhan mentah dataset (diabaikan git)
│   ├── processed/                       # splits.csv, metadata audit, forensics_summary.json, images/ (diabaikan git)
│   └── ood/                             # Dataset Out-of-Distribution stress test (Phase 4)
├── experiments/
│   ├── experiment_log.csv               # Log global Ref-0 hingga Exp-3B
│   ├── phase2_phase3_ablation_table.csv # Tabel ablasi 22 kolom siap untuk Final Report
│   ├── phase2_summary.json & phase3_summary.json
│   └── exp{1a,1b,2a,2b,3a,3b}_history.json
├── models/                              # Checkpoint terlatih .keras (diabaikan git; tersinkron di Drive)
├── notebooks/                           # 01_phase1_data_forensics, 02_phase2_baseline_cnn, 03_phase3_transfer_learning
├── reports/
│   ├── phase3_progress_report.md        # Laporan detail Phase 3 (cerminan Report 04)
│   └── figures/                         # Grafik forensik, learning curves, ringkasan ablasi & Pareto
├── src/
│   ├── __init__.py
│   ├── config.py                        # Konfigurasi path, seed locking, konvensi label
│   ├── data_pipeline.py                 # Pipeline tf.data + preprocessing layer per backbone
│   └── phase1_forensics.py              # Audit integritas, pHash deduplication, stratified split
├── tracker/                             # Progress tracker lokal (progress.json & UI D3.js)
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
