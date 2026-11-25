# Klasifikasi Citra Real Art vs AI-Generated Art

Proyek riset dan implementasi komparatif (*Deep Learning Side Project*) yang membandingkan **CNN Baseline (From Scratch)** dan **Transfer Learning (`MobileNetV2` & `ResNet50`)** untuk membedakan lukisan asli (*Real Art*) vs karya generatif AI (*AI-Generated Art*) pada dataset [`DataScienceProject/Art_Images_Ai_And_Real_`](https://huggingface.co/datasets/DataScienceProject/Art_Images_Ai_And_Real_).

**Status Proyek:** **Selesai (`29/29` task wajib selesai; `1` task opsional `p3_modern` di-skip)**

---

## Tautan Dokumentasi, Colab & Artefak Cloud

- **Folder Google Drive Proyek:** [Real-vs-AI-Art-Classification](https://drive.google.com/drive/folders/1-H_FJGoVT5PmSaYm0xVSzQhkO9Bn9pWu)
- **Living Research Log (Google Docs):** [Research Log & Progress Tracker — Real vs AI Art Classification](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)
- **Final Technical Report (Report 06, Google Docs):** [Report 06 — Final Technical Report](https://docs.google.com/document/d/1kYit1b04AvphjMMXWh_rxTZf6lmox0fMzWnFRBPa1Qc/edit) · [Salinan Lokal Markdown](reports/final_technical_report.md)
- **Laporan Progress Phase 3 (Report 04, Google Docs):** [Report 04 — Transfer Learning & Ablation](https://docs.google.com/document/d/11YJE47lC4AV0nDrFhVB0ug2vMSyKHh1c2GjKs7E_9TM/edit) · [Markdown](reports/phase3_progress_report.md)
- **Laporan Progress Phase 4 (Report 05, Google Docs):** [Report 05 — Deep Evaluation, Grad-CAM & OOD](https://docs.google.com/document/d/1x9Pa3yWlTYrP42CiYI_1Bq65I3_sp8uWmbFdYv8H5xk/edit) · [Markdown](reports/phase4_progress_report.md)
- **Colab Demo Launcher (Phase 5):** [05_phase5_demo_gradio.ipynb](https://colab.research.google.com/drive/1ac06t0JHi2grKGyPYAkuORKXQk8V5ER7) · [Salinan Repo](notebooks/05_phase5_demo_gradio.ipynb)
- **Dataset Cache Colab:** [GitHub Release v1.0-dataset (`processed_images_224.zip`)](https://github.com/Indraku19/real-art-vs-ai-generated-art/releases/tag/v1.0-dataset)

---

## Ringkasan Hasil Benchmark (Test Set `N = 282` & OOD Stress Test `N = 30`)

Seluruh model menggunakan pipeline dan split identik (`2.256` Train / `282` Val / `282` Test setelah deduplikasi pHash) serta *center-crop* `1:1` -> `224 x 224` dengan preprocessing khusus backbone tertanam di dalam grafik model (`input RGB [0..255]`).

| Run ID | Model & Strategi | Size Asli -> Export | Test Acc | Test Loss | Test F1 | Test ROC-AUC | GPU T4 Latency | OOD Acc (`N=30`) | OOD ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref-0** | DataScienceProject/CNN *(HF)* | N/A | 77.00% | 0.4900 | 0.7700 | N/A | N/A | N/A | N/A |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) | 273.64 MB | 69.50% | 0.5893 | 0.6161 | 0.8085 | 8.72 ms | 73.33% | 0.7333 |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) | 4.92 MB | 80.14% | 0.4658 | 0.7705 | 0.9198 | 16.97 ms | 73.33% | 0.9156 |
| **Exp-2A** | MobileNetV2 — Frozen | 11.06 MB | 93.26% | 0.1607 | 0.9309 | 0.9845 | 162.24 ms | 83.33% | 0.9378 |
| **Exp-2B** | MobileNetV2 — Selective FT | 24.65 MB -> **9.80 MB** | **97.87%** | 0.0855 | **0.9786** | **0.9948** | **160.99 ms** | 83.33% | 0.9600 |
| **Exp-3A** | ResNet50 — Frozen | 93.62 MB | 98.23% | 0.0588 | 0.9822 | 0.9981 | 232.86 ms | **90.00%** | **0.9956** |
| **Exp-3B** | ResNet50 — Selective FT | 224.75 MB -> **91.61 MB** | **98.58%** | **0.0414** | **0.9858** | **0.9985** | 232.69 ms | **90.00%** | **0.9956** |

### Kesimpulan Utama RQ1–RQ4
- **RQ1 (Scratch vs Transfer Learning):** Transfer Learning mengungguli *fair baseline* `Exp-1B` (`80.14%`) hingga **`+18.44 pp`** (`Exp-3B`: `98.58%`, `ROC-AUC = 0.9985`, hanya `4` error dari `282` Test images) sekaligus terbukti invarian terhadap perbedaan interpolasi resize (`PIL-LANCZOS` vs *bilinear*).
- **RQ2 (Frozen vs Selective Fine-Tuning):** Membuka `30%` layer akhir sambil mengunci `100%` layer `BatchNormalization` menstabilkan training; MobileNetV2 meningkat signifikan (`93.26% -> 97.87%` Test Acc), sedangkan ResNet50 menyempurnakan kalibrasi (`Test Loss 0.0588 -> 0.0414`). Uji **McNemar** `Exp-2B` vs `Exp-3B` (`p = 0.625`) menunjukkan `Exp-2B` (`9.80 MB` export) setara secara statistik *in-distribution*.
- **RQ3 (Shortcut vs Real Features):** Audit Phase 1 menemukan bias ekstrem citra persegi `1:1` (`79.01%` AI vs `1.55%` Real) yang dimitigasi dengan `center_crop=True`. **Grad-CAM pre-sigmoid logits** membuktikan `Exp-1B` masih bergantung pada *frame-border shortcut*, sedangkan `Exp-2B` dan `Exp-3B` memfokuskan aktivasi pada tekstur sapuan kuas dan pencahayaan interior karya.
- **RQ4 (OOD Generalization, `N = 30`):** Pada citra eksternal beresolusi tinggi (`pHash >= 14`), **`Exp-3B`** dan `Exp-3A` mempertahankan **`90.00%` OOD Accuracy** (`100%` Recall pada Midjourney v6, FLUX.1-Dev, DALL-E 3; `80%` pada WikiArt museum).

---

## Quickstart Lokal: Ekspor Model & Demo Gradio + Grad-CAM

### 1. Setup Environment (`Python 3.12` via `uv`)
TensorFlow membutuhkan Python <= 3.12 pada macOS `arm64`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Ekspor Ulang Model Self-Contained (Tanpa Optimizer State)
Jika seluruh checkpoint Phase 3 berada di `models/` (atau diunduh dari Drive `04_Models/`), jalankan:

```bash
python -m src.export_models
```

Skrip ini mengekspor `art_classifier_resnet50_ft.keras` (`91.61 MB`, `-59.24%`) dan `art_classifier_mobilenetv2_ft.keras` (`9.80 MB`, `-60.22%`) ke `models/export/`, memverifikasi paritas numerik (`max_abs_diff = 0.0`), dan mencatat hash SHA-256 ke [`experiments/exported_models_manifest.json`](experiments/exported_models_manifest.json).

### 3. Jalankan Aplikasi Demo Interaktif (Gradio + Grad-CAM)
- **Di Lokal:**
  ```bash
  python app.py --port 7860
  ```
  Lalu buka `http://127.0.0.1:7860` di browser untuk mengunggah citra seni, membandingkan `Exp-3B` vs `Exp-2B`, mengatur *threshold* / `Center-Crop 1:1`, serta melihat heatmap **Pre-Sigmoid Logit Grad-CAM** berdampingan.
- **Di Google Colab:** Buka [05_phase5_demo_gradio.ipynb](https://colab.research.google.com/drive/1ac06t0JHi2grKGyPYAkuORKXQk8V5ER7) yang terhubung ke model di Google Drive dan meluncurkan `python app.py --share`.

---

## Limitasi & Rencana Pengembangan (*Future Work*)

- **Domain Shift Lukisan Klasik:** Seluruh `3` error `Exp-3B` di OOD merupakan *False Positive* pada lukisan museum gelap/berkontras tinggi (`80.0%` WikiArt).
- **Kehilangan Sinyal Resolusi Tinggi:** Resizing ke `224 x 224` dapat mengaburkan tekstur halus dan artefak kompresi pada karya besar.
- **Hybrid Frequency-Domain (`FFT / DCT`) & `ELA`:** Menambahkan fitur spektrum **2D FFT/DCT** atau peta residu **Error Level Analysis (ELA)** serta evaluasi *multi-patch* resolusi penuh dapat memperkuat ketahanan terhadap generator modern dan seni digital manusia.

---

## Struktur Direktori Proyek

```text
deep-learning/
├── app.py                                    # Aplikasi UI interaktif Gradio + Grad-CAM
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                                  # Unduhan mentah dataset (diabaikan git)
│   ├── processed/                            # splits.csv, metadata audit, forensics_summary.json, images/ (diabaikan git)
│   └── ood/                                  # ood_metadata.csv (30 citra OOD), images/ & .zip (diabaikan git)
├── experiments/
│   ├── experiment_log.csv                    # Log global lengkap Ref-0 hingga Exp-3B
│   ├── final_master_benchmark_table.csv      # Tabel master 31 kolom Phase 2-4
│   ├── exported_models_manifest.json         # Manifest ekspor inferensi Phase 5 + SHA-256 + Drive IDs
│   ├── phase2_phase3_ablation_table.csv
│   ├── phase{2,3,4}_summary.json
│   ├── phase4_{test,ood}_predictions.csv
│   └── exp{1a,1b,2a,2b,3a,3b}_history.json
├── models/                                   # Checkpoints & models/export/*.keras (diabaikan git; tersinkron di Drive)
├── notebooks/
│   ├── 01_phase1_data_forensics.ipynb
│   ├── 02_phase2_baseline_cnn.ipynb
│   ├── 03_phase3_transfer_learning.ipynb
│   ├── 04_phase4_deep_evaluation.ipynb
│   └── 05_phase5_demo_gradio.ipynb           # Colab launcher untuk app.py --share
├── reports/
│   ├── final_technical_report.md             # Laporan teknis final (cerminan Report 06)
│   ├── phase3_progress_report.md             # Laporan Phase 3 (cerminan Report 04)
│   ├── phase4_progress_report.md             # Laporan Phase 4 (cerminan Report 05)
│   └── figures/                              # 15 grafik forensik, learning curves, ROC/PR, CM, Error, Grad-CAM & OOD
├── src/
│   ├── __init__.py
│   ├── config.py                             # Konfigurasi path, seed locking (42), konvensi label
│   ├── data_pipeline.py                      # Pipeline tf.data, preprocessing layer & compat loader
│   ├── phase1_forensics.py                   # Audit integritas, pHash deduplication, stratified split
│   ├── curate_ood.py                         # Kurasi & pHash deduplication 30 citra OOD eksternal
│   ├── export_models.py                      # Ekspor model inference-only (.keras) + verifikasi paritas
│   └── inference.py                          # Engine inferensi & Pre-Sigmoid Logit Grad-CAM
└── tracker/                                  # Progress tracker lokal (progress.json & UI D3.js)
```

## Konvensi Label
- `0`: `Real Art`
- `1`: `AI-Generated Art`
