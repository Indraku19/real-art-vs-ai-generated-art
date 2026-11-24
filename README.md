# Klasifikasi Citra Real Art vs AI-Generated Art

Proyek riset dan eksperimen komparatif (*Deep Learning Side Project*) untuk membandingkan **CNN Baseline (From Scratch)** dan **Transfer Learning (MobileNetV2 & ResNet50)** pada dataset [`DataScienceProject/Art_Images_Ai_And_Real_`](https://huggingface.co/datasets/DataScienceProject/Art_Images_Ai_And_Real_).

## Tautan Dokumentasi & Artefak Cloud
- **Folder Google Drive:** [Real-vs-AI-Art-Classification](https://drive.google.com/drive/folders/1-H_FJGoVT5PmSaYm0xVSzQhkO9Bn9pWu)
- **Living Research Log (Google Docs):** [Research Log & Progress Tracker — Real vs AI Art Classification](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)

## Struktur Direktori Lokal
```text
deep-learning/
├── data/
│   ├── raw/            # Unduhan mentah dataset (diabaikan git)
│   ├── processed/      # splits.csv, metadata audit, citra terproses
│   └── ood/            # Dataset Out-of-Distribution stress test
├── experiments/
│   └── experiment_log.csv
├── models/             # Checkpoint model terlatih (.keras)
├── notebooks/          # Notebook eksplorasi & eksperimen
├── reports/
│   └── figures/        # Grafik forensik, learning curves, Grad-CAM
├── src/
│   ├── __init__.py
│   └── config.py       # Konfigurasi path, seed locking, konvensi label
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
