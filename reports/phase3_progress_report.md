# Report 04 — Phase 3: Transfer Learning & Ablation Study (MobileNetV2 & ResNet50)

**Proyek:** Klasifikasi Citra Real Art vs AI-Generated Art  
**Dokumen Induk:** [Research Log & Progress Tracker](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)  
**Notebook Colab:** [03_phase3_transfer_learning.ipynb](https://colab.research.google.com/drive/16ALd9AbYMTcDHBR26RB2OAZWWH-YdmEZ)  
**Dataset Cache Release:** [GitHub Release v1.0-dataset](https://github.com/Indraku19/real-art-vs-ai-generated-art/releases/tag/v1.0-dataset)  
**Pull Requests:** [PR #4](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/4) · [PR #6](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/6) · [PR #7](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/7) · [PR #8](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/8)

---

## 1. Protokol Eksperimen Terkontrol (Phase 3)

Seluruh eksperimen Transfer Learning dijalankan pada Google Colab dengan akselerasi GPU **NVIDIA Tesla T4** menggunakan pembagian data `splits.csv` dan pipeline `tf.data` yang identik persis dengan Phase 2 sehingga perbandingan antar-fase bersifat adil dan bebas kebocoran data (*zero data leakage*):

- **Dataset & Split Identik:** `2.256` citra Train (augmentasi konservatif `RandomFlip`, `RandomRotation(0.05)`, `RandomZoom(0.1)` + `center_crop=True`, `224 × 224`) dan `282` citra Validation (`140` Real Art, `142` AI-Generated Art). **Test Set (`282` citra: `141` Real / `141` AI) tetap terkunci (`LOCKED`) hingga Phase 4.**
- **Distribusi Dataset Cepat di Colab:** Arsip bersih `processed_images_224.zip` (`235,08 MiB`, berisi `2.820` PNG RGB `224 × 224` hasil *center-crop*) diambil dari GitHub Release `v1.0-dataset` tanpa perlu mengunduh ulang 2.839 berkas satu per satu dari Hugging Face.
- **Lingkungan Komputasi:** Google Colab **NVIDIA Tesla T4 GPU**, TensorFlow `2.21.0`, Keras `3.13.2`, `BATCH_SIZE = 32` (`71` Train batches, `9` Val batches), `SEED = 42`.
- **Classification Head Identik & In-Model Preprocessing:** Setiap backbone ImageNet (`include_top=False`, `weights="imagenet"`) menerima input mentah `[0..255]` RGB dan menerapkan layer preprocessing khusus di dalam grafik model (`Rescaling(1/127.5, -1)` untuk **MobileNetV2**, `ResNet50PreprocessLayer` untuk **ResNet50**), lalu diteruskan ke *head* seragam: `GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(1, sigmoid)`.
- **Dua Tahap Ablasi (Stage 1 vs Stage 2):**
  - **Stage 1 — Frozen Feature Extractor (`Exp-2A` & `Exp-3A`):** Seluruh bobot backbone dibekukan (`trainable=False`), optimizer `Adam(lr=3e-4)`, `BinaryCrossentropy`, maksimal `20` epoch.
  - **Stage 2 — Selective Fine-Tuning (`Exp-2B` & `Exp-3B`):** Memuat checkpoint terbaik Stage 1, membuka `30%` layer paling atas pada backbone (`107..153` pada MobileNetV2; `122..174` pada ResNet50) sembari **menjaga 100% layer `BatchNormalization` tetap terkunci (`trainable=False`)** — `52` layer BN pada MobileNetV2 dan `53` layer BN pada ResNet50 — lalu melatih ulang menggunakan `AdamW(lr=1e-5, weight_decay=1e-4)`, maksimal `20` epoch.
- **Callbacks Standar:** `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)`, dan `ModelCheckpoint(save_best_only=True)`.
- **Perbaikan Kompatibilitas Keras 3 (`PR #7`):** Helper `count_model_params()` diperbarui dari `keras.backend.count_params()` (yang dihapus di Keras 3) menjadi `int(np.prod(w.shape))`, sehingga notebook berjalan mulus dari Cell 1 hingga Cell 9 (`exec 7..14`) tanpa error.

---

## 2. Tabel Ablasi Gabungan Phase 2 & Phase 3 (Validation Set, N = 282)

| **Run ID** | **Arsitektur & Strategi** | **Params (Trainable)** | **Ukuran (.keras)** | **Best / Total Ep** | **Val Loss** | **Val Acc (Error / 282)** | **Val Prec** | **Val Rec** | **Val F1** | **Val ROC-AUC** | **Overfit Gap** | **Waktu/Ep (Mean / Steady Med)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ref-0** | DataScienceProject/CNN (Scratch HF) | N/A | N/A | N/A | 0.4900 (Test) | 77.00% (Test) | 0.7700 (Test) | 0.7700 (Test) | 0.7700 (Test) | N/A | N/A | ~25–35s (hardware HF) |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) — Scratch | 23,907,521 | 273.64 MB | 2 / 8 | 0.4869 | 79.43% (58 err) | 0.8387 | 0.7324 | 0.7820 | 0.8554 | -8.64% | 20.57s / 18.93s |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) — Scratch | 422,881 | 4.92 MB | 9 / 15 | 0.3547 | 86.17% (39 err) | 0.8872 | 0.8310 | 0.8582 | 0.9306 | +1.99% | 57.00s / 57.20s |
| **Exp-2A** | MobileNetV2 — Frozen Feature Extractor | 2,422,081 (164,097) | 11.06 MB | 12 / 17 | 0.1277 | 94.33% (16 err) | 0.9375 | 0.9507 | 0.9441 | 0.9909 | +0.35% | 30.41s / 20.64s |
| **Exp-2B** | MobileNetV2 — Selective Fine-Tuning (top 30%, BN frozen) | 2,422,081 (1,943,233) | 24.65 MB | 8 / 13 | 0.0643 | **97.52% (7 err)** | 0.9720 | 0.9789 | **0.9754** | **0.9975** | +0.26% | 26.99s / 21.64s |
| **Exp-3A** | ResNet50 — Frozen Feature Extractor | 23,850,113 (262,401) | 93.62 MB | 6 / 11 | 0.0612 | 97.87% (6 err) | 0.9722 | 0.9859 | 0.9790 | 0.9983 | -1.19% | 31.38s / 22.54s |
| **Exp-3B** | ResNet50 — Selective Fine-Tuning (top 30%, BN frozen) | 23,850,113 (17,447,169) | 224.75 MB | 2 / 7 | 0.0519 | **98.23% (5 err)** | 0.9790 | 0.9859 | **0.9825** | **0.9989** | -0.09% | 37.79s / 32.41s |

**Catatan Metrik Tambahan pada Checkpoint Terbaik:**
- **Exp-2A (MobileNetV2 Frozen):** Best Train Acc `94.68%`, Final Train Acc `95.30%`, `16` error validasi dari `282` (`9` FP, `7` FN).
- **Exp-2B (MobileNetV2 Fine-Tuned):** Best Train Acc `97.78%`, Final Train Acc `99.20%`, `7` error validasi dari `282` (`4` FP, `3` FN); membuka `31` layer non-BN teratas sambil mengunci `52` layer BN.
- **Exp-3A (ResNet50 Frozen):** Best Train Acc `96.68%`, Final Train Acc `98.05%`, `6` error validasi dari `282` (`4` FP, `2` FN).
- **Exp-3B (ResNet50 Fine-Tuned):** Best Train Acc `98.14%`, Final Train Acc `99.65%`, `5` error validasi dari `282` (`3` FP, `2` FN); membuka `37` layer non-BN teratas sambil mengunci `53` layer BN.

---

## 3. Log Training Lengkap per Epoch (`Exp-2A`, `Exp-2B`, `Exp-3A`, `Exp-3B`)

### 3.1 Exp-2A — MobileNetV2 Frozen Feature Extractor (`164.097` trainable dari `2.422.081` params; Best Epoch 12/17)

| **Epoch** | **Train Loss** | **Val Loss** | **Train Acc** | **Val Acc** | **Val Prec** | **Val Rec** | **Val AUC** | **LR** | **Waktu (s)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4413 | 0.2218 | 79.26% | 91.49% | 0.9041 | 0.9296 | 0.9776 | 3.00e-04 | 165.48 |
| 2 | 0.2617 | 0.1761 | 89.41% | 92.91% | 0.9236 | 0.9366 | 0.9845 | 3.00e-04 | 20.97 |
| 3 | 0.2244 | 0.1590 | 90.91% | 92.91% | 0.9178 | 0.9437 | 0.9887 | 3.00e-04 | 21.67 |
| 4 | 0.2007 | 0.1436 | 92.07% | 95.39% | 0.9640 | 0.9437 | 0.9895 | 3.00e-04 | 20.27 |
| 5 | 0.1807 | 0.1422 | 92.95% | 93.62% | 0.9306 | 0.9437 | 0.9890 | 3.00e-04 | 21.91 |
| 6 | 0.1876 | 0.1490 | 92.46% | 93.62% | 0.9247 | 0.9507 | 0.9884 | 3.00e-04 | 41.23 |
| 7 | 0.1613 | 0.1663 | 93.31% | 92.91% | 0.9013 | 0.9648 | 0.9890 | 3.00e-04 | 19.46 |
| 8 | 0.1551 | 0.1623 | 94.19% | 92.91% | 0.9013 | 0.9648 | 0.9891 | 1.50e-04 | 19.99 |
| 9 | 0.1580 | 0.1392 | 93.35% | 93.62% | 0.9247 | 0.9507 | 0.9902 | 1.50e-04 | 21.53 |
| 10 | 0.1432 | 0.1341 | 94.68% | 93.97% | 0.9310 | 0.9507 | 0.9900 | 1.50e-04 | 20.07 |
| 11 | 0.1288 | 0.1622 | 95.12% | 92.91% | 0.9013 | 0.9648 | 0.9901 | 1.50e-04 | 21.69 |
| 12 | 0.1311 | 0.1277 | 94.68% | 94.33% | 0.9375 | 0.9507 | 0.9908 | 1.50e-04 | 20.31 |
| 13 | 0.1276 | 0.1350 | 94.95% | 93.62% | 0.9247 | 0.9507 | 0.9900 | 1.50e-04 | 19.53 |
| 14 | 0.1277 | 0.1501 | 95.21% | 92.91% | 0.9067 | 0.9577 | 0.9903 | 1.50e-04 | 21.24 |
| 15 | 0.1304 | 0.1449 | 94.64% | 93.62% | 0.9133 | 0.9648 | 0.9906 | 7.50e-05 | 19.64 |
| 16 | 0.1233 | 0.1385 | 95.12% | 94.33% | 0.9257 | 0.9648 | 0.9905 | 7.50e-05 | 22.55 |
| 17 | 0.1185 | 0.1370 | 95.30% | 94.33% | 0.9257 | 0.9648 | 0.9906 | 3.75e-05 | 19.42 |

### 3.2 Exp-2B — MobileNetV2 Selective Fine-Tuning (`1.943.233` trainable dari `2.422.081` params, `52` BN frozen; Best Epoch 8/13)

| **Epoch** | **Train Loss** | **Val Loss** | **Train Acc** | **Val Acc** | **Val Prec** | **Val Rec** | **Val AUC** | **LR** | **Waktu (s)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1252 | 0.1273 | 95.08% | 93.26% | 0.9184 | 0.9507 | 0.9922 | 1.00e-05 | 56.76 |
| 2 | 0.1058 | 0.1508 | 95.66% | 93.26% | 0.8917 | 0.9859 | 0.9939 | 1.00e-05 | 20.28 |
| 3 | 0.0785 | 0.0970 | 97.12% | 95.39% | 0.9388 | 0.9718 | 0.9955 | 1.00e-05 | 21.03 |
| 4 | 0.0765 | 0.0799 | 97.16% | 95.74% | 0.9577 | 0.9577 | 0.9962 | 1.00e-05 | 21.60 |
| 5 | 0.0703 | 0.0874 | 97.21% | 96.81% | 0.9586 | 0.9789 | 0.9959 | 1.00e-05 | 19.96 |
| 6 | 0.0551 | 0.0791 | 97.78% | 96.45% | 0.9783 | 0.9507 | 0.9962 | 1.00e-05 | 22.30 |
| 7 | 0.0586 | 0.0796 | 97.70% | 97.16% | 0.9926 | 0.9507 | 0.9943 | 1.00e-05 | 23.17 |
| 8 | 0.0575 | 0.0643 | 97.78% | 97.52% | 0.9720 | 0.9789 | 0.9975 | 1.00e-05 | 22.99 |
| 9 | 0.0384 | 0.0724 | 98.80% | 96.81% | 0.9586 | 0.9789 | 0.9978 | 1.00e-05 | 39.64 |
| 10 | 0.0339 | 0.0983 | 98.80% | 96.10% | 0.9396 | 0.9859 | 0.9965 | 1.00e-05 | 19.90 |
| 11 | 0.0283 | 0.0829 | 99.02% | 96.45% | 0.9521 | 0.9789 | 0.9974 | 5.00e-06 | 20.05 |
| 12 | 0.0354 | 0.0728 | 98.80% | 97.52% | 0.9720 | 0.9789 | 0.9976 | 5.00e-06 | 21.67 |
| 13 | 0.0256 | 0.0815 | 99.20% | 96.45% | 0.9521 | 0.9789 | 0.9974 | 2.50e-06 | 41.49 |

### 3.3 Exp-3A — ResNet50 Frozen Feature Extractor (`262.401` trainable dari `23.850.113` params; Best Epoch 6/11)

| **Epoch** | **Train Loss** | **Val Loss** | **Train Acc** | **Val Acc** | **Val Prec** | **Val Rec** | **Val AUC** | **LR** | **Waktu (s)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3333 | 0.1513 | 84.80% | 93.97% | 0.8981 | 0.9930 | 0.9954 | 3.00e-04 | 85.35 |
| 2 | 0.1818 | 0.1692 | 93.04% | 91.13% | 0.8545 | 0.9930 | 0.9958 | 3.00e-04 | 20.92 |
| 3 | 0.1267 | 0.0858 | 94.99% | 96.10% | 0.9396 | 0.9859 | 0.9972 | 3.00e-04 | 23.94 |
| 4 | 0.1158 | 0.1459 | 95.66% | 93.97% | 0.8931 | 1.0000 | 0.9979 | 3.00e-04 | 21.36 |
| 5 | 0.0969 | 0.0797 | 96.28% | 96.81% | 0.9524 | 0.9859 | 0.9979 | 3.00e-04 | 23.59 |
| 6 | 0.0883 | 0.0612 | 96.68% | 97.87% | 0.9722 | 0.9859 | 0.9983 | 3.00e-04 | 41.70 |
| 7 | 0.0770 | 0.0616 | 97.16% | 97.52% | 0.9655 | 0.9859 | 0.9985 | 3.00e-04 | 21.18 |
| 8 | 0.0672 | 0.0696 | 97.30% | 97.16% | 0.9527 | 0.9930 | 0.9984 | 3.00e-04 | 41.20 |
| 9 | 0.0570 | 0.0787 | 97.83% | 96.81% | 0.9524 | 0.9859 | 0.9982 | 1.50e-04 | 22.35 |
| 10 | 0.0533 | 0.0762 | 98.01% | 96.81% | 0.9524 | 0.9859 | 0.9983 | 1.50e-04 | 20.85 |
| 11 | 0.0530 | 0.0825 | 98.05% | 97.52% | 0.9530 | 1.0000 | 0.9983 | 7.50e-05 | 22.73 |

### 3.4 Exp-3B — ResNet50 Selective Fine-Tuning (`17.447.169` trainable dari `23.850.113` params, `53` BN frozen; Best Epoch 2/7)

| **Epoch** | **Train Loss** | **Val Loss** | **Train Acc** | **Val Acc** | **Val Prec** | **Val Rec** | **Val AUC** | **LR** | **Waktu (s)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0711 | 0.0695 | 97.34% | 96.81% | 0.9524 | 0.9859 | 0.9984 | 1.00e-05 | 71.61 |
| 2 | 0.0472 | 0.0519 | 98.14% | 98.23% | 0.9790 | 0.9859 | 0.9988 | 1.00e-05 | 25.74 |
| 3 | 0.0318 | 0.0636 | 99.02% | 97.87% | 0.9658 | 0.9930 | 0.9958 | 1.00e-05 | 39.96 |
| 4 | 0.0221 | 0.0574 | 99.34% | 97.87% | 0.9595 | 1.0000 | 0.9994 | 1.00e-05 | 39.07 |
| 5 | 0.0217 | 0.0667 | 99.11% | 97.87% | 0.9595 | 1.0000 | 0.9958 | 5.00e-06 | 23.78 |
| 6 | 0.0129 | 0.0860 | 99.65% | 97.16% | 0.9467 | 1.0000 | 0.9961 | 5.00e-06 | 40.04 |
| 7 | 0.0104 | 0.0568 | 99.65% | 98.23% | 0.9660 | 1.0000 | 0.9960 | 2.50e-06 | 24.34 |

---

## 4. Analisis Ilmiah & Jawaban Sementara RQ1–RQ2

1. **Jawaban RQ1 (Scratch CNN vs Transfer Learning):** Fitur *pre-trained* ImageNet memberikan lompatan performa yang sangat besar di atas *fair baseline* `Exp-1B` (`86.17%`, `val_loss=0.3547`, `ROC-AUC=0.9306`, `39` error validasi):
   - Bahkan tanpa membuka satu pun layer konvolusi (*frozen feature extraction*), **MobileNetV2 (`Exp-2A`)** langsung melesat ke `94.33%` (`+8.16 pp`, `ROC-AUC=0.9909`, error turun dari `39` menjadi `16`), dan **ResNet50 (`Exp-3A`)** menembus `97.87%` (`+11.70 pp`, `ROC-AUC=0.9983`, error tersisa `6`).
   - Setelah *selective fine-tuning*, **Exp-2B** mencapai `97.52%` (`+11.35 pp`, `val_loss` turun `5.5×` ke `0.0643`, error berkurang `82%` menjadi `7`), dan **Exp-3B** mencapai puncak validasi `98.23%` (`+12.06 pp`, `val_loss` turun `6.8×` ke `0.0519`, `ROC-AUC=0.9989`, error berkurang `87%` menjadi `5` dari `282`).

2. **Jawaban RQ2 (Frozen vs Selective Fine-Tuning dengan `BatchNormalization` Terkunci):**
   - Menjaga seluruh `52` layer BN (MobileNetV2) dan `53` layer BN (ResNet50) tetap `trainable=False` terbukti krusial: osilasi ekstrem `val_loss` seperti yang terjadi di Phase 2 (`Exp-1B` sempat melonjak ke `3.4914`) **hilang sepenuhnya** pada keempat eksperimen Phase 3.
   - Dampak *fine-tuning* bergantung pada kapasitas representasi awal backbone:
     - **MobileNetV2 (`Exp-2A → Exp-2B`):** Mendapat manfaat sangat besar dan bersih — akurasi naik `+3.19 pp` (`94.33% → 97.52%`), `val_loss` terpangkas hampir separuh (`0.1277 → 0.0643`), F1 naik ke `0.9754`, dan *overfitting gap* di epoch terbaik tetap sangat kecil (`+0.26%`).
     - **ResNet50 (`Exp-3A → Exp-3B`):** Mengalami *diminishing returns* dan konvergensi sangat cepat — akurasi naik `+0.36 pp` (`97.87% → 98.23%`, berkurang `1` kesalahan validasi) karena fitur beku ResNet50 sudah bernilai `97.87%` (`ROC-AUC=0.9983`). Dengan `17.45M` parameter terbuka di `Exp-3B`, model mencapai `val_loss` minimum seketika pada **Epoch 2 dari 7**, lalu `train_acc` naik cepat hingga `99.65%` di Epoch 7 sementara `val_loss` mulai merangkak naik (`0.0519 → 0.0860` di Epoch 6) — membuktikan pola *capacity-dependent overfitting onset* dan memvalidasi peran `EarlyStopping(restore_best_weights=True)`.

3. **Trade-off Efisiensi vs Akurasi (Pareto Frontier):**
   - **Exp-2B (`MobileNetV2 Fine-Tuned`)** adalah *sweet spot* efisiensi pada Pareto frontier: hanya berukuran **`24.65 MB`** (`2.42M` total parameter), meraih **`97.52%` Val Accuracy** (`val_f1=0.9754`, `ROC-AUC=0.9975`, `7` error).
   - **Exp-3B (`ResNet50 Fine-Tuned`)** meraih akurasi validasi tertinggi (**`98.23%`**, `ROC-AUC=0.9989`, `5` error), namun ukuran modelnya **`224.75 MB`** (`23.85M` total parameter) — **`9.1×` lebih besar** dibandingkan `Exp-2B` untuk selisih hanya **`+0.71 pp`** (`2` citra validasi). Keduanya (`Exp-2B` dan `Exp-3B`) layak dibawa ke **Phase 4** untuk adu performa pada *Test Set* yang masih terkunci, uji latensi inferensi, Grad-CAM, dan OOD Stress Test.

4. **Analisis Runtime GPU NVIDIA Tesla T4:**
   - Pada keempat eksperimen, **Epoch 1** jauh lebih lama (`56.76s`–`165.48s`) karena mencakup kompilasi grafik XLA/cuDNN dan pengisian `tf.data.Dataset.cache()` pertama kali.
   - Setelah cache terisi, waktu **steady-state median** (Epoch 2 ke atas) sangat konsisten: `20.64s` (`Exp-2A`), `21.64s` (`Exp-2B`), `22.54s` (`Exp-3A`), dan `32.41s` (`Exp-3B`), dengan sisa waktu didominasi oleh augmentasi spasial CPU per epoch dan penulisan checkpoint `.keras` ke disk (terlihat jeda ~40s setiap kali checkpoint besar disimpan).

5. **Sikap Kritis Ilmiah Menuju Phase 4 (RQ3 & RQ4):**
   - Nilai `ROC-AUC` validasi yang nyaris jenuh (`0.9975–0.9989`) **belum boleh dianggap bukti final** bahwa model benar-benar memahami artefak generatif AI. Meskipun bias aspek rasio telah dimitigasi dengan `center_crop=True`, sisa pola kurasi distribusi *in-distribution* (misalnya resolusi asli sebelum *downscale*, kompresi JPEG historis pada lukisan vs tekstur halus AI) masih berpotensi membantu prediksi.
   - Oleh karena itu, **Phase 4** wajib menguji kedua kandidat utama (`Exp-2B` dan `Exp-3B`, bersama pembanding `Exp-1B`) pada **Test Set (`282` citra)**, menginspeksi seluruh kesalahan melalui **Error Analysis**, memverifikasi area fokus dengan **Grad-CAM** (RQ3), dan melakukan **OOD Stress Test** terhadap lukisan museum serta generator modern seperti Midjourney v6, Flux, dan DALL-E 3 (RQ4).

---

## 5. Inventaris Artefak Phase 3 (Siap untuk Final Report & Phase 4)

| **Artefak** | **Path Lokal (Repo)** | **Tautan Google Drive / Colab** | **Keterangan** |
| --- | --- | --- | --- |
| **Notebook Phase 3 (Executed)** | `notebooks/03_phase3_transfer_learning.ipynb` | [Buka di Colab](https://colab.research.google.com/drive/16ALd9AbYMTcDHBR26RB2OAZWWH-YdmEZ) | Seluruh 9 cell dieksekusi di GPU T4 tanpa error |
| **Summary Metrik Phase 3** | `experiments/phase3_summary.json` | [phase3_summary.json](https://drive.google.com/file/d/1uzXrwkyExx6PAtH4LW_33rASOsgB3bsH/view?usp=drivesdk) | Ringkasan parameter, ukuran, best epoch, dan metrik val |
| **Histori Per-Epoch** | `experiments/exp2a_history.json` · `exp2b` · `exp3a` · `exp3b` | [Folder phase3_outputs](https://drive.google.com/drive/folders/1tFeT0V2s_gwEJm3g6Ii-1bdGQ_fiGUwR) | Log loss, akurasi, prec, rec, AUC, LR, dan waktu/epoch |
| **Log Eksperimen Global** | `experiments/experiment_log.csv` | [experiment_log.csv](https://drive.google.com/file/d/1Dgn_IaogzkvhkRXdHKZKb2pK1bE5ViRP/view?usp=drivesdk) | `Ref-0` hingga `Exp-3B` (timestamp WIB diselaraskan) |
| **Tabel Ablasi Gabungan** | `experiments/phase2_phase3_ablation_table.csv` | Di GitHub (`main`) | Tabel komparasi 22 kolom siap olah untuk Final Report |
| **Grafik Kurva Pembelajaran** | `reports/figures/phase3_transfer_learning_curves.png` | [phase3_transfer_learning_curves.png](https://drive.google.com/file/d/1F4plj5uFzvUSzymMLbdk7srXhWTk_Tzs/view?usp=drivesdk) | 6-panel Loss, Accuracy, ROC-AUC (`Exp-2A` s.d. `Exp-3B`) |
| **Grafik Ringkasan & Pareto** | `reports/figures/phase3_ablation_summary.png` | [phase3_ablation_summary.png](https://drive.google.com/file/d/1eUrZOY5FuMjWmaY-fNQbFmOZHvIoEOIS/view?usp=drivesdk) | Progresi metrik + kurva Pareto ukuran vs akurasi |
| **Checkpoint Exp-2A** | `models/exp2a_mobilenetv2_frozen.keras` (`11.06 MB`) | [exp2a_mobilenetv2_frozen.keras](https://drive.google.com/file/d/1wabABZAKWwxscOBETVxKp7HHVLV_zE1J/view?usp=drivesdk) | MobileNetV2 Frozen (`gitignored`, tersimpan lokal & Drive) |
| **Checkpoint Exp-2B** | `models/exp2b_mobilenetv2_finetuned.keras` (`24.65 MB`) | [exp2b_mobilenetv2_finetuned.keras](https://drive.google.com/file/d/1gxF8FB8aKag8sHO-VVtiNnBu8akb0bIu/view?usp=drivesdk) | **Kandidat Pareto Efisien** (terverifikasi load lokal) |
| **Checkpoint Exp-3A** | `models/exp3a_resnet50_frozen.keras` (`93.62 MB`) | [exp3a_resnet50_frozen.keras](https://drive.google.com/file/d/1c1G1lpSXvBq7InJHccK879hRVsfvScgB/view?usp=drivesdk) | ResNet50 Frozen (`gitignored`, tersimpan lokal & Drive) |
| **Checkpoint Exp-3B** | `models/exp3b_resnet50_finetuned.keras` (`224.75 MB`) | [exp3b_resnet50_finetuned.keras](https://drive.google.com/file/d/1tZYyYAOwWkhKV-fU1wL2jSvPHIqUaICa/view?usp=drivesdk) | **Kandidat Akurasi Puncak** (terverifikasi load lokal) |

---

## 6. Visualisasi Hasil Eksperimen

### 6.1 Kurva Pembelajaran Phase 3 (Exp-2A hingga Exp-3B)

### 6.2 Ringkasan Ablasi Gabungan & Pareto Efisiensi (Phase 2 + Phase 3)
