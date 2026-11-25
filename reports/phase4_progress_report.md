# Report 05 — Phase 4: Deep Evaluation, Error Analysis, Grad-CAM & OOD Stress Test

**Proyek:** Klasifikasi Citra Real Art vs AI-Generated Art  
**Dokumen Induk:** [Research Log & Progress Tracker](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)  
**Notebook Colab (Executed):** [04_phase4_deep_evaluation.ipynb](https://colab.research.google.com/drive/1qdSI0HbOMgv02Np8kZ-bedR6HWV7APR5)  
**Dataset OOD Drive:** [Folder 05_OOD_Dataset (`ood_stress_test_30.zip`)](https://drive.google.com/drive/folders/1UMYD84ZHvNObSUtlkg7I6wvmUho2N-uy)  
**Pull Requests:** [PR #9](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/9) · [PR #10](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/10) · [PR #11](https://github.com/Indraku19/real-art-vs-ai-generated-art/pull/11)

---

## 1. Protokol Evaluasi Mendalam Phase 4

Seluruh 6 model hasil Phase 2 (`Exp-1A`, `Exp-1B`) dan Phase 3 (`Exp-2A`, `Exp-2B`, `Exp-3A`, `Exp-3B`) dievaluasi bersama secara terkontrol di **Google Colab NVIDIA Tesla T4 GPU** (`TensorFlow 2.21.0`, `Keras 3.13.2`, seluruh 9 code cell `exec 3..11` berjalan bersih tanpa error):

- **Kompatibilitas Lintas-Versi Keras (`PR #10`):** Dua model *scratch* Phase 2 yang dilatih di macOS (`Keras 3.15.1`) dimuat secara aman di Colab (`Keras 3.13.2`) melalui helper `load_keras_model_compat()` di `src/data_pipeline.py` tanpa memodifikasi arsip `.keras` asli.
- **Disiplin One-Shot Test Set Lock:** *Threshold tuning* (`t_opt`) dilakukan **eksklusif pada Validation Set (`282` citra)** dengan target memaksimalkan `F1` sambil mengunci `Recall >= 0.80` (tie-breaker terdekat ke `0.50`), lalu **Test Set (`282` citra: `141` Real / `141` AI)** dibuka **tepat satu kali** untuk seluruh model.
- **Interpretabilitas Grad-CAM Pre-Sigmoid Logits (`RQ3`):** Gradien dihitung terhadap *pre-sigmoid logit* (`kernel @ gap_out + bias`) guna mencegah saturasi gradien saat probabilitas mendekati `0.0` atau `1.0`.
- **OOD Stress Test Independen (`RQ4`):** 30 citra eksternal (`15` lukisan museum resolusi `750–2.327 px` dari `huggan/wikiart` + `5` Midjourney v6 + `5` FLUX.1-Dev + `5` DALL-E 3 dengan jarak pHash minimum `14` terhadap seluruh 2.820 citra dataset utama).

---

## 2. Hasil Threshold Tuning (Val Set) & Evaluasi Final Test Set (`N = 282`)

| **Run ID** | **Model & Strategi** | **Val Acc (P4 Colab)** | **Test Acc @0.50 (Err / 282)** | **Test Loss** | **Prec** | **Rec** | **F1** | **ROC-AUC** | **PR-AUC** | **Tuned t*** | **Test Acc @t* (F1)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ref-0** | DataScienceProject/CNN (HF Benchmark) | N/A | 77.00% | 0.4900 | 0.7700 | 0.7700 | 0.7700 | N/A | N/A | 0.50 | N/A |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) — Scratch | 71.28% | 69.50% (86 err) | 0.5893 | 0.8313 | 0.4894 | 0.6161 | 0.8085 | 0.8091 | 0.31 | 73.76% (0.7338) |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) — Scratch | 80.50% | 80.14% (56 err) | 0.4658 | 0.9126 | 0.6667 | 0.7705 | 0.9198 | 0.9086 | 0.11 | 82.27% (0.8344) |
| **Exp-2A** | MobileNetV2 — Frozen Feature Extractor | 95.04% | 93.26% (19 err) | 0.1607 | 0.9552 | 0.9078 | 0.9309 | 0.9845 | 0.9868 | 0.38 | 93.62% (0.9357) |
| **Exp-2B** | MobileNetV2 — Selective Fine-Tuning (top 30%, BN frozen) | 97.52% | **97.87% (6 err)** | 0.0855 | 0.9856 | 0.9716 | **0.9786** | **0.9948** | 0.9953 | 0.55 | 97.16% (0.9712) |
| **Exp-3A** | ResNet50 — Frozen Feature Extractor | 97.87% | 98.23% (5 err) | 0.0588 | 0.9857 | 0.9787 | 0.9822 | 0.9981 | 0.9983 | 0.49 | 97.87% (0.9787) |
| **Exp-3B** | ResNet50 — Selective Fine-Tuning (top 30%, BN frozen) | 98.58% | **98.58% (4 err)** | 0.0414 | 0.9858 | 0.9858 | **0.9858** | **0.9985** | 0.9987 | 0.47 | 98.58% (0.9859) |

**Temuan Utama Evaluasi Test Set (`N = 282`):**

- **Dominasi Transfer Learning vs Target Proposal (`>= 85% Acc`, `>= 80% Recall`):** Ketiga model utama Transfer Learning (`Exp-2B`, `Exp-3A`, `Exp-3B`) melampaui target dengan selisih besar — puncaknya **`Exp-3B` (ResNet50 Fine-Tuned)** meraih **`98.58%` Test Accuracy (`4` error: `2` FP, `2` FN)**, **`Test Loss = 0.0414`**, **`F1 = 0.9858`**, **`ROC-AUC = 0.9985`**, dan **`PR-AUC = 0.9987`**.
- **Perbandingan Statistik McNemar (`Exp-2B` vs `Exp-3B`):** Pada Test Set `N = 282`, kedua model sama-sama benar pada `275` citra, sama-sama salah pada `3` citra, `Exp-2B` unggul sendiri pada `1` citra, dan `Exp-3B` unggul sendiri pada `3` citra (`4` pasangan diskordan → **two-sided exact binomial `p = 0.625`**). Secara statistik akurasi *in-distribution* keduanya setara, namun `Exp-3B` unggul signifikan dalam *probability calibration* (`Test Loss 0.0414` vs `0.0855`) dan ketahanan OOD.
- **Catatan Metodologis Penting — Sensitivitas Interpolasi Model Scratch (`Exp-1A` & `Exp-1B`):** Pada lingkungan Colab, dataset utama dimuat dari arsip cache `224 × 224` (`processed_images_224.zip` hasil *downscale* `PIL.Image.Resampling.LANCZOS`), sedangkan saat validasi lokal Phase 2 citra penuh di-*resize* oleh `tf.image.resize(..., method="bilinear")`. Model *scratch* yang tidak memiliki *pre-training* ImageNet mengalami pergeseran tajam pada **Val Colab** (`Exp-1A`: `79.43% → 71.28%`, `Exp-1B`: `86.17% → 80.50%`) dan **Test Set** (`Exp-1A`: `69.50%`, `Exp-1B`: `80.14%`), sementara model Transfer Learning tetap kokoh (`Exp-2B`: `97.52% → 97.87%`, `Exp-3B`: `98.23% → 98.58%`). Ini menjadi bukti tambahan bahwa CNN *scratch* sensitif terhadap perbedaan *kernel* interpolasi tepi frekuensi tinggi, sedangkan representasi pra-latih ImageNet jauh lebih invarian.
- **Dampak Threshold Tuning (`t*`):** Pada model yang *under-recall* (`Exp-1A` `t*=0.31`, `Exp-1B` `t*=0.11`), *threshold tuning* memulihkan Test F1 secara drastis (`0.6161 → 0.7338` dan `0.7705 → 0.8344`). Sebaliknya, pada model fine-tuned berkalibrasi baik (`Exp-2B`, `Exp-3B`), ambang batas default `t = 0.50` sudah optimal.

---

## 3. Benchmark Efisiensi Inferensi (GPU NVIDIA Tesla T4)

| **Run ID** | **Model** | **Params Total (Trainable)** | **Ukuran (.keras)** | **Single Mean (ms)** | **Single Median (ms)** | **Single P95 (ms)** | **Batch-32 (ms/img)** | **Throughput (img/s)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) | 23,907,521 (23,907,521) | 273.64 MB | 8.88 | **8.72** | 9.99 | 0.56 | 1785.7 |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) | 422,881 (421,921) | 4.92 MB | 17.44 | **16.97** | 20.95 | 0.88 | 1136.4 |
| **Exp-2A** | MobileNetV2 | 2,422,081 (164,097) | 11.06 MB | 174.91 | **162.24** | 228.56 | 4.98 | 200.8 |
| **Exp-2B** | MobileNetV2 | 2,422,081 (1,943,233) | 24.65 MB | 174.35 | **160.99** | 225.13 | 6.42 | 155.8 |
| **Exp-3A** | ResNet50 | 23,850,113 (262,401) | 93.62 MB | 252.35 | **232.86** | 327.61 | 7.05 | 141.8 |
| **Exp-3B** | ResNet50 | 23,850,113 (17,447,169) | 224.75 MB | 246.93 | **232.69** | 316.53 | 9.25 | 108.1 |

**Analisis Trade-Off Kecepatan & Memori:**

- Di GPU T4 melalui panggilan langsung Python/Keras (`single-image` termasuk overhead dispatch Python + TensorFlow), **MobileNetV2 Fine-Tuned (`Exp-2B`)** berukuran **`24.65 MB`** (`9.1×` lebih kecil daripada `Exp-3B` `224.75 MB`), **`1.45×` lebih cepat** per citra tunggal (`160.99 ms` vs `232.69 ms`), serta memberi *batch-32 throughput* **`155.8 img/s`** vs **`108.1 img/s`** pada `Exp-3B`.
- Untuk produksi *edge/mobile/hemat biaya*, **`Exp-2B`** adalah pemenang Pareto; sedangkan untuk akurasi puncak dan ketahanan OOD di *server/demo cloud*, **`Exp-3B`** adalah pilihan utama.

---

## 4. Error Analysis & Interpretabilitas Grad-CAM (Jawaban RQ3)

| **Kelompok Sampel Test** | **Jumlah** | **Real / AI** | **Mean Aspect Ratio** | **% Square (0.95–1.05)** | **% Format PNG** | **Median Resolusi (W×H)** |
| --- | --- | --- | --- | --- | --- | --- |
| **Error (2B or 3B)** | 7 | 2 / 5 | 0.969 | 57.1% | 14.3% | 614×625 |
| **Correct Both (2B & 3B)** | 275 | 139 / 136 | 1.010 | 36.7% | 2.9% | 614×614 |

**Temuan Kunci Error Analysis & Grad-CAM (`RQ3`):**

1. **Ketidaksinkronan Shortcut Format/Aspek (`RQ3`):** Pada seluruh `7` kesalahan gabungan (`Exp-2B` atau `Exp-3B`) di Test Set (`2` Real / `5` AI), `mean aspect ratio = 0.969` (vs `1.010` pada `275` prediksi benar). Tidak ada bukti ketergantungan ke *aspect ratio* ekstrem setelah `center_crop=True`.
2. **Penemuan Label Noise di Dataset Asli (`600 × 625 JPEG`):** Inspeksi visual *Error Analysis* mengungkap 1 sampel berlabel `AI-Generated Art` di Test Set yang ternyata merupakan **meme komik digital 4-panel** bertema Stable Diffusion/Colab (`2B P(AI)=0.014`, `3B P(AI)=0.988`). Kesalahan lainnya bersifat sangat wajar secara artistik: lukisan laut bergaya klasik yang merupakan AI (`BOTH FN`: `2B=0.008`, `3B=0.019`), lukisan potret/pemandangan bergaya *chiaroscuro* / warna kontras cerah (`BOTH FP`), dan gambar berbingkai kayu tebal (`3B FN`: `P(AI)=0.488`).
3. **Verifikasi Visual Grad-CAM (*Shortcut* Bingkai vs Tekstur Karya):** Pada sampel AI dengan bingkai kayu tebal (baris ke-8 `phase4_gradcam_comparison.png`), **`Exp-1B` Scratch CNN memusatkan aktivasi tinggi tepat pada garis bingkai atas dan bawah** (*border/frame shortcut*), sedangkan **`Exp-2B` dan `Exp-3B` mengabaikan bingkai dan memusatkan fokus pada area bangunan, salju, serta permukaan karya**. Pada sampel lukisan real dan AI futuristik/arsitektur, kedua model Transfer Learning secara konsisten membaca tekstur kuas, bayangan wajah, dan transisi pencahayaan interior.

---

## 5. Out-of-Distribution (OOD) Stress Test (`N = 30`, Jawaban RQ4)

| **Run ID** | **Model** | **Test Acc** | **OOD Acc @0.50 (Err / 30)** | **Generalization Drop (pp)** | **OOD F1** | **OOD ROC-AUC** | **OOD PR-AUC** | **WikiArt (15 Real)** | **Midjourney v6 (5 AI)** | **FLUX.1-Dev (5 AI)** | **DALL-E 3 (5 AI)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) | 69.50% | 73.33% (8 err) | +3.83 pp | 0.7333 | 0.7333 | 0.7550 | 73.3% | 40.0% | 100.0% | 80.0% |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) | 80.14% | 73.33% (8 err) | -6.81 pp | 0.7778 | 0.9156 | 0.9343 | 53.3% | 100.0% | 100.0% | 80.0% |
| **Exp-2A** | MobileNetV2 | 93.26% | 83.33% (5 err) | -9.93 pp | 0.8485 | 0.9378 | 0.9485 | 73.3% | 80.0% | 100.0% | 100.0% |
| **Exp-2B** | MobileNetV2 | 97.87% | 83.33% (5 err) | -14.54 pp | 0.8485 | 0.9600 | 0.9750 | 73.3% | 80.0% | 100.0% | 100.0% |
| **Exp-3A** | ResNet50 | 98.23% | **90.00% (3 err)** | -8.23 pp | 0.9091 | **0.9956** | 0.9958 | 80.0% | 100.0% | 100.0% | 100.0% |
| **Exp-3B** | ResNet50 | 98.58% | **90.00% (3 err)** | -8.58 pp | 0.9091 | **0.9956** | 0.9958 | 80.0% | 100.0% | 100.0% | 100.0% |

**Temuan Kunci OOD Stress Test (`RQ4`):**

- **Superioritas ResNet50 pada Generalisasi Lintas-Domain:** **`Exp-3A` & `Exp-3B`** mempertahankan **`90.00%` OOD Accuracy (`27/30` benar)** dengan **`ROC-AUC = 0.9956`** dan **`PR-AUC = 0.9958`** — penurunan hanya **`-8.23 pp` hingga `-8.58 pp`** dari Test Set. Luar biasanya, kedua varian ResNet50 mencatat **`100%` Recall pada seluruh 15 citra AI modern** (`5/5` Midjourney v6, `5/5` FLUX.1-Dev, `5/5` DALL-E 3)!
- **MobileNetV2 Lebih Rentan terhadap Pergeseran Domain (`-14.54 pp`):** `Exp-2B` turun ke **`83.33%` (`ROC-AUC = 0.9600`)**, gagal pada `4/15` lukisan WikiArt dan `1/5` Midjourney v6 (meski tetap `100%` pada FLUX.1-Dev dan DALL-E 3).
- **Pola Kegagalan Utama (`3` False Positive di WikiArt):** Seluruh `3` kesalahan `Exp-3B` pada OOD terjadi pada lukisan asli museum (`80.0%` WikiArt) yang memiliki pencahayaan dramatis/halus seperti lukisan malam berbulan (`OOD-R12`, `P(AI)=0.984`) dan lukisan bunga berlatar merah gelap (`OOD-R11`, `P(AI)=0.933`) yang visual pencahayaannya menyerupai estetika *diffusion rendering*.

---

## 6. Rekomendasi Model Juara & Inventaris Artefak (Siap Phase 5)

- **Model Utama (Server / Demo Gradio + Grad-CAM Phase 5):** **`Exp-3B` (`exp3b_resnet50_finetuned.keras`, `224.75 MB`)** — terbaik di hampir seluruh metrik kunci: Test Acc **`98.58%`**, Test Loss **`0.0414`**, Test F1 **`0.9858`**, Test ROC-AUC **`0.9985`**, OOD Acc **`90.00%`**, dan OOD ROC-AUC **`0.9956`**.
- **Model Alternatif Efisien (Deployment Ringan):** **`Exp-2B` (`exp2b_mobilenetv2_finetuned.keras`, `24.65 MB`)** — `9.1×` lebih kecil dan `1.45×` lebih cepat dengan Test Acc setara secara statistik (`97.87%`, McNemar `p = 0.625`).

| **Artefak** | **Path Lokal (Repo)** | **Tautan Google Drive / Colab** |
| --- | --- | --- |
| **Notebook Phase 4 (Executed)** | `notebooks/04_phase4_deep_evaluation.ipynb` | [Buka di Colab](https://colab.research.google.com/drive/1qdSI0HbOMgv02Np8kZ-bedR6HWV7APR5) |
| **Summary JSON Phase 4** | `experiments/phase4_summary.json` | [phase4_summary.json](https://drive.google.com/file/d/11V_cEzLTmul22Xc1odqWajCrX5r8mvn1/view?usp=drivesdk) |
| **Master Benchmark Table** | `experiments/final_master_benchmark_table.csv` | Di GitHub (`main`) |
| **Prediksi Test & OOD (CSV)** | `experiments/phase4_test_predictions.csv` · `phase4_ood_predictions.csv` | [Folder phase4_outputs](https://drive.google.com/drive/folders/11qS7Mjr-CuD6Js_Bv26H3GsVWu1coCVa) |
| **Experiment Log Global** | `experiments/experiment_log.csv` | [experiment_log.csv](https://drive.google.com/file/d/1imjyQgHk-Cji68sZ6aV270obxRaXO3M7/view?usp=drivesdk) |
| **8 Grafik Visualisasi Phase 4** | `reports/figures/phase4_*.png` | [Folder 03_Figures_&_Visuals](https://drive.google.com/drive/folders/1W5ZO-s3XMNbSwTUMjZAstrQ36zzCQLpY) |

---

## 7. Visualisasi Lengkap Phase 4

### 7.1 Threshold Tuning pada Validation Set

### 7.2 Kurva ROC & Precision-Recall Test Set (N = 282)

### 7.3 Confusion Matrix Test Set (6 Model)

### 7.4 Benchmark Efisiensi Inferensi & Pareto Footprint

### 7.5 Error Analysis Grid Test Set (FP, FN & Borderline)

### 7.6 Perbandingan Grad-CAM Pre-Sigmoid Logits (Exp-1B vs Exp-2B vs Exp-3B)

### 7.7 Ringkasan OOD Stress Test & Breakdown per Sumber

### 7.8 Sampel Grad-CAM Out-of-Distribution (WikiArt vs Modern AI)
