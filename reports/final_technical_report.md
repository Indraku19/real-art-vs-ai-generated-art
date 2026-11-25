# Report 06 — Final Technical Report: Klasifikasi Citra Real Art vs AI-Generated Art (CNN From Scratch vs Transfer Learning)

**Proyek:** Klasifikasi Citra Real Art vs AI-Generated Art (`CNN Baseline vs Transfer Learning`)  
**Dataset Utama:** [`DataScienceProject/Art_Images_Ai_And_Real_`](https://huggingface.co/datasets/DataScienceProject/Art_Images_Ai_And_Real_)  
**Repositori GitHub:** [`Indraku19/real-art-vs-ai-generated-art`](https://github.com/Indraku19/real-art-vs-ai-generated-art)  
**Folder Google Drive:** [`Real-vs-AI-Art-Classification`](https://drive.google.com/drive/folders/1-H_FJGoVT5PmSaYm0xVSzQhkO9Bn9pWu)  
**Dokumen Induk:** [Research Log & Progress Tracker](https://docs.google.com/document/d/1RcadCdzhNb0HJgKf5FN61nsJZgXwhUWTLxxa2DlPSxM/edit)  
**Status Proyek:** **Selesai (`29/29` task wajib selesai, `1` task opsional di-skip)**

---

## 1. Executive Summary & Jawaban Research Questions (RQ1–RQ4)

Proyek ini mengevaluasi secara terkontrol perbandingan antara **Convolutional Neural Network (CNN) yang dilatih dari nol (*From Scratch*)** dan **Transfer Learning berbasis backbone ImageNet (`MobileNetV2` & `ResNet50`)** untuk membedakan lukisan asli buatan manusia (*Real Art*, label `0`) dan karya generatif AI (*AI-Generated Art*, label `1`). Dengan mengunci pipeline bebas kebocoran data (*pHash deduplication*, *stratified split* `80/10/10`, dan *center-crop* `1:1` untuk memitigasi bias aspek rasio), model terbaik **`Exp-3B` (ResNet50 Selective Fine-Tuning)** meraih **`98,58%` Test Accuracy** (`4` error dari `282` citra), **`0,9985` ROC-AUC**, dan **`90,00%` Out-of-Distribution (OOD) Accuracy** — melampaui target awal proposal (`>= 85%` Accuracy, `>= 80%` Recall) sekaligus jauh mengungguli benchmark rujukan Hugging Face `Ref-0` (`77,00%` Test Accuracy).

Ringkasan jawaban empiris untuk keempat *Research Questions*:

1. **RQ1 — Scratch CNN vs Transfer Learning:** Representasi fitur *pre-trained* ImageNet menghasilkan lompatan performa sangat besar di atas *fair baseline* `Exp-1B` (`80,14%` Test Acc, `0,9198` ROC-AUC, `56` error dari `282`). Bahkan tanpa *fine-tuning* konvolusi, **MobileNetV2 Frozen (`Exp-2A`)** naik ke **`93,26%` Test Acc** (`+13,12 pp`) dan **ResNet50 Frozen (`Exp-3A`)** menembus **`98,23%` Test Acc** (`+18,09 pp`). Setelah *selective fine-tuning*, **`Exp-2B` (MobileNetV2 FT)** mencapai **`97,87%` Test Acc** (`6` error) dan **`Exp-3B` (ResNet50 FT)** mencapai puncak **`98,58%` Test Acc** (`4` error, `Test Loss = 0,0414`). Selain akurasi, model Transfer Learning juga kebal terhadap pergeseran kernel interpolasi resize (`PIL-LANCZOS` vs *bilinear*), sedangkan model *scratch* merosot `-6` hingga `-8 pp`.
2. **RQ2 — Frozen vs Selective Fine-Tuning:** Membuka `30%` layer atas backbone sambil **menjaga 100% layer `BatchNormalization` tetap terkunci (`trainable=False`)** menghilangkan fluktuasi ekstrem `val_loss` sekaligus menurunkan *overfitting*. Dampaknya bergantung kapasitas awal backbone: pada **MobileNetV2**, *fine-tuning* (`Exp-2A -> Exp-2B`) memberi lonjakan signifikan di Test Set dari **`93,26%` -> `97,87%`** (`+4,61 pp`, `Test Loss 0,1607 -> 0,0855`, error turun dari `19` ke `6`). Pada **ResNet50** yang fitur bekunya sudah bernilai `98,23%`, *fine-tuning* (`Exp-3A -> Exp-3B`) tetap memperbaiki kalibrasi probabilitas (`Test Loss 0,0588 -> 0,0414`) dan menekan error menjadi `4/282`, dengan puncak validasi tercapai cepat pada **Epoch 2 dari 7** akibat kapasitas `17,45M` parameter terbuka.
3. **RQ3 — Shortcut vs Real Features (Error Analysis & Grad-CAM):** Analisis metadata pada `7` kesalahan gabungan `Exp-2B`/`Exp-3B` di Test Set menunjukkan rata-rata *aspect ratio* **`0,969`** (vs `1,010` pada `275` prediksi benar), membuktikan mitigasi `center_crop=True` menghilangkan ketergantungan ke *aspect ratio* ekstrem. Inspeksi visual menemukan `1` kasus *label noise* bawaan dataset (meme komik 4-panel berlabel AI). **Grad-CAM pre-sigmoid logits** membuktikan secara visual bahwa *scratch* CNN (`Exp-1B`) sempat mengandalkan **jalan pintas bingkai kayu (*frame-border shortcut*)**, sedangkan **`Exp-2B` dan `Exp-3B` mengabaikan bingkai** dan memusatkan aktivasi pada sapuan kuas, shading wajah, dan pencahayaan interior karya seni.
4. **RQ4 — Out-of-Distribution (OOD) Generalization (`N = 30`):** Pada `30` citra eksternal beresolusi tinggi dan terverifikasi `pHash >= 14` (`15` lukisan `huggan/wikiart` + `5` Midjourney v6 + `5` FLUX.1-Dev + `5` DALL-E 3), **`Exp-3B` (ResNet50 FT)** dan `Exp-3A` mempertahankan **`90,00%` OOD Accuracy (`27/30`, `ROC-AUC = 0,9956`)** dengan **`100%` Recall pada seluruh 15 citra generator AI modern (`15/15`)** dan `80,0%` (`12/15`) pada lukisan museum WikiArt. Sebaliknya, `Exp-2B` (MobileNetV2 FT) mengalami penurunan lebih tajam (`-14,54 pp` menjadi **`83,33%` OOD Acc**, `ROC-AUC = 0,9600`).

---

## 2. Dataset & Temuan Forensik Phase 1 (Mitigasi Kebocoran & Shortcut)

Audit menyeluruh di Phase 1 (`src/phase1_forensics.py`) sebelum pelatihan model mengungkap tiga temuan kritis pada dataset publik `DataScienceProject/Art_Images_Ai_And_Real_`:

1. **Koreksi Label Terbalik pada Hugging Face:** Metadata `ClassLabel` asli bernilai `['fake', 'real']` (`0 = fake`, `1 = real`), berlawanan dengan konvensi standar (`0 = Real Art`, `1 = AI-Generated Art`). Pemetaan ulang eksplisit `{0 -> 1, 1 -> 0}` dikunci di awal pipeline dan dikonfirmasi melalui inspeksi visual 16 panel (`phase1_visual_sanity_check.png`).
2. **Integritas & Deduplikasi Anti-Leakage:** Dari `2.839` citra yang dapat dibaca (`0` korup; `1.419` Real Art, `1.420` AI Art), seluruh citra (`3` Grayscale `L`, `3` `RGBA`, `71` PNG) dinormalkan ke **RGB 3-channel PNG**. Pemindaian *perceptual hash* (`imagehash.phash`, `Hamming <= 4`) mendeteksi **`6` pasangan duplikat persis** dan **`15` pasangan *near-duplicate***, lalu menghapus **`19` duplikat intra-kelas** (`0` konflik lintas kelas). Sisa **`2.820` citra bersih** dibagi secara *stratified* (`80/10/10`, `SEED = 42`):
   - **Train:** `2.256` citra (`1.124` Real Art, `1.132` AI-Generated Art)
   - **Validation:** `282` citra (`140` Real Art, `142` AI-Generated Art)
   - **Test (Terkunci hingga Phase 4):** `282` citra (`141` Real Art, `141` AI-Generated Art) — **0 exact pHash overlap** antar split.
3. **Bias Aspek Rasio & Resolusi (`79,01%` vs `1,55%` Persegi):**

| Kelas | Jumlah Bersih | Median Resolusi (`W x H`) | Median Ukuran File | % Citra Persegi (`0,95–1,05`) | Format Asli |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Real Art (`0`)** | 1.405 | `482 x 484` px | `64,8 KB` | **`1,55%`** | `100%` JPEG (`1.419`) |
| **AI-Generated Art (`1`)** | 1.415 | `1024 x 1024` px | `137,2 KB` | **`79,01%`** | `1.349` JPEG + `71` PNG |

Ketimpangan aspek rasio persegi (`79,01%` AI vs `1,55%` Real) merupakan sumber utama *shortcut learning* jika citra langsung di-*squish* menjadi `224 x 224`. Untuk mengatasinya, `src/data_pipeline.py` menerapkan **square center crop (`center_crop=True`)** sebelum resize ke `224 x 224`, ditambah augmentasi konservatif khusus Train (`RandomFlip`, `RandomRotation(0.05)`, `RandomZoom(0.1)`).

---

## 3. Desain Eksperimen Terkontrol & Arsitektur 6 Model

Seluruh eksperimen memakai pembagian `splits.csv`, batch size `32`, dan kontrak input seragam `float32 RGB [0.0, 255.0]` berukuran `(224, 224, 3)` dengan **layer preprocessing tertanam langsung di dalam model**:

- **Phase 2 — Controlled Baseline (CNN From Scratch, `notebooks/02_phase2_baseline_cnn.ipynb`):**
  - **`Exp-1A` (Naive CNN):** Replikasi arsitektur rujukan `2` blok `Conv2D(32 -> 64)` + `MaxPool2D` + `Flatten()` + `Dense(128)` (`23.907.521` params, `273,64 MB`), normalisasi `[0, 1]`. Membuktikan hipotesis ledakan parameter pada `Flatten()` (`54 x 54 x 64 = 186.624` fitur ke `Dense(128)`) yang langsung *overfit* setelah Epoch 2.
  - **`Exp-1B` (Proper Scratch CNN):** *Fair baseline* `4` blok `Conv2D(32 -> 64 -> 128 -> 256)` + `BatchNormalization` + `MaxPool2D` + `GlobalAveragePooling2D` + `Dropout(0.4)` (`422.881` params, `4,92 MB`, `56,5x` lebih ringkas dari `Exp-1A`), normalisasi `[0, 1]`.
- **Phase 3 — Transfer Learning & Ablation (`notebooks/03_phase3_transfer_learning.ipynb`, GPU T4 Colab):**
  - **Stage 1 Frozen (`Exp-2A` MobileNetV2, `Exp-3A` ResNet50):** Backbone ImageNet dibekukan penuh (`trainable=False`), *classification head* seragam (`GAP -> Dropout(0.3) -> Dense(128, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`), optimizer `Adam(lr=3e-4)`. Preprocessing: `Rescaling(1/127.5, -1)` untuk MobileNetV2; `ResNet50PreprocessLayer` (RGB->BGR + ImageNet mean subtraction) untuk ResNet50.
  - **Stage 2 Selective Fine-Tuning (`Exp-2B` MobileNetV2, `Exp-3B` ResNet50):** Memuat bobot terbaik Stage 1, membuka `30%` layer paling atas (`31` layer non-BN MobileNetV2; `37` layer non-BN ResNet50) sambil mengunci `100%` layer `BatchNormalization` (`52` BN pada MobileNetV2, `53` BN pada ResNet50), lalu melatih dengan `AdamW(lr=1e-5, weight_decay=1e-4)`.
  - *(Catatan Ruang Lingkup)* Task opsional `p3_modern` (*EfficientNetV2B0 / ConvNeXt*) diputuskan untuk **di-skip** guna menjaga kedalaman ablasi pada arsitektur CNN vs Transfer Learning klasik (`Exp-1A` s.d. `Exp-3B`).

---

## 4. Tabel Master Benchmark 6 Model (Validation, Test Set `N = 282`, GPU T4, & OOD `N = 30`)

Seluruh angka evaluasi akhir berasal dari eksekusi terkontrol satu kali (*one-shot Test unlock*) pada Google Colab GPU NVIDIA Tesla T4 (`experiments/final_master_benchmark_table.csv` & `experiments/exported_models_manifest.json`):

| Run ID | Model & Strategi | Params (Trainable) | Ukuran Asli -> Export | Val Acc (P2/P3) | Test Acc @0.50 (Err / 282) | Test Loss | Test F1 | Test ROC-AUC | PR-AUC | Tuned `t*` (Test Acc) | GPU T4 Latency (Throughput) | OOD Acc `N=30` (AUC) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref-0** | DataScienceProject/CNN *(HF)* | N/A | N/A | N/A | 77,00% | 0,4900 | 0,7700 | N/A | N/A | N/A | ~25–35s/ep (HF) | N/A |
| **Exp-1A** | Naive CNN (2 Conv + Flatten) | 23.907.521 (23.907.521) | 273,64 MB | 79,43% | 69,50% (86 err) | 0,5893 | 0,6161 | 0,8085 | 0,8091 | `0,31` (73,76%) | 8,72 ms (1.785,7 img/s) | 73,33% (0,7333) |
| **Exp-1B** | Proper CNN (4 Conv + BN + GAP) | 422.881 (421.921) | 4,92 MB | 86,17% | 80,14% (56 err) | 0,4658 | 0,7705 | 0,9198 | 0,9086 | `0,11` (82,27%) | 16,97 ms (1.136,4 img/s) | 73,33% (0,9156) |
| **Exp-2A** | MobileNetV2 — Frozen | 2.422.081 (164.097) | 11,06 MB | 94,33% | 93,26% (19 err) | 0,1607 | 0,9309 | 0,9845 | 0,9868 | `0,38` (93,62%) | 162,24 ms (200,8 img/s) | 83,33% (0,9378) |
| **Exp-2B** | MobileNetV2 — Selective FT | 2.422.081 (1.943.233) | 24,65 MB -> **9,80 MB** | 97,52% | **97,87% (6 err)** | 0,0855 | **0,9786** | **0,9948** | **0,9953** | `0,55` (97,16%) | **160,99 ms (155,8 img/s)** | 83,33% (0,9600) |
| **Exp-3A** | ResNet50 — Frozen | 23.850.113 (262.401) | 93,62 MB | 97,87% | 98,23% (5 err) | 0,0588 | 0,9822 | 0,9981 | 0,9983 | `0,49` (97,87%) | 232,86 ms (141,8 img/s) | **90,00% (0,9956)** |
| **Exp-3B** | ResNet50 — Selective FT | 23.850.113 (17.447.169) | 224,75 MB -> **91,61 MB** | 98,23% | **98,58% (4 err)** | **0,0414** | **0,9858** | **0,9985** | **0,9987** | `0,47` (98,58%) | 232,69 ms (108,1 img/s) | **90,00% (0,9956)** |

### Uji Statistik McNemar & Sensitivitas Threshold (`Exp-2B` vs `Exp-3B`)

- **McNemar Exact Two-Sided Test (`N = 282`):** `Exp-2B` dan `Exp-3B` sama-sama benar pada `275` citra, sama-sama salah pada `3` citra, `Exp-2B` unggul sendiri pada `1` citra, dan `Exp-3B` unggul sendiri pada `3` citra (`4` pasangan diskordan, **`p = 0,625`**). Secara statistik akurasi *in-distribution* keduanya setara, sehingga **`Exp-2B`** sah menjadi pemenang **Pareto Efisiensi** (`9,3x` lebih kecil setelah ekspor: `9,80 MB` vs `91,61 MB`; `1,45x` lebih cepat per gambar).
- **Kalibrasi & Threshold Tuning:** Model *scratch* yang *under-recall* pada `t = 0,50` membutuhkan ambang batas rendah (`t* = 0,11` untuk `Exp-1B`, F1 membaik `0,7705 -> 0,8344`). Sebaliknya, `Exp-2B` (`t* = 0,55`) dan `Exp-3B` (`t* = 0,47`) sudah terkalibrasi optimal langsung pada threshold default `0,50`.

---

## 5. Analisis Diagnostik Mendalam: Error Analysis, Grad-CAM & OOD Stress Test

### 5.1 Error Analysis & Sensitivitas Interpolasi Resize
- **Ketiadaan Bias Rasio Aspek pada Sisa Error:** Ketujuh sampel Test Set tempat `Exp-2B` atau `Exp-3B` berbuat salah (`2` Real Art / `5` AI Art) memiliki `mean aspect ratio = 0,969` (dibanding `1,010` pada `275` citra yang benar keduanya).
- **Temuan *Label Noise*:** Salah satu kesalahan `Exp-2B` (`P(AI)=0,014` vs `Exp-3B P(AI)=0,988`) di Test Set adalah gambar `600 x 625` JPEG berupa **meme komik digital 4-panel** bertema Stable Diffusion yang dilabeli sebagai `AI-Generated Art` di dataset asli. Kasus lain yang membingungkan kedua model berupa lukisan laut *synthetic-classical* yang sangat halus (`BOTH FN`: `2B=0,008`, `3B=0,019`) dan potret lukisan *chiaroscuro* kontras tinggi (`BOTH FP`).
- **Sensitivitas Kernel Interpolasi:** Saat arsip Colab (`224 x 224` hasil *downscale* `PIL.Image.Resampling.LANCZOS`) dibandingkan dengan validasi lokal (*bilinear* `tf.image.resize`), performa *scratch* `Exp-1A` (`79,43% -> 71,28%`) dan `Exp-1B` (`86,17% -> 80,50%`) turun tajam di Val Colab, sementara `Exp-2B` (`97,52%`) dan `Exp-3B` (`98,23% -> 98,58%`) stabil. Model berbasis ImageNet terbukti lebih invarian terhadap perbedaan kernel interpolasi.

### 5.2 Interpretabilitas Grad-CAM Pre-Sigmoid Logits (`RQ3`)
Menggunakan formulasi gradien terhadap **pre-sigmoid logit** (`W @ gap_out + b`) untuk menghindari kejenuhan gradien fungsi sigmoid (`P(AI) ~ 1.0` atau `0.0`), perbandingan peta panas antara `Exp-1B`, `Exp-2B`, dan `Exp-3B` menunjukkan:
- **Bukti Shortcut pada Scratch CNN (`Exp-1B`):** Pada citra karya berbingkai kayu tebal, `Exp-1B` mengaktifkan area tepi bingkai atas dan bawah secara dominan (*frame-border shortcut*).
- **Fokus Fitur Sejati pada Transfer Learning (`Exp-2B` & `Exp-3B`):** Kedua model Transfer Learning mengabaikan bingkai eksternal dan memfokuskan bobot aktivasi pada struktur arsitektur, pencahayaan volumetrik, sapuan kuas, serta transisi tekstur permukaan di bagian interior karya.

### 5.3 Out-of-Distribution (OOD) Stress Test per Sumber (`N = 30`, `RQ4`)

| Run ID | Model | Test Acc | OOD Acc (`N=30`) | Drop (pp) | OOD ROC-AUC | `huggan/wikiart` (15 Real) | Midjourney v6 (5 AI) | FLUX.1-Dev (5 AI) | DALL-E 3 (5 AI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp-1A** | Naive CNN | 69,50% | 73,33% | +3,83 pp | 0,7333 | 73,3% (`11/15`) | 40,0% (`2/5`) | 100,0% (`5/5`) | 80,0% (`4/5`) |
| **Exp-1B** | Proper CNN | 80,14% | 73,33% | -6,81 pp | 0,9156 | 53,3% (`8/15`) | 100,0% (`5/5`) | 100,0% (`5/5`) | 80,0% (`4/5`) |
| **Exp-2A** | MobileNetV2 Frozen | 93,26% | 83,33% | -9,93 pp | 0,9378 | 73,3% (`11/15`) | 80,0% (`4/5`) | 100,0% (`5/5`) | 100,0% (`5/5`) |
| **Exp-2B** | MobileNetV2 FT | 97,87% | 83,33% | -14,54 pp | 0,9600 | 73,3% (`11/15`) | 80,0% (`4/5`) | 100,0% (`5/5`) | 100,0% (`5/5`) |
| **Exp-3A** | ResNet50 Frozen | 98,23% | **90,00%** | -8,23 pp | **0,9956** | **80,0% (`12/15`)** | **100,0% (`5/5`)** | **100,0% (`5/5`)** | **100,0% (`5/5`)** |
| **Exp-3B** | ResNet50 FT | **98,58%** | **90,00%** | -8,58 pp | **0,9956** | **80,0% (`12/15`)** | **100,0% (`5/5`)** | **100,0% (`5/5`)** | **100,0% (`5/5`)** |

**Insight Kunci OOD:** Seluruh `3` kesalahan `Exp-3B` pada uji OOD adalah **False Positive pada lukisan museum WikiArt (`12/15`)**, bukan gagal mendeteksi generator modern (`15/15` benar di Midjourney v6, FLUX.1-Dev, dan DALL-E 3). Lukisan malam berbulan (`OOD-R12`, `P(AI)=0,984`) dan lukisan bunga berlatar merah gelap (`OOD-R11`, `P(AI)=0,933`) memiliki kontras pencahayaan halus yang oleh model dikira karakteristik *diffusion lighting*.

---

## 6. Deployment Packaging (`p5_export`) & Demo Interaktif (`p5_gradio`)

Pada **Phase 5**, kedua model produksi diekspor ulang sebagai arsip `.keras` **self-contained khusus inferensi** (*optimizer state* dibersihkan via `src/export_models.py`), menerima input mentah `float32 RGB [0..255]` berukuran `224 x 224` dan diverifikasi memiliki **paritas numerik sempurna (`max_abs_diff = 0.00e+00`)** terhadap checkpoint asli:

| Model Key | Run ID | File Ekspor Inferensi | Ukuran Asli -> Ekspor (Reduksi) | SHA-256 (Prefix 16) | Paritas (`N=5`) | Google Drive ID (`04_Models/`) |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| **`resnet50_ft`** | `Exp-3B` | `models/export/art_classifier_resnet50_ft.keras` | `224,75 MB` -> **`91,61 MB`** (**`-59,24%`**) | `e7d12a8183d4befe` | `0.00e+00` | `1Zuat2bQ_iM0iqb2zAKp8vouQlyGHoZu9` |
| **`mobilenetv2_ft`** | `Exp-2B` | `models/export/art_classifier_mobilenetv2_ft.keras` | `24,65 MB` -> **`9,80 MB`** (**`-60,22%`**) | `58113a282bf56438` | `0.00e+00` | `1auHmCzLn9UIB44nIBEs6YxnczzdwqOS3` |

Komponen aplikasi demo yang dibangun dan terverifikasi:
1. **Inference & Explainability Engine (`src/inference.py`):** Menyediakan `ArtClassifierEngine` dengan lazy caching model, `load_keras_model_compat()` lintas-versi Keras (`3.13` / `3.15`), pra-proses EXIF transpose + opsional `center_crop=True`, dan `compute_gradcam_heatmap()` berbasis *pre-sigmoid logits*.
2. **Gradio Interactive UI (`app.py`, Gradio `6.32.0`):** Antarmuka unggah gambar dengan pilihan model (`Exp-3B Primary` vs `Exp-2B Lightweight`), *slider threshold* `P(AI)`, toggle `Center-Crop 1:1`, *slider* opasitas heatmap Grad-CAM, tampilan citra berdampingan (`Input 224x224` vs `Grad-CAM Overlay`), bar probabilitas, dan kartu diagnostik latensi. Terverifikasi HTTP `200 OK` di lokal dan inferensi akurat pada sampel Test Set (`~200–241 ms` ResNet50, `~90 ms` MobileNetV2 di CPU).
3. **Colab Demo Launcher (`notebooks/05_phase5_demo_gradio.ipynb`):** Notebook peluncur Colab 5 langkah (`Gradio install -> GitHub clone -> salin export model dari Drive 04_Models -> smoke test Grad-CAM -> python app.py --share`), tersimpan di Drive `02_Notebooks/` (`1ac06t0JHi2grKGyPYAkuORKXQk8V5ER7`).

---

## 7. Limitasi & Rencana Pengembangan (*Future Work*)

1. **Domain Shift pada Lukisan Klasik Kontras Tinggi (WikiArt FP):** `Exp-3B` masih menghasilkan `20%` False Positive (`3/15`) pada lukisan museum gelap atau berpencahayaan dramatis saat threshold `0,50` dipakai di luar domain pelatihan. *Mitigasi ke depan:* kalibrasi *temperature scaling* dan penambahan dataset lukisan museum yang lebih beragam pada tahap *fine-tuning*.
2. **Kehilangan Detail Frekuensi Tinggi Akibat Resize `224 x 224`:** Input `224 x 224` membuang spektrum mikrotekstur dan jejak kompresi *high-resolution* pada lukisan berukuran besar (`> 1024 px`), sekaligus membuat model tanpa *pre-training* rentan terhadap metode interpolasi (`LANCZOS` vs `bilinear`).
3. **Analisis Domain Frekuensi (FFT / DCT):** Menambahkan cabang fitur **2D Fast Fourier Transform (FFT)** atau spektrum **DCT** (*hybrid spatial + frequency CNN / dual-branch network*) untuk menangkap artefak periodik *upsampling* generator difusi secara lebih langsung dan invarian terhadap pencahayaan artistik.
4. **Error Level Analysis (ELA) & Klasifikasi Multi-Patch:** Mengombinasikan peta residu **Error Level Analysis (ELA)** pada kanal input tambahan atau menggunakan pemungutan suara multi-crop (*multi-scale patch ensemble*) pada resolusi asli agar detail sapuan kuas kanvas tidak tergerus proses pengecilan gambar.
5. **Pembaruan Berkala Benchmark Generator Terbaru:** Walau meraih `100%` Recall terhadap sampel Midjourney v6, FLUX.1-Dev, dan DALL-E 3 pada uji OOD `N = 30`, evaluasi OOD perlu diperluas ke lebih banyak sampel (`> 200` citra) dan arsitektur generator terbaru serta karya *digital painting* buatan manusia (seperti ArtStation / DeviantArt) yang estetika warnanya dekat dengan ilustrasi AI.

---

## 8. Reproduksibilitas & Visualisasi Utama

Cara menjalankan pipeline, ekspor ulang model, dan meluncurkan demo secara lokal:

```bash
# 1. Setup environment Python 3.12 (macOS arm64 / Linux)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Ekspor model inference-only (.keras) + verifikasi paritas & SHA-256
python -m src.export_models

# 3. Jalankan aplikasi interaktif Gradio + Grad-CAM
python app.py --port 7860
```

Grafik kunci hasil eksperimen lintas fase yang disematkan pada bagian visualisasi laporan ini:

### 8.1 Ringkasan Ablasi & Kurva Pareto (Phase 2 + Phase 3)

### 8.2 Kurva ROC & Precision-Recall Final Test Set (N = 282)

### 8.3 Verifikasi Grad-CAM Pre-Sigmoid Logits (Scratch vs Transfer Learning)

### 8.4 Hasil OOD Stress Test Lintas Generator Modern (N = 30)
