# Streamlit CRNN Multimodal 8 Kelas

Aplikasi menggunakan satu model Keras:

- `models/model_lr01f12_best.keras`
- learning rate `0.0001`;
- 12 frame visual;
- validation accuracy `79.86%`;
- test accuracy `75%`;
- skala fusion saat training: visual `0.5` dan audio `0.5`.

Model merupakan HDF5 legacy TensorFlow/Keras 2.10 yang disimpan dengan ekstensi
`.keras`. Aplikasi membangun ulang arsitektur yang sama agar dapat dimuat pada
Keras 3.

## Fitur

- prediksi webcam dan mikrofon tanpa batas waktu;
- kotak wajah, label emosi, confidence, dan probabilitas delapan kelas;
- slider fokus visual-suara saat inferensi;
- interval prediksi live 1-5 detik;
- unggah video dan analisis timeline per segmen;
- unduh hasil lengkap ke CSV.

Slider fokus bukan `class_weight` training. Posisi 50/50 mempertahankan skala
asli model, yaitu `0.5` untuk cabang visual dan `0.5` untuk cabang audio.

## Menjalankan secara lokal

```powershell
conda activate env_skripsi
cd C:\streamlit_crnn_multimodal
streamlit run streamlit_app.py
```

Atau:

```powershell
powershell -ExecutionPolicy Bypass -File C:\streamlit_crnn_multimodal\scripts\run_streamlit.ps1
```

## Smoke test

```powershell
conda activate env_skripsi
cd C:\streamlit_crnn_multimodal
python scripts\smoke_test.py
```

Untuk memeriksa satu video:

```powershell
python scripts\smoke_test.py --video "C:\path\video.mp4"
```

## Deploy ke Streamlit Community Cloud

1. Unggah seluruh proyek ke sebuah repositori GitHub, termasuk folder `models`.
2. Pastikan main file adalah `streamlit_app.py`.
3. Hubungkan repositori tersebut di Streamlit Community Cloud.
4. Tunggu instalasi paket dari `requirements.txt` dan pemuatan model selesai.

Webcam dan mikrofon browser memerlukan deployment HTTPS. Streamlit Community
Cloud menyediakan HTTPS secara otomatis.

## Preprocessing

- visual: 12 frame, Haar Cascade, margin wajah 10%, RGB 64 x 64, normalisasi 0-1;
- audio: mono 22.050 Hz, 40 MFCC, `n_fft=1024`, window 551 sampel,
  hop 220 sampel, panjang target 531 langkah waktu;
- normalisasi MFCC menggunakan mean dan standard deviation training;
- kelas: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Model dilatih pada ekspresi terkontrol RAVDESS. Hasil dunia nyata dapat berbeda
dan tidak dimaksudkan sebagai diagnosis psikologis.
