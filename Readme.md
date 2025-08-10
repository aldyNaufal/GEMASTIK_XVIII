# Prediksi Produksi Padi dan Beras dengan Multi-Output Regression

## 📌 Deskripsi
Penelitian ini bertujuan untuk membangun model prediksi produksi **padi** dan **beras** secara simultan menggunakan pendekatan **Multi-Output Regression** dengan algoritma inti **Random Forest Regressor** dan **XGBoost Regressor**.

Data yang digunakan terdiri dari:
- **Fitur Utama**: Intensitas curah hujan dan kelembapan tanah (sumber: HDX)
- **Target**: Produksi padi dan beras (sumber: Kementerian Pertanian)

Hasil penelitian diharapkan menjadi dasar pengembangan sistem rekomendasi tanaman pangan adaptif untuk membantu petani dan pembuat kebijakan dalam mengambil keputusan berbasis prediksi iklim.

---

## 🗂 Dataset
1. **Intensitas Curah Hujan** (HDX - CHIRPS & CHIRPS-GEFS)
2. **Kelembapan Tanah** (HDX)
3. **Produksi Padi dan Beras** (Kementerian Pertanian)

---

## ⚙️ Metodologi
1. **Pembersihan & Penggabungan Data**
2. **Analisis Eksploratori Data (EDA)**  
   - Mengidentifikasi fluktuasi musiman
   - Mendeteksi hubungan non-linear antara variabel iklim dan produksi
3. **Pemilihan Algoritma**
   - **Random Forest Regressor**
   - **XGBoost Regressor**
4. **Pencarian Parameter Optimal**
   - Menggunakan `MultiOutputRegressor` + `RandomizedSearchCV`
5. **Pembagian Data**
   - **Train set**: 80%
   - **Test set**: 20%
6. **Evaluasi Model**
   - Mean Squared Error (MSE)
   - Koefisien Determinasi (R²)

---

## 🧮 Algoritma yang Digunakan
- **Random Forest Regressor**  
  Membuat banyak pohon keputusan dan menggabungkan hasil prediksi untuk mengurangi variansi.
- **XGBoost Regressor**  
  Menggunakan boosting dan regularisasi untuk meningkatkan akurasi dan efisiensi.

---

## 📊 Hasil Evaluasi

### Tabel I — Random Forest Regressor
| Metrik               | Skor               |
|----------------------|--------------------|
| Mean Squared Error   | 3.8179 × 10¹²       |
| R-squared (R²)       | -0.4103             |

**Interpretasi:**  
- MSE tinggi → selisih kuadrat prediksi dan nilai aktual besar  
- R² negatif → kinerja lebih buruk dari baseline (rata-rata target)

---

### Tabel II — XGBoost Regressor
| Metrik               | Skor               |
|----------------------|--------------------|
| Mean Squared Error   | 2.6987 × 10¹²       |
| R-squared (R²)       | -0.0122             |

**Interpretasi:**  
- MSE lebih rendah dari Random Forest → sedikit perbaikan akurasi
- R² mendekati nol → kemampuan menjelaskan variasi target masih rendah

---

## 📌 Kesimpulan
- Kedua model **belum memberikan performa memuaskan** untuk memprediksi produksi padi dan beras dari variabel iklim.
- Kemungkinan penyebab:
  - Korelasi fitur-target rendah
  - Keterbatasan data dan resolusi bulanan
  - Tidak dimasukkannya faktor non-iklim (misalnya hama, kualitas tanah)
- Potensi pengembangan:
  - Menambah variabel non-iklim
  - Menggunakan metode ensemble hybrid
  - Integrasi ke dashboard interaktif real-time

---

## 🔗 Repository
Hasil training model dapat diakses di:  
[https://github.com/aldyNaufal/GEMASTIK_XVIII](https://github.com/aldyNaufal/GEMASTIK_XVIII)

