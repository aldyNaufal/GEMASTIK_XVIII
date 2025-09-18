# 🌾 Prediksi Produksi Padi & Beras Indonesia Menggunakan Multi-Output Regression

**Analisis Dampak Iklim Terhadap Ketahanan Pangan dengan Machine Learning**

---



## 📌 1. Domain Proyek: Pertanian Presisi & Ketahanan Pangan

Indonesia, sebagai salah satu negara agraris terbesar di dunia, sangat bergantung pada produksi padi untuk menjaga **ketahanan pangan** nasional. Padi dan beras bukan hanya komoditas ekonomi, tetapi juga pilar stabilitas sosial dan politik. Namun, sektor pertanian dihadapkan pada tantangan besar akibat **perubahan iklim**, yang menyebabkan pergeseran pola curah hujan, peningkatan frekuensi cuaca ekstrem, dan perubahan kelembapan tanah. Fenomena ini secara langsung memengaruhi siklus tanam dan hasil panen, menciptakan ketidakpastian bagi jutaan petani dan mengancam pasokan pangan nasional.

Data dari Badan Pusat Statistik (BPS) menunjukkan bahwa fluktuasi produksi padi seringkali berkorelasi dengan anomali iklim. Dalam konteks ini, kemampuan untuk **memprediksi hasil panen** berdasarkan data iklim menjadi sangat krusial. Prediksi yang akurat dapat menjadi dasar bagi pemerintah untuk merumuskan kebijakan impor/ekspor, stabilisasi harga, dan alokasi bantuan. Bagi petani, informasi ini dapat membantu dalam pengambilan keputusan terkait jadwal tanam, pemilihan varietas, dan manajemen irigasi.

Proyek ini bertujuan untuk menjawab tantangan tersebut dengan membangun model *machine learning* yang mampu memprediksi **produksi padi dan beras secara simultan** berdasarkan dua variabel iklim kunci: **intensitas curah hujan** dan **kelembapan tanah**. Dengan menggunakan pendekatan **Multi-Output Regression**, model ini diharapkan dapat menangkap hubungan kompleks antara iklim dan hasil pertanian, serta menjadi fondasi untuk sistem pertanian presisi yang lebih adaptif dan berbasis data.

---

## 🎯 2. Business Understanding

### 🔍 Problem Statements

1.  Bagaimana cara membangun model tunggal yang mampu memprediksi dua target yang saling terkait—**produksi padi** (hasil panen mentah) dan **produksi beras** (hasil olahan)—secara bersamaan berdasarkan data iklim?
2.  Seberapa besar pengaruh variabel iklim, khususnya **intensitas curah hujan** dan **kelembapan tanah**, terhadap fluktuasi produksi padi dan beras di Indonesia pada skala bulanan?
3.  Algoritma *ensemble learning* manakah, **Random Forest** atau **XGBoost**, yang memberikan performa lebih baik dalam kerangka *Multi-Output Regression* untuk kasus prediksi agrikultur?

### 🎯 Objectives

1.  Mengimplementasikan dan membandingkan dua model **Multi-Output Regression** berbasis **Random Forest Regressor** dan **XGBoost Regressor** untuk prediksi produksi padi dan beras.
2.  Mengevaluasi kinerja model secara kuantitatif menggunakan metrik **Mean Squared Error (MSE)** dan **Koefisien Determinasi (R²)** untuk mengukur akurasi dan kemampuan model dalam menjelaskan variasi data.
3.  Menganalisis keterbatasan model yang hanya berbasis data iklim dan memberikan rekomendasi strategis untuk pengembangan sistem prediksi yang lebih holistik dan akurat di masa depan.

### 💡 Solusi yang Diusulkan

Menerapkan pendekatan **Multi-Output Regression**. Alih-alih membangun dua model terpisah untuk padi dan beras, pendekatan ini menggunakan *wrapper* `MultiOutputRegressor` dari Scikit-learn yang melatih satu model regresi inti (seperti Random Forest atau XGBoost) untuk setiap target secara independen namun dalam satu *pipeline* yang terintegrasi. Ini menyederhanakan proses pemodelan dan memungkinkan evaluasi gabungan.

---

## 📁 3. Dataset Overview

* **Sumber Data Iklim**: Humanitarian Data Exchange (HDX), mencakup data CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) dan CHIRPS-GEFS untuk curah hujan, serta data kelembapan tanah.
* **Sumber Data Produksi**: Kementerian Pertanian Republik Indonesia, berisi data historis produksi padi dan beras bulanan.
* **Granularitas**: Data diagregasi dalam skala waktu **bulanan**.

---

## 📋 4. Fitur & Target Dataset

### 📥 Fitur Input (Prediktor)

| Fitur | Deskripsi |
| :--- | :--- |
| `intensitas_curah_hujan` | Rata-rata curah hujan bulanan (mm), indikator utama ketersediaan air untuk tanaman. |
| `kelembapan_tanah` | Rata-rata tingkat kelembapan tanah bulanan (%), memengaruhi kesehatan akar dan penyerapan nutrisi. |

### 🎯 Target Output (yang Diprediksi)

| Target | Deskripsi |
| :--- | :--- |
| `produksi_padi` | Total produksi padi (gabah kering giling) dalam satuan ton. |
| `produksi_beras` | Total produksi beras (hasil setelah penggilingan) dalam satuan ton. |

---

## 🔍 5. Data Understanding & EDA

Analisis data eksplorasi (EDA) dilakukan untuk memahami pola dan hubungan dalam data sebelum pemodelan.

* **Analisis Runtun Waktu (Time Series)**: Visualisasi data produksi menunjukkan adanya **pola musiman** yang jelas, dengan puncak panen raya pada bulan-bulan tertentu. Pola ini sesuai dengan siklus tanam padi di Indonesia.
* **Analisis Korelasi**: Diagram sebar (scatter plot) antara fitur iklim dan target produksi menunjukkan hubungan yang **tidak linear dan cenderung lemah**. Tidak ditemukan korelasi garis lurus yang kuat antara curah hujan/kelembapan tanah dengan jumlah produksi.



* **Insight Kunci**: Korelasi yang lemah ini menjadi **indikasi awal** bahwa variabel iklim saja mungkin **tidak cukup informatif** untuk menjelaskan seluruh variasi dalam produksi padi. Faktor-faktor lain yang tidak ada dalam dataset (seperti penggunaan pupuk, serangan hama, atau luas tanam) kemungkinan besar memainkan peran yang sangat signifikan. Hal ini memberi sinyal bahwa model mungkin akan kesulitan untuk mencapai akurasi tinggi.

---

## 🧹 6. Data Preparation

| Langkah | Penjelasan |
| :--- | :--- |
| **Pembersihan Data** | Menangani nilai yang hilang (*missing values*) pada data iklim atau produksi, jika ada, dengan metode imputasi yang sesuai (misalnya, *forward fill* atau rata-rata). |
| **Penggabungan Data** | Menggabungkan dataset iklim dan dataset produksi menjadi satu *dataframe* terpadu berdasarkan kolom waktu (bulan dan tahun). |
| **Train-Test Split** | Membagi dataset secara acak menjadi **80% data latih** untuk melatih model dan **20% data uji** untuk mengevaluasi performanya pada data baru, dengan `random_state` untuk reproduktifitas. |

---

## ⚙️ 7. Model Development

### 🔑 **Pendekatan Kunci: Multi-Output Regression**

Pendekatan ini memungkinkan satu model untuk memprediksi beberapa variabel target secara simultan. Kerangka `MultiOutputRegressor` dari Scikit-learn bekerja dengan cara melatih satu regressor independen untuk setiap target. Jadi, ketika kita menggunakan `MultiOutputRegressor(XGBoostRegressor())`, sistem akan melatih satu model XGBoost untuk memprediksi `produksi_padi` dan satu model XGBoost lain untuk memprediksi `produksi_beras`.

### 🔹 Model 1: **Random Forest Regressor**

* **Prinsip**: Bekerja dengan membangun banyak pohon keputusan (*decision trees*) secara paralel pada sampel data yang berbeda (*bagging*). Prediksi akhir adalah rata-rata dari prediksi semua pohon, yang membantu mengurangi *overfitting* dan meningkatkan stabilitas.

### 🔹 Model 2: **XGBoost Regressor**

* **Prinsip**: Merupakan implementasi canggih dari *gradient boosting*. Model ini membangun pohon keputusan secara berurutan, di mana setiap pohon baru dilatih untuk memperbaiki kesalahan (*residual*) dari pohon sebelumnya. Dilengkapi dengan teknik regularisasi untuk mencegah *overfitting* dan seringkali memberikan akurasi yang lebih tinggi.

### 🔧 **Pencarian Parameter Optimal (Hyperparameter Tuning)**

Untuk kedua model, `RandomizedSearchCV` digunakan untuk secara efisien mencari kombinasi hyperparameter terbaik (seperti `n_estimators`, `max_depth`, `learning_rate`) yang menghasilkan error prediksi terendah.

---

## 📏 8. Evaluation

### 🎯 Tujuan Evaluasi

Evaluasi bertujuan untuk mengukur seberapa akurat prediksi model dibandingkan dengan data aktual dan untuk memahami apakah model tersebut lebih baik daripada sekadar tebakan sederhana.

### 📐 Metrik Evaluasi

| Metrik | Rumus | Keterangan |
| :--- | :--- | :--- |
| **MSE** | $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | **Mean Squared Error**. Mengukur rata-rata kuadrat selisih antara nilai aktual ($y_i$) dan prediksi ($\hat{y}_i$). Semakin kecil, semakin baik. |
| **R² Score** | $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **Koefisien Determinasi**. Mengukur seberapa baik model dapat menjelaskan variasi data. Nilai 1 sempurna, 0 berarti setara dengan menebak rata-rata, dan **negatif berarti model lebih buruk daripada menebak rata-rata**. |

---

## 📊 9. Hasil Evaluasi & Analisis

Hasil evaluasi pada data uji menunjukkan performa kedua model yang **jauh dari memuaskan**.

### 📉 **Random Forest Regressor**

| Metrik | Skor |
| :--- | :--- |
| Mean Squared Error | 3.8179 × 10¹² |
| R-squared (R²) | **-0.4103** |

* **Analisis**: Nilai MSE yang sangat besar menunjukkan error prediksi yang tinggi. Yang lebih penting, **skor R² negatif (-0.41)** secara definitif menyatakan bahwa performa model ini **lebih buruk daripada sekadar memprediksi nilai rata-rata produksi**. Ini adalah indikator kuat bahwa model gagal menangkap pola yang berguna dari data.

### 📉 **XGBoost Regressor**

| Metrik | Skor |
| :--- | :--- |
| Mean Squared Error | 2.6987 × 10¹² |
| R-squared (R²) | **-0.0122** |

* **Analisis**: XGBoost menunjukkan MSE yang sedikit lebih rendah daripada Random Forest, menandakan perbaikan minor. Namun, **skor R² yang masih negatif (-0.01)** menegaskan bahwa model ini juga gagal total. Performanya hanya sedikit lebih baik dari baseline, tetapi tetap tidak dapat digunakan untuk prediksi yang andal.

---

## 💡 10. Pembahasan & Rekomendasi

Kegagalan kedua model untuk menghasilkan prediksi yang valid (ditandai dengan R² negatif) memberikan pelajaran penting tentang kompleksitas pemodelan agrikultur.

### 📉 **Analisis Kegagalan Model**

1.  **Korelasi Fitur-Target Sangat Rendah**: Seperti yang diindikasikan pada tahap EDA, hubungan antara curah hujan/kelembapan tanah bulanan dengan total produksi sangat lemah. Model tidak dapat menemukan sinyal yang cukup kuat dari fitur yang tersedia.
2.  **Keterbatasan Fitur (Missing Variable Bias)**: Model ini **tidak memperhitungkan faktor-faktor non-iklim** yang krusial, seperti:
    * **Luas Lahan Tanam**: Variabel paling fundamental yang memengaruhi total produksi.
    * **Penggunaan Pupuk dan Pestisida**: Sangat memengaruhi produktivitas per hektar.
    * **Serangan Hama dan Penyakit**: Dapat menyebabkan gagal panen massal.
    * **Kebijakan Pemerintah**: Subsidi pupuk, program irigasi, dll.
3.  **Resolusi Data Bulanan**: Data bulanan mungkin terlalu kasar (*coarse*) untuk menangkap dampak cuaca pada fase kritis pertumbuhan tanaman (misalnya, hujan lebat selama beberapa hari pada fase pembungaan).

### 🚀 **Rekomendasi Strategis untuk Pengembangan Selanjutnya**

1.  **Tambahkan Fitur Non-Iklim**: Langkah paling prioritas adalah mengintegrasikan data **luas lahan tanam**, **penggunaan pupuk**, data **serangan hama (OPT)**, dan bahkan **data harga gabah** sebagai fitur tambahan.
2.  **Gunakan Data Resolusi Tinggi**: Jika memungkinkan, gunakan data iklim harian atau mingguan untuk menangkap peristiwa cuaca yang lebih spesifik dan dampaknya pada siklus tanam.
3.  **Eksplorasi Model Time-Series**: Mengingat data memiliki komponen waktu yang kuat, model yang dirancang khusus untuk data runtun waktu seperti **ARIMA**, **SARIMA**, atau **LSTM (Long Short-Term Memory)** dapat memberikan hasil yang lebih baik.
4.  **Kembangkan Dashboard Interaktif**: Sebagai tujuan akhir, model yang telah divalidasi dapat diintegrasikan ke dalam *dashboard* interaktif yang menampilkan prediksi dan memberikan rekomendasi adaptif bagi para pemangku kepentingan.

---

## ✅ 11. Kesimpulan

Penelitian ini menunjukkan bahwa model prediksi produksi padi dan beras yang **hanya mengandalkan data iklim (curah hujan dan kelembapan tanah) tidaklah cukup** untuk menghasilkan prediksi yang akurat dan andal di Indonesia. Kedua model yang diuji, **Random Forest** dan **XGBoost**, gagal total dalam tugas ini, yang dibuktikan dengan **skor R² negatif** pada kedua model.

Meskipun modelnya gagal, proyek ini berhasil memberikan sebuah wawasan penting: **sistem pertanian adalah ekosistem yang kompleks** di mana iklim hanyalah salah satu dari banyak faktor yang saling berinteraksi. Kegagalan ini bukan akhir, melainkan sebuah titik awal yang valid, yang mengarahkan penelitian di masa depan untuk membangun model yang lebih holistik dengan mengintegrasikan variabel agronomis, ekonomis, dan kebijakan untuk menciptakan alat prediksi ketahanan pangan yang benar-benar bermanfaat.
