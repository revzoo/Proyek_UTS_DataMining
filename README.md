# 🩺 Proyek UTS Data Mining: Prediksi Penyakit Berdasarkan Gejala
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/revzoo/Proyek_UTS_DataMining)

### 🌐 Live Demo (Streamlit Community Cloud)
Aplikasi ini sudah di-deploy dan dapat diakses secara langsung melalui tautan berikut:

👉 **[DEMO Klik disni](https://triasegejalamandiri.streamlit.app/)**

### 📓 Notebook Pelatihan Model (Google Colab)
Proses eksperimen, eksplorasi data, dan pelatihan model dapat dilihat pada tautan berikut:

👉 **[COLAB Klik disini](https://colab.research.google.com/drive/1V0GrvU5tZQbFhxFVfE33-Vm-44dprldr?usp=sharing)**

---

Proyek ini adalah aplikasi web sederhana berbasis **Streamlit** yang dibuat untuk memenuhi tugas Ujian Tengah Semester (UTS) mata kuliah **Proyek Data Mining**. 

Aplikasi ini menggunakan algoritma *Machine Learning* **Multinomial Naive Bayes** untuk mencoba memprediksi kemungkinan penyakit berdasarkan beberapa gejala yang diinputkan.

---

## 👨‍🎓 Identitas Mahasiswa
- **Nama**: Mukti Cahyo Pamungkas
- **NIM**: 23.11.5435

---

## 📌 Deskripsi Singkat
- 🧠 Model dilatih menggunakan algoritma Multinomial Naive Bayes.
- 🎯 **Akurasi Model: 95.00%** (berdasarkan hasil evaluasi klasifikasi pada 20 jenis penyakit).
- 📝 Input berupa pilihan gejala (sudah diterjemahkan ke bahasa Indonesia untuk memudahkan penggunaan).
- 📊 Output berupa 3 kemungkinan penyakit teratas berdasarkan perhitungan probabilitas dari model.
- ⚠️ *Catatan: Aplikasi ini dibuat murni untuk keperluan tugas kuliah dan eksperimen pemodelan data mining, bukan untuk diagnosis medis sungguhan.*

---

## 🚀 Cara Menjalankan Aplikasi

### 💻 Menjalankan Secara Lokal (Di Komputer Sendiri)
Jika ingin menjalankan aplikasi ini secara lokal:

1. Kloning repositori ini:
   ```bash
   git clone https://github.com/revzoo/Proyek_UTS_DataMining.git
   cd Proyek_UTS_DataMining
   ```

2. Install library pendukung:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi menggunakan Streamlit:
   ```bash
   streamlit run app.py
   ```
