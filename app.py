import streamlit as st
import joblib
import numpy as np
import os


# 1. Konfigurasi Halaman & UI
st.set_page_config(
    page_title="Sistem Deteksi Penyakit",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# 2. Fungsi Memuat Model (dengan Cache)
@st.cache_resource
def load_models():
    """
    Memuat file .pkl yang diekspor dari Google Colab.
    Gunakan st.cache_resource agar model tidak di-load ulang setiap interaksi.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model_artifacts")
    
    # Path file
    model_path = os.path.join(model_dir, "model_naive_bayes.pkl")
    mlb_path = os.path.join(model_dir, "mlb_encoder.pkl")
    feature_path = os.path.join(model_dir, "symptom_features.pkl")
    
    # Load menggunakan joblib
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    symptom_features = joblib.load(feature_path)
    
    return model, mlb, symptom_features

# Tangkap error jika file tidak ditemukan
try:
    model, mlb, valid_symptoms = load_models()
except Exception as e:
    st.error("❌ **Gagal memuat model Machine Learning!**")
    st.info("Pastikan folder `model_artifacts` beserta isi file `.pkl`-nya berada di lokasi yang sama dengan `app.py`.")
    st.exception(e)
    st.stop()


# 3. Antarmuka Web Utama (Frontend)
st.title("🩺 Asisten Triase Gejala Mandiri")
st.markdown("""
Selamat datang di Sistem Rekomendasi Penyakit Berbasis Gejala Ringan.  
Aplikasi ini memproses gejala yang Anda rasakan menggunakan algoritma **Multinomial Naive Bayes** untuk memprediksi probabilitas penyakit.
""")

st.divider()

st.subheader("📝 Formulir Pengecekan Gejala")
# Widget multiselect lebih rapi daripada menampilkan 80 checkbox
gejala_input = st.multiselect(
    label="Pilih gejala yang sedang Anda alami:",
    options=valid_symptoms,
    help="Ketik untuk mencari gejala (misal: fever, headache, nausea)",
    placeholder="Pilih satu atau lebih gejala..."
)


# 4. Logika Prediksi (Backend)
if st.button("Analisis Penyakit", type="primary", use_container_width=True):
    if not gejala_input:
        st.warning("⚠️ Silakan pilih minimal satu gejala untuk dianalisis!")
    else:
        with st.spinner("Model sedang memproses data..."):
            # A. Encoding input gejala menjadi vektor biner menggunakan MLB
            # Logika ini sama persis dengan fungsi predict_disease() di Colab
            input_vector = mlb.transform([gejala_input])
            
            # B. Kalkulasi probabilitas
            probabilitas = model.predict_proba(input_vector)[0]
            classes = model.classes_
            
            # C. Ambil 3 probabilitas tertinggi
            top_indices = np.argsort(probabilitas)[::-1][:3]
            
            # Ekstraksi hasil
            prediksi_utama = classes[top_indices[0]]
            prob_utama = probabilitas[top_indices[0]] * 100
            
            # 5. Menampilkan Hasil Prediksi
            st.divider()
            st.subheader("📋 Hasil Diagnosis Sistem")
            
            # Tampilan prediksi utama yang menonjol
            st.success(f"### 🛑 Kemungkinan Terbesar: **{prediksi_utama}**")
            st.metric(label="Tingkat Keyakinan Model (Probabilitas)", value=f"{prob_utama:.2f}%")
            
            # Tampilan prediksi alternatif
            st.markdown("#### Kemungkinan Lainnya (Top 3):")
            
            # Tampilkan ranking 2 dan 3
            for i in top_indices[1:]:
                nama_penyakit = classes[i]
                prob_persen = probabilitas[i] * 100
                
                # Hanya tampilkan jika probabilitasnya relevan (> 0%)
                if prob_persen > 0:
                    st.write(f"- **{nama_penyakit}** ({prob_persen:.2f}%)")
            
            # Disclaimer medis (Wajib ada untuk etika aplikasi kesehatan)
            st.caption("---")
            st.info("""
            **Peringatan:** Hasil prediksi ini dihasilkan oleh model *Machine Learning* untuk keperluan tugas akademik (Proyek Data Mining) dan **TIDAK** dapat menggantikan diagnosis, nasihat, atau perawatan medis dari dokter profesional.
            """)