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


def terjemahkan_gejala(gejala_en):
    kamus = {
        'abdominal_pain': 'Sakit Perut',
        'anxiety': 'Kecemasan',
        'aura': 'Aura (Gejala Awal Migrain)',
        'back_pain': 'Sakit Punggung',
        'belching': 'Sering Sendawa',
        'blisters': 'Lepuh',
        'bloating': 'Perut Kembung',
        'blurred_vision': 'Penglihatan Kabur',
        'body_ache': 'Badan Pegal-pegal',
        'burning_urination': 'Buang Air Kecil Terasa Panas',
        'chest_discomfort': 'Dada Terasa Tidak Nyaman',
        'chest_pain': 'Nyeri Dada',
        'chest_tightness': 'Dada Terasa Sesak',
        'chills': 'Menggigil',
        'cloudy_urine': 'Urin Keruh',
        'cold_hands_feet': 'Tangan dan Kaki Dingin',
        'constipation': 'Sembelit',
        'cough': 'Batuk',
        'dark_urine': 'Urin Gelap',
        'depression': 'Depresi',
        'difficulty_breathing': 'Kesulitan Bernapas',
        'dizziness': 'Pusing / Kleyengan',
        'dry_cough': 'Batuk Kering',
        'dry_skin': 'Kulit Kering',
        'ear_pain': 'Sakit Telinga',
        'fast_heartbeat': 'Detak Jantung Cepat',
        'fatigue': 'Kelelahan',
        'fever': 'Demam',
        'frequent_urination': 'Sering Buang Air Kecil',
        'headache': 'Sakit Kepala',
        'heartburn': 'Nyeri Ulu Hati (Heartburn)',
        'high_fever': 'Demam Tinggi',
        'hunger': 'Mudah Lapar',
        'increased_thirst': 'Sering Haus',
        'insomnia': 'Insomnia / Sulit Tidur',
        'itching': 'Gatal-gatal',
        'jaundice': 'Penyakit Kuning (Jaundice)',
        'joint_pain': 'Nyeri Sendi',
        'limited_movement': 'Gerakan Terbatas',
        'limited_range_of_motion': 'Rentang Gerak Terbatas',
        'loss_of_appetite': 'Kehilangan Nafsu Makan',
        'lower_abdominal_pain': 'Sakit Perut Bagian Bawah',
        'mild_fever': 'Demam Ringan',
        'muscle_pain': 'Nyeri Otot',
        'nasal_congestion': 'Hidung Tersumbat',
        'nausea': 'Mual',
        'night_sweats': 'Keringat Malam',
        'nocturnal_symptoms': 'Gejala Memburuk di Malam Hari',
        'nosebleed': 'Mimisan',
        'pale_skin': 'Kulit Pucat',
        'palpitations': 'Jantung Berdebar',
        'persistent_cough': 'Batuk Berkepanjangan',
        'phonophobia': 'Sensitif Terhadap Suara',
        'photophobia': 'Sensitif Terhadap Cahaya',
        'productive_cough': 'Batuk Berdahak',
        'prolonged_fever': 'Demam Berkepanjangan',
        'rash': 'Ruam',
        'rebound_tenderness': 'Nyeri Lepas (Perut)',
        'red_rash': 'Ruam Merah',
        'redness': 'Kemerahan',
        'right_lower_abdominal_pain': 'Sakit Perut Kanan Bawah',
        'rose_spots': 'Bintik Kemerahan',
        'runny_nose': 'Hidung Meler',
        'scaly_skin': 'Kulit Bersisik',
        'sensitivity': 'Sensitif / Nyeri Saat Ditekan',
        'shortness_of_breath': 'Sesak Napas',
        'slow_healing': 'Luka Sulit Sembuh',
        'sneezing': 'Bersin-bersin',
        'sore_throat': 'Sakit Tenggorokan',
        'stiffness': 'Kaku',
        'sudden_joint_pain': 'Nyeri Sendi Tiba-tiba',
        'sweating': 'Berkeringat',
        'swelling': 'Pembengkakan',
        'tenderness': 'Nyeri Saat Ditekan',
        'throbbing_headache': 'Sakit Kepala Berdenyut',
        'urgency': 'Keinginan Kuat Buang Air Kecil',
        'vomiting': 'Muntah',
        'warmth_over_joint': 'Terasa Hangat di Area Sendi',
        'weight_loss': 'Penurunan Berat Badan',
        'wheezing': 'Napas Berbunyi (Mengi)'
    }
    if gejala_en in kamus:
        return kamus[gejala_en]
    return gejala_en.replace('_', ' ').title()


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
    format_func=terjemahkan_gejala,
    help="Ketik gejala yang Anda rasakan untuk mencari...",
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