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
        'itching': 'Gatal', 'skin_rash': 'Ruam Kulit', 'nodal_skin_eruptions': 'Benjolan Kulit',
        'continuous_sneezing': 'Bersin Terus-menerus', 'shivering': 'Gemetar', 'chills': 'Menggigil',
        'joint_pain': 'Nyeri Sendi', 'stomach_pain': 'Sakit Perut', 'acidity': 'Asam Lambung',
        'ulcers_on_tongue': 'Sariawan di Lidah', 'muscle_wasting': 'Otot Mengecil', 'vomiting': 'Muntah',
        'burning_micturition': 'Buang Air Kecil Terasa Panas', 'spotting_ urination': 'Bercak Saat Buang Air Kecil',
        'fatigue': 'Kelelahan', 'weight_gain': 'Berat Badan Naik', 'anxiety': 'Kecemasan',
        'cold_hands_and_feets': 'Tangan dan Kaki Dingin', 'mood_swings': 'Perubahan Mood',
        'weight_loss': 'Berat Badan Turun', 'restlessness': 'Gelisah', 'lethargy': 'Lesu',
        'patches_in_throat': 'Bercak di Tenggorokan', 'irregular_sugar_level': 'Gula Darah Tidak Teratur',
        'cough': 'Batuk', 'high_fever': 'Demam Tinggi', 'sunken_eyes': 'Mata Cekung',
        'breathlessness': 'Sesak Napas', 'sweating': 'Berkeringat', 'dehydration': 'Dehidrasi',
        'indigestion': 'Gangguan Pencernaan', 'headache': 'Sakit Kepala', 'yellowish_skin': 'Kulit Menguning',
        'dark_urine': 'Urin Gelap', 'nausea': 'Mual', 'loss_of_appetite': 'Hilang Nafsu Makan',
        'pain_behind_the_eyes': 'Sakit di Belakang Mata', 'back_pain': 'Sakit Punggung', 'constipation': 'Sembelit',
        'abdominal_pain': 'Sakit Perut Bagian Bawah', 'diarrhoea': 'Diare', 'mild_fever': 'Demam Ringan',
        'yellow_urine': 'Urin Kuning', 'yellowing_of_eyes': 'Mata Menguning', 'acute_liver_failure': 'Gagal Hati Akut',
        'swelling_of_stomach': 'Perut Bengkak', 'swelled_lymph_nodes': 'Kelenjar Getah Bening Bengkak',
        'malaise': 'Tidak Enak Badan', 'blurred_and_distorted_vision': 'Penglihatan Kabur',
        'phlegm': 'Berlendir/Dahak', 'throat_irritation': 'Iritasi Tenggorokan', 'redness_of_eyes': 'Mata Merah',
        'sinus_pressure': 'Tekanan Sinus', 'runny_nose': 'Hidung Meler', 'congestion': 'Hidung Tersumbat',
        'chest_pain': 'Nyeri Dada', 'weakness_in_limbs': 'Tungkai Lemah', 'fast_heart_rate': 'Detak Jantung Cepat',
        'pain_during_bowel_movements': 'Sakit Saat BAB', 'pain_in_anal_region': 'Sakit di Area Anus',
        'bloody_stool': 'BAB Berdarah', 'irritation_in_anus': 'Iritasi di Anus', 'neck_pain': 'Sakit Leher',
        'dizziness': 'Pusing', 'cramps': 'Kram', 'bruising': 'Memar', 'obesity': 'Obesitas',
        'swollen_legs': 'Kaki Bengkak', 'swollen_blood_vessels': 'Pembuluh Darah Bengkak',
        'puffy_face_and_eyes': 'Wajah dan Mata Bengkak', 'enlarged_thyroid': 'Tiroid Membesar',
        'brittle_nails': 'Kuku Rapuh', 'swollen_extremeties': 'Ekstremitas Bengkak',
        'excessive_hunger': 'Rasa Lapar Berlebihan', 'extra_marital_contacts': 'Kontak Seksual Ekstramarital',
        'drying_and_tingling_lips': 'Bibir Kering dan Kesemutan', 'slurred_speech': 'Bicara Cadel',
        'knee_pain': 'Sakit Lutut', 'hip_joint_pain': 'Sakit Sendi Panggul', 'muscle_weakness': 'Otot Lemah',
        'stiff_neck': 'Leher Kaku', 'swelling_joints': 'Sendi Bengkak', 'movement_stiffness': 'Kaku Bergerak',
        'spinning_movements': 'Gerakan Berputar (Vertigo)', 'loss_of_balance': 'Kehilangan Keseimbangan',
        'unsteadiness': 'Goyah', 'weakness_of_one_body_side': 'Satu Sisi Tubuh Lemah', 'loss_of_smell': 'Kehilangan Penciuman',
        'bladder_discomfort': 'Ketidaknyamanan Kandung Kemih', 'foul_smell_of urine': 'Urin Berbau Busuk',
        'continuous_feel_of_urine': 'Rasa Ingin Buang Air Kecil Terus', 'passage_of_gases': 'Sering Buang Gas',
        'internal_itching': 'Gatal Internal', 'toxic_look_(typhos)': 'Terlihat Sangat Sakit (Tifus)',
        'depression': 'Depresi', 'irritability': 'Mudah Marah', 'muscle_pain': 'Nyeri Otot',
        'altered_sensorium': 'Kesadaran Menurun', 'red_spots_over_body': 'Bintik Merah di Tubuh',
        'belly_pain': 'Nyeri Perut', 'abnormal_menstruation': 'Menstruasi Tidak Normal',
        'dischromic _patches': 'Bercak Perubahan Warna Kulit', 'watering_from_eyes': 'Mata Berair',
        'increased_appetite': 'Nafsu Makan Meningkat', 'polyuria': 'Sering Buang Air Kecil',
        'family_history': 'Riwayat Keluarga', 'mucoid_sputum': 'Dahak Kental',
        'rusty_sputum': 'Dahak Berwarna Karat', 'lack_of_concentration': 'Kurang Konsentrasi',
        'visual_disturbances': 'Gangguan Penglihatan', 'receiving_blood_transfusion': 'Menerima Transfusi Darah',
        'receiving_unsterile_injections': 'Menerima Suntikan Tidak Steril', 'coma': 'Koma',
        'stomach_bleeding': 'Pendarahan Lambung', 'distention_of_abdomen': 'Perut Membengkak',
        'history_of_alcohol_consumption': 'Riwayat Konsumsi Alkohol', 'fluid_overload': 'Kelebihan Cairan',
        'blood_in_sputum': 'Darah dalam Dahak', 'prominent_veins_on_calf': 'Urat Menonjol di Betis',
        'palpitations': 'Jantung Berdebar', 'painful_walking': 'Sakit Saat Berjalan',
        'pus_filled_pimples': 'Jerawat Bernanah', 'blackheads': 'Komedo', 'scurring': 'Bekas Luka/Bopeng',
        'skin_peeling': 'Kulit Terkelupas', 'silver_like_dusting': 'Sisik Perak di Kulit',
        'small_dents_in_nails': 'Lekukan Kecil di Kuku', 'inflammatory_nails': 'Kuku Meradang',
        'blister': 'Lepuh', 'red_sore_around_nose': 'Luka Merah Sekitar Hidung',
        'yellow_crust_ooze': 'Kerak Kuning Mengalir'
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