import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="CardioCare AI", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold; background-color: #003f5c; color: white;}
    </style>
    """, unsafe_allow_html=True)

# Cache model biar enteng
@st.cache_resource
def load_model():
    return joblib.load('model_kardio_rf.pkl')

model = load_model()

# ==========================================
# SIDEBAR: PEMILIHAN BAHASA & INPUT
# ==========================================
st.sidebar.markdown("### 🌐 Language / Bahasa")
lang = st.sidebar.radio("", ["🇮🇩 Indonesia", "🇬🇧 English"], label_visibility="collapsed")

# Kamus Teks UI
if lang == "🇮🇩 Indonesia":
    ui = {
        "title": "🫀 Sistem Cerdas Prediksi Kardiovaskular",
        "subtitle": "**Berbasis Random Forest & XAI (SHAP) untuk Rekomendasi Mandiri**",
        "sidebar_header": "📋 Masukkan Data Pasien",
        "age": "Umur (Tahun)",
        "gender": "Jenis Kelamin",
        "gender_opt": ["Wanita", "Pria"],
        "height": "Tinggi Badan (cm)",
        "weight": "Berat Badan (kg)",
        "ap_hi": "Tensi Sistolik (Atas)",
        "ap_lo": "Tensi Diastolik (Bawah)",
        "cholesterol": "Kolesterol",
        "gluc": "Glukosa",
        "level_opt": ["Normal", "Tinggi", "Sangat Tinggi"],
        "smoke": "Merokok?",
        "alco": "Konsumsi Alkohol?",
        "active": "Rutin Olahraga?",
        "yes_no": ["Tidak", "Ya"],
        "btn_diagnose": "🚀 Diagnosa Sekarang",
        "loading": "AI sedang menganalisis data...",
        "res_title": "📊 Hasil Diagnosa",
        "high_risk": "⚠️ RISIKO TINGGI",
        "low_risk": "✅ RISIKO RENDAH",
        "prob": "Probabilitas Risiko",
        "rec_title": "💊 Rekomendasi Gaya Hidup Mandiri",
        "xai_title": "🧠 Mengapa Sistem Memprediksi Demikian?",
        "xai_desc": "Grafik *SHAP Waterfall* membongkar kontribusi setiap parameter tubuh.",
        "wait_msg": "👈 Silakan atur parameter di sebelah kiri, lalu klik tombol **Diagnosa Sekarang**."
    }
    feat_names = ['Jenis Kelamin', 'Tensi Sistolik', 'Tensi Diastolik', 'Kolesterol', 'Glukosa', 'Merokok', 'Alkohol', 'Aktivitas Fisik', 'Umur (Tahun)', 'BMI']
else:
    ui = {
        "title": "🫀 Smart Cardiovascular Prediction System",
        "subtitle": "**Based on Random Forest & XAI (SHAP) for Independent Recommendations**",
        "sidebar_header": "📋 Enter Patient Data",
        "age": "Age (Years)",
        "gender": "Gender",
        "gender_opt": ["Female", "Male"],
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "ap_hi": "Systolic BP (Upper)",
        "ap_lo": "Diastolic BP (Lower)",
        "cholesterol": "Cholesterol",
        "gluc": "Glucose",
        "level_opt": ["Normal", "High", "Very High"],
        "smoke": "Smoker?",
        "alco": "Alcohol Consumption?",
        "active": "Regular Exercise?",
        "yes_no": ["No", "Yes"],
        "btn_diagnose": "🚀 Diagnose Now",
        "loading": "AI is analyzing data...",
        "res_title": "📊 Diagnosis Results",
        "high_risk": "⚠️ HIGH RISK",
        "low_risk": "✅ LOW RISK",
        "prob": "Risk Probability",
        "rec_title": "💊 Independent Lifestyle Recommendations",
        "xai_title": "🧠 Why Did the System Predict This?",
        "xai_desc": "The *SHAP Waterfall* chart uncovers the contribution of each body parameter.",
        "wait_msg": "👈 Please adjust the parameters on the left, then click the **Diagnose Now** button."
    }
    feat_names = ['Gender', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Glucose', 'Smoker', 'Alcohol', 'Physical Activity', 'Age (Years)', 'BMI']

st.title(ui["title"])
st.markdown(ui["subtitle"])
st.markdown("---")

st.sidebar.header(ui["sidebar_header"])
age = st.sidebar.number_input(ui["age"], 20, 100, 50)
gender = st.sidebar.selectbox(ui["gender"], [1, 2], format_func=lambda x: ui["gender_opt"][x-1])
height = st.sidebar.number_input(ui["height"], 100, 220, 160)
weight = st.sidebar.number_input(ui["weight"], 30, 200, 65)
ap_hi = st.sidebar.number_input(ui["ap_hi"], 80, 250, 120)
ap_lo = st.sidebar.number_input(ui["ap_lo"], 50, 180, 80)
cholesterol = st.sidebar.selectbox(ui["cholesterol"], [1, 2, 3], format_func=lambda x: ui["level_opt"][x-1])
gluc = st.sidebar.selectbox(ui["gluc"], [1, 2, 3], format_func=lambda x: ui["level_opt"][x-1])
smoke = st.sidebar.selectbox(ui["smoke"], [0, 1], format_func=lambda x: ui["yes_no"][x])
alco = st.sidebar.selectbox(ui["alco"], [0, 1], format_func=lambda x: ui["yes_no"][x])
active = st.sidebar.selectbox(ui["active"], [0, 1], format_func=lambda x: ui["yes_no"][x])

prediksi_btn = st.sidebar.button(ui["btn_diagnose"])

# ==========================================
# MAIN PAGE: HASIL & REKOMENDASI
# ==========================================
if prediksi_btn:
    with st.spinner(ui["loading"]):
        # Feature Engineering (BMI)
        bmi = round(weight / ((height / 100) ** 2), 2)
        
        # Susunan Data Input (Harus sesuai X_train Colab)
        nama_fitur_model = ['gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi']
        data_input = pd.DataFrame([[gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age, bmi]], columns=nama_fitur_model)
        
        # Prediksi
        prediksi = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0][1] * 100
        
        col1, col2 = st.columns(2)
        
        # 1. KOTAK DIAGNOSA
        with col1:
            st.subheader(ui["res_title"])
            if prediksi == 1:
                st.error(f"### {ui['high_risk']}\n**{ui['prob']}: {probabilitas:.2f}%**")
            else:
                st.success(f"### {ui['low_risk']}\n**{ui['prob']}: {probabilitas:.2f}%**")
            st.info(f"💡 **BMI:** {bmi}")
            
        # 2. KOTAK REKOMENDASI (KNOWLEDGE BASE FINAL)
        with col2:
            st.subheader(ui["rec_title"])
            rek_count = 0
            
            # RULE 1: HIPERTENSI
            status_hipertensi = 0
            if ap_hi >= 160 or ap_lo >= 100:
                status_hipertensi = 2
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.error("🚨 **KRISIS HIPERTENSI:** JANGAN olahraga berat! Singkirkan garam/kecap 100%. Segera ke dokter untuk stabilisasi obat.")
                else:
                    st.error("🚨 **HYPERTENSIVE CRISIS:** DO NOT do heavy exercise! Eliminate salt/soy sauce 100%. See a doctor immediately for medication.")
            elif ap_hi >= 130 or ap_lo >= 80:
                status_hipertensi = 1
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.warning("⚠️ **WASPADA HIPERTENSI:** Batasi garam maks 1 sdt/hari. Stop mi instan & makanan kaleng.")
                else:
                    st.warning("⚠️ **HYPERTENSION WARNING:** Limit salt to max 1 tsp/day. Stop instant noodles & canned food.")

            # RULE 2: AKTIVITAS FISIK (CONFLICT RESOLUTION)
            if active == 0:
                rek_count += 1
                if status_hipertensi == 2:
                    if lang == "🇮🇩 Indonesia":
                        st.warning("🏃 **AKTIVITAS FISIK:** Tensi sedang krisis. Cukup jalan santai di rumah 10-15 menit. JANGAN lari/kardio.")
                    else:
                        st.warning("🏃 **PHYSICAL ACTIVITY:** BP is critical. Just do light walking at home for 10-15 mins. NO running/cardio.")
                else:
                    if lang == "🇮🇩 Indonesia":
                        st.warning("🏃 **AKTIVITAS FISIK:** Targetkan jalan kaki cepat/bersepeda 30 menit sehari (min. 5x seminggu).")
                    else:
                        st.warning("🏃 **PHYSICAL ACTIVITY:** Target brisk walking/cycling for 30 mins daily (min. 5x a week).")

            # RULE 3 & 4: OBESITAS & GULA
            if bmi >= 25.0:
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.warning("⚖️ **PENURUNAN BERAT BADAN:** Lakukan defisit kalori (kurangi porsi 20%). Stop makan berat 3 jam sebelum tidur.")
                else:
                    st.warning("⚖️ **WEIGHT LOSS:** Apply a caloric deficit (reduce portion by 20%). Stop heavy meals 3 hours before bed.")
            
            if gluc >= 2:
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.warning("🩸 **WASPADA GULA DARAH:** Risiko Diabetes. Jadikan sayur porsi utama, kurangi nasi jadi 1/4 piring. Stop minuman manis.")
                else:
                    st.warning("🩸 **BLOOD SUGAR WARNING:** Diabetes risk. Make veggies the main portion, reduce rice to 1/4 plate. Stop sugary drinks.")

            # RULE 5: KOLESTEROL
            if cholesterol >= 2:
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.warning("🍔 **KOLESTEROL JAHAT:** Hindari gorengan & jeroan. Ubah teknik masak jadi rebus/panggang.")
                else:
                    st.warning("🍔 **BAD CHOLESTEROL:** Avoid fried food & offal. Change cooking methods to boiling/baking.")

            # RULE 6: MEROKOK
            if smoke == 1:
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.warning("🚭 **PENGHENTIAN ROKOK:** Nikotin menyempitkan pembuluh darah. Tahan hasrat dengan permen karet bebas gula.")
                else:
                    st.warning("🚭 **SMOKING CESSATION:** Nicotine narrows blood vessels. Curb cravings with sugar-free gum.")

            # RULE 7: ALKOHOL
            if alco == 1:
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.warning("🍷 **HENTIKAN ALKOHOL:** Memicu aritmia & pelemahan otot jantung. Ganti dengan air mineral/teh hijau.")
                else:
                    st.warning("🍷 **STOP ALCOHOL:** Triggers arrhythmias & heart muscle weakness. Switch to mineral water/green tea.")

            # RULE 8: LANSIA
            if age >= 55:
                rek_count += 1
                if lang == "🇮🇩 Indonesia":
                    st.info("🫀 **PEMANTAUAN LANSIA:** Usia > 55 tahun. Lakukan rekam jantung (EKG) rutin minimal 1 tahun sekali.")
                else:
                    st.info("🫀 **ELDERLY MONITORING:** Age > 55. Perform routine ECG checkups at least once a year.")

            # RULE 9: DEFAULT SAFE
            if rek_count == 0:
                if lang == "🇮🇩 Indonesia":
                    st.success("🌟 **LUAR BIASA:** Parameter tubuh & gaya hidup sangat sehat. Pertahankan!")
                else:
                    st.success("🌟 **EXCELLENT:** Body parameters & lifestyle are very healthy. Keep it up!")

        st.markdown("---")
        
        # ==========================================
        # XAI SHAP VISUALIZATION
        # ==========================================
        st.subheader(ui["xai_title"])
        st.write(ui["xai_desc"])
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_input)
        
        if isinstance(shap_values, list):
            shap_local = shap_values[1][0]
            base_val = explainer.expected_value[1]
        elif len(np.shape(shap_values)) == 3:
            shap_local = shap_values[0, :, 1]
            base_val = explainer.expected_value[1]
        else:
            shap_local = shap_values[0]
            base_val = explainer.expected_value
            
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        shap.waterfall_plot(shap.Explanation(values=shap_local, 
                                             base_values=base_val, 
                                             data=data_input.iloc[0], 
                                             feature_names=feat_names), 
                            max_display=10, show=False)
        # (Kode SHAP kamu sebelumnya)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300) # Pastikan dpi=300 di sini
        shap.waterfall_plot(shap.Explanation(values=shap_local, 
                                             base_values=base_val, 
                                             data=data_input.iloc[0], 
                                             feature_names=feat_names), 
                            max_display=10, show=False)
                            
        # TAMBAHKAN KODE INI UNTUK MENYIMPAN GAMBAR 300 DPI SECARA OTOMATIS
        plt.savefig('shap_waterfall_300dpi.png', dpi=300, bbox_inches='tight')
        
        st.pyplot(fig)
        st.pyplot(fig)
else:
    st.info(ui["wait_msg"])