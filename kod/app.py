import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Telco Churn Tahmin Paneli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık ve Tasarım
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 5px;
        color: #991B1B;
        font-weight: bold;
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 5px;
        color: #065F46;
        font-weight: bold;
    }
    </style>
""", unsafe_ok=True)

st.markdown('<div class="main-title">Telco Müşteri Kaybı (Churn) Tahmin Modeli</div>', unsafe_ok=True)
st.markdown('<div class="sub-title">Yapay Zeka Destekli Müşteri Terk Analizi ve Sadakat Tahmini</div>', unsafe_ok=True)

# Gerekli dosyaların varlığını kontrol et
model_path = "models/best_model.pkl"
scaler_path = "models/scaler.pkl"
data_path = "veri/processed_churn.csv"

# models ve veri klasörleri kod/ üstünde olduğundan, çalışma dizini root iken doğru çalışması için bu yolları kullanıyoruz
if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(data_path)):
    st.error("Gerekli model, ölçekleyici veya veri dosyası bulunamadı! Lütfen önce preprocess.py ve model_training.py dosyalarını çalıştırın.")
    st.stop()

# Dosyaları yükle
@st.cache_resource
def load_assets():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    df_cols = pd.read_csv(data_path, nrows=1)
    feature_cols = df_cols.drop(columns=["Churn"]).columns.tolist()
    return model, scaler, feature_cols

model, scaler, feature_cols = load_assets()

# Sidebar: Kullanıcı Girişleri
st.sidebar.header("Müşteri Bilgilerini Girin")

# Demografik ve Temel Bilgiler
st.sidebar.subheader("Demografik & Sözleşme")
gender = st.sidebar.selectbox("Cinsiyet (gender)", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Yaşlı Müşteri (SeniorCitizen)", ["Hayır (0)", "Evet (1)"])
partner = st.sidebar.selectbox("Evli mi? (Partner)", ["Yes", "No"])
dependents = st.sidebar.selectbox("Bakmakla Yükümlü Olduğu Biri Var mı? (Dependents)", ["Yes", "No"])
contract = st.sidebar.selectbox("Sözleşme Tipi (Contract)", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Kağıtsız Fatura (PaperlessBilling)", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Ödeme Yöntemi (PaymentMethod)", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Hizmetler
st.sidebar.subheader("Alınan Hizmetler")
phone_service = st.sidebar.selectbox("Telefon Servisi (PhoneService)", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Çoklu Hat (MultipleLines)", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("İnternet Servis Tipi (InternetService)", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Çevrimiçi Güvenlik (OnlineSecurity)", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Çevrimiçi Yedekleme (OnlineBackup)", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Cihaz Koruması (DeviceProtection)", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Teknik Destek (TechSupport)", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Canlı TV Yayını (StreamingTV)", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Film Yayını (StreamingMovies)", ["Yes", "No", "No internet service"])

# Sürekli Değişkenler (Ana Sayfada)
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Müşteri Ömrü (tenure - Ay olarak)", min_value=1, max_value=72, value=12, step=1)
with col2:
    monthly_charges = st.slider("Aylık Ücret (MonthlyCharges - USD)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
with col3:
    total_charges = st.slider("Toplam Ücret (TotalCharges - USD)", min_value=18.0, max_value=9000.0, value=1000.0, step=10.0)

# Tahmin Butonu
st.markdown("<br>", unsafe_ok=True)
if st.button("Müşteri Durumunu Tahmin Et", type="primary", use_container_width=True):
    # 45 Kolonlu Boş Giriş Sözlüğü Oluştur
    input_dict = {col: 0 for col in feature_cols}
    
    # Sayısal değerleri aktar
    input_dict["SeniorCitizen"] = 1 if senior_citizen == "Evet (1)" else 0
    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly_charges
    input_dict["TotalCharges"] = total_charges
    
    # One-Hot değişkenleri güncelle
    def set_one_hot(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_dict:
            input_dict[col_name] = 1

    set_one_hot("gender", gender)
    set_one_hot("Partner", partner)
    set_one_hot("Dependents", dependents)
    set_one_hot("PhoneService", phone_service)
    set_one_hot("MultipleLines", multiple_lines)
    set_one_hot("InternetService", internet_service)
    set_one_hot("OnlineSecurity", online_security)
    set_one_hot("OnlineBackup", online_backup)
    set_one_hot("DeviceProtection", device_protection)
    set_one_hot("TechSupport", tech_support)
    set_one_hot("StreamingTV", streaming_tv)
    set_one_hot("StreamingMovies", streaming_movies)
    set_one_hot("Contract", contract)
    set_one_hot("PaperlessBilling", paperless_billing)
    set_one_hot("PaymentMethod", payment_method)
    
    # DataFrame oluştur
    input_df = pd.DataFrame([input_dict])
    
    # Sayısal değerleri ölçeklendir
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Modeli çalıştır ve olasılığı al
    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = model.predict(input_df)[0]
    
    # Sonuç Alanı
    st.markdown("### Tahmin Sonuçları")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(label="Müşteri Kayıp (Churn) Olasılığı", value=f"%{churn_prob*100:.2f}")
        st.progress(churn_prob)
        
    with col_res2:
        if churn_prob > 0.5:
            st.markdown(f"""
                <div class="risk-high">
                    ⚠️ YÜKSEK RİSK: Müşteri her an ayrılabilir! <br>
                    Kayıp Olasılığı: %{churn_prob*100:.2f} <br>
                    Öneri: Müşteriye özel indirimler, uzun dönem taahhüt yenileme veya özel müşteri hizmetleri desteği sağlanması önerilir.
                </div>
            """, unsafe_ok=True)
        else:
            st.markdown(f"""
                <div class="risk-low">
                    ✅ GÜVENLİ: Sadık Müşteri. <br>
                    Kayıp Olasılığı: %{churn_prob*100:.2f} <br>
                    Öneri: Müşteri mevcut hizmet kalitesinden memnun görünüyor. İlişkileri korumak için standart bültenler ve sadakat programları yeterlidir.
                </div>
            """, unsafe_ok=True)

# Bilgi Bölümü
st.markdown("<br><hr>", unsafe_ok=True)
model_name = type(model).__name__
st.markdown(f"""
    **Model Hakkında:**
    Bu web uygulaması, Kaggle Telco Customer Churn veri seti üzerinde eğitilmiş en başarılı model olan **{model_name}** modelini kullanmaktadır. 
    Arka planda veriler RobustScaler ile ölçeklenmekte ve girilen özelliklere göre olasılık puanı hesaplanmaktadır.
""")
