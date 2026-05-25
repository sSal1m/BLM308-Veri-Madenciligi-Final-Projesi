========================================================================
Telco Müşteri Kaybı (Churn) Analizi ve Karar Destek Sistemi
Ders: BLM308 Veri Madenciliği Final Projesi
========================================================================

Hazırlayan: Seha Salim (Öğrenci No: 231041017)

Bu proje, Kaggle üzerinde yer alan Telco Customer Churn veri setini temel
alarak müşteri kayıplarını proaktif bir şekilde tahmin etmek ve önlemek
amacıyla CRISP-DM metodolojisine uygun olarak geliştirilmiş uçtan uca bir
makine öğrenimi projesidir.

Proje; veri ön işleme, sınıf dengesizliğinin yansız biçimde giderilmesi,
10 katlı çapraz doğrulama ile model eğitimi, modeller arası istatistiksel
karşılaştırma (McNemar Testi), Açıklanabilir Yapay Zeka (XAI - SHAP)
entegrasyonu ve canlı bir Streamlit karar destek paneli içermektedir.

------------------------------------------------------------------------
METODOLOJİK VE MİMARİ TASARIM
------------------------------------------------------------------------

Proje geliştirilirken akademik standartlara ve veri madenciliği
prensiplerine tam uyum hedeflenmiştir:

1. Veri Sızıntısının (Data Leakage) Önlenmesi:
   SMOTE (Sentetik Azınlık Aşırı Örneklendirme) işlemi, veri ön işleme
   aşamasında global olarak uygulanmamış; veri sızıntısını önlemek
   amacıyla imblearn.pipeline.Pipeline yapısı kullanılarak sadece
   çapraz doğrulama katmanlarının eğitim (train) alt kümelerine
   uygulanmıştır. Sınama katındaki gerçek test verileri sentetik
   verilerden tamamen izole edilerek yansız (unbiased) metrikler
   elde edilmiştir.

2. McNemar Testi Uyumu:
   Modeller arası adil istatistiksel karşılaştırma yapılabilmesi için
   tüm modeller (Logistic Regression, Random Forest, MLP)
   KFold(shuffle=True, random_state=42) ile tamamen aynı katman (fold)
   bölünmeleri ve indeksleri üzerinden eğitilmiş ve test edilmiştir.

------------------------------------------------------------------------
PROJE KLASÖR YAPISI
------------------------------------------------------------------------

├── veri/
│   ├── processed_churn.csv   # Ön işlemesi yapılmış temiz veri
│   ├── processed_churn.arff  # Weka analizi için dönüştürülmüş ARFF formatı
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Orijinal veri seti
├── models/
│   ├── best_model.pkl        # En başarılı model (Logistic Regression)
│   └── scaler.pkl            # Giriş verilerini ölçeklemek için kaydedilmiş RobustScaler
├── plots/
│   ├── eda_churn_dist.png    # EDA: Churn sınıf dağılımı grafiği
│   ├── eda_tenure_churn.png  # EDA: Müşteri ömrü (tenure) ve Churn kutu grafiği
│   ├── eda_correlation.png   # EDA: Sürekli sayısal değişkenlerin korelasyon matrisi
│   ├── shap_summary.png      # SHAP: Küresel özellik önem sıralaması grafiği
│   └── shap_local.png        # SHAP: Waterfall yerel müşteri tahmin açıklaması grafiği
├── kod/
│   ├── preprocess.py         # Uçtan uca veri ön işleme ve temizlik betiği
│   ├── model_training.py     # OOP tabanlı model eğitimi, Pipeline SMOTE ve 10-Fold CV
│   ├── shap_analysis.py      # Küresel ve yerel SHAP analiz kodları
│   ├── generate_eda.py       # Keşifsel Veri Analizi (EDA) grafik üretim betiği
│   ├── generate_docx.py      # Word formatındaki akademik raporu derleyen betik
│   └── app.py                # Streamlit Canlı Karar Destek Arayüzü
├── rapor.docx                # Word formatında hazırlanmış 10 sayfalık akademik proje raporu
├── README.md                 # GitHub dökümantasyonu (Markdown)
├── README.txt                # Plain-text formatında proje dökümantasyonu
└── requirements.txt          # Gerekli Python kütüphaneleri listesi

------------------------------------------------------------------------
GEREKSİNİMLER VE KURULUM
------------------------------------------------------------------------

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları
sırasıyla uygulayınız:

1. Depoyu İndirin ve Çalışma Dizinine Geçin:
   Terminali açıp projenin bulunduğu kök dizine geçiş yapın.

2. Gerekli Kütüphaneleri Yükleyin:
   Proje için gerekli olan kütüphaneleri yüklemek için aşağıdaki komutu
   çalıştırın:
   
   pip install -r requirements.txt

------------------------------------------------------------------------
ADIM ADIM ÇALIŞTIRMA KILAVUZU
------------------------------------------------------------------------

Proje modülleri birbirine bağımlı çalışmaktadır. Sırasıyla şu komutları
terminalde çalıştırın:

1. Adım: Veri Ön İşleme (Preprocessing)
   Bu adım, veri kümesini otomatik olarak indirir, temizler, eksik
   verileri ortanca (median) ile doldurur, kategorik sütunları One-Hot
   kodlar ve sürekli sayısal değerleri ölçekler (RobustScaler). Veri
   sızıntısını önlemek için SMOTE bu aşamada uygulanmaz. Sonuçları
   CSV ve Weka uyumlu ARFF olarak kaydeder:
   
   python kod/preprocess.py

2. Adım: Model Eğitimi ve Değerlendirme (Cross-Validation & İstatistik)
   3 farklı modeli (Logistic Regression, Random Forest, MLP)
   imblearn.pipeline.Pipeline kullanarak SMOTE entegrasyonu ile 10 katlı
   çapraz doğrulamadan geçirir. Her model için yansız skorları tablo
   olarak basar. En iyi iki model arasında aynı fold indeksleriyle McNemar
   Testi yaparak istatistiksel farkı ölçer ve en iyi modeli
   models/best_model.pkl olarak kaydeder:
   
   python kod/model_training.py

3. Adım: Açıklanabilir Yapay Zeka (SHAP Analizi)
   Eğitilen en iyi model (Logistic Regression) üzerinde küresel (Summary
   Plot) ve yerel (Waterfall Plot) SHAP değerlerini hesaplayıp grafikler
   üretir ve plots/ dizinine kaydeder.
   
   python kod/shap_analysis.py

4. Adım: Canlı Web Uygulamasını Başlatın
   Tahmin yapmak amacıyla hazırlanan Streamlit kullanıcı arayüzünü
   başlatmak için aşağıdaki komutu çalıştırın:
   
   streamlit run kod/app.py
   
   Komut çalıştıktan sonra tarayıcınızda otomatik olarak
   http://localhost:8501 adresi açılacaktır.

------------------------------------------------------------------------
DENEYSEL SONUÇLAR ÖZETİ
------------------------------------------------------------------------

10-Fold Çapraz Doğrulama Performansı (Yansız Sonuçlar):
+-----------------------------+------------+------------+-----------+
| Model                       | Accuracy   | F1-Score   | ROC-AUC   |
+-----------------------------+------------+------------+-----------+
| Logistic Regression (En İyi)|  %79.07     |  %62.52     |  0.8432   |
| Random Forest               |  %77.68     |  %58.73     |  0.8201   |
| Multilayer Perceptron (MLP) |  %74.87     |  %54.66     |  0.7848   |
+-----------------------------+------------+------------+-----------+

Not: Veri sızıntısı engellendiği için elde edilen metrikler tamamen
gerçekçi ve yansız (unbiased) seviyelerdedir.

McNemar Testi Sonucu (Logistic Regression vs Random Forest):
* Test İstatistiği: 11.6448
* p-Değeri (p-value): 6.4382 x 10^-4 (p < 0.05)
* Yorum: İki modelin tahmin performansları arasındaki fark
  istatistiksel olarak anlamlıdır ve Logistic Regression modelinin
  başarısı Random Forest modeline göre üstündür.

------------------------------------------------------------------------
AKADEMİK RAPOR
------------------------------------------------------------------------

Projenin tüm akademik ve teknik detaylarını içeren, CRISP-DM süreçlerine
göre yapılandırılmış 10 sayfalık akademik rapor rapor.docx formatında
kök dizinde hazır durumdadır.