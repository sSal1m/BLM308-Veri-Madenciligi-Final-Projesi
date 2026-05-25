import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# Veri setinin indirileceği URL (IBM Resmi Deposundan)
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_FILE = "veri/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def download_dataset(url=DATASET_URL, target_path=LOCAL_FILE):
    """
    Telco Customer Churn veri setini yerelde bulunmuyorsa otomatik olarak indirir.
    """
    if not os.path.exists(target_path):
        print(f"[INFO] '{target_path}' dosyası bulunamadı. İnternetten indiriliyor...")
        try:
            # Klasörü kontrol et
            dir_name = os.path.dirname(target_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            # Veriyi indir
            urllib.request.urlretrieve(url, target_path)
            print("[INFO] Veri seti başarıyla indirildi.")
        except Exception as e:
            print(f"[ERROR] Veri seti indirilemedi: {e}")
            print("[INFO] Lütfen veri setini manuel olarak indirin ve bu dizine koyun.")
    else:
        print(f"[INFO] '{target_path}' dosyası yerelde mevcut.")

def load_and_clean_data(filepath=LOCAL_FILE):
    """
    1. Veriyi yükler, gereksiz kolonları düşürür.
    2. TotalCharges kolonundaki boşlukları yakalar, float tipine çevirir ve eksik değerleri median ile doldurur.
    """
    print("[1/6] Veri yükleniyor ve temizleniyor...")
    df = pd.read_csv(filepath)
    
    # "customerID" gibi makine öğrenimi için gereksiz kolonu düşür
    df = df.drop(columns=["customerID"], errors="ignore")
    
    # "TotalCharges" kolonunda boşluk içeren satırları yakalamak için NaN ile değiştir
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Eksik değerleri median (ortanca) ile doldur
    total_charges_median = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)
    
    return df

def encode_data(df, target_col="Churn"):
    """
    2. Kategorik değişkenleri One-Hot Encoding yöntemiyle dönüştürür.
    Hedef değişken olan "Churn" kolonunu 1 ve 0 olarak map eder.
    """
    print("[2/6] Kategorik kolonlar One-Hot Encoding yöntemiyle dönüştürülüyor...")
    
    # Hedef değişken Churn'ü 1 ve 0 olarak eşle
    if target_col in df.columns:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    
    # Hedef değişken dışındaki kategorik sütunları tespit et (object tipindekiler)
    features_df = df.drop(columns=[target_col], errors="ignore")
    cat_cols = features_df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # One-Hot Encoding uygula (Weka uyumluluğu için veri tipini int/float yapıyoruz)
    encoded_features = pd.get_dummies(features_df, columns=cat_cols, drop_first=False, dtype=int)
    
    # Hedef değişkeni tekrar ekle
    encoded_features[target_col] = df[target_col]
    return encoded_features

def scale_features(df, scaler_type="robust", target_col="Churn"):
    """
    3. Sayısal özellikleri RobustScaler veya StandardScaler kullanarak ölçekler.
    (Sadece sürekli sayısal değerleri ölçekler: tenure, MonthlyCharges, TotalCharges)
    """
    print(f"[3/6] Sayısal özellikler {scaler_type} ölçekleyici ile ölçekleniyor...")
    
    # Sürekli sayısal kolonlar
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    # Seçilen ölçekleyiciyi belirle
    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    elif scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Ölçekleyici tipi 'robust' veya 'standard' olmalıdır.")
    
    # Ölçekleme işlemini uygula
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Ölçekleyiciyi kaydet (Streamlit uygulaması için)
    os.makedirs("models", exist_ok=True)
    import pickle
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[INFO] Ölçekleyici 'models/scaler.pkl' olarak kaydedildi.")
    
    return df

def save_to_csv(df, filepath="veri/processed_churn.csv"):
    """
    5. Hazırlanan temiz veriyi CSV olarak kaydeder.
    """
    print(f"[5/6] Temizlenmiş veri '{filepath}' olarak kaydediliyor...")
    
    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print("[INFO] CSV dosyası başarıyla kaydedildi.")

def save_to_arff(df, relation_name="telco_customer_churn", filepath="veri/processed_churn.arff"):
    """
    6. Veriyi standart bir ARFF formatına dönüştürür ve kaydeder.
    @relation, @attribute ve @data blokları hatasız oluşturulur ve tipler doğru yansıtılır.
    """
    print(f"[6/6] Veri Weka ARFF formatına dönüştürülüyor ve '{filepath}' olarak kaydediliyor...")
    
    # Klasör yoksa oluştur
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        # İlişki adı (Relation)
        f.write(f"@relation {relation_name}\n\n")
        
        # Özellikler (Attributes)
        for col in df.columns:
            # Weka'da hata oluşmaması için tırnak işareti içeren isimleri temizle
            col_clean = col.replace('"', '')
            safe_col_name = f'"{col_clean}"'
            
            # Hedef kolon 'Churn' nominal (kategorik) olmalıdır: {0, 1}
            if col == "Churn":
                unique_vals = sorted(df[col].dropna().unique())
                vals_str = ",".join(map(str, [int(v) for v in unique_vals]))
                f.write(f"@attribute {safe_col_name} {{{vals_str}}}\n")
            # Sayısal veri tipleri
            elif pd.api.types.is_numeric_dtype(df[col]):
                f.write(f"@attribute {safe_col_name} numeric\n")
            # Kategorik / nominal veri tipleri
            else:
                unique_vals = sorted(df[col].dropna().unique())
                vals_str = ",".join([f"'{str(v)}'" for v in unique_vals])
                f.write(f"@attribute {safe_col_name} {{{vals_str}}}\n")
        
        # Veri bloğu (Data)
        f.write("\n@data\n")
        for row in df.itertuples(index=False):
            row_str_list = []
            for val in row:
                if pd.isna(val):
                    row_str_list.append("?")
                # Boolean değerleri Weka için sayıya veya nominale dönüştür
                elif isinstance(val, bool):
                    row_str_list.append("1" if val else "0")
                else:
                    row_str_list.append(str(val))
            f.write(",".join(row_str_list) + "\n")
            
    print("[INFO] ARFF dosyası başarıyla kaydedildi.")

if __name__ == "__main__":
    # 0. Veri setini yerelde ara, yoksa indir
    download_dataset(DATASET_URL, LOCAL_FILE)
    
    # 1. Yükle ve temizle
    df_cleaned = load_and_clean_data(LOCAL_FILE)
    
    # 2. One-hot encoding ve hedef kolon eşleme
    df_encoded = encode_data(df_cleaned, target_col="Churn")
    
    # 3. Ölçeklendirme (RobustScaler varsayılan olarak seçildi)
    df_scaled = scale_features(df_encoded, scaler_type="robust", target_col="Churn")
    
    # 4. Küresel SMOTE kaldırıldı (Veri sızıntısını önlemek için artık eğitim aşamasında fold bazlı uygulanacak)
    
    # 5. CSV olarak kaydet (SMOTE uygulanmamış veri)
    save_to_csv(df_scaled, "veri/processed_churn.csv")
    
    # 6. ARFF olarak kaydet (SMOTE uygulanmamış veri)
    save_to_arff(df_scaled, relation_name="telco_customer_churn", filepath="veri/processed_churn.arff")
    
    print("\n[SUCCESS] Tüm veri ön işleme adımları başarıyla tamamlandı (SMOTE eğitim aşamasına kaydırıldı)!")
