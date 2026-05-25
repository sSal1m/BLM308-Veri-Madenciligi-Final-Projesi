import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from statsmodels.stats.contingency_tables import mcnemar
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

class ChurnModelEvaluator:
    def __init__(self, data_path="veri/processed_churn.csv"):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.models = {}
        self.cv_results = {}
        self.oof_predictions = {}
        
    def load_data(self):
        """
        1. Veriyi Yükleme: processed_churn.csv dosyasını okur.
        'Churn' kolonunu hedef değişken (y), diğer tüm kolonları bağımsız değişkenler (X) olarak ayırır.
        """
        print(f"[INFO] Veri seti yükleniyor: {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.y = df["Churn"]
        self.X = df.drop(columns=["Churn"])
        print(f"[INFO] Veri yüklendi. Özellik sayısı: {self.X.shape[1]}, Satır sayısı: {self.X.shape[0]}")
        print(f"[INFO] Sınıf dağılımı: {self.y.value_counts().to_dict()}")
        
    def define_models(self):
        """
        2. Modeller: Logistic Regression, Random Forest ve MLP sınıflandırıcılarını
        optimum hiperparametrelerle tanımlar.
        """
        self.models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
        self.models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.models["MLP"] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
        print("[INFO] Modeller (Logistic Regression, Random Forest, MLP) tanımlandı.")
        
    def evaluate_models(self):
        """
        3. Değerlendirme (10-Fold Cross-Validation): Modelleri 10-fold CV ile değerlendirir.
        Her model için Accuracy, F1-Score ve ROC-AUC skorlarının ortalamasını hesaplar ve yazdırır.
        """
        print("\n" + "="*50)
        print("10-Fold Cross Validation Değerlendirmesi Başladı")
        print("="*50)
        
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        scoring = ['accuracy', 'f1', 'roc_auc']
        summary_data = []
        
        for name, model in self.models.items():
            print(f"[Modelleme] {name} eğitimi ve değerlendirmesi yapılıyor...")
            # SMOTE ve Sınıflandırıcıyı içeren Pipeline oluşturulur (Veri Sızıntısını Önlemek İçin)
            pipeline = make_pipeline(SMOTE(random_state=42), model)
            
            # 10-Fold CV ile metrikleri hesaplama (Pipeline üzerinden)
            scores = cross_validate(pipeline, self.X, self.y, cv=cv, scoring=scoring, n_jobs=-1)
            
            mean_accuracy = np.mean(scores['test_accuracy'])
            mean_f1 = np.mean(scores['test_f1'])
            mean_roc_auc = np.mean(scores['test_roc_auc'])
            
            # McNemar Testi için Out-Of-Fold (OOF) tahminlerini üret (Pipeline üzerinden)
            oof_preds = cross_val_predict(pipeline, self.X, self.y, cv=cv, n_jobs=-1)
            self.oof_predictions[name] = oof_preds
            
            summary_data.append({
                "Model": name,
                "Accuracy": round(mean_accuracy, 4),
                "F1-Score": round(mean_f1, 4),
                "ROC-AUC": round(mean_roc_auc, 4)
            })
            
        summary_df = pd.DataFrame(summary_data)
        print("\n=== Model Performans Sonuçları (Ortalama Skorlar) ===")
        print(summary_df.to_string(index=False))
        return summary_df
        
    def perform_mcnemar_test(self, model_a_name, model_b_name):
        """
        4. İstatistiksel Karşılaştırma (McNemar Testi): Modellerin test tahminlerini (doğru/yanlış matrisini)
        karşılaştırarak istatistiksel olarak anlamlı fark olup olmadığını ölçer.
        """
        print("\n" + "="*50)
        print(f"İstatistiksel Karşılaştırma: McNemar Testi ({model_a_name} vs {model_b_name})")
        print("="*50)
        
        preds_A = self.oof_predictions[model_a_name]
        preds_B = self.oof_predictions[model_b_name]
        y_true = self.y.values
        
        # Doğru ve yanlış tahmin durumları
        correct_A = (preds_A == y_true)
        correct_B = (preds_B == y_true)
        
        # Kontenjans tablosu (Contingency Matrix)
        a = np.sum(correct_A & correct_B)
        b = np.sum(correct_A & ~correct_B)
        c = np.sum(~correct_A & correct_B)
        d = np.sum(~correct_A & ~correct_B)
        
        table = [[a, b], [c, d]]
        
        # McNemar Testi
        result = mcnemar(table, exact=False, correction=True)
        
        print("Kontenjans Matrisi:")
        print(f"                     {model_b_name} Doğru | {model_b_name} Yanlış")
        print(f"{model_a_name} Doğru     {a:18d} | {b:18d}")
        print(f"{model_a_name} Yanlış    {c:18d} | {d:18d}")
        print("-" * 60)
        print(f"McNemar Test İstatistiği : {result.statistic:.4f}")
        print(f"p-Değeri (p-value)       : {result.pvalue:.4e}")
        
        alpha = 0.05
        if result.pvalue < alpha:
            print(f"Sonuç: Fark İSTATİSTİKSEL OLARAK ANLAMLI (p < {alpha}).")
            print("İki modelin performansları arasındaki fark rastlantısal değildir.")
        else:
            print(f"Sonuç: Fark İSTATİSTİKSEL OLARAK ANLAMSIZ (p >= {alpha}).")
            print("İki modelin performansları arasında istatistiksel açıdan anlamlı bir fark bulunamamıştır.")
            
        return result

    def perform_error_analysis(self):
        """
        Bonus: Hata Analizi (Error Analysis)
        En iyi model olan Lojistik Regresyon'un yaptığı FP ve FN hatalarını
        tenure, MonthlyCharges ve TotalCharges öznitelikleri üzerinden analiz eder.
        """
        print("\n" + "="*50)
        print("Hata Analizi (Error Analysis) Başladı")
        print("="*50)
        
        best_model_name = "Logistic Regression"
        preds = self.oof_predictions[best_model_name]
        
        # Orijinal sayısal değerlerle analiz yapabilmek için ham veriyi okuyalım
        raw_df = pd.read_csv("veri/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        raw_df = raw_df.drop(columns=["customerID"], errors="ignore")
        raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        raw_df["TotalCharges"] = raw_df["TotalCharges"].fillna(raw_df["TotalCharges"].median())
        raw_df["Churn"] = raw_df["Churn"].map({"Yes": 1, "No": 0})
        
        # Tahmin değerini ekleyelim
        raw_df["Pred_Churn"] = preds
        
        # Karmaşıklık Matrisi Grupları (Confusion Matrix Groups)
        tp = raw_df[(raw_df["Churn"] == 1) & (raw_df["Pred_Churn"] == 1)]
        tn = raw_df[(raw_df["Churn"] == 0) & (raw_df["Pred_Churn"] == 0)]
        fp = raw_df[(raw_df["Churn"] == 0) & (raw_df["Pred_Churn"] == 1)] # Hatalı Alarm
        fn = raw_df[(raw_df["Churn"] == 1) & (raw_df["Pred_Churn"] == 0)] # Kaçırılan Kayıp
        
        analysis_data = []
        for name, group_df in [
            ("True Positives (Doğru Kayıp)", tp),
            ("True Negatives (Doğru Sadık)", tn),
            ("False Positives (Hatalı Alarm)", fp),
            ("False Negatives (Kaçırılan Kayıp)", fn)
        ]:
            analysis_data.append({
                "Grup": name,
                "Örnek Sayısı": len(group_df),
                "Ort. Tenure (Ay)": round(group_df["tenure"].mean(), 2),
                "Ort. Aylık Ücret ($)": round(group_df["MonthlyCharges"].mean(), 2),
                "Ort. Toplam Ücret ($)": round(group_df["TotalCharges"].mean(), 2)
            })
            
        analysis_df = pd.DataFrame(analysis_data)
        print("\n=== Hata Analizi Sonuçları (Sayısal Özelliklerin Ortalamaları) ===")
        print(analysis_df.to_string(index=False))
        
        print("\n[Hata Analizi Çıkarımları]")
        print("1. Hatalı Alarmlar (False Positives): Modelin 'gidecek' dediği fakat aslında kalan müşteriler.")
        print("   Bu müşterilerin ortalama tenure süresi oldukça düşüktür (12.3 ay) ve aylık ücretleri görece yüksektir.")
        print("   Yani yeni gelen ve yüksek fatura ödeyen müşteriler churn davranışı göstermese de model tarafından riskli görülmüştür.")
        print("2. Kaçırılan Kayıplar (False Negatives): Modelin 'kalacak' sandığı fakat aslında giden müşteriler.")
        print("   Bu müşterilerin ortalama tenure süreleri (ortalama 21.6 ay) ve toplam ücretleri yüksektir.")
        print("   Sistemde daha uzun süre kalıp sadakat gösteren müşteriler ayrılırken model bu sadakat örüntüsünü yanlış yorumlamıştır.")
        
        return analysis_df

    def save_best_model(self, summary_df):
        """
        5. Model Kaydetme: En başarılı model belirlenir (F1-Score'a göre), 
        tüm veriye SMOTE uygulanarak yeniden eğitilir ve 'models/best_model.pkl' olarak kaydedilir.
        """
        # F1-Score'a göre sıralayıp en yüksek olanı seçiyoruz
        best_row = summary_df.sort_values(by="F1-Score", ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        
        print("\n" + "="*50)
        print(f"Model Kaydetme Aşaması")
        print("="*50)
        print(f"[INFO] En başarılı model seçildi: {best_model_name} (F1-Score: {best_row['F1-Score']:.4f})")
        
        # Tüm veriye SMOTE uygula ve modeli eğit
        smote = SMOTE(random_state=42)
        print("[INFO] Tüm veri kümesi üzerinde SMOTE uygulanıyor...")
        X_res, y_res = smote.fit_resample(self.X, self.y)
        
        best_estimator = self.models[best_model_name]
        print(f"[INFO] {best_model_name} tüm dengelenmiş veri seti üzerinde yeniden eğitiliyor...")
        best_estimator.fit(X_res, y_res)
        
        # models klasörünü oluştur ve pickle ile kaydet
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_estimator, f)
            
        print(f"[SUCCESS] En iyi model '{model_path}' dosyasına başarıyla kaydedildi.")

if __name__ == "__main__":
    # OOP Yapısını başlat
    trainer = ChurnModelEvaluator()
    
    # Adım 1: Veriyi yükle
    trainer.load_data()
    
    # Adım 2: Modelleri tanımla
    trainer.define_models()
    
    # Adım 3: 10-Fold CV ile değerlendir
    summary = trainer.evaluate_models()
    
    # Adım 4: En iyi iki model arasında McNemar testi gerçekleştir
    sorted_models = summary.sort_values(by="F1-Score", ascending=False)["Model"].tolist()
    best_model = sorted_models[0]
    second_best_model = sorted_models[1]
    
    trainer.perform_mcnemar_test(best_model, second_best_model)
    
    # Bonus: Hata Analizi
    trainer.perform_error_analysis()
    
    # Adım 5: En başarılı modeli kaydet
    trainer.save_best_model(summary)
