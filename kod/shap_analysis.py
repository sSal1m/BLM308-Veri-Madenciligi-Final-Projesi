import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression

def run_shap_analysis():
    """
    Kaydedilen en iyi model (Logistic Regression, Random Forest veya MLP) üzerinde 
    dinamik olarak uygun SHAP Explainer'ı seçerek küresel ve yerel analizleri yapar.
    Grafikleri plots/ klasörüne kaydeder.
    """
    print("[XAI] SHAP analizi başlatılıyor...")
    
    # 1. Model ve verileri yükleme
    model_path = "models/best_model.pkl"
    data_path = "veri/processed_churn.csv"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    
    # plots klasörünü oluştur
    os.makedirs("plots", exist_ok=True)
    
    # Hızlı hesaplama için veriden 200 adet rastgele örnek seçelim (Bütün veri yerine örneklem kullanımı)
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], size=200, replace=False)
    X_sample = X.iloc[sample_indices]
    
    # 2. Model türüne göre dinamik Explainer seçimi
    print(f"[XAI] Yüklenen model tipi: {type(model).__name__}")
    
    if isinstance(model, LogisticRegression):
        # Lojistik Regresyon için LinearExplainer kullanılır
        explainer = shap.LinearExplainer(model, X_sample)
        print("[XAI] LinearExplainer oluşturuldu.")
        explanation = explainer(X_sample)
        explanation_class1 = explanation
    elif hasattr(model, "tree_") or hasattr(model, "estimators_"):
        # Ağaç tabanlı modeller (Random Forest vb.) için TreeExplainer
        explainer = shap.TreeExplainer(model)
        print("[XAI] TreeExplainer oluşturuldu.")
        explanation = explainer(X_sample)
        # Sınıf 1 (Churn = 1) değerlerini al (Scikit-learn ağaç modelleri için N, M, 2 formatındadır)
        if len(explanation.shape) == 3:
            explanation_class1 = explanation[:, :, 1]
        else:
            explanation_class1 = explanation
    else:
        # Diğer modeller (MLP vb.) için genel Explainer
        explainer = shap.Explainer(model, X_sample)
        print("[XAI] Genel Explainer oluşturuldu.")
        explanation = explainer(X_sample)
        explanation_class1 = explanation

    # 3. Küresel Yorumlama (Summary Plot)
    print("[XAI] Küresel yorumlama grafiği (Summary Plot) çiziliyor...")
    plt.figure(figsize=(12, 8))
    # SHAP kütüphanesinin summary_plot fonksiyonunu kullanıyoruz
    shap.summary_plot(explanation_class1, X_sample, show=False)
    plt.title(f"SHAP Küresel Özellik Etkisi (Model: {type(model).__name__})", fontsize=14, pad=15)
    plt.tight_layout()
    summary_plot_path = "plots/shap_summary.png"
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Küresel SHAP grafiği '{summary_plot_path}' olarak kaydedildi.")
    
    # 4. Yerel Yorumlama (Waterfall Plot)
    print("[XAI] Yerel yorumlama grafiği (Waterfall Plot) çiziliyor...")
    
    # Örneklem içerisinden ilk müşteriyi seçip kararı açıklayalım
    local_idx = 0
    
    plt.figure(figsize=(10, 6))
    # SHAP kütüphanesinin standart waterfall plot grafiğini kullanıyoruz
    shap.plots.waterfall(explanation_class1[local_idx], max_display=10, show=False)
    plt.title(f"Müşteri Karar Açıklaması (Yerel Görünüm - Waterfall Plot)", fontsize=12, pad=15)
    plt.tight_layout()
    
    local_plot_path = "plots/shap_local.png"
    plt.savefig(local_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Yerel SHAP Waterfall grafiği '{local_plot_path}' olarak kaydedildi.")

if __name__ == "__main__":
    run_shap_analysis()
