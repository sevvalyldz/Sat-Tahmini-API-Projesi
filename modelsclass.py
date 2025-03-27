import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """
    Verilen X (özellikler) ve y (etiketler) ile farklı sınıflandırma modellerini eğitir ve karşılaştırır.
    
    Parametreler:
    - X: Özellikler (Pandas DataFrame veya NumPy Array)
    - y: Hedef değişken (Pandas Series veya NumPy Array)
    - test_size: Test verisinin oranı (varsayılan: %20)
    - random_state: Rastgelelik kontrolü için sabit değer

    Dönüş:
    - Sonuçları içeren Pandas DataFrame
    """
    X = df[["product_id", "unit_price_y", "Month","customer_id","Segment"]]

    # Veri setini eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Veriyi ölçeklendir (KNN ve SVM için önemli)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Kullanılacak modeller
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Karar Ağacı": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Lojistik Regresyon": LogisticRegression(max_iter=500),
        "SVM": SVC(kernel='rbf')
    }

    # Sonuçları saklamak için liste
    results = []

    # Modelleri eğit ve test et
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()

        # Metrikleri hesapla
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        train_time = end_time - start_time

        # Sonuçları kaydet
        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Training Time (s)": round(train_time, 4)
        })

    # Sonuçları DataFrame olarak döndür
    return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

