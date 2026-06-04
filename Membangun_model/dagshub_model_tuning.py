import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub

# =================================================================
# 0. KONEKSI ONLINE KE DAGSHUB VIA LIBRARY (KRITERIA ADVANCE)
# =================================================================
# Otomatisasi konfigurasi remote tracking URI ke server cloud
dagshub.init(repo_owner='3mailmrizal-bit', repo_name='Eksperimen_SML_Mochamad-Rizal', mlflow=True)

# =================================================================
# 1. OTOMATISASI SISTEM PENDETEKSI & PENYALIN DATA
# =================================================================
sumber_data = 'preprocessing/CSS_preprocessing.csv'
tujuan_data = 'Membangun_model/CSS_preprocessing.csv'

if os.path.exists(sumber_data):
    shutil.copy(sumber_data, tujuan_data)
    print("✅ Sukses: Data preprocessing siap digunakan untuk tuning cloud.")

# =================================================================
# 2. PENYIAPAN DATA
# =================================================================
df_model = pd.read_csv(tujuan_data)
X = df_model.drop(columns=['Credit_Mix'])
y = df_model['Credit_Mix']

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =================================================================
# 3. HYPERPARAMETER TUNING (GRID SEARCH)
# =================================================================
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

print("\n--- Memulai Proses Hyperparameter Tuning ---")
rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# =================================================================
# 4. EVALUASI MODEL
# =================================================================
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# =================================================================
# 5. PEMBUATAN ARTEFAK TAMBAHAN (WAJIB UNTUK ADVANCE)
# =================================================================
# Artefak Tambahan 1: Grafik Feature Importance (.png)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
grafik_path = "Membangun_model/feature_importance.png"
plt.savefig(grafik_path)
plt.close()

# Artefak Tambahan 2: File Metadata Model (.json)
meta_data = {
    "model_type": "RandomForestClassifier",
    "target_column": "Credit_Mix",
    "best_parameters": best_params
}
json_path = "Membangun_model/model_meta.json"
with open(json_path, "w") as f:
    json.dump(meta_data, f, indent=4)

# =================================================================
# 6. MANUAL LOGGING KE SERVER CLOUD DAGSHUB
# =================================================================
mlflow.set_experiment("Eksperimen_Tuning_Online_Rizal")

with mlflow.start_run(run_name="RF_Tuning_Advance_DagsHub"):
    print("\n--- Mencatat Log ke DagsHub secara Real-time ---")
    
    # A. Log Parameter Terbaik
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
    
    # B. Log Metrik Evaluasi
    mlflow.log_metric("testing_accuracy", acc)
    mlflow.log_metric("testing_precision", prec)
    mlflow.log_metric("testing_recall", rec)
    mlflow.log_metric("testing_f1_score", f1)
    
    # C. Log Model Utama
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="best_tuned_model")
    
    # D. Log Artefak Tambahan (Syarat Kriteria Advance)
    mlflow.log_artifact(grafik_path)
    mlflow.log_artifact(json_path)
    
    print("✅ Sukses: Seluruh parameter, metrik, dan model terkirim ke cloud DagsHub!")