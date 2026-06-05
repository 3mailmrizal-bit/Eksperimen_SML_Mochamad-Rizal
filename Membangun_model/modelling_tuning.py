import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =================================================================
# 1. OTOMATISASI SISTEM PENDETEKSI & PENYALIN DATA
# =================================================================
sumber_data = '../preprocessing/CSS_preprocessing.csv'
tujuan_data = 'Membangun_model/CSS_preprocessing.csv'

if os.path.exists(sumber_data):
    shutil.copy(sumber_data, tujuan_data)
    print("✅ Sukses: Data preprocessing siap digunakan untuk tuning.")
else:
    print("❌ Error: File sumber tidak ditemukan.")

# =================================================================
# 2. PENYIAPAN DATA
# =================================================================
df_model = pd.read_csv(tujuan_data)
X = df_model.drop(columns=['Credit_Mix'])
y = df_model['Credit_Mix']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =================================================================
# 3. KANDIDAT PARAMETER & PROSES GRID SEARCH (TUNING)
# =================================================================
# Menentukan variasi parameter yang ingin diuji coba
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'criterion': ['gini', 'entropy']
}

print("\n--- Memulai Proses Hyperparameter Tuning (GridSearchCV) ---")
rf_base = RandomForestClassifier(random_state=42)

# Mencari kombinasi terbaik dengan cross-validation sebanyak 3 lipatan (cv=3)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mengambil model terbaik hasil tuning
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("✅ Tuning Selesai!")
print(f"Kombinasi Parameter Terbaik: {best_params}")

# =================================================================
# 4. EVALUASI MODEL TERBAIK
# =================================================================
y_pred = best_model.predict(X_test)

# Menghitung metrik performa secara manual
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# =================================================================
# 5. MANUAL LOGGING KE MLFLOW (KRITERIA SKILLED)
# =================================================================
# Menentukan nama eksperimen khusus tuning
mlflow.set_experiment("Eksperimen_Tuning_Rizal")

# Memulai perekaman manual
with mlflow.start_run(run_name="RandomForest_Tuning_Manual"):
    
    print("\n--- Mencatat Parameter dan Metrik ke MLflow secara Manual ---")
    
    # A. Mencatat Parameter Terbaik satu per satu (Manual Log)
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
    
    # B. Mencatat Metrik Evaluasi satu per satu (Manual Log)
    mlflow.log_metric("testing_accuracy", acc)
    mlflow.log_metric("testing_precision", prec)
    mlflow.log_metric("testing_recall", rec)
    mlflow.log_metric("testing_f1_score", f1)
    
    # C. Menyimpan file biner Model Terbaik sebagai Artefak secara Manual
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="best_tuned_model")
    
    print("✅ Sukses: Seluruh parameter, metrik, dan model berhasil dicatat!")