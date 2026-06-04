import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =================================================================
# 1. OTOMATISASI SISTEM PENDETEKSI & PENYALIN DATA
# =================================================================
sumber_data = 'preprocessing/CSS_preprocessing.csv'
tujuan_folder = 'Membangun_model'
tujuan_data = 'Membangun_model/CSS_preprocessing.csv'

print("--- Memeriksa Ketersediaan File Data ---")

# Memastikan folder tujuan ada
if not os.path.exists(tujuan_folder):
    os.makedirs(tujuan_folder)
    print(f"Folder '{tujuan_folder}' berhasil dibuat.")

# Proses copy otomatis jika file sumber tersedia
if os.path.exists(sumber_data):
    shutil.copy(sumber_data, tujuan_data)
    print("✅ Sukses: File 'CSS_preprocessing.csv' disalin ke folder Membangun_model!")
else:
    print("❌ Error: File sumber tidak ditemukan di 'preprocessing/'.")
    print("Pastikan pipeline Kriteria 1 sudah dijalankan!")

# =================================================================
# 2. PENYIAPAN DATA (Langkah 9.3)
# =================================================================
# Membaca data yang sudah disalin
df_model = pd.read_csv(tujuan_data)

# 1. Memisahkan Fitur Utama (X) dan Target Label Prediksi (y)
X = df_model.drop(columns=['Credit_Mix'])
y = df_model['Credit_Mix']

# 2. Melakukan Data Splitting (Train 80% : Test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =================================================================
# 3. MEMBANGUN MODEL & LOGGING MLFLOW (KRITERIA 2)
# =================================================================
# Mengaktifkan autolog agar MLflow mencatat semua parameter & metrik secara otomatis
mlflow.set_experiment("Eksperimen_ML_Rizal")
mlflow.sklearn.autolog()

# 3. Menginisialisasi Model Klasifikasi Random Forest
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

print("\n--- Memulai Proses Pelatihan Model ---")

# 4. Memulai Proses Fitting Model di dalam MLflow Run
with mlflow.start_run():
    model_rf.fit(X_train, y_train)
    print("✅ Sukses: Model telah dilatih dan dicatat ke MLflow!")

print("\n--- Verifikasi Hasil Akhir ---")
print(f"Ukuran Data Latih : {X_train.shape}")
print(f"Ukuran Data Uji   : {X_test.shape}")