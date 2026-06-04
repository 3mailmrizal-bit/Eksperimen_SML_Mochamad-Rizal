# ==============================================================================
# Deskripsi: Script otomatisasi Data Preprocessing untuk proyek Dicoding MSML
# ==============================================================================

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing_pipeline(input_filename="CSS_raw.csv", output_path="CSS_preprocessing.csv"):
    """
    Fungsi utama untuk membaca dataset mentah secara dinamis berdasarkan lokasi script,
    melakukan seluruh tahapan preprocessing, dan menyimpan hasilnya.
    """
    # 1. Mendapatkan path absolut dari folder tempat script ini berada
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Mendapatkan path folder utama
    main_dir = os.path.dirname(script_dir)
    
    # 3. Menggabungkan path folder utama dengan nama file
    input_path = os.path.join(main_dir, input_filename)
    output_path = os.path.join(script_dir, output_path)

    print(f" Mengunggah data mentah dari: {input_path}...")
    try:
        # Membaca dataset menggunakan path dinamis yang sudah dibuat
        df = pd.read_csv(input_path)
        print(f"Berhasil memuat data! Ukuran awal dataset: {df.shape}")
    except FileNotFoundError:
        print(f"⚠️ Error: File '{input_filename}' tidak ditemukan di {main_dir}.")
        return None

    # Membuat salinan agar dataframe asli tidak rusak
    df_clean = df.copy()

    # =========================================================
    # TAHAP 5.9 & 5.1: PEMBERSIHAN SAPU JAGAT KARAKTER & REGEX
    # =========================================================
    print(" Menjalankan pembersihan Sapu Jagat Karakter...")
    dirty_cols = ['Age', 'Num_of_Loan', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Delay_from_due_date', 'Annual_Income', 'Outstanding_Debt']
    for col in dirty_cols:
        if col in df_clean.columns:
            # Menghapus simbol pengganggu, sisakan angka, titik desimal, dan minus
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True).str.strip()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Imputasi khusus nilai kosong Monthly_Inhand_Salary
    if 'Monthly_Inhand_Salary' in df_clean.columns:
        salary_median = df_clean['Monthly_Inhand_Salary'].median()
        df_clean['Monthly_Inhand_Salary'] = df_clean['Monthly_Inhand_Salary'].fillna(salary_median)

    # Imputasi masal menggunakan nilai median untuk kolom numerik hasil pembersihan regex
    numeric_features = df_clean.select_dtypes(include=['int64', 'float64', 'int8', 'int16', 'int32']).columns
    df_clean[numeric_features] = df_clean[numeric_features].fillna(df_clean[numeric_features].median())

    # =========================================================
    # TAHAP 5.2: HANDLING DUPLICATE & IMPUTASI MODUS TARGET
    # =========================================================
    print(" Memeriksa dan menghapus data duplikat...")
    total_duplicate = df_clean.duplicated().sum()
    if total_duplicate > 0:
        df_clean = df_clean.drop_duplicates()
        
    if 'Credit_Mix' in df_clean.columns:
        credit_mix_mode = df_clean['Credit_Mix'].mode()[0]
        df_clean['Credit_Mix'] = df_clean['Credit_Mix'].replace('_', credit_mix_mode)

    # =========================================================
    # TAHAP ENKODING DATA KATEGORIKAL (TARGET & FITUR SISA)
    # =========================================================
    print(" Melakukan encoding pada fitur kategorikal...")
    # Mapping manual untuk target Credit_Mix (Ordinal Encoding)
    credit_mix_mapping = {'Bad': 0, 'Standard': 1, 'Good': 2}
    if 'Credit_Mix' in df_clean.columns and df_clean['Credit_Mix'].dtype == 'object':
        df_clean['Credit_Mix'] = df_clean['Credit_Mix'].map(credit_mix_mapping)

    # LabelEncoder untuk Occupation (Fitur Sisa)
    if 'Occupation' in df_clean.columns:
        le_occupation = LabelEncoder()
        df_clean['Occupation'] = le_occupation.fit_transform(df_clean['Occupation'])
        
    # Faktor keamanan untuk kolom bertipe objek sisa lainnya
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col not in ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan', 'Changed_Credit_Limit']:
            df_clean[col] = df_clean[col].astype('category').cat.codes

    # =========================================================
    # TAHAP BINNING FEATURE ENGINEERING
    # =========================================================
    print(" Membuat fitur baru lewat teknik Binning...")
    if 'Delay_from_due_date' in df_clean.columns:
        df_clean['Delay_Group'] = pd.qcut(df_clean['Delay_from_due_date'], q=3, labels=[0, 1, 2]).astype(int)

    # =========================================================
    # TAHAP 5.3: STANDARISASI FITUR NUMERIK
    # =========================================================
    print(" Menyetarakan skala data (Standarisasi)...")
    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Delay_from_due_date', 'Credit_Utilization_Ratio', 'Total_EMI_per_month']
    available_cols = [col for col in numeric_cols if col in df_clean.columns]
    
    scaler = StandardScaler()
    df_clean[available_cols] = scaler.fit_transform(df_clean[available_cols])

    # =========================================================
    # TAHAP 5.4: OUTLIER HANDLING CAPPING (IQR)
    # =========================================================
    print(" Membatasi nilai ekstrem (Outlier Capping)...")
    for col in available_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    # =========================================================
    # TAHAP 5.7: SELEKSI FITUR / BUANG KOLOM IDENTITAS
    # =========================================================
    print(" Menghapus fitur identitas yang tidak diperlukan...")
    unused_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan', 'Changed_Credit_Limit']
    cols_to_drop = [col for col in unused_cols if col in df_clean.columns]
    df_model = df_clean.drop(columns=cols_to_drop)

    # =========================================================
    # [Penambahan]: EKSPOR HASIL PREPROCESSING MENJADI FILE BARU
    # =========================================================
    print(f" Menyimpan hasil pembersihan ke file baru: {output_path}...")
    df_model.to_csv(output_path, index=False)
    
    print("\n--- PIPELINE DATA PREPROCESSING BERHASIL TOTAL ---")
    print(f"File '{output_path}' sukses dibuat dengan ukuran final: {df_model.shape}")
    
    return df_model

# Perintah pengeksekusian otomatis saat script .py ini dijalankan langsung
if __name__ == "__main__":
    run_preprocessing_pipeline(input_filename="CSS_raw.csv", output_path="CSS_preprocessing.csv")
#

