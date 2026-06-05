import json
import requests

# ============================================
# KONFIGURASI
# ============================================
# Ganti port jika Anda menjalankan di port lain (misal 5000)
URL_MODEL = "http://127.0.0.1:5002/invocations"

# ============================================
# DATA DUMMY (Sesuai dengan 20 Fitur Model Anda)
# ============================================
# Kita hardcode datanya agar tidak perlu baca CSV eksternal.
# Pastikan nilai-nilai ini masuk akal (numerik, karena CSV preprocessing sudah di-encode).
sample_data = {
    "Age": 32,
    "Occupation": 1,                     # Sudah di-encode (misal: 1 = Engineer)
    "Annual_Income": 8500000,
    "Monthly_Inhand_Salary": 700000,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 2,
    "Interest_Rate": 5.5,
    "Num_of_Loan": 1,
    "Delay_from_due_date": 10,
    "Num_of_Delayed_Payment": 2,
    "Num_Credit_Inquiries": 5,
    "Outstanding_Debt": 10000,
    "Credit_Utilization_Ratio": 30.5,
    "Credit_History_Age": 120,
    "Payment_of_Min_Amount": 100,
    "Total_EMI_per_month": 500,
    "Amount_invested_monthly": 1000,
    "Payment_Behaviour": 2,              # Sudah di-encode
    "Monthly_Balance": 2000,
    "Delay_Group": 1                     # Sudah di-encode
}

def predict_credit(input_data):
    """Fungsi untuk mengirim data ke model MLflow"""
    
    # Format payload untuk MLflow (dataframe_records adalah yang paling aman)
    payload = {"dataframe_records": [input_data]}
    headers = {"Content-Type": "application/json"}

    print(f"🚀 Mengirim data ke {URL_MODEL}...")

    try:
        response = requests.post(
            URL_MODEL, 
            data=json.dumps(payload), 
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            hasil = response.json()
            print("✅ Koneksi ke Model Sukses!")
            return hasil
        else:
            print(f"❌ Model merespon Error! Kode: {response.status_code}")
            print(f"Pesan: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Gagal terhubung! Pastikan model sudah di-serve (mlflow models serve).")
        return None

if __name__ == "__main__":
    print("=== PENGUJIAN INFERENCE MODEL ===")
    
    # Jalankan prediksi
    hasil = predict_credit(sample_data)

    # Tampilkan hasil
    if hasil is not None:
        print("\n--- HASIL PREDIKSI ---")
        print(f"Respon Mentah: {hasil}")
        
        if "predictions" in hasil:
            prediksi = hasil["predictions"][0]
            print(f"🎯 Nilai Prediksi (Credit_Mix): {prediksi}")