import time
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# ==============================================================================
# DEFINISI 10 METRIKS UNTUK KRITERIA ADVANCE (4 PTS)
# ==============================================================================

# 1. Counter: Total request prediksi yang masuk
PREDICTION_REQUESTS = Counter(
    'model_prediction_requests_total', 
    'Total number of prediction requests received'
)

# 2. Counter: Total sukses prediksi
PREDICTION_SUCCESS = Counter(
    'model_prediction_success_total', 
    'Total number of successful predictions'
)

# 3. Counter: Total error/gagal prediksi
PREDICTION_ERRORS = Counter(
    'model_prediction_errors_total', 
    'Total number of prediction errors'
)

# 4. Gauge: Status keaktifan model (1 = Aktif, 0 = Mati)
MODEL_STATUS = Gauge(
    'model_status_active', 
    'Current status of the model serving instance (1 for active)'
)

# 5. Gauge: Penggunaan Memori simulasi (dalam MB)
MEMORY_USAGE = Gauge(
    'model_memory_usage_megabytes', 
    'Simulated memory usage of the model instance in MB'
)

# 6. Gauge: Penggunaan CPU simulasi (dalam persen)
CPU_USAGE = Gauge(
    'model_cpu_usage_percentage', 
    'Simulated CPU usage percentage of the model instance'
)

# 7. Histogram: Latensi / Waktu pemrosesan prediksi
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds', 
    'Time taken to process prediction request',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# 8. Summary: Ukuran data input (misal jumlah fitur atau panjang string)
INPUT_DATA_SIZE = Summary(
    'model_input_data_size_bytes', 
    'Summary of input data size processed'
)

# 9. Gauge: Nilai rata-rata probabilitas/skor prediksi terbaru
LATEST_PREDICTION_SCORE = Gauge(
    'model_latest_prediction_score', 
    'The prediction score/probability of the latest inference request'
)

# 10. Counter: Total deteksi anomali atau data out-of-bounds
OUT_OF_BOUNDS_INPUTS = Counter(
    'model_out_of_bounds_inputs_total', 
    'Total number of input features that fall outside expected bounds'
)

# ==============================================================================
# FUNGSI SIMULASI JALANNYA MODEL
# ==============================================================================
def process_prediction_simulated():
    # Set status model aktif
    MODEL_STATUS.set(1)
    
    # Simulasi penggunaan resource sistem
    MEMORY_USAGE.set(random.uniform(150.0, 250.0))  # Antara 150MB - 250MB
    CPU_USAGE.set(random.uniform(5.0, 45.0))       # Antara 5% - 45%

    # Mulai hitung latensi
    start_time = time.time()
    
    PREDICTION_REQUESTS.inc()
    
    try:
        # Simulasi waktu komputasi model (latensi)
        processing_time = random.uniform(0.02, 0.4)
        time.sleep(processing_time)
        
        # Simulasi ukuran data input
        INPUT_DATA_SIZE.observe(random.randint(50, 500))
        
        # Simulasi hasil score prediksi model (misal probabilitas 0.0 - 1.0)
        score = random.uniform(0.1, 0.99)
        LATEST_PREDICTION_SCORE.set(score)
        
        # Simulasi jika ada input yang aneh / out of bounds (probabilitas kecil)
        if random.random() > 0.95:
            OUT_OF_BOUNDS_INPUTS.inc()
            
        # Simulasi error acak (misal 2% kemungkinan error)
        if random.random() < 0.02:
            raise Exception("Simulated Model Inference Timeout")
            
        PREDICTION_SUCCESS.inc()
        
    except Exception as e:
        PREDICTION_ERRORS.inc()
        print(f"Error encountered: {e}")
        
    finally:
        # Catat durasi pemrosesan ke histogram
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)

if __name__ == '__main__':
    # Jalankan server exporter di port 8000
    print("Starting Prometheus Exporter on port 8000...")
    start_http_server(8000)
    
    # Loop terus-menerus untuk memperbarui data metrik secara dinamis
    while True:
        process_prediction_simulated()
        time.sleep(2)  # Simulasi request datang setiap 2 detik