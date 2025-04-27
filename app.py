from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Set backend ke Agg untuk server tanpa GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Pastikan folder static ada
if not os.path.exists('static'):
    os.makedirs('static')

# Inisialisasi variabel global
model = None
scaler = None
data_tahunan = None

def initialize_model():
    global model, scaler, data_tahunan
    
    print("Memuat model ANN...")
    
    try:
        # Load model dengan recompile
        model = tf.keras.models.load_model('pneumonia_prediction_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model ANN berhasil dimuat")
        
        # Load dan preprocess data
        df = pd.read_csv('dinkes-od_18513_jml_kasus_penyakit_pneumonia__kabupatenkota_v2_data.csv')
        data_tahunan = df.groupby('tahun')['jumlah_kasus'].sum().reset_index()
        
        # Inisialisasi scaler
        scaler = MinMaxScaler()
        scaler.fit(data_tahunan)
        print("Data berhasil diproses")
        
    except Exception as e:
        print(f"Error saat memuat model: {str(e)}")
        raise

def predict_cases(year):
    # Normalisasi input
    year_scaled = scaler.transform([[year, 0]])[0, 0].reshape(-1, 1)
    
    # Prediksi
    prediction_scaled = model.predict(year_scaled, verbose=0)
    
    # Denormalisasi hasil
    prediction = scaler.inverse_transform([[year_scaled[0, 0], prediction_scaled[0, 0]]])[0, 1]
    
    return int(prediction)

# Inisialisasi model saat startup
print("Memulai inisialisasi...")
initialize_model()
print("Inisialisasi selesai")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input tahun dari form
        year = int(request.form['year'])
        
        # Lakukan prediksi
        prediction = predict_cases(year)
        
        # Buat dua subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Data Aktual
        sns.scatterplot(data=data_tahunan, x='tahun', y='jumlah_kasus', color='blue', label='Data Aktual', ax=ax1)
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Jumlah Kasus')
        ax1.set_title('Tren Kasus Pneumonia di Jawa Barat')
        
        # Plot 2: Hasil Prediksi vs Data Aktual
        ax2.scatter(data_tahunan['tahun'], data_tahunan['jumlah_kasus'], color='blue', label='Data Aktual')
        ax2.scatter(year, prediction, color='red', s=100, label='Prediksi ANN')
        
        # Atur range tahun untuk plot kedua
        min_year = min(data_tahunan['tahun'].min(), year)
        max_year = max(data_tahunan['tahun'].max(), year)
        ax2.set_xlim(min_year - 1, max_year + 1)
        
        # Tambahkan garis trend
        all_years = np.append(data_tahunan['tahun'].values, year)
        all_cases = np.append(data_tahunan['jumlah_kasus'].values, prediction)
        z = np.polyfit(all_years, all_cases, 1)
        p = np.poly1d(z)
        trend_x = np.linspace(min_year, max_year, 100)
        ax2.plot(trend_x, p(trend_x), "r--", alpha=0.8, label='Trend')
        
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Jumlah Kasus')
        ax2.set_title('Hasil Prediksi ANN vs Data Aktual')
        ax2.legend()
        
        # Atur layout dan simpan
        plt.tight_layout()
        plt.savefig('static/prediction_plot.png', bbox_inches='tight', dpi=300, facecolor='#1a1a1a')
        plt.close()
        
        return jsonify({
            'status': 'success',
            'year': year,
            'prediction': prediction,
            'plot_url': '/static/prediction_plot.png'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 