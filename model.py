import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Baca dataset
df = pd.read_csv('dinkes-od_18513_jml_kasus_penyakit_pneumonia__kabupatenkota_v2_data.csv')

# Preprocessing data
# Ambil data tahun dan jumlah kasus
data = df.groupby('tahun')['jumlah_kasus'].sum().reset_index()

# Visualisasi Data Awal
plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x='tahun', y='jumlah_kasus', color='blue', label='Data Aktual')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Kasus')
plt.title('Tren Kasus Pneumonia di Jawa Barat')
plt.legend()
plt.savefig('static/data_aktual.png')
plt.close()

# Normalisasi data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Pisahkan data menjadi input (X) dan output (Y)
X = data_scaled[:, 0].reshape(-1, 1)  # Tahun sebagai input
Y = data_scaled[:, 1]  # Jumlah kasus sebagai output

# Split data menjadi training dan testing set (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Membangun model ANN
def build_model():
    model = Sequential([
        Input(shape=(1,)),  # Input layer dengan shape yang eksplisit
        Dense(8, activation='relu'),  # Hidden layer pertama dengan ukuran lebih kecil
        Dense(4, activation='relu'),  # Hidden layer kedua dengan ukuran lebih kecil
        Dense(1, activation='linear')  # Output layer
    ])
    
    # Kompilasi model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Fungsi untuk melatih model
def train_model(model, X_train, Y_train, X_test, Y_test):
    # Tambahkan early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Latih model dengan epochs yang lebih sedikit
    history = model.fit(
        X_train, Y_train,
        epochs=50,  # Kurangi epochs
        batch_size=4,  # Batch size kecil untuk dataset kecil
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# Fungsi untuk melakukan prediksi
def predict_cases(model, year):
    # Normalisasi tahun input
    year_scaled = scaler.transform([[year, 0]])[0, 0].reshape(-1, 1)
    
    # Prediksi dengan model
    prediction_scaled = model.predict(year_scaled, verbose=0)  # Tambahkan verbose=0 untuk mengurangi output
    
    # Balikkan skala hasil prediksi
    prediction = scaler.inverse_transform([[year_scaled[0, 0], prediction_scaled[0, 0]]])[0, 1]
    
    return int(prediction)

if __name__ == '__main__':
    # Build dan train model
    model = build_model()
    history = train_model(model, X_train, Y_train, X_test, Y_test)
    
    # Evaluasi model
    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Simpan model
    model.save('pneumonia_prediction_model.h5')
    print("Model berhasil disimpan") 