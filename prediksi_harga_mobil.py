import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import pickle

# Fungsi untuk memuat model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Muat model
model = load_model()

# Fitur yang digunakan dalam model
features = ['year', 'mileage', 'cylinder_size', 'price_credit']

# Antarmuka Streamlit
st.title('Prediksi Harga Mobil Bekas')

year = st.number_input('Tahun Pembuatan', min_value=1990, max_value=2024, value=2010)
mileage = st.number_input('Jarak Tempuh', min_value=0, value=50000)
cylinder_size = st.number_input('Kapasitas Silinder', min_value=1000, max_value=5000, value=2000)
price_credit = st.number_input('Harga Kredit', min_value=0, value=100000000)

input_data = pd.DataFrame([[year, mileage, cylinder_size, price_credit]], columns=features)
prediction = model.predict(input_data)[0]

st.write(f'Prediksi Harga Tunai: {prediction}')
