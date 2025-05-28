from flask import Flask, jsonify, render_template
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('/kaggle/working/model.h5')

# Initialize the scaler
scaler = StandardScaler()

# Dummy data to fit the scaler (use the same data used for training the scaler)
dummy_data = pd.DataFrame({
    'PM2.5': [85],
    'PM10': [124],
    'NO': [1.44],
    'NO2': [20],
    'NOx': [12],
    'NH3': [10],
    'CO': [0.1],
    'SO2': [15],
    'O3': [127],
    'Benzene': [0.20],
    'Toluene': [6],
    'Xylene': [0.2]
})
scaler.fit(dummy_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Example input data
    example_input = pd.DataFrame({
        'PM2.5': [85],
        'PM10': [124],
        'NO': [1.44],
        'NO2': [20],
        'NOx': [12],
        'NH3': [10],
        'CO': [0.1],
        'SO2': [15],
        'O3': [127],
        'Benzene': [0.20],
        'Toluene': [6],
        'Xylene': [0.2]
    })
    input_scaled = scaler.transform(example_input)
    prediction = model.predict(input_scaled)
    aqi = float(prediction[0][0])

    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Moderate"
    elif aqi <= 150:
        category = "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        category = "Unhealthy"
    elif aqi <= 300:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"

    return jsonify({'AQI': aqi, 'Category': category})

if __name__ == '__main__':
    app.run(debug=True)
