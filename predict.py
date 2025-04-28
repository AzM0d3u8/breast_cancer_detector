
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model('models/breast_cancer_detector.h5')
scaler = joblib.load('models/scaler.pkl')

# Example patient (random input here, replace with actual input)
patient = np.random.rand(1, 30)

# Scale
patient_scaled = scaler.transform(patient)

# Predict
prediction = model.predict(patient_scaled)
predicted_class = (prediction > 0.5).astype('int32')

print(f'Predicted class: {predicted_class[0][0]} (1=Benign, 0=Malignant)')
