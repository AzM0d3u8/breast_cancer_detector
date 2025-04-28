
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f'Fold {fold+1}')
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(30,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    
    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=32,
              callbacks=[early_stop, reduce_lr], verbose=0)
    
    # Evaluate
    preds = (model.predict(X_val) > 0.5).astype('int32')
    acc = accuracy_score(y_val, preds)
    acc_scores.append(acc)
    print(f'Fold {fold+1} Accuracy: {acc:.4f}')

# Final model on all data
model.save('models/breast_cancer_detector.h5')
print(f'Mean Cross-Validation Accuracy: {np.mean(acc_scores):.4f}')
