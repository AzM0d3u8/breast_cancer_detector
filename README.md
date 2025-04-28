
# 🧠 Breast Cancer Detection using Deep Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

# 🚀 Breast Cancer Detection using Deep Neural Networks



## 🧬 Dataset

- **Source**: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- **Features**: 30 numerical input features derived from digitized images of fine needle aspirates (FNAs) of breast masses (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension - mean, standard error, and worst values for each).
- **Target Classes**:
  - `0`: Malignant
  - `1`: Benign
- **Total Samples**: 569 instances (357 Benign, 212 Malignant).

---

## 📂 Project Structure

```text
breast_cancer_detector/
├── models/                  # Saved models and scalers
│   ├── breast_cancer_detector_model.keras
│   └── scaler.pkl
├── notebooks/               # Jupyter notebooks
│   └── breast_cancer_detection.ipynb
├── scripts/                 # Python scripts for training and prediction
│   ├── train.py
│   └── predict.py
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License file
---

⚙️ Features of This Project
Deep Learning Model: Sequential Deep Neural Network built with Keras (TensorFlow backend). Features 5 layers with Dropout regularization to prevent overfitting.
Training Callbacks: Implements EarlyStopping to halt training when validation performance degrades and ReduceLROnPlateau to adjust the learning rate dynamically.
Preprocessing: Utilizes StandardScaler from Scikit-learn for feature scaling, saved for consistent application during inference.
Model Persistence: Saves the trained Keras model (.keras format) and the scaler object (.pkl using joblib).
Evaluation & Visualization: Generates and displays:
Confusion Matrix
ROC Curve and AUC Score
Training & Validation Accuracy/Loss History Plots
Prediction Interfaces:
Batch Prediction: Accepts CSV file uploads for prediction (via Notebook or script).

Modularity: Separate Python scripts (train.py, predict.py) for streamlined training and inference workflows.
