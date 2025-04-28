
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
├── assets/                  # Images, banners, plots, README assets
│   ├── banner_cyberpunk_ai_theme.jpg
│   └── # Add paths to saved plots like confusion_matrix.png, roc_curve.png, training_history.png
├── data/                    # (Optional) Directory for raw/processed data if not loading directly
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License file

