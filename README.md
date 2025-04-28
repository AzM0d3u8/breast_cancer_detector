
# 🧠 Breast Cancer Detection using Deep Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 📖 Overview

This project focuses on **early and accurate detection of breast cancer** using a deep learning-based neural network.  
It classifies tumors as either **Benign** or **Malignant** based on fine-grained features extracted from biopsy data.

The aim is to help automate preliminary diagnosis and assist healthcare professionals.

---

## 📂 Repository Structure

breast_cancer_detector/ ├── models/ # Saved models and scalers │ ├── breast_cancer_detector_model.keras │ └── scaler.pkl │ ├── notebooks/ # Jupyter notebooks │ └── breast_cancer_detection.ipynb │ ├── scripts/ # Python scripts for training and prediction │ ├── train.py │ └── predict.py │ ├── assets/ # Images, banners, README assets │ └── banner_cyberpunk_ai_theme.jpg │ ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── LICENSE # MIT License file

---

## 🧬 Dataset

- **Source**: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- **Features**: 30 numerical input features (e.g., radius, texture, smoothness).
- **Target Classes**: 
  - `0`: Malignant
  - `1`: Benign
- **Total Samples**: 569 instances.

---

## ⚙️ Features of This Project

- Deep Neural Network with 5 layers and **Dropout Regularization**.
- **EarlyStopping** and **ReduceLROnPlateau** callbacks to prevent overfitting.
- **Keras** model saving (`.keras`) and **scaler** serialization (`.pkl`).
- **Confusion Matrix**, **ROC Curve**, and **Training History** visualization.
- CSV Upload for batch predictions inside Colab.
- Manual patient input using **ipywidgets**.
- Highly modular scripts for training and inference.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AzM0d3u8/breast_cancer_detector.git
cd breast_cancer_detector
