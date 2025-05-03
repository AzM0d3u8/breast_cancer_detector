
# 🧠 Breast Cancer Detection using Deep Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🧬 Dataset

- **Source**: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- **Features**: 30 numerical input features derived from digitized images of fine needle aspirates (FNAs) of breast masses (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension - mean, standard error, and worst values for each).
- **Target Classes**:
  - `0`: Malignant
  - `1`: Benign
- **Total Samples**: 569 instances (357 Benign, 212 Malignant).

---

⚙️ Features of This Project
- Deep Learning Model: Sequential Deep Neural Network built with Keras (TensorFlow backend). Features 5 layers with Dropout regularization to prevent overfitting.
- Training Callbacks: Implements EarlyStopping to halt training when validation performance degrades and ReduceLROnPlateau to adjust the learning rate dynamically.
- Preprocessing: Utilizes StandardScaler from Scikit-learn for feature scaling, saved for consistent application during inference.
- Model Persistence: Saves the trained Keras model (.keras format) and the scaler object (.pkl using joblib).
- Evaluation & Visualization: Generates and displays:
- Confusion Matrix
- ROC Curve and AUC Score
- Training & Validation Accuracy/Loss History Plots
- Prediction Interfaces:
- Batch Prediction: Accepts CSV file uploads for prediction (via Notebook or script).
- Manual Input: Interactive patient data input using ipywidgets within the Jupyter Notebook.
- Modularity: Separate Python scripts (train.py, predict.py) for streamlined training and inference workflows.

---

## 📊 Model Evaluation

| Metric          | Value    |
|-----------------|----------|
| Accuracy        | 96.49%   |
| Precision       | ~96%     |
| Recall          | ~96%     |
| F1 Score        | ~96%     |
| ROC AUC Score   | ~99%     |

**Key Points:**
- High **precision** ensures fewer false positives (important for cancer diagnosis).
- High **recall** ensures fewer false negatives (catching almost all true cancer cases).
- AUC close to **1.0** indicates the model discriminates extremely well between benign and malignant cases.

---

🧠 Challenges & Solutions
Designing a neural network for medical diagnostics involves technical and ethical precision. Below are key complexities tackled during development:

Challenge	Details & Solution
- 🔄 Data Normalization	Raw features span different scales, which destabilize training. We applied StandardScaler to normalize all features to zero mean and unit variance.
- 🧱 Architecture Design	Overly simple networks underfit; too complex ones overfit. After experimentation, a 5-layer dense network with gradually decreasing neurons and dropout regularization was used for optimal generalization.
- 📉 Overfitting on Small Dataset	With only 569 samples, deep models risk memorizing data. Used Dropout, EarlyStopping, and ReduceLROnPlateau callbacks to regularize and prevent overfitting.
- 🧾 Output Interpretation	The model outputs probabilities (via sigmoid). These are thresholded at 0.5 to convert into binary classifications (0 = malignant, 1 = benign).
- 📊 Metric Selection	Accuracy can mislead in imbalanced datasets. We computed Precision, Recall, F1 Score, and AUC-ROC to assess reliability—especially for detecting malignancy (false negatives are risky).
- 💾 Model + Scaler Persistence	Saved both the trained model (.h5) and scaler (.pkl) to allow seamless reuse in inference pipelines and web deployment.
- 🧑‍💻 Usability in Colab	Non-technical users may struggle with code. We used ipywidgets for GUI-based input (single or batch via CSV), ensuring broader accessibility.
├── README.md                # Project documentation
└── LICENSE                  # MIT License file
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

```
## 🧪 How It Works

1. The neural network receives 30 features from breast mass images.
2. The input is scaled using a trained `StandardScaler`.
3. The DNN processes the data through nonlinear transformations.
4. A final sigmoid layer outputs the probability of malignancy.
5. Prediction threshold (0.5) is used to classify as Benign (1) or Malignant (0).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AzM0d3u8/breast_cancer_detector.git
cd breast_cancer_detector

```
2. Install Requirements
```bash
pip install -r requirements.txt
# or install manually
pip install tensorflow scikit-learn pandas matplotlib joblib ipywidgets

```
## 🧠 Future Enhancements

- Integration with Flask or Streamlit for web-based prediction.

- Extended dataset compatibility and feature exploration.

- AutoML integration to optimize hyperparameters.

- Explainable AI (XAI) integration using SHAP or LIME.

## ❤️ Purpose

This project is built with a vision to assist early-stage detection of breast cancer using AI, making diagnostics faster, reliable, and accessible.

“Let this not just be a model, but a shield in someone’s fight against cancer.”

## 📜 License 

MIT License — free to use with attribution.
