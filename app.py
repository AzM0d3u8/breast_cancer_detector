import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import io
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_models():
    model = load_model('models/breast_cancer_detector_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

# Initialize SHAP explainer
@st.cache_resource
def get_shap_explainer(model, background_data):
    explainer = shap.DeepExplainer(model, background_data)
    return explainer

# Load feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def plot_shap_summary(shap_values, feature_names):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    return plt.gcf()

def plot_shap_force(shap_values, feature_names, feature_values):
    plt.figure(figsize=(10, 4))
    shap.force_plot(
        shap_values[0],
        feature_values,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    return plt.gcf()

def main():
    st.title("🏥 Breast Cancer Detection using Deep Neural Networks")
    st.write("This app uses a deep neural network to predict whether a breast mass is benign or malignant based on various features.")

    # Load model and scaler
    try:
        model, scaler = load_models()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["CSV Upload", "Manual Input", "SHAP Analysis"])

    with tab1:
        st.header("Upload CSV File")
        st.write("Upload a CSV file containing the features. The file should have 30 columns with the following features:")
        st.write(", ".join(feature_names))
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check if all required features are present
                if len(df.columns) != 30:
                    st.error("The CSV file must contain exactly 30 features.")
                    return
                
                # Scale the features
                X_scaled = scaler.transform(df)
                
                # Make predictions
                predictions = model.predict(X_scaled)
                predictions_binary = (predictions > 0.5).astype(int)
                
                # Display results
                st.subheader("Predictions")
                results_df = pd.DataFrame({
                    'Prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions_binary],
                    'Confidence': [f"{float(p[0])*100:.2f}%" for p in predictions]
                })
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    with tab2:
        st.header("Manual Input")
        st.write("Enter the values for each feature:")
        
        # Create input fields for each feature
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1
                )
        
        if st.button("Predict"):
            # Convert input to array and scale
            input_array = np.array([list(input_data.values())])
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_binary = (prediction > 0.5).astype(int)
            
            # Display result
            st.subheader("Prediction Result")
            result = "Malignant" if prediction_binary[0] == 1 else "Benign"
            confidence = float(prediction[0]) * 100
            
            # Create a colored box for the result
            if result == "Benign":
                st.success(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
            else:
                st.error(f"Prediction: {result} (Confidence: {confidence:.2f}%)")

    with tab3:
        st.header("SHAP Analysis")
        st.write("This section provides insights into how each feature contributes to the model's predictions.")
        
        # Load sample data for background
        try:
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            X_sample = scaler.transform(data.data[:100])  # Use first 100 samples as background
            
            # Initialize SHAP explainer
            explainer = get_shap_explainer(model, X_sample)
            
            # Create two columns for different SHAP visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Importance Summary")
                # Calculate SHAP values for the background data
                shap_values = explainer.shap_values(X_sample)
                # Plot summary
                fig = plot_shap_summary(shap_values, feature_names)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Feature Impact Analysis")
                st.write("Enter values to see how each feature impacts the prediction:")
                
                # Create input fields for features
                analysis_input = {}
                for feature in feature_names:
                    analysis_input[feature] = st.number_input(
                        f"{feature} (Analysis)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        key=f"shap_{feature}"
                    )
                
                if st.button("Analyze Feature Impact"):
                    # Convert input to array and scale
                    analysis_array = np.array([list(analysis_input.values())])
                    analysis_scaled = scaler.transform(analysis_array)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(analysis_scaled)
                    
                    # Plot force plot
                    fig = plot_shap_force(shap_values, feature_names, analysis_scaled[0])
                    st.pyplot(fig)
                    plt.close()
                    
                    # Display feature contributions
                    st.subheader("Feature Contributions")
                    contributions = pd.DataFrame({
                        'Feature': feature_names,
                        'Contribution': shap_values[0]
                    })
                    contributions = contributions.sort_values('Contribution', key=abs, ascending=False)
                    st.dataframe(contributions)
        
        except Exception as e:
            st.error(f"Error in SHAP analysis: {str(e)}")

if __name__ == "__main__":
    main() 