import streamlit as st
import pandas as pd
import joblib
import os
from src.data_processing import preprocess_data
from src.model_training import auto_train
from src.evaluation import evaluate_model
from src.explainability import explain_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="AutoML Disaster Prediction", layout="wide")

st.title("AutoML for Natural Disaster Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    target_column = st.selectbox("Select Target Column", df.columns)
    
    if st.button("Run AutoML Pipeline"):
        with st.spinner("Preprocessing data..."):
            try:
                X_train, X_test, y_train, y_test, scaler, label_encoders, imputer = preprocess_data(df, target_column)
                st.success("Data preprocessing complete.")
            except Exception as e:
                st.error(f"Error in preprocessing: {e}")
                st.stop()
        
        with st.spinner("Training and optimizing models... (This may take a while)"):
            best_model = auto_train(X_train, y_train)
            
        if best_model:
            st.success("Model training complete!")
            
            st.subheader("Model Evaluation")
            metrics = evaluate_model(best_model, X_test, y_test)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            
            st.subheader("Model Explainability (SHAP)")
            with st.spinner("Generating SHAP plot..."):
                explain_model(best_model, X_test, "shap_plot.png")
                st.image("shap_plot.png")
            
            # Save model
            joblib.dump(best_model, "best_model_app.pkl")
            with open("best_model_app.pkl", "rb") as f:
                st.download_button("Download Best Model", f, file_name="best_model.pkl")
        else:
            st.error("Model training failed.")
