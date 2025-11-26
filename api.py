from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from src.data_processing import generate_features

app = FastAPI(title="AutoML Disaster Prediction API")

# Load pipeline
try:
    pipeline_data = joblib.load("best_model.pkl")
    model = pipeline_data['model']
    scaler = pipeline_data['scaler']
    label_encoders = pipeline_data['label_encoders']
    imputer = pipeline_data['imputer']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionRequest(BaseModel):
    data: list[dict]

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(request.data)
        
        # Preprocessing steps (must match training)
        # 1. Encode categoricals
        for col, le in label_encoders.items():
            if col in df.columns:
                # Handle unknown labels
                df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
        
        # 2. Impute
        # Ensure columns match what imputer expects (might need to reorder or select)
        # Imputer was fitted on X_train (numericals + encoded categoricals)
        # We assume input df has same columns
        df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
        
        # 3. Generate Features
        df_features = generate_features(df_imputed)
        
        # 4. Scale
        # Scaler was fitted on X_train (after feature gen)
        X_scaled = scaler.transform(df_features)
        
        # Predict
        prediction = model.predict(X_scaled)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AutoML Disaster Prediction API is running"}
