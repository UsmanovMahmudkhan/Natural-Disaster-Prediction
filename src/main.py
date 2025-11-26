import argparse
import joblib
import os
from src.data_processing import load_data, preprocess_data
from src.model_training import auto_train
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="AutoML for Natural Disaster Prediction")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--target_column', type=str, required=True, help="Name of the target column")
    parser.add_argument('--output_model_path', type=str, default='best_model.pkl', help="Path to save the best model")
    parser.add_argument('--time_series', action='store_true', help="Enable Time Series Cross-Validation")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    try:
        df = load_data(args.data_path)
    except Exception as e:
        print(e)
        return

    print("Preprocessing data...")
    try:
        X_train, X_test, y_train, y_test, scaler, label_encoders, imputer = preprocess_data(df, args.target_column)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    print("Starting model training and selection...")
    best_model = auto_train(X_train, y_train, time_series=args.time_series)
    
    if best_model:
        print("Evaluating best model...")
        evaluate_model(best_model, X_test, y_test)
        
        print(f"Saving pipeline to {args.output_model_path}...")
        pipeline_data = {
            'model': best_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'imputer': imputer
        }
        joblib.dump(pipeline_data, args.output_model_path)
        print("Pipeline saved successfully.")
    else:
        print("Model training failed.")

if __name__ == "__main__":
    main()
