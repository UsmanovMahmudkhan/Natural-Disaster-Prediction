import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os
import subprocess
import joblib

def create_dummy_data(filepath):
    """
    Creates a dummy dataset for testing.
    """
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    # Add some missing values
    df.iloc[0:10, 0] = np.nan
    
    # Add a categorical column
    df['category'] = np.random.choice(['A', 'B', 'C'], size=200)
    
    df.to_csv(filepath, index=False)
    print(f"Dummy dataset created at {filepath}")

def run_pipeline(data_path, target_column, output_model_path):
    """
    Runs the AutoML pipeline using the main script.
    """
    cmd = [
        "python", "src/main.py",
        "--data_path", data_path,
        "--target_column", target_column,
        "--output_model_path", output_model_path,
        "--time_series" # Test time series flag
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("Pipeline executed successfully.")
        
        # Verify output file
        if os.path.exists(output_model_path):
            print(f"Model file found at {output_model_path}")
            try:
                pipeline_data = joblib.load(output_model_path)
                print("Pipeline keys:", pipeline_data.keys())
                print("Model type:", type(pipeline_data['model']))
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("Model file NOT found.")
    else:
        print("Pipeline execution failed.")

if __name__ == "__main__":
    data_path = "data/dummy_data.csv"
    target_column = "target"
    output_model_path = "best_model.pkl"
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    create_dummy_data(data_path)
    run_pipeline(data_path, target_column, output_model_path)
