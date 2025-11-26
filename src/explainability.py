import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def explain_model(model, X_test, output_path="shap_summary.png"):
    """
    Generates SHAP summary plot for the model.
    """
    print("Generating SHAP explanation...")
    
    # SHAP explainer depends on the model type
    try:
        # Tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Kernel explainer for other models (slower)
        # Using a subset of X_test for speed if needed
        explainer = shap.KernelExplainer(model.predict, X_test.iloc[:100])
        shap_values = explainer.shap_values(X_test.iloc[:100])
        X_test = X_test.iloc[:100]

    # Handle binary classification case where shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Positive class

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")
