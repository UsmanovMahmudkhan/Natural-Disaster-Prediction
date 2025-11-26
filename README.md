# AutoML for Natural Disaster Prediction

An advanced AutoML system for automatically selecting, training, and optimizing machine learning models to predict natural disasters.

## Features

- **Advanced Data Processing**
  - SMOTE for handling class imbalance
  - KNN Imputer for robust missing value handling
  - Automatic feature engineering (interaction terms, polynomial features)

- **State-of-the-Art Models**
  - Random Forest, SVM, Logistic Regression
  - CatBoost, LightGBM (optional), XGBoost (optional)
  - Ensemble methods (Voting & Stacking Classifiers)

- **Optimization & Explainability**
  - Optuna for Bayesian hyperparameter optimization
  - SHAP for model interpretability

- **User Interfaces**
  - Streamlit web dashboard
  - FastAPI REST API
  - Command-line interface

- **Time Series Support**
  - Time Series Cross-Validation

## Installation

```bash
# Clone the repository
git clone https://github.com/UsmanovMahmudkhan/Natural-Disaster-Prediction.git
cd Natural-Disaster-Prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Note: this repository currently contains a focused evaluation utility (see Project Structure). The original higher-level CLI/web/API entrypoints referenced below may not be present in this minimal snapshot.

### Evaluating a trained model
The included utility provides a simple evaluate_model function which prints common classification metrics and returns them as a dictionary.

Example:

```python
from src.evaluation import evaluate_model

# model: a fitted scikit-learn-like estimator
# X_test, y_test: test features and labels
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

The function prints accuracy, precision, recall, F1 and a full classification report, and returns a dict with the metric values.

(If you expected CLI, Streamlit or API entrypoints, those are not present in this repository snapshot.)

## Project Structure

```
├── src/
│   └── evaluation.py        # Model evaluation metrics (evaluate_model)
└── README.md               # This file
```

Notes:
- This README has been updated to reflect the current repository contents. Other modules referenced in earlier documentation (e.g., main.py, model_training.py, explainability.py, app.py, api.py) are not present in this snapshot.
- If you have a fuller project layout elsewhere, merge or add those files to restore the full workflow (training, explainability, web/API interfaces).