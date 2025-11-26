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

### CLI
```bash
python src/main.py --data_path data/your_data.csv --target_column target
```

### Web Dashboard
```bash
streamlit run app.py
```

### API Server
```bash
uvicorn api:app --reload
```

## Project Structure

```
├── src/
│   ├── data_processing.py   # Data preprocessing & feature engineering
│   ├── model_training.py    # Model training & optimization
│   ├── evaluation.py        # Model evaluation metrics
│   ├── explainability.py    # SHAP explanations
│   └── main.py             # CLI entry point
├── app.py                   # Streamlit web app
├── api.py                   # FastAPI server
├── verify_pipeline.py       # Verification script
└── requirements.txt         # Dependencies
```

## License

MIT
