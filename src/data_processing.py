import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def generate_features(X):
    """
    Generates interaction terms and polynomial features.
    """
    # Select numerical columns for interaction
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(X[numerical_cols])
        feature_names = poly.get_feature_names_out(numerical_cols)
        
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
        
        # Drop original columns to avoid duplication if PolynomialFeatures keeps them (it doesn't with interaction_only=True usually, but let's be safe)
        # Actually interaction_only=True keeps x1, x2, x1*x2. We want to keep original X and add interactions.
        # Let's just add the new interaction columns that are not in X
        
        X = pd.concat([X, poly_df], axis=1)
        
        # Remove duplicate columns if any
        X = X.loc[:, ~X.columns.duplicated()]
        
    return X

def preprocess_data(df, target_column):
    """
    Preprocesses the data:
    - Handles missing values (KNN Imputer).
    - Encodes categorical variables.
    - Generates features.
    - Scales numerical features.
    - Handles class imbalance (SMOTE).
    - Splits into train and test sets.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data first to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Separate numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unknown labels in test set
        # For simplicity, we fit on combined unique values or handle exceptions. 
        # A robust way is to fit on train and map unknown to a special category or mode.
        # Here we'll fit on train and use a custom transform that handles unknowns? 
        # Or just concat for fitting (slightly leaky but common for LabelEncoder if domain is fixed).
        # Let's stick to fitting on train and hoping test has same labels, or use OneHotEncoder.
        # For tree models, LabelEncoder is okay.
        
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        
        # For test, we need to handle unseen labels. 
        # A simple hack: map unseen to the first class or mode.
        X_test[col] = X_test[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
        X_test[col] = le.transform(X_test[col])
        
        label_encoders[col] = le

    # Encode target if categorical
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        target_le = LabelEncoder()
        y_train = target_le.fit_transform(y_train)
        y_test = target_le.transform(y_test)
    
    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5)
    if len(numerical_cols) > 0:
        # We need to impute on the whole X_train (including encoded categoricals if we want, but usually just numericals)
        # KNNImputer works on numericals. Label encoded categoricals are treated as numericals (ordinal).
        # Let's impute everything.
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Feature Engineering
    X_train = generate_features(X_train)
    X_test = generate_features(X_test)
    
    # Align columns (in case feature generation produced different columns due to data differences, though unlikely with standard PolynomialFeatures)
    # But PolynomialFeatures depends on input columns. If X_train and X_test have same cols, output is same.
    
    # Scale numerical features
    scaler = StandardScaler()
    # Re-identify numerical columns after feature engineering (all generated are numerical)
    # Actually, all columns are numerical now after encoding and imputation.
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Handle Class Imbalance (SMOTE) - ONLY on Train data
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler, label_encoders, imputer
