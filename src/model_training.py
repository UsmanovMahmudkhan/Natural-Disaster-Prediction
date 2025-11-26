from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
except Exception:
    LIGHTGBM_AVAILABLE = False

import catboost as cb
import optuna

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
except Exception:
    XGBOOST_AVAILABLE = False

def get_model(model_name, params=None):
    """
    Returns a model instance based on the model name and parameters.
    """
    if params is None:
        params = {}
        
    if model_name == 'random_forest':
        return RandomForestClassifier(random_state=42, **params)
    elif model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not available.")
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
    elif model_name == 'svm':
        return SVC(probability=True, random_state=42, **params)
    elif model_name == 'logistic_regression':
        return LogisticRegression(random_state=42, max_iter=1000, **params)
    elif model_name == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not available.")
        return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
    elif model_name == 'catboost':
        return cb.CatBoostClassifier(random_state=42, verbose=0, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def objective(trial, model_name, X_train, y_train, cv=3):
    """
    Objective function for Optuna optimization.
    """
    params = {}
    if model_name == 'random_forest':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
        params['max_depth'] = trial.suggest_int('max_depth', 5, 30)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
    elif model_name == 'xgboost':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
    elif model_name == 'svm':
        params['C'] = trial.suggest_float('C', 0.1, 10, log=True)
        params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf'])
    elif model_name == 'logistic_regression':
        params['C'] = trial.suggest_float('C', 0.1, 10, log=True)
        params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    elif model_name == 'lightgbm':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)
    elif model_name == 'catboost':
        params['iterations'] = trial.suggest_int('iterations', 50, 200)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        params['depth'] = trial.suggest_int('depth', 4, 10)

    model = get_model(model_name, params)
    
    # Use cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    return np.mean(scores)

def tune_hyperparameters(model_name, X_train, y_train, cv=3):
    """
    Tunes hyperparameters using Optuna.
    """
    print(f"Tuning {model_name} with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, cv), n_trials=10)
    
    print(f"Best params for {model_name}: {study.best_params}")
    return get_model(model_name, study.best_params)

def create_ensemble(estimators):
    """
    Creates Stacking and Voting classifiers from a list of (name, model) tuples.
    """
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return voting_clf, stacking_clf

def auto_train(X_train, y_train, time_series=False):
    """
    Iterates through available models, tunes them using Optuna, and selects the best one.
    Also trains ensemble models.
    """
    models = ['random_forest', 'svm', 'logistic_regression', 'catboost']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        models.append('lightgbm')
    
    best_model = None
    best_score = -1
    best_model_name = ""
    
    tuned_models = []

    print("Starting AutoML process with Optuna...")
    
    cv_strategy = 3
    if time_series:
        from sklearn.model_selection import TimeSeriesSplit
        print("Using Time Series Cross-Validation...")
        cv_strategy = TimeSeriesSplit(n_splits=3)

    for model_name in models:
        try:
            # Note: tune_hyperparameters also needs to know about cv_strategy
            # For simplicity, we'll pass it or update tune_hyperparameters
            # Let's update tune_hyperparameters to accept cv
            tuned_model = tune_hyperparameters(model_name, X_train, y_train, cv=cv_strategy)
            
            # Evaluate using cross-validation
            scores = cross_val_score(tuned_model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
            mean_score = np.mean(scores)
            
            print(f"{model_name} CV Accuracy: {mean_score:.4f}")
            
            tuned_models.append((model_name, tuned_model))

            if mean_score > best_score:
                best_score = mean_score
                best_model = tuned_model
                best_model_name = model_name
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Train Ensemble Models
    if len(tuned_models) > 1:
        print("Training Ensemble Models...")
        voting_clf, stacking_clf = create_ensemble(tuned_models)
        
        for name, clf in [('Voting Classifier', voting_clf), ('Stacking Classifier', stacking_clf)]:
            try:
                scores = cross_val_score(clf, X_train, y_train, cv=cv_strategy, scoring='accuracy')
                mean_score = np.mean(scores)
                print(f"{name} CV Accuracy: {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = clf
                    best_model_name = name
                    
                    # Fit the best ensemble model on the whole training set
                    best_model.fit(X_train, y_train)
            except Exception as e:
                print(f"Error training {name}: {e}")

    if best_model is not None:
        print(f"Fitting best model ({best_model_name}) on full training set...")
        best_model.fit(X_train, y_train)

    print(f"Best model selected: {best_model_name} with accuracy: {best_score:.4f}")
    return best_model
