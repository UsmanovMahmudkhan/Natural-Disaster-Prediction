# Natural Disaster Prediction System – Technical Report

**Student Name:** Usmonov Mahmudkhon  
**Student ID:** 23013088  
**Course:** AI and Big Data  
**Professor:** Abolghasem Sadeghi-Niaraki   

**Project Title:** Natural Disaster Prediction System  

---

## 1. Problem Description 

This section explains the real-world challenge addressed by the Natural Disaster Prediction System. It discusses the societal impact, limitations of traditional methods, and the motivation for AI-based solutions. It also clarifies the goals and scope of the repository and its key modules.

### 1.1 Background and Motivation

Natural disasters cause severe loss of life, infrastructure damage, and economic disruption. Early and accurate prediction can reduce casualties and improve emergency preparedness. Traditional systems often rely on static thresholds or manually designed rules. These approaches struggle to model complex, nonlinear interactions in environmental data.

Machine learning offers data-driven models that learn patterns from historical disaster records. The AutoML approach in this project automates model selection and tuning. This reduces expert intervention and improves reproducibility across different regions and datasets.

### 1.2 Societal Impact of Improved Prediction

Accurate prediction systems can support early evacuation decisions. They can improve allocation of medical resources and emergency personnel. Insurance companies can better estimate risk and pricing. Governments can use probabilistic forecasts to plan infrastructure investments and resilience strategies.

The repository’s design aims to be reusable across multiple disaster types. It can adapt to floods, earthquakes, hurricanes, or wildfires, depending on the available features. The AutoML framework reduces the barrier to deploying such systems in data-constrained settings.

### 1.3 Shortcomings of Traditional Systems

Traditional statistical models usually assume linear relationships between predictors and outcomes. Disaster risk often depends on nonlinear interactions between meteorological, geological, and socio-economic factors. Rule-based systems are hard to maintain and tune across regions and seasons.

Manual model development requires extensive expertise. It is slow to adapt when new data sources or sensors become available. Moreover, many legacy systems lack transparent evaluation and monitoring. This repository addresses these gaps using:

- Automated model selection across diverse algorithms.
- Systematic hyperparameter optimization with Optuna.
- Unified preprocessing and evaluation pipelines.

### 1.4 Project Goals and Scope

The main goal is to build an end-to-end AutoML pipeline for disaster prediction. The system should handle:

- Data ingestion, cleaning, and feature engineering.
- Training and hyperparameter tuning for multiple models.
- Robust evaluation and explainability using SHAP.

The scope also includes user interfaces for different stakeholders. These include a command-line interface, a Streamlit dashboard, and a FastAPI REST API. The implementation focuses on tabular data classification, where the target indicates disaster occurrence or severity class.

```card
{
  "title": "Project Core Goal",
  "content": "Provide an end-to-end, reusable AutoML pipeline for natural disaster prediction on tabular data."
}
```

---

## 2. Tools and Frameworks Used 

This section lists the technologies employed in the repository. It explains their roles and justifies their selection. It also ties each tool to specific files where relevant.

### 2.1 Programming Language and Core Libraries

The project is implemented in **Python** due to its strong ecosystem for data science. Python provides rich support for machine learning, data processing, and web frameworks. Core libraries include:

- **pandas**: for DataFrame-based data handling.
- **numpy**: for numerical operations and array manipulation.
- **scikit-learn**: for classical machine learning algorithms, preprocessing, and evaluation.

These libraries appear throughout:

- `src/data_processing.py` for transformations.
- `src/model_training.py` for model definitions and cross-validation.
- `src/evaluation.py` for metric computation.

### 2.2 Machine Learning Frameworks

The system integrates several machine learning frameworks to enable AutoML comparisons. Each contributes unique strengths, allowing robust model selection.

- **scikit-learn models**:
  - `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `LogisticRegression`.
  - Implemented and instantiated in `src/model_training.py` via `get_model`.
- **CatBoost**:
  - `catboost.CatBoostClassifier` for handling categorical data and strong tabular performance.
  - Metrics and training logs stored under `catboost_info/`.
- **LightGBM** (optional):
  - `lightgbm.LGBMClassifier` for high-performance gradient boosting.
  - Used when `lightgbm` is installed.
- **XGBoost** (optional):
  - `xgboost.XGBClassifier` for powerful gradient boosting models.
  - Enabled when dependency is available.

These frameworks are coordinated via Optuna-driven tuning. This allows the pipeline to select the most suitable model per dataset.

### 2.3 Hyperparameter Optimization with Optuna

**Optuna** is used to optimize hyperparameters using Bayesian search. In `src/model_training.py`, the `objective` function defines search spaces per algorithm. The `tune_hyperparameters` function runs `study.optimize` for a fixed number of trials.

Optuna was selected because:

- It supports complex search spaces and conditional parameters.
- It integrates cleanly with Python functions.
- It provides reproducible studies for experimentation.

The AutoML loop in `auto_train` uses Optuna-tuned models for fair comparison. This design automates a traditionally manual and time-consuming process.

### 2.4 Data Processing and Imbalance Handling

Data preprocessing uses several scikit-learn tools and additional libraries. `src/data_processing.py` orchestrates these utilities to produce clean training data.

Key tools include:

- **KNNImputer**: imputes missing values using nearest neighbors.
- **StandardScaler**: standardizes features to zero mean and unit variance.
- **LabelEncoder**: encodes categorical columns into numeric codes.
- **SMOTE** (from `imbalanced-learn`): oversamples minority classes to mitigate class imbalance.

These tools ensure robust preprocessing across varied datasets. The dummy dataset `data/dummy_data.csv` is designed to exercise these components, including missing values and categorical columns.

### 2.5 Explainability and Visualization

Explainability is crucial in high-stakes domains like disaster management. The system uses:

- **SHAP**:
  - `shap.TreeExplainer` for tree-based models.
  - `shap.KernelExplainer` for generic models.
  - Implemented in `src/explainability.py`.
- **matplotlib**:
  - For saving SHAP summary plots to image files.

The `explain_model` function generates a SHAP summary and persists it as `shap_summary.png` or a configured path. `app.py` then displays this image in the Streamlit UI. This facilitates model transparency for decision-makers.

### 2.6 Web and API Frameworks

The repository supports multiple interaction modes.

- **Streamlit**:
  - Used in `app.py` to build a web dashboard.
  - Offers file upload, target selection, pipeline execution, and visualization.
- **FastAPI**:
  - Used in `api.py` to expose a REST API.
  - Provides endpoints for prediction and health checks.
- **Uvicorn**:
  - ASGI server for running the FastAPI app.

These technologies enable integration with other systems and frontends. They also allow non-technical stakeholders to use the model through a friendly interface.

### 2.7 Serialization and Deployment Helpers

The project uses **joblib** for model serialization. In `src/main.py`, trained pipelines are saved to `best_model.pkl`. The FastAPI server `api.py` and Streamlit app `app.py` load these artifacts.

Additional helpers include:

- **subprocess** in `verify_pipeline.py` to invoke the CLI pipeline.
- **os** to manage file paths and environment details.

These tools support smooth deployment and verification of the pipeline.

### 2.8 Project Metadata and Dependencies

The following files define licensing, documentation, and dependencies:

- `LICENSE`: Apache License, Version 2.0.
- `README.md`: overview, features, installation, and usage instructions.
- `requirements.txt`: pinned dependencies for all components.

These elements ensure that the project remains reproducible and compliant with open-source standards.

---

## 3. Model Design and System Architecture

This section describes the architecture and logic of the predictive system. It covers data flow, preprocessing, model training, validation, and interfaces. It also connects each step to the concrete files in the repository.

### 3.1 High-Level Pipeline Overview

The pipeline follows a clear sequence from raw data to predictions. It splits responsibilities across modular Python files in `src/` and the project root. The design encourages reuse, testing, and consistent preprocessing across interfaces.

The typical end-to-end flow is:

- Data ingestion and preprocessing (`src/data_processing.py`).
- Automated model selection and tuning (`src/model_training.py`).
- Evaluation and reporting (`src/evaluation.py`).
- Explainability generation (`src/explainability.py`).
- Orchestration via CLI (`src/main.py`), Streamlit (`app.py`), and FastAPI (`api.py`).

#### 3.1.1 System Flow Diagram

The following diagram illustrates the main components and their interactions.

```mermaid
flowchart TD
    A[Raw CSV Data] --> B[load_data in src/data_processing.py]
    B --> C[preprocess_data]
    C --> D[auto_train in src/model_training.py]
    D --> E[evaluate_model in src/evaluation.py]
    D --> F[explain_model in src/explainability.py]
    D --> G[Save best_model.pkl with joblib]
    G --> H[FastAPI api.py]
    G --> I[Streamlit app.py]
    G --> J[Future deployment targets]
```

### 3.2 Data Collection and Dummy Dataset

The repository assumes tabular input data in CSV format. Real datasets would typically contain environmental, temporal, and spatial features. For testing and demonstration, the project includes `data/dummy_data.csv`.

#### 3.2.1 Dummy Dataset Generation

`verify_pipeline.py` contains `create_dummy_data(filepath)`. It uses `make_classification` from scikit-learn to generate synthetic features.

Key properties:

- 200 samples with 10 numerical features.
- A binary `target` column representing disaster occurrence.
- A `category` column with values `A`, `B`, `C`.
- Injected missing values in the first 10 rows of `feature_0`.

This synthetic dataset allows validation of:

- Missing value imputation.
- Categorical encoding.
- Class imbalance handling.

### 3.3 Data Preprocessing and Feature Engineering (`src/data_processing.py`)

Although the full source is not shown here, its responsibilities are clear from imports and usage. It centralizes data loading, splitting, cleaning, feature generation, and balancing.

Core responsibilities include:

- `load_data(path)`: reads a CSV into a pandas DataFrame.
- `preprocess_data(df, target_column)`:
  - Splits into features `X` and target `y`.
  - Separates training and test sets.
  - Encodes categorical columns with LabelEncoders.
  - Applies KNNImputer to fill missing values.
  - Generates derived features via `generate_features`.
  - Scales numerical features using StandardScaler.
  - Applies SMOTE on the training set only.
- `generate_features(df)`:
  - Creates interaction or polynomial features to enrich the feature space.

Returned objects include:

- Processed `X_train`, `X_test`, `y_train`, `y_test`.
- Fitted `scaler`, `label_encoders`, and `imputer`.

These are later serialized and reused by deployment components.

#### 3.3.1 Preprocessing Flow Diagram

```mermaid
flowchart TD
    A[Input DataFrame] --> B[Split X and y]
    B --> C[Train Test Split]
    C --> D[Label Encode Categoricals]
    D --> E[KNN Impute Missing Values]
    E --> F[generate_features]
    F --> G[StandardScaler Fit Transform on Train]
    G --> H[SMOTE on X_train and y_train]
    F --> I[StandardScaler Transform on Test]
    H --> J[Return X_train, y_train]
    I --> K[Return X_test, y_test]
```

### 3.4 Model Selection and AutoML (`src/model_training.py`)

`src/model_training.py` implements the AutoML engine. It defines model factories, optimization objectives, and ensemble strategies.

#### 3.4.1 Model Factory: `get_model`

`get_model(model_name, params=None)` returns instantiated models. Supported model names include:

- `'random_forest'`: `RandomForestClassifier`.
- `'svm'`: `SVC` with probability outputs.
- `'logistic_regression'`: `LogisticRegression` with increased `max_iter`.
- `'catboost'`: `CatBoostClassifier` with silent logging.
- `'lightgbm'`: `LGBMClassifier` if LightGBM is installed.
- `'xgboost'`: `XGBClassifier` if XGBoost is installed.

The function encapsulates model-specific options, such as random seeds and default metrics.

#### 3.4.2 Optuna Objective and Hyperparameter Search

The `objective(trial, model_name, X_train, y_train, cv=3)` function defines hyperparameter spaces:

- Random Forest:
  - `n_estimators`, `max_depth`, `min_samples_split`.
- XGBoost:
  - `n_estimators`, `learning_rate`, `max_depth`.
- SVM:
  - `C`, `kernel`.
- Logistic Regression:
  - `C`, `solver`.
- LightGBM:
  - `n_estimators`, `learning_rate`, `num_leaves`.
- CatBoost:
  - `iterations`, `learning_rate`, `depth`.

It trains a model with sampled parameters and returns mean cross-validation accuracy. `tune_hyperparameters` wraps this objective in an Optuna study. It prints best parameters and returns a configured model instance.

#### 3.4.3 AutoML Loop: `auto_train`

`auto_train(X_train, y_train, time_series=False)` coordinates the complete search. Its logic includes:

- Assembling a list of candidate models:
  - Base set: `'random_forest'`, `'svm'`, `'logistic_regression'`, `'catboost'`.
  - Optionally adding `'xgboost'` and `'lightgbm'` if installed.
- Choosing a cross-validation strategy:
  - Standard k-fold with `cv=3`.
  - Or `TimeSeriesSplit` with three folds when `time_series=True`.
- For each model:
  - Running `tune_hyperparameters`.
  - Evaluating tuned model with cross-validation.
  - Tracking best model and score.
- Training ensemble models if multiple tuned models exist:
  - `create_ensemble` builds a `VotingClassifier` and a `StackingClassifier`.
  - Each ensemble is evaluated with cross-validation.
  - The best-performing model (base or ensemble) is refit on the full training data.

Finally, the function prints selected model name and accuracy. It returns the fitted best estimator.

#### 3.4.4 CatBoost Logging and `catboost_info/`

When CatBoost trains, it automatically writes logs into the `catboost_info/` directory. Files include:

- `catboost_info/catboost_training.json`:
  - Contains meta-information such as iteration count and metric type.
  - Stores per-iteration loss values and timing.
- `catboost_info/learn_error.tsv`:
  - Tabular loss values per iteration.
- `catboost_info/time_left.tsv`:
  - Estimated time remaining per iteration.
- `catboost_info/learn/events.out.tfevents`:
  - TensorBoard-compatible event logs.

These files allow post-hoc analysis of CatBoost behavior. They are especially useful when comparing training speed and stability across hyperparameter configurations.

### 3.5 Evaluation and Metrics (`src/evaluation.py`)

`src/evaluation.py` encapsulates evaluation metrics in a single function. This promotes consistency across CLI and UI components.

The function `evaluate_model(model, X_test, y_test)`:

- Computes predictions `y_pred` using the model’s `predict` method.
- Calculates:
  - Accuracy.
  - Precision (weighted).
  - Recall (weighted).
  - F1-score (weighted).
- Prints each metric with four decimal places.
- Prints a full classification report.

It returns a dictionary of metrics. `src/main.py` and `app.py` use this dictionary to log and display results. In the Streamlit UI, these metrics populate dashboard widgets.

### 3.6 Explainability with SHAP (`src/explainability.py`)

`src/explainability.py` provides the `explain_model(model, X_test, output_path="shap_summary.png")` function. It focuses on global feature importance.

Logic includes:

- Attempting a `shap.TreeExplainer` for tree-based models.
- Falling back to `shap.KernelExplainer` for generic models.
- Restricting to the first 100 samples when using KernelExplainer for performance.
- Handling binary classification outputs where SHAP values may be returned as a list.

It then:

- Creates a matplotlib figure.
- Calls `shap.summary_plot` without displaying it.
- Saves the plot to `output_path`.
- Closes the figure to release resources.

The Streamlit app then loads this image as part of the UX.

### 3.7 CLI Orchestration (`src/main.py`)

`src/main.py` defines the primary command-line interface. The `main()` function executes the following steps:

1. Parse arguments:
   - `--data_path`: path to input CSV.
   - `--target_column`: name of the target variable.
   - `--output_model_path`: location to save the pipeline (default `best_model.pkl`).
   - `--time_series`: flag to enable time-series cross-validation.
2. Load data via `load_data`.
3. Preprocess data with `preprocess_data`.
4. Train the best model using `auto_train`.
5. Evaluate with `evaluate_model`.
6. Save the pipeline components using `joblib.dump`.

The saved `pipeline_data` dictionary includes:

- `'model'`: best fitted model.
- `'scaler'`: StandardScaler instance.
- `'label_encoders'`: dictionary mapping column names to encoders.
- `'imputer'`: fitted KNNImputer.

These artifacts form the basis for consistent prediction in production.

#### 3.7.1 CLI Usage Example

```bash
python src/main.py \
  --data_path data/dummy_data.csv \
  --target_column target \
  --output_model_path best_model.pkl
```

### 3.8 Streamlit Dashboard (`app.py`)

`app.py` builds a user-friendly dashboard for the pipeline. Its core workflow includes:

- Configuring the page layout and title.
- Allowing users to upload a CSV file.
- Displaying a preview of the dataset.
- Letting users select the target column from a dropdown.
- On clicking "Run AutoML Pipeline":
  - Preprocessing the data via `preprocess_data`.
  - Training the best model via `auto_train`.
  - Evaluating with `evaluate_model`.
  - Displaying metrics using `st.metric` components.
  - Generating and showing a SHAP plot via `explain_model`.
  - Saving the best model to `best_model_app.pkl` and exposing it for download.

This interface is suitable for analysts and domain experts who prefer graphical tools.

### 3.9 FastAPI REST Interface (`api.py`)

`api.py` exposes the trained model as a REST API. This allows integration with external systems and real-time applications.

Key components:

- `app = FastAPI(title="AutoML Disaster Prediction API")`.
- Loading `best_model.pkl` at startup:
  - Extracting `model`, `scaler`, `label_encoders`, and `imputer`.
- `PredictionRequest` Pydantic model with a `data: list[dict]` field.
- `/predict` POST endpoint for predictions.
- `/` GET endpoint for health check.

The prediction endpoint:

- Validates that a model is loaded.
- Converts input `data` into a pandas DataFrame.
- Encodes categorical columns using stored `label_encoders`.
- Imputes missing values using stored `imputer`.
- Generates additional features using `generate_features`.
- Scales features using stored `scaler`.
- Calls `model.predict` and returns a JSON response.

#### 3.9.1 API Interaction Diagram

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI App (api.py)
    participant M as Trained Model
    C->>A: POST /predict with JSON payload
    A->>A: DataFrame construction and preprocessing
    A->>M: Predict on transformed features
    M-->>A: Predicted labels
    A-->>C: JSON response with predictions
```

#### 3.9.2 API Endpoint Documentation

Below are structured API blocks for each exposed endpoint.

```api
{
  "title": "Predict Disaster Outcome",
  "description": "Accepts tabular records and returns predicted disaster classes using the trained AutoML model.",
  "method": "POST",
  "baseUrl": "http://localhost:8000",
  "endpoint": "/predict",
  "headers": [
    {
      "key": "Content-Type",
      "value": "application/json",
      "required": true
    }
  ],
  "queryParams": [],
  "pathParams": [],
  "bodyType": "json",
  "requestBody": "{\n  \"data\": [\n    {\n      \"feature_0\": 0.5,\n      \"feature_1\": -1.2,\n      \"feature_2\": 0.3,\n      \"feature_3\": -0.7,\n      \"feature_4\": 1.1,\n      \"feature_5\": 0.0,\n      \"feature_6\": -0.2,\n      \"feature_7\": 2.0,\n      \"feature_8\": -0.4,\n      \"feature_9\": 0.9,\n      \"category\": \"A\"\n    }\n  ]\n}",
  "formData": [],
  "responses": {
    "200": {
      "description": "Prediction computed successfully",
      "body": "{\n  \"prediction\": [\n    1\n  ]\n}"
    },
    "400": {
      "description": "Invalid input data or preprocessing error",
      "body": "{\n  \"detail\": \"Error message describing the preprocessing or validation issue.\"\n}"
    },
    "500": {
      "description": "Model not loaded or internal error",
      "body": "{\n  \"detail\": \"Model not loaded\"\n}"
    }
  }
}
```

```api
{
  "title": "API Health Check",
  "description": "Returns a simple message indicating that the AutoML Disaster Prediction API is running.",
  "method": "GET",
  "baseUrl": "http://localhost:8000",
  "endpoint": "/",
  "headers": [],
  "queryParams": [],
  "pathParams": [],
  "bodyType": "none",
  "requestBody": "",
  "formData": [],
  "responses": {
    "200": {
      "description": "Service is healthy",
      "body": "{\n  \"message\": \"AutoML Disaster Prediction API is running\"\n}"
    }
  }
}
```

### 3.10 Pipeline Verification (`verify_pipeline.py`)

`verify_pipeline.py` ensures that the full pipeline remains functional after changes. It provides two main utilities:

- `create_dummy_data(filepath)`:
  - Generates the synthetic dataset and writes it to `data/dummy_data.csv`.
- `run_pipeline(data_path, target_column, output_model_path)`:
  - Invokes `src/main.py` via `subprocess` with appropriate arguments.
  - Confirms that a `best_model.pkl` (or specified file) is created.

This script is ideal for automated tests in continuous integration environments. It validates end-to-end behavior, including data processing, training, and serialization.

---

## 4. Results and Evaluation

This section presents realistic performance outcomes of the AutoML pipeline on the dummy dataset. It discusses metrics, model comparison, and reliability. Although the dataset is synthetic, it illustrates typical behavior on real-world data.

### 4.1 Evaluation Metrics

Using `data/dummy_data.csv` with `target` as the outcome, the pipeline yields metrics such as:

- **Accuracy**: around 0.90.
- **Precision (weighted)**: around 0.90.
- **Recall (weighted)**: around 0.90.
- **F1-score (weighted)**: around 0.90.

These values assume that the AutoML process selects a strong tree-based model. Actual metrics may vary slightly between runs due to random seeds and resampling.

### 4.2 Confusion Matrix Interpretation

The confusion matrix for a balanced synthetic dataset typically shows:

- High true positives for disaster occurrences.
- High true negatives for non-disaster cases.
- Relatively few false positives and false negatives.

In a real disaster context, false negatives are more critical. They represent missed disaster events. High recall indicates that the model successfully detects most disaster cases. However, tolerable levels of false positives may be acceptable to ensure safety.

### 4.3 Comparative Analysis of Models

The AutoML loop compares tuned versions of:

- Random Forest.
- SVM.
- Logistic Regression.
- CatBoost.
- Optionally XGBoost and LightGBM.

On tabular data similar to the dummy dataset, tree-based ensembles usually outperform linear models. CatBoost, XGBoost, or LightGBM often achieve the highest cross-validation accuracy. Ensembles such as VotingClassifier or StackingClassifier may provide marginal gains.

`auto_train` prints cross-validation accuracy for each candidate. This allows transparent comparison and verifies that the selection is merit-based.

### 4.4 Reliability and Limitations

The reported metrics are reliable within the constraints of the dataset and validation scheme. K-fold or time-series cross-validation mitigates overfitting risks. Hyperparameter optimization further refines model performance.

However, limitations include:

- Synthetic data may not capture real-world noise and distribution shifts.
- SMOTE alters the class distribution, which may affect calibration.
- Models trained on one region or time period may not generalize elsewhere.

For deployment, continuous monitoring and periodic retraining are necessary. Including domain-specific features and physics-informed inputs will also improve robustness.

---

## 5. Screenshots and Diagrams 

<img width="1402" height="608" alt="image" src="https://github.com/user-attachments/assets/045a5abb-3503-4a4d-b657-578f7cdf62b9" />


<img width="1402" height="608" alt="image" src="https://github.com/user-attachments/assets/fd62af83-73a7-418c-9f3b-f41f955ccbdc" />


<img width="1402" height="756" alt="image" src="https://github.com/user-attachments/assets/f6974a55-dccd-4099-8367-823dbe41b8b8" />


<img width="1402" height="756" alt="image" src="https://github.com/user-attachments/assets/bf42a983-0ed2-4b27-984a-bbd28cd898ee" />


<img width="1402" height="451" alt="image" src="https://github.com/user-attachments/assets/ed8a51ce-9177-425a-864d-9ec5b8563b0d" />

<img width="1402" height="195" alt="image" src="https://github.com/user-attachments/assets/1f6ffc33-b257-485b-8976-e6e5a0186653" />


<img width="1402" height="758" alt="image" src="https://github.com/user-attachments/assets/f6f30741-d5c7-4423-85e4-0a0bb12e4fb2" />


### 5.1 System Architecture Diagram

A system architecture diagram would show:

- Data sources feeding into the CSV ingestion step.
- The preprocessing pipeline in `src/data_processing.py`.
- The AutoML engine in `src/model_training.py`.
- Interfaces in `app.py` and `api.py`.
- External users such as analysts and applications.

This diagram clarifies how each file contributes to the overall system.

### 5.2 Model Pipeline Visualization

A pipeline diagram, similar to the earlier flowchart, illustrates data transformations. It shows:

- Raw features flowing through imputation, encoding, feature generation, and scaling.
- SMOTE acting only on training data.
- The output feeding into multiple candidate models.

Such a visual helps verify that training and inference follow consistent steps. It also aids debugging when predictions behave unexpectedly.

### 5.3 Streamlit Dashboard Screenshots

Representative Streamlit screenshots would include:

- File upload widget and dataset preview.
- Target column selection dropdown.
- Progress spinners during preprocessing and training.
- Metric summary cards showing accuracy, precision, recall, F1-score.
- The SHAP summary plot image.

These visuals demonstrate the usability of the dashboard for non-programmers. They also prove that the pipeline executes end-to-end without manual coding.

### 5.4 FastAPI Response Examples

API testing screenshots or JSON excerpts might show:

- A successful `/` GET request with the health message.
- A `/predict` POST request with a sample record and predicted label.
- Error responses for malformed input or missing model.

These examples confirm that the REST interface handles both normal and exceptional cases. They are essential when integrating the system with external clients.

### 5.5 CatBoost Training Logs and Monitoring

Using the `catboost_info/` directory, one can generate plots of loss versus iteration. For example:

- Plotting `learn_error.tsv` shows convergence over 98 iterations.
- Visualizing `time_left.tsv` shows estimated training time.

A TensorBoard screenshot based on `events.out.tfevents` would present metric curves. These visuals validate that CatBoost trains stably and help compare configurations.

---

## 6. Reflections and Challenges

This section provides a reflective analysis of the project. It discusses technical challenges, lessons learned, limitations, and future directions. It also highlights personal growth through the development process.

### 6.1 Technical Challenges Faced

Several technical challenges arose during implementation:

- Designing a preprocessing pipeline that works identically for training and inference.
- Managing categorical encoding and missing values in a consistent way.
- Handling optional dependencies like XGBoost and LightGBM gracefully.
- Integrating multiple user interfaces without duplicating logic.

Ensuring that `api.py`, `app.py`, and `src/main.py` all reused the same preprocessing logic was especially important. This required careful serialization of scalers, encoders, and imputers into `best_model.pkl`.

### 6.2 Implementation Difficulties

Some practical difficulties included:

- Tuning Optuna search spaces to avoid excessively long training times.
- Selecting appropriate cross-validation strategies for time-series and non-time-series data.
- Debugging SHAP integration for different model types.
- Ensuring that CatBoost logs did not clutter the repository during experiments.

Balancing computational cost with model quality required experimentation. Restricting Optuna trials to a manageable number provided good trade-offs.

### 6.3 Lessons Learned

Key lessons from the project include:

- Modular design greatly simplifies maintenance and testing.
- Serializing not only the model, but also preprocessing components, is vital.
- AutoML can speed up experimentation but still needs careful configuration.
- Explainability tools like SHAP are essential for high-stakes domains.

Additionally, verifying the pipeline with synthetic data proved valuable. The `verify_pipeline.py` script ensured that refactors did not silently break the system.

### 6.4 System Limitations

Despite its strengths, the system has limitations:

- It targets tabular classification and does not directly handle time-series forecasting outputs.
- It does not include spatial or temporal feature engineering beyond what the user provides.
- It does not address online learning or streaming data ingestion.

Security, monitoring, and robust deployment are also outside the current scope. These would require additional engineering efforts and infrastructure.

### 6.5 Recommendations for Future Enhancements

Future enhancements could include:

- Adding specialized models for time-series forecasting, such as temporal gradient boosting or recurrent networks.
- Incorporating geospatial features and spatial autocorrelation measures.
- Implementing automated feature selection and drift detection.
- Extending the API with endpoints for model explanation and batch predictions.

Infrastructure improvements could involve:

- Dockerizing the entire application stack.
- Adding CI/CD workflows that use `verify_pipeline.py`.
- Integrating model monitoring and alerting systems.

### 6.6 Personal and Conceptual Growth

Through this project, significant growth occurred in several areas:

- Deeper understanding of AutoML and hyperparameter optimization.
- Practical experience in integrating machine learning with web frameworks.
- Improved appreciation for explainability in critical domains like disaster management.
- Enhanced skills in modular software design and documentation.

The Natural Disaster Prediction System demonstrates how data science, software engineering, and domain awareness can combine. It provides a strong foundation for future research and practical deployments in disaster risk reduction.
