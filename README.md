# Diabetes Prediction System

A machine learning project that predicts whether a patient is likely to have diabetes using diagnostic measurements from the Pima Indians Diabetes dataset. The repository includes the source dataset, a Jupyter notebook that documents the full training workflow, and persisted model artifacts that can be loaded for inference.

> **Important:** This project is intended for learning and experimentation only. It is not a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Contents](#repository-contents)
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Performance](#model-performance)
- [Getting Started](#getting-started)
- [Running the Notebook](#running-the-notebook)
- [Using the Saved Model](#using-the-saved-model)
- [Project Structure](#project-structure)
- [Recommended Improvements](#recommended-improvements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

The goal of this project is to build a binary classification model that predicts diabetes status from common diagnostic attributes such as glucose level, blood pressure, insulin level, body mass index (BMI), age, and diabetes pedigree function.

The current implementation uses:

- **Python** for data processing and modeling
- **NumPy** and **pandas** for numerical and tabular data handling
- **scikit-learn** for preprocessing, model training, splitting data, and evaluation
- **Support Vector Machine (SVM)** with a linear kernel for classification
- **joblib** for persisting the trained model artifacts

## Repository Contents

| Path | Description |
| --- | --- |
| `diabetes_prediction_model.ipynb` | Main notebook containing data loading, exploratory analysis, preprocessing, model training, evaluation, example prediction, and model persistence. |
| `sample_data/diabetes.csv` | Pima Indians Diabetes dataset used to train and evaluate the model. |
| `diabetes_prediction_model.pkl` | Serialized scikit-learn SVM classifier saved with `joblib`. |
| `diabetes_prediction_model_columns.json` | Serialized list describing the feature order expected by the saved model. Despite the `.json` extension, this file is currently stored with `joblib`. |
| `README.md` | Project documentation. |

## Dataset

The dataset contains **768 rows** and **9 columns**. Eight columns are input features, and the final column is the binary target label.

### Feature Columns

| Column | Description |
| --- | --- |
| `Pregnancies` | Number of pregnancies. |
| `Glucose` | Plasma glucose concentration. |
| `BloodPressure` | Diastolic blood pressure. |
| `SkinThickness` | Triceps skinfold thickness. |
| `Insulin` | 2-hour serum insulin. |
| `BMI` | Body mass index. |
| `DiabetesPedigreeFunction` | Diabetes pedigree function score. |
| `Age` | Patient age in years. |

### Target Column

| Value | Meaning |
| --- | --- |
| `0` | Patient is classified as non-diabetic. |
| `1` | Patient is classified as diabetic. |

The notebook separates the dataset into feature matrix `X` and target vector `Y`, standardizes the feature values, and then trains a classifier on the standardized data.

## Machine Learning Workflow

The notebook follows these main steps:

1. **Import dependencies**
   - Imports NumPy, pandas, scikit-learn preprocessing utilities, train/test splitting, SVM model support, accuracy scoring, and joblib.

2. **Load the dataset**
   - Reads `sample_data/diabetes.csv` into a pandas DataFrame.

3. **Explore the data**
   - Displays the first rows of the dataset.
   - Checks the dataset shape.
   - Computes summary statistics.
   - Reviews the distribution of the target labels.
   - Compares average feature values grouped by diabetes outcome.

4. **Split features and labels**
   - Drops the `Outcome` column from the feature matrix.
   - Uses the `Outcome` column as the target label.

5. **Standardize features**
   - Fits a `StandardScaler` on the input features.
   - Transforms the feature matrix so that values are scaled before training.

6. **Create train/test split**
   - Uses an 80/20 split.
   - Uses stratification to preserve the original class distribution in both training and test sets.
   - Uses `random_state=2` for reproducibility.

7. **Train the model**
   - Trains an SVM classifier with a linear kernel.

8. **Evaluate the model**
   - Computes accuracy on both training and test data.

9. **Run an example prediction**
   - Converts a single patient record to a NumPy array.
   - Reshapes it for one-row prediction.
   - Applies the same scaling process.
   - Predicts whether the patient is diabetic or non-diabetic.

10. **Persist model artifacts**
    - Saves the trained classifier as `diabetes_prediction_model.pkl`.
    - Saves the model input column order as `diabetes_prediction_model_columns.json`.

## Model Performance

The notebook reports the following accuracy values for the current SVM classifier:

| Split | Accuracy |
| --- | ---: |
| Training data | ~78.66% |
| Test data | ~77.27% |

These results are useful for a baseline educational model. For real-world use, additional validation, better handling of missing or zero-coded medical measurements, threshold analysis, and clinical review would be required.

## Getting Started

### Prerequisites

Install Python 3.9 or later, then install the required Python packages:

```bash
pip install numpy pandas scikit-learn joblib jupyter
```

If you prefer to isolate dependencies, create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn joblib jupyter
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Clone the Repository

```bash
git clone <repository-url>
cd Diabetes-prediction-system
```

## Running the Notebook

Start Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

Then open:

```text
diabetes_prediction_model.ipynb
```

Run the cells from top to bottom to reproduce the training workflow and regenerate the model artifacts.

## Using the Saved Model

The repository includes a saved SVM classifier. You can load it with `joblib` and run predictions on new patient data.

```python
import joblib
import numpy as np

model = joblib.load("diabetes_prediction_model.pkl")

# Feature order:
# Pregnancies, Glucose, BloodPressure, SkinThickness,
# Insulin, BMI, DiabetesPedigreeFunction, Age
sample = np.array([[0, 137, 40, 35, 168, 43.1, 2.288, 33]])

prediction = model.predict(sample)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

### Important Note About Scaling

The notebook trains the model on standardized features, but the saved artifact currently contains only the classifier. For the most reliable inference workflow, persist the fitted `StandardScaler` together with the classifier, for example by saving a scikit-learn `Pipeline`.

A future improvement could replace the current model-saving logic with:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import joblib

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", svm.SVC(kernel="linear")),
])

pipeline.fit(X_train_raw, y_train)
joblib.dump(pipeline, "diabetes_prediction_pipeline.pkl")
```

This ensures that raw input values are scaled exactly the same way during training and prediction.

## Project Structure

```text
Diabetes-prediction-system/
├── README.md
├── diabetes_prediction_model.ipynb
├── diabetes_prediction_model.pkl
├── diabetes_prediction_model_columns.json
└── sample_data/
    └── diabetes.csv
```

## Recommended Improvements

Potential next steps for this project include:

- Save the fitted `StandardScaler` or a complete scikit-learn `Pipeline` for safer inference.
- Rename `diabetes_prediction_model_columns.json` or save it as actual JSON instead of a joblib artifact.
- Add a `requirements.txt` or `pyproject.toml` file for reproducible dependency installation.
- Add a standalone prediction script or web interface.
- Add model evaluation metrics beyond accuracy, such as precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Use cross-validation for a more robust estimate of model performance.
- Investigate zero values in medical columns where zero may represent missing data.
- Add tests for data loading, preprocessing, and prediction behavior.

## Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'` or `No module named 'sklearn'`

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn joblib jupyter
```

### Model predictions look inconsistent

Make sure the input feature order matches the training dataset exactly:

```text
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
```

Also note that the current saved model was trained on standardized values. For production-style inference, save and use the same fitted scaler or a single preprocessing-and-model pipeline.

### Jupyter cannot find the CSV file

Run Jupyter from the repository root so the relative dataset path resolves correctly:

```bash
jupyter notebook
```

## License

No license file is currently included in this repository. Add a license before distributing or reusing this project publicly.
