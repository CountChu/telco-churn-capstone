"""Project settings in one place: paths, column groups, validation rules and model settings.

Every other module imports these values, so a column list or a number such as
the test size is never repeated - an experiment is changed here, in one spot.
"""
from pathlib import Path

# --- Paths: relative to the project folder, so the code runs on any machine ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --- Columns ---
ID_COLUMN = "customerID"  # identifies a customer; kept for the output, never a feature
TARGET_COLUMN = "Churn"  # "Yes"/"No" in the raw file, 1/0 after cleaning
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]  # already stored as 0/1
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# --- Validation rules, taken from the data dictionary ---
TARGET_LABELS = {"Yes", "No"}
TARGET_MAPPING = {"No": 0, "Yes": 1}
YES_NO = {"Yes", "No"}
INTERNET_ADDON = {"Yes", "No", "No internet service"}
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
ALLOWED_VALUES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
}
TENURE_RANGE = (0, 120)  # months; a generous sanity bound (10 years)
NON_NEGATIVE_COLUMNS = ["MonthlyCharges", "TotalCharges"]

# --- Model settings ---
TEST_SIZE = 0.20
RANDOM_STATE = 42
PRIMARY_MODEL = "logistic_regression"
LOGISTIC_REGRESSION_PARAMS = {"max_iter": 1000}
DECISION_TREE_PARAMS = {"max_depth": 5, "min_samples_leaf": 20}

# --- Risk bands for the predictions file (simple cut-offs to prioritise outreach) ---
RISK_BINS = [0.0, 0.3, 0.6, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]
