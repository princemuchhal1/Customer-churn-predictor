import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


# -----------------------------
# 1. Load dataset
# -----------------------------

DATA_PATH = "../data/Telco_customer_churn.xlsx"

df = pd.read_excel(DATA_PATH)


# -----------------------------
# 2. Select final features
# -----------------------------

features = [
    "Payment Method",
    "Paperless Billing",
    "Contract",
    "Streaming TV",
    "Streaming Movies",
    "Tech Support",
    "Device Protection",
    "Online Backup",
    "Online Security",
    "Internet Service",
    "Dependents",
    "Senior Citizen",
    "Partner",
    "Tenure Months",
    "Monthly Charges"
]

X = df[features]
y = df["Churn Value"]


# -----------------------------
# 3. Define feature types
# -----------------------------

categorical_features = [
    "Payment Method",
    "Paperless Billing",
    "Contract",
    "Streaming TV",
    "Streaming Movies",
    "Tech Support",
    "Device Protection",
    "Online Backup",
    "Online Security",
    "Internet Service",
    "Dependents",
    "Senior Citizen",
    "Partner"
]

numerical_features = [
    "Tenure Months",
    "Monthly Charges"
]


# -----------------------------
# 4. Create preprocessing
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)


# -----------------------------
# 5. Create final model
# -----------------------------

dt_lr = VotingClassifier(
    estimators=[
        (
            "dt",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ],
    voting="soft"
)


# -----------------------------
# 6. Create complete pipeline
# -----------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", dt_lr)
    ]
)


# -----------------------------
# 7. Train on complete dataset
# -----------------------------

model_pipeline.fit(X, y)


# -----------------------------
# 8. Save trained pipeline
# -----------------------------

MODEL_PATH = "../models/churn_model.pkl"

joblib.dump(model_pipeline, MODEL_PATH)

print("Model trained successfully.")
print(f"Model saved to: {MODEL_PATH}")