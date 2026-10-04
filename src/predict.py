import pandas as pd
import joblib


# -----------------------------
# 1. Load trained model
# -----------------------------

MODEL_PATH = "../models/churn_model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------
# 2. New customer data
# -----------------------------

customer = pd.DataFrame([{
    "Payment Method": "Mailed check",
    "Paperless Billing": "No",
    "Contract": "Month-to-month",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Tech Support": "No",
    "Device Protection": "No",
    "Online Backup": "No",
    "Online Security": "No",
    "Internet Service": "DSL",
    "Dependents": "No",
    "Senior Citizen": "No",
    "Partner": "No",
    "Tenure Months": 2,
    "Monthly Charges": 50.00
}])


# -----------------------------
# 3. Predict churn probability
# -----------------------------


churn_probability = model.predict_proba(customer)[0][1]


# -----------------------------
# 4. Display result
# -----------------------------

print(f"Churn Probability: {churn_probability:.4f}")
print(f"Churn Probability: {churn_probability * 100:.2f}%")