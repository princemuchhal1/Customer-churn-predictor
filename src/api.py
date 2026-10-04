from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import json
import io
from .database import get_connection

# CORS Handling

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# 1. Load trained model
# -----------------------------

from .model import model


# -----------------------------
# 3. Define customer input
# -----------------------------

class Customer(BaseModel):
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]

    paperless_billing: Literal["No", "Yes"]

    contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    streaming_tv: Literal[
        "No",
        "No internet service",
        "Yes"
    ]

    streaming_movies: Literal[
        "No",
        "No internet service",
        "Yes"
    ]

    tech_support: Literal[
        "No",
        "No internet service",
        "Yes"
    ]

    device_protection: Literal[
        "No",
        "No internet service",
        "Yes"
    ]

    online_backup: Literal[
        "No",
        "No internet service",
        "Yes"
    ]

    online_security: Literal[
        "No",
        "No internet service",
        "Yes"
    ]

    internet_service: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    dependents: Literal["No", "Yes"]
    senior_citizen: Literal["No", "Yes"]
    partner: Literal["No", "Yes"]

    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)

    
# -----------------------------
# 4. Home endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# -----------------------------
# 5. Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict(customer: Customer):
    try:
        customer_df = pd.DataFrame([{
            "Payment Method": customer.payment_method,
            "Paperless Billing": customer.paperless_billing,
            "Contract": customer.contract,
            "Streaming TV": customer.streaming_tv,
            "Streaming Movies": customer.streaming_movies,
            "Tech Support": customer.tech_support,
            "Device Protection": customer.device_protection,
            "Online Backup": customer.online_backup,
            "Online Security": customer.online_security,
            "Internet Service": customer.internet_service,
            "Dependents": customer.dependents,
            "Senior Citizen": customer.senior_citizen,
            "Partner": customer.partner,
            "Tenure Months": customer.tenure_months,
            "Monthly Charges": customer.monthly_charges
        }])

        churn_probability = model.predict_proba(customer_df)[0][1]

        churn_prediction = "Yes" if churn_probability >= 0.5 else "No"

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (
                        churn_probability,
                        churn_prediction
                    )
                    VALUES (%s, %s)
                    """,
                    (float(churn_probability), churn_prediction)
                )

        return {
            "churn_probability": round(float(churn_probability), 4),
            "churn_prediction": churn_prediction
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )


# -----------------------------
# 6. Batch prediction endpoint
# -----------------------------

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        filename = file.filename.lower()

        # -----------------------------
        # Read CSV
        # -----------------------------

        if filename.endswith(".csv"):

            df = pd.read_csv(io.BytesIO(contents))

        # -----------------------------
        # Read JSON
        # -----------------------------

        elif filename.endswith(".json"):

            data = json.loads(contents.decode("utf-8"))

            if not isinstance(data, list):
                raise HTTPException(
                    status_code=400,
                    detail="JSON file must contain a list of customers."
                )

            df = pd.DataFrame(data)

        else:

            raise HTTPException(
                status_code=400,
                detail="Only CSV and JSON files are supported."
            )

        # -----------------------------
        # Check empty file
        # -----------------------------

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        results = []

        # -----------------------------
        # Validate and predict each row
        # -----------------------------

        for index, row in df.iterrows():

            try:

                customer = Customer.model_validate(
                    row.to_dict()
                )

                customer_df = pd.DataFrame([{
                    "Payment Method": customer.payment_method,
                    "Paperless Billing": customer.paperless_billing,
                    "Contract": customer.contract,
                    "Streaming TV": customer.streaming_tv,
                    "Streaming Movies": customer.streaming_movies,
                    "Tech Support": customer.tech_support,
                    "Device Protection": customer.device_protection,
                    "Online Backup": customer.online_backup,
                    "Online Security": customer.online_security,
                    "Internet Service": customer.internet_service,
                    "Dependents": customer.dependents,
                    "Senior Citizen": customer.senior_citizen,
                    "Partner": customer.partner,
                    "Tenure Months": customer.tenure_months,
                    "Monthly Charges": customer.monthly_charges
                }])

                churn_probability = model.predict_proba(
                    customer_df
                )[0][1]

                churn_prediction = (
                    "Yes"
                    if churn_probability >= 0.5
                    else "No"
                )

                results.append({
                    "row": index + 1,
                    "churn_probability": round(
                        float(churn_probability), 4
                    ),
                    "churn_prediction": churn_prediction
                })

            except Exception as e:

                results.append({
                    "row": index + 1,
                    "error": "Invalid customer data."
                })

        return {
            "total_rows": len(df),
            "results": results
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed."
        )

@app.get("/predictions")
def get_predictions():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id,
                        churn_probability,
                        churn_prediction,
                        created_at
                    FROM predictions
                    ORDER BY created_at DESC
                """)

                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "churn_probability": round(float(row[1]), 4),
                "churn_prediction": row[2],
                "created_at": row[3]
            }
            for row in rows
        ]

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve prediction history."
        )