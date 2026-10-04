# Telco Customer Churn — Backend

Backend and machine learning service for an end-to-end customer churn prediction application.

The project takes a trained Scikit-learn model and exposes it through a FastAPI REST API, with PostgreSQL used to store prediction history.

## Live

- **API:** https://churn-api-ka1q.onrender.com/
- **Swagger Docs:** https://churn-api-ka1q.onrender.com/docs
- **Frontend:** https://customer-churn-frontend-qtem.onrender.com/

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- FastAPI
- Pydantic
- PostgreSQL
- psycopg
- Joblib
- Docker / Docker Compose
- Render

## ML Model

Dataset: **IBM Telco Customer Churn**

The model uses 15 customer features including:

- Contract
- Payment Method
- Internet Service
- Tenure Months
- Monthly Charges
- Technical Support
- Online Security
- Online Backup
- Device Protection
- Streaming Services
- Customer demographics

### Preprocessing

- Categorical features → `OneHotEncoder`
- Numerical features → `StandardScaler`

### Final Model

Soft-voting ensemble:

```text
Decision Tree + Logistic Regression
