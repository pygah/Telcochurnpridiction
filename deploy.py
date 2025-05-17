import mlflow.pyfunc
import pandas as pd

# Load model from MLflow model registry (Production stage)
model_uri = "models:/CustomerChurnModel/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Example input (replace this with real user input or data source)
input_dict = {
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [5],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["DSL"],
    "OnlineSecurity": ["Yes"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["Yes"],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [75.35],
    "TotalCharges": [385.6]
}

# Convert to DataFrame
input_data = pd.DataFrame(input_dict)

# Predict
predictions = model.predict(input_data)
print("Predicted churn probability/class:", predictions)

