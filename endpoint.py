import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# The same label encoder mapping as in your utils.py (fit on full dataset)
def encode_input(df):
    # Normally you fit LabelEncoders on full training data, but here for demo, fit on input
    # For real app: persist encoders fitted on full train and reuse here
    label_cols = df.select_dtypes(include='object').columns
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Sample input data
data = pd.DataFrame({
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
})

# Preprocess input the same way your model expects
data_encoded = encode_input(data)

json_data = {
    "dataframe_split": data_encoded.to_dict(orient="split")
}

response = requests.post(
    url="http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    json=json_data
)

# Process and print the prediction result
prediction = response.json()["predictions"][0]
churn_label = "Yes" if prediction == 1 else "No"

print(f"Predicted class: {prediction}")
print(f"Will the customer churn? {churn_label}")