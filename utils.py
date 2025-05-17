import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv(r"C:\Users\DELL\Desktop\MLFLOW PROJECT(NOW)\telco_churn.csv")
    return df

def preprocess_data(df):
    df = df.dropna()

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode categorical variables
    label_cols = df.select_dtypes(include='object').columns
    for col in label_cols:
        if col != 'customerID':
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

