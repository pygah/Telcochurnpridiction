# Telco Customer Churn Prediction with MLflow

This project builds a machine learning model to predict customer churn using the Telco Customer Churn dataset. It uses `scikit-learn` for model training and `MLflow` for experiment tracking, model deployment, and serving.

---

## 📁 Project Structure

MLFLOW PROJECT(NOW)/
│
├── telco_churn.csv # Dataset file
├── main.py # Script to preprocess, train, and log the model with MLflow
├── deploy.py # Script to load the best model from MLflow registry and deploy it
├── endpoint.py # Script to send sample inference requests to the deployed model
├── utils.py # Helper functions for loading and preprocessing data
├── requirements.txt # Project dependencies
└── README.md # Project documentation
2. Install dependencies
pip install -r requirements.txt
3. Start MLflow UI
mlflow ui
Then go to http://127.0.0.1:5000 to see the experiment dashboard.
Training the Model:
Run the training and tracking script:
python main.py
This will:

Load and preprocess the dataset

Train a classification model (e.g., RandomForestClassifier)

Log metrics, model, and parameters to MLflow

Register the model to the MLflow Model Registry
Deploy the Model
To deploy the best model version for local REST API inference:
python deploy.py
This serves the model on http://127.0.0.1:5001/invocations
Make Predictions (Endpoint):
Once the model is deployed, use endpoint.py to send a test request:
python endpoint.py
Example output:
Predicted class: 1
Will the customer churn? Yes
 Sample Input Format
The input sent to the model is a pandas DataFrame in "split" orientation JSON:
{
  "dataframe_split": {
    "columns": ["gender", "SeniorCitizen", ...],
    "index": [0],
    "data": [["Female", 0, "Yes", "No", 5, ...]]
  }
}
Requirements
Key dependencies:

Python 3.7+

pandas

scikit-learn

mlflow

requests

See requirements.txt for the full list.
Troubleshooting
❌ ConnectionRefusedError
If you see this error:
ConnectionRefusedError: [WinError 10061] No connection could be made because the target machine actively refused it
Make sure:

You ran deploy.py to serve the model

The URL and port in endpoint.py match your deployment (usually http://127.0.0.1:5001/invocations)
Model Evaluation:
All training runs are tracked and visualized in the MLflow UI. You can compare accuracy, parameters, and models across different experiments.
