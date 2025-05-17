import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from utils import load_data, preprocess_data

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

mlflow.sklearn.autolog()
mlflow.set_experiment("Telco_Churn_Experiment")


def train_model(params):
    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        return {'loss': -acc, 'status': STATUS_OK}


space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150]),
    'max_depth': hp.choice('max_depth', [5, 10, None])
}

with mlflow.start_run(run_name="RF_Hyperopt"):
    trials = Trials()
    best_result = fmin(fn=train_model, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

    best_params = {
        'n_estimators': [50, 100, 150][best_result['n_estimators']],
        'max_depth': [5, 10, None][best_result['max_depth']]
    }

    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train, y_train)
    mlflow.sklearn.log_model(final_model, "model")
