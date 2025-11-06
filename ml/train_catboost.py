import os
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

# Load data
df = pd.read_csv("data/sample.csv")
X = df.drop("target", axis=1).astype("float64")
y = df["target"]

# Ensure you have at least 2 test samples
test_size = 0.2 if len(df) >= 10 else 0.5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train
model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=False)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = float(mean_absolute_error(y_test, preds))
r2 = float(r2_score(y_test, preds)) if len(y_test) > 1 else float("nan")

# Log to MLflow
with mlflow.start_run(run_name="latest_catboost") as run:
    # signature + example help downstream serving
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:2]

    # New-style: use 'name' instead of deprecated artifact_path
    mlflow.catboost.log_model(
        cb_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=None  # stays local
    )
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print(f"Logged model. MAE={mae:.4f} R2={r2:.4f} run_id={run.info.run_id}")