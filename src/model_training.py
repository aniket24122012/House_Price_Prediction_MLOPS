import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import infer_signature
from logger_config import get_logger
import yaml

logger = get_logger("model_training")

def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def ensure_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def main():
    # Ensure MLflow folder exists
    # mlflow_dir = "mlflow_tracking"
    # ensure_folder(mlflow_dir)
    # mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("HousePricePrediction3")

    config = load_config()
    X_train = pd.read_csv("artifacts/X_train.csv")
    y_train = pd.read_csv("artifacts/y_train.csv")

    param_grid = config["model_training"]["param_grid"]
    best_r2 = float("-inf")
    best_model_uri = None
    best_params = None

    for params in ParameterGrid(param_grid):
        with mlflow.start_run():
            logger.info(f"Training with params: {params}")
            
            model = RandomForestRegressor(**params, random_state=config["model_training"]["random_state"])
            model.fit(X_train, y_train.values.ravel())
            
            preds = model.predict(X_train)
            mse = mean_squared_error(y_train, preds)
            r2 = r2_score(y_train, preds)
            
            # Log params & metrics
            mlflow.log_params(params)
            mlflow.log_metrics({"train_mse": mse, "train_r2": r2})

            # Log model with signature & input_example
            signature = infer_signature(X_train, preds)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="house_price_model",
                signature=signature,
                input_example=X_train.head(3)
            )
            
            logger.info(f"Run finished. MSE: {mse}, R2: {r2}")

            if r2 > best_r2:
                best_r2 = r2
                best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/house_price_model"
                best_params = params

    # Register best model & move to Production
    client = MlflowClient()
    model_details = mlflow.register_model(best_model_uri, "HousePriceModel")
    client.transition_model_version_stage(
        name="HousePriceModel",
        version=model_details.version,
        stage="Production"
    )
    logger.info(f"Best model registered and moved to Production: {best_params}")

if __name__ == "__main__":
    main()
