import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from logger_config import get_logger

logger = get_logger("model_evaluation")

def ensure_folder(folder_path):
    import os
    os.makedirs(folder_path, exist_ok=True)

def main():
    # Ensure MLflow folder exists
    # mlflow_dir = "mlflow_tracking"
    # ensure_folder(mlflow_dir)
    # mlflow.set_tracking_uri(mlflow_dir)
    mlflow.set_experiment("HousePricePrediction")

    X_test = pd.read_csv("artifacts/X_test.csv")
    y_test = pd.read_csv("artifacts/y_test.csv")

    # Load best production model
    model = mlflow.sklearn.load_model("models:/HousePriceModel/Production")
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log metrics inside MLflow run
    with mlflow.start_run():
        mlflow.log_metrics({"test_mse": mse, "test_r2": r2})

    logger.info(f"Test MSE: {mse}, R2: {r2}")
    logger.info("Evaluation complete and metrics logged in MLflow.")

if __name__ == "__main__":
    main()
