import pandas as pd
import yaml
from logger_config import get_logger
from utils import ensure_folder

logger = get_logger("feature_engineering")

def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    target_col = config["feature_engineering"]["target_col"]
    
    train = pd.read_csv("artifacts/train.csv")
    test = pd.read_csv("artifacts/test.csv")
    
    X_train = train.drop(columns=[target_col])
    y_train = train[[target_col]]
    X_test = test.drop(columns=[target_col])
    y_test = test[[target_col]]
    
    ensure_folder("artifacts/X_train.csv")
    X_train.to_csv("artifacts/X_train.csv", index=False)
    ensure_folder("artifacts/y_train.csv")
    y_train.to_csv("artifacts/y_train.csv", index=False)
    ensure_folder("artifacts/X_test.csv")
    X_test.to_csv("artifacts/X_test.csv", index=False)
    ensure_folder("artifacts/y_test.csv")
    y_test.to_csv("artifacts/y_test.csv", index=False)
    
    logger.info("Feature engineering done.")

if __name__ == "__main__":
    main()
