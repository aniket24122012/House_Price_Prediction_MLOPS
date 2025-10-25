import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from logger_config import get_logger
from utils import ensure_folder

logger = get_logger("preprocessing")

def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    df = pd.read_csv("artifacts/processed_data.csv")
    
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]
    
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    ensure_folder("artifacts/train.csv")
    train.to_csv("artifacts/train.csv", index=False)
    ensure_folder("artifacts/test.csv")
    test.to_csv("artifacts/test.csv", index=False)
    
    logger.info(f"Train/Test split done. Train shape: {train.shape}, Test shape: {test.shape}")

if __name__ == "__main__":
    main()
