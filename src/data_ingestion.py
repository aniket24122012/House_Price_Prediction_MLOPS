import pandas as pd
import yaml
from logger_config import get_logger
from utils import ensure_folder

logger = get_logger("data_ingestion")

def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_path = config["data_ingestion"]["raw_data_path"]
    
    df = pd.read_csv(raw_path)
    
    out_file = "artifacts/processed_data.csv"
    ensure_folder(out_file)
    df.to_csv(out_file, index=False)
    
    logger.info(f"Data ingested from {raw_path}, shape: {df.shape}")

if __name__ == "__main__":
    main()
