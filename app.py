from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import mlflow

# MLflow setup
mlflow.set_tracking_uri("mlflow_tracking")
model_name = "HousePriceModel"
model_stage = "Production"
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")

app = FastAPI(title="House Price Prediction API")

# Enable CORS for all origins (change "*" to specific domains if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class HouseFeatures(BaseModel):
    Bedrooms: int
    Bathrooms: int
    Sqft_Living: float
    Sqft_Lot: float
    Floors: float
    Age: float

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)
    return {"predicted_price": float(prediction[0])}
