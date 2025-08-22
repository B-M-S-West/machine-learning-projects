from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

with open("ml_API_example/house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="House Price Prediction API")

class HouseData(BaseModel):
    size_sqft: float
    bedrooms: int

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict_price(data: HouseData):
    features = np.array([[data.size_sqft, data.bedrooms]])
    prediction = model.predict(features)

    return {"predicted_price": float(prediction[0])}
