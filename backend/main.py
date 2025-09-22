from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("model.joblib")

app = FastAPI(title="Car Price Prediction API")

# Request schema
class CarFeatures(BaseModel):
    symboling: int
    fueltype: str
    aspiration: str
    doornumber: str
    carbody: str
    drivewheel: str
    enginelocation: str
    wheelbase: float
    carlength: float
    carwidth: float
    carheight: float
    curbweight: int
    enginetype: str
    cylindernumber: str
    enginesize: int
    fuelsystem: str
    boreratio: float
    stroke: float
    compressionratio: float
    horsepower: int
    peakrpm: int
    citympg: int
    highwaympg: int

@app.post("/predict")
def predict_price(features: CarFeatures):
    # Convert to DataFrame
    data = pd.DataFrame([features.model_dump()])

    # Predict
    prediction = model.predict(data)[0]
    return {"predicted_price": round(float(prediction), 2)}
