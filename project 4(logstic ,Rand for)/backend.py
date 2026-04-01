from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pickle 
import numpy as np
from typing import Literal

app=FastAPI()
class Transaction(BaseModel):
    distance_from_home: float
    ratio_to_median_purchase_price: float
    used_pin_number: Literal[0,1]
    online_order: Literal[0,1]

with open("credit_card_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("credit_card_model.pkl", "rb") as f:
    classifier_lr = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict/")
def predict(transaction: Transaction):
    transaction_array = np.array([transaction.distance_from_home, transaction.ratio_to_median_purchase_price, transaction.used_pin_number, transaction.online_order]).reshape(1, -1)
    transaction_scaled = scaler.transform(transaction_array)
    prediction = classifier_lr.predict(transaction_scaled)
    probability = classifier_lr.predict_proba(transaction_scaled)[0][1]
    return {"prediction": "Fraud" if prediction[0] == 1 else "Not Fraud", "probability": f'{probability*100:.2f}%'}


