from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pickle 
import numpy as np
from typing import Literal

app=FastAPI()
class Transaction(BaseModel):
    amount: float
    transaction_hour: float
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age:int
    foreign_transaction: Literal[0,1]
    location_mismatch: Literal[0,1]
    merchant_category_encoded:Literal[1,2,3,4]

with open("credit_card_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("credit_card_model.pkl", "rb") as f:
    classifier_lr = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    classifier_rf = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict/")
def predict(transaction: Transaction):
    transaction_array = np.array([transaction.amount, transaction.transaction_hour, transaction.device_trust_score, transaction.velocity_last_24h, transaction.cardholder_age, transaction.foreign_transaction, transaction.location_mismatch, transaction.merchant_category_encoded]).reshape(1,-1)
    transaction_scaled = scaler.transform(transaction_array)
    prediction = classifier_rf.predict(transaction_scaled)
    probability = classifier_rf.predict_proba(transaction_scaled)[0][1]
    return {"prediction": "Fraud" if prediction[0] == 1 else "Not Fraud", "probability": f'{probability*100:.2f}%'}


