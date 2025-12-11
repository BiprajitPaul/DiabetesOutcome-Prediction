# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import Dict
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles


app = FastAPI()



# load artifacts
scaler = joblib.load("scaler.joblib")
W = np.load("mlar_W.npy")        # shape (n_features, 1)
b = float(np.load("mlar_b.npy")[0])

# load dataset for medians/column order (alternatively save medians to disk in train_and_save)
data = pd.read_csv("PIMA_diabetes_Dataset.csv")
cols_maybe_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols_maybe_zero_missing:
    data[c] = data[c].replace(0, np.nan)
medians = {c: float(data[c].median()) for c in cols_maybe_zero_missing}

# column order expected
FEATURE_ORDER = list(data.drop(columns="Outcome").columns)  # ensure consistent ordering

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class InputFeatures(BaseModel):
    # create optional fields with float. Keep names exactly as in FEATURE_ORDER
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(payload: InputFeatures):
    # convert input to dict ordered by FEATURE_ORDER
    x_dict = payload.dict()
    # impute zeros for the specified columns the same way you did in training
    for c in cols_maybe_zero_missing:
        val = x_dict.get(c)
        # if user gives 0 or missing, treat as NaN => replace with median
        if val is None:
            x_dict[c] = medians[c]
        else:
            # convert to float (pydantic already did)
            if float(val) == 0.0:
                x_dict[c] = medians[c]

    # build feature vector in correct order
    x_list = [float(x_dict[col]) for col in FEATURE_ORDER]

    # log1p for skewed features
    for idx, col in enumerate(FEATURE_ORDER):
        if col in ['Insulin','SkinThickness','Glucose','BMI']:
            x_list[idx] = np.log1p(x_list[idx])

    X_arr = np.array(x_list).reshape(1, -1)
    # scale
    X_scaled = scaler.transform(X_arr)

    # predict with MLAR
    probs = sigmoid(X_scaled @ W + b).ravel()[0]
    pred = int(probs >= 0.5)
    return {"prediction": int(pred), "probability": float(probs)}
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")

