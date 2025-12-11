# train_and_save.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ---- copy/paste your MLAR / predict_mlar from main.py (shortened here) ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_mlar_np(X, W, b, threshold=0.5):
    probs = sigmoid(X @ W + b).ravel()
    return (probs >= threshold).astype(int), probs

def MLAR(X, y, lr=0.1, epochs=2000, pos_weight=2.0, l2=0.0, seed=42):
    np.random.seed(seed)
    m, n = X.shape
    W = np.random.randn(n, 1) * 0.01
    b = 0.0
    for i in range(epochs):
        z = X @ W + b
        y_hat = sigmoid(z)
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
        w = np.where(y == 1, pos_weight, 1.0)
        dz = (y_hat - y) * w
        dW = (1/m) * (X.T @ dz) + (l2/m) * W
        db = (1/m) * np.sum(dz)
        lr_i = lr / (1 + 0.001 * i)
        W -= lr_i * dW
        b -= lr_i * db
    return W, b
# -------------------------------------------------------------------------

# Load data
data = pd.read_csv("PIMA_diabetes_Dataset.csv")  # ensure this file exists
# Replace 0 with NaN for selected columns, same as your notebook
cols_maybe_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols_maybe_zero_missing:
    data[c] = data[c].replace(0, np.nan)
for c in cols_maybe_zero_missing:
    data[c].fillna(data[c].median(), inplace=True)

# features & labels
features = data.drop(columns="Outcome")
labels = data["Outcome"]

# log-transform same columns
skewed = ['Insulin','SkinThickness','Glucose','BMI']
for c in skewed:
    features[c] = np.log1p(features[c])

# split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# scale using MinMax (fit on train)
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# convert to numpy for MLAR
X_train_np = X_train_scaled.values
y_train_np = y_train.values.reshape(-1,1)

# Train MLAR (choose hyperparams or load tuned ones)
W, b = MLAR(X_train_np, y_train_np, lr=0.05, epochs=3000, pos_weight=1.5, l2=1e-4, seed=42)

# Save artifacts
joblib.dump(scaler, "scaler.joblib")
np.save("mlar_W.npy", W)
np.save("mlar_b.npy", np.array([b]))   # save as array for easy load

print("Saved scaler.joblib, mlar_W.npy, mlar_b.npy")
