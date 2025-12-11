# 📘 **Diabetes MLAR Predictor**  
*A Machine Learning + Logistic Regression Hybrid Model for Diabetes Risk Prediction With API + Web UI*

---

## 📌 **Overview**

This project is a complete end-to-end **Diabetes Prediction System** built using:

- **MLAR (Machine Learning Assisted Regression)** – a customized logistic-regression–style model using scaled inputs and manually learned weights  
- **PIMA Indian Diabetes Dataset**  
- **FastAPI Backend** providing a `/predict` API  
- **Modern Frontend Web UI** for taking realtime inputs  
- **Explainability** through feature-wise analysis & permutation importance  

The system accepts **8 clinical features** and returns:

- **Binary prediction** → *Diabetic / Not Diabetic*  
- **Probability score** (0–1) computed using MLAR  

This makes the project ideal for demonstrating model deployment, preprocessing pipelines, and modern API development.

---

## 🧠 **What is MLAR?**

**MLAR = Machine Learning Assisted Regression**

It is a manually trained logistic regression–style model where parameters:

- `W` → weight vector  
- `b` → bias  

are learned using a custom gradient-based update loop.

### MLAR Prediction Formula

scaled_input = scaler.transform(features)
z = scaled_input · W + b
probability = sigmoid(z)

Where:

sigmoid(z) = 1 / (1 + exp(-z))


During training, model artifacts are saved as:

| File | Description |
|------|-------------|
| `mlar_W.npy` | Learned weight vector |
| `mlar_b.npy` | Learned bias value |
| `scaler.joblib` | StandardScaler used during training |

This ensures **consistent preprocessing** during inference.

---

## 📊 **Dataset Used**

**PIMA Diabetes Dataset** (768 samples, 9 total columns)

Features:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  

Target:

Outcome (1 = Diabetic, 0 = Not Diabetic)


### Handling Missing Values

Certain columns contain zero values that represent missing medical measurements:



['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


These zeros are replaced with **median values** during both training and prediction.

### Fixing Skewness

The following features undergo a **log1p() transformation** for better scaling:



['Insulin', 'SkinThickness', 'Glucose', 'BMI']

FinalCode/
│
├── api.py                     # FastAPI backend
├── train_and_save.py          # Trains MLAR model & saves scaler, W, b
├── Feature_Wise_Compute.ipynb # Feature importance analysis
├── main.ipynb                 # Full exploratory analysis
│
├── PIMA_diabetes_Dataset.csv
├── scaler.joblib
├── mlar_W.npy
├── mlar_b.npy
│
├── static/
│    ├── index.html            # Frontend UI
│    └── styles.css            # Modern responsive design
│
├── requirements.txt
├── pyproject.toml
└── README.md

---

## 🚀 **Running the Backend (FastAPI)**

Start server:

```bash
uv run uvicorn api:app --reload --port 8000
🌐 Frontend Web Application

The UI is located at:

static/index.html
static/styles.css


Start backend, then open:

http://127.0.0.1:8000/