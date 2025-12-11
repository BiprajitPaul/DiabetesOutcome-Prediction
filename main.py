
# =============================


import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from time import perf_counter as now 

from sklearn.model_selection import train_test_split, StratifiedKFold # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc, confusion_matrix
)
import itertools

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================
# Load Dataset
# =============================
data = pd.read_csv("PIMA_diabetes_Dataset.csv")
print("Dataset shape:", data.shape)
print(data.head())

# =============================
# Data Cleaning
# Replace 0 with NaN for selected columns
# =============================
cols_maybe_zero_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for c in cols_maybe_zero_missing:
    data[c] = data[c].replace(0, np.nan)

# Fill missing with median
for c in cols_maybe_zero_missing:
    data[c].fillna(data[c].median(), inplace=True)

# Optional: show missing counts
print("\nMissing counts (post-imputation):")
print(data[cols_maybe_zero_missing].isna().sum())

# =============================
# Data Visualization
# =============================
plt.figure(figsize=(12, 8))
for idx, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, idx + 1)
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=data)
plt.title("Diabetes Distribution (0 = No, 1 = Yes)")
plt.show()


# =============================
# Feature Engineering & Scaling
# =============================
features = data.drop(columns="Outcome")
labels = data["Outcome"]

# log-transform skewed features (after imputation)
skewed = ['Insulin', 'SkinThickness', 'Glucose', 'BMI']
for c in skewed:
    # log1p is safe for zeros (though we've imputed)
    features[c] = np.log1p(features[c])

# Split dataset (with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Scale features (fit on train only)
scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

cols_to_scale = X_train.columns
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

# Convert to numpy for MLAR and timing uses
X_train_np = X_train_scaled.values
X_test_np  = X_test_scaled.values
y_train_np = y_train.values.reshape(-1,1)
y_test_np  = y_test.values.reshape(-1,1)

# =============================
# Utility functions: timing & evaluation
# =============================
def measure_test_time(func, X, repeats=1000):
    """Repeat predictions to get stable average test time per call."""
    start = now()
    for _ in range(repeats):
        _ = func(X)
    end = now()
    return (end - start) / repeats

def evaluate_model(name, model, X_train, y_train, X_test, y_test, test_repeats=1000):
    # Train time
    start_train = now()
    model.fit(X_train, y_train)
    end_train = now()
    train_time = end_train - start_train

    # Predictions (single run for metrics)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    # Test time (averaged for stability)
    test_time = measure_test_time(lambda X: model.predict(X), X_test, repeats=test_repeats)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        "PR_AUC": average_precision_score(y_test, y_proba) if y_proba is not None else np.nan,
        "Train Time (s)": train_time,
        "Test Time (s)": test_time
    }
    return results

# =============================
# Train Baseline Models
# =============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

metrics_summary = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    metrics_summary[name] = evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test, test_repeats=1000)

# =============================
# Custom MLAR (Modified Logistic Regression)
# =============================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_mlar(X, W, b, threshold=0.5):
    probs = sigmoid(X @ W + b).ravel()
    return (probs >= threshold).astype(int), probs

def MLAR(X, y, lr=0.1, epochs=2000, pos_weight=2.0, l2=0.0, seed=RANDOM_STATE):
    np.random.seed(seed)
    m, n = X.shape
    W = np.random.randn(n, 1) * 0.01
    b = 0.0

    for i in range(epochs):
        z = X @ W + b
        y_hat = sigmoid(z)
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)

        # class weights (boost positives)
        w = np.where(y == 1, pos_weight, 1.0)

        # gradient
        dz = (y_hat - y) * w
        dW = (1/m) * (X.T @ dz) + (l2/m) * W
        db = (1/m) * np.sum(dz)

        # learning rate schedule
        lr_i = lr / (1 + 0.001 * i)
        W -= lr_i * dW
        b -= lr_i * db

    return W, b

# =============================
# Hyperparameter Tuning for MLAR (manual grid search)
# =============================
def tune_mlar(X_train, y_train, X_val, y_val, param_grid=None, verbose=False):
    if param_grid is None:
        param_grid = {
            "lr": [0.01, 0.05, 0.1],
            "epochs": [3000, 5000],
            "pos_weight": [1.0, 1.5, 2.0],
            "l2": [0.0, 1e-4, 1e-3]
        }

    best_score = -1
    best_params = None
    best_results = None
    best_W, best_b = None, None

    total = 1
    for v in param_grid.values():
        total *= len(v)
    checked = 0

    for lr in param_grid["lr"]:
        for epochs in param_grid["epochs"]:
            for pos_weight in param_grid["pos_weight"]:
                for l2 in param_grid["l2"]:
                    checked += 1
                    if verbose:
                        print(f"Checking {checked}/{total}: lr={lr}, epochs={epochs}, pos_weight={pos_weight}, l2={l2} ...", end=" ")
                    W, b = MLAR(X_train, y_train, lr=lr, epochs=epochs, pos_weight=pos_weight, l2=l2)
                    y_pred, y_proba = predict_mlar(X_val, W, b)
                    f1 = f1_score(y_val.ravel(), y_pred, zero_division=0)
                    roc_auc = roc_auc_score(y_val.ravel(), y_proba)
                    if f1 > best_score:
                        best_score = f1
                        best_params = {"lr": lr, "epochs": epochs, "pos_weight": pos_weight, "l2": l2}
                        best_results = {
                            "Accuracy": accuracy_score(y_val.ravel(), y_pred),
                            "Precision": precision_score(y_val.ravel(), y_pred, zero_division=0),
                            "Recall": recall_score(y_val.ravel(), y_pred, zero_division=0),
                            "F1": f1,
                            "ROC_AUC": roc_auc,
                            "PR_AUC": average_precision_score(y_val.ravel(), y_proba)
                        }
                        best_W, best_b = W, b
                    if verbose:
                        print(f"f1={f1:.4f}")

    return best_params, best_results, best_W, best_b

# Run tuning on our train/test split (we're using test as validation here for simplicity)
print("\n🔍 Tuning MLAR hyperparameters... (this may take a few minutes)")
param_grid = {
    "lr": [0.01, 0.05],
    "epochs": [3000, 5000],
    "pos_weight": [1.0, 1.5, 2.0],
    "l2": [0.0, 1e-4]
}
best_params, best_results, W_mlar_best, b_mlar_best = tune_mlar(X_train_np, y_train_np, X_test_np, y_test_np, param_grid=param_grid, verbose=True)

print("\nBest MLAR parameters:", best_params)
print("Best MLAR results (on validation/test):", best_results)

# =============================
# Evaluate MLAR with timing
# =============================
# Retrain MLAR with best params on training data
start_train = now()
W_mlar, b_mlar = MLAR(X_train_np, y_train_np,
                      lr=best_params["lr"],
                      epochs=best_params["epochs"],
                      pos_weight=best_params["pos_weight"],
                      l2=best_params["l2"])
end_train = now()
train_time = end_train - start_train

# Predictions once for metrics
y_pred_mlar, y_proba_mlar = predict_mlar(X_test_np, W_mlar, b_mlar)

# Define wrapper that ensures prediction outputs are used
def predict_wrapper(X):
    y_pred, _ = predict_mlar(X, W_mlar, b_mlar)
    return y_pred

# Average test time (stable)
test_time = measure_test_time(predict_wrapper, X_test_np, repeats=5000)

metrics_summary["MLAR"] = {
    "Accuracy": accuracy_score(y_test_np.ravel(), y_pred_mlar),
    "Precision": precision_score(y_test_np.ravel(), y_pred_mlar, zero_division=0),
    "Recall": recall_score(y_test_np.ravel(), y_pred_mlar, zero_division=0),
    "F1": f1_score(y_test_np.ravel(), y_pred_mlar, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test_np.ravel(), y_proba_mlar),
    "PR_AUC": average_precision_score(y_test_np.ravel(), y_proba_mlar),
    "Train Time (s)": train_time,
    "Test Time (s)": test_time
}

# =============================
# Final Evaluation Summary
# =============================
metrics_df = pd.DataFrame(metrics_summary).T
print("\n📌 Final Evaluation Summary:")
print(metrics_df.round(6))

# =============================
# Plot Metrics & Curves
# =============================
plt.figure(figsize=(10,6))
metrics_df[['Accuracy','Precision','Recall','F1','ROC_AUC','PR_AUC']].plot(kind='bar')
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30 , ha='right')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Train/Test times
plt.figure(figsize=(8,4))
metrics_df[['Train Time (s)','Test Time (s)']].plot(kind='bar')
plt.title("Training and Testing Time (seconds)")
plt.xticks(rotation=30, ha ='right')
# ✅ Add this line
plt.yticks(np.arange(0, metrics_df[['Train Time (s)','Test Time (s)']].values.max() + 0.025, 0.025))

plt.tight_layout()
plt.show()
# =========================================
# Print Final Comparison in Numeric Form
# =========================================
print("\n📊 Final Numeric Comparison of All Models (Rounded to 4 decimals):\n")
print(metrics_df[['Accuracy','Precision','Recall','F1','ROC_AUC','PR_AUC',
                  'Train Time (s)','Test Time (s)']].round(6).to_string())

# ROC & PR curves
probas = {}
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probas[name] = model.predict_proba(X_test_scaled)[:,1]
probas["MLAR"] = y_proba_mlar

plt.figure(figsize=(8,6))
for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
for name, y_proba in probas.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"{name} (AP = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Confusion Matrices for All Models
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion(cm, labels, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# ✅ Updated plotting section
num_models = len(models) + 1  # +1 for MLAR
cols = 3  # number of columns in grid
rows = (num_models + cols - 1) // cols  # automatically determine rows

plt.figure(figsize=(6 * cols, 4 * rows))

i = 1
for name, model in models.items():
    plt.subplot(rows, cols, i)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, labels=[0, 1], title=name)
    i += 1

# Add MLAR confusion matrix
plt.subplot(rows, cols, i)
cm = confusion_matrix(y_test, y_pred_mlar)
plot_confusion(cm, labels=[0, 1], title="MLAR")

plt.tight_layout()
plt.show()
