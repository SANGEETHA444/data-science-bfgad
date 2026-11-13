import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from lightgbm import LGBMClassifier

print("=" * 70)
print("ADAPTIVE RISK DETECTION")
print("=" * 70)

data = pd.read_csv("dataset/fusion_data.csv")

print("Dataset Shape :", data.shape)

features = [
    "BCS",
    "GraphTrustScore",
    "AAS",
    "FusionScore",
    "LoginFrequency",
    "TransactionRisk",
    "UserReputation",
    "DeviceTrustScore",
    "BrowserConsistency"
]

X = data[features]
y = data["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Fix class imbalance
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 2)
print(f"scale_pos_weight : {spw}  (neg={neg}, pos={pos})")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    scale_pos_weight=spw,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

prediction  = model.predict(X_test)
probability = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, zero_division=0)
recall    = recall_score(y_test, prediction, zero_division=0)
f1        = f1_score(y_test, prediction, zero_division=0)
roc_auc   = roc_auc_score(y_test, probability)
pr_auc    = average_precision_score(y_test, probability)
cm        = confusion_matrix(y_test, prediction)
tn, fp, fn, tp = cm.ravel()
fpr = round(fp / (fp + tn), 4)
fnr = round(fn / (fn + tp), 4)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"FPR       : {fpr}")
print(f"FNR       : {fnr}")
print(f"\n{classification_report(y_test, prediction)}")

# Risk score and decision
result = X_test.copy()
result["Actual"]     = y_test.values
result["Prediction"] = prediction
result["RiskScore"]  = (probability * 100).round(2)

def category(score):
    if score < 30:   return "Safe"
    elif score < 60: return "OTP Verification"
    elif score < 80: return "Biometric Verification"
    else:            return "Blocked"

result["Decision"] = result["RiskScore"].apply(category)
result.to_csv("dataset/final_predictions.csv", index=False)

# Save metrics
summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score",
               "ROC-AUC", "PR-AUC", "FPR", "FNR"],
    "Value":  [round(accuracy, 4), round(precision, 4), round(recall, 4),
               round(f1, 4), round(roc_auc, 4), round(pr_auc, 4), fpr, fnr]
})
summary.to_csv("results/tables/model_performance.csv", index=False)

# Save confusion matrix
cm_df = pd.DataFrame(cm,
    columns=["Pred Normal", "Pred Fraud"],
    index=["Actual Normal", "Actual Fraud"])
cm_df.to_csv("results/tables/confusion_matrix.csv")

print("\nAdaptive Risk Detection Completed")
