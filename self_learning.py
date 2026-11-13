import pandas as pd
import numpy as np
import os

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

print("=" * 70)
print("SELF-LEARNING MODULE")
print("=" * 70)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

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

X = data[features].values
y = data["isFraud"].values

# -------------------------------------------------------
# Train / Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------------------------------
# Scale
# -------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------------------------------------
# Incremental Learning with SGD (simulates online updates)
# -------------------------------------------------------

weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}

model = SGDClassifier(
    loss="log",
    max_iter=1,
    tol=None,
    warm_start=True,
    class_weight=class_weight_dict,
    random_state=42
)

batch_size = 500
n_batches  = len(X_train) // batch_size

print(f"\nTraining with {n_batches} incremental batches (batch size = {batch_size})...")

round_results = []

for i in range(n_batches):

    start = i * batch_size
    end   = start + batch_size

    X_batch = X_train[start:end]
    y_batch = y_train[start:end]

    model.partial_fit(X_batch, y_batch, classes=[0, 1])

    if (i + 1) % 5 == 0 or i == n_batches - 1:

        pred = model.predict(X_test)

        acc  = round(accuracy_score(y_test, pred), 4)
        prec = round(precision_score(y_test, pred, zero_division=0), 4)
        rec  = round(recall_score(y_test, pred, zero_division=0), 4)
        f1   = round(f1_score(y_test, pred, zero_division=0), 4)

        round_results.append({
            "Batch": i + 1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

        print(f"  Batch {i+1:>3} | Acc: {acc} | Prec: {prec} | Rec: {rec} | F1: {f1}")

# -------------------------------------------------------
# Feedback Loop Simulation
# -------------------------------------------------------
# Simulate analyst feedback: flagged high-risk predictions
# are confirmed and fed back into the model for retraining.

print("\nSimulating Feedback Loop...")

pred_all   = model.predict(X_test)
prob_all   = model.predict_proba(X_test)[:, 1]

# Feedback: samples where model predicted fraud (high confidence)
feedback_mask = prob_all >= 0.70
X_feedback    = X_test[feedback_mask]
y_feedback    = y_test[feedback_mask]   # ground truth labels as analyst confirmation

if len(X_feedback) > 0:
    model.partial_fit(X_feedback, y_feedback, classes=[0, 1])
    print(f"  Feedback samples used for retraining : {len(X_feedback)}")
else:
    print("  No high-confidence feedback samples found.")

# -------------------------------------------------------
# Final Evaluation after Feedback
# -------------------------------------------------------

final_pred = model.predict(X_test)

final_acc  = round(accuracy_score(y_test, final_pred), 4)
final_prec = round(precision_score(y_test, final_pred, zero_division=0), 4)
final_rec  = round(recall_score(y_test, final_pred, zero_division=0), 4)
final_f1   = round(f1_score(y_test, final_pred, zero_division=0), 4)

print(f"\nFinal Performance After Feedback Loop")
print(f"  Accuracy  : {final_acc}")
print(f"  Precision : {final_prec}")
print(f"  Recall    : {final_rec}")
print(f"  F1 Score  : {final_f1}")

# -------------------------------------------------------
# Save Learning Curve
# -------------------------------------------------------

curve_df = pd.DataFrame(round_results)
curve_df.to_csv("results/tables/self_learning_curve.csv", index=False)

# -------------------------------------------------------
# Save Final Metrics
# -------------------------------------------------------

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Feedback Samples", "Total Batches"],
    "Value":  [final_acc, final_prec, final_rec, final_f1, int(feedback_mask.sum()), n_batches]
})

summary.to_csv("results/tables/self_learning_results.csv", index=False)

print("\nSelf-Learning Results Saved")
print("Self-Learning Module Completed Successfully")
