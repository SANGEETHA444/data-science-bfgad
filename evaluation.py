import pandas as pd
import numpy as np
import time

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

print("=" * 70)
print("EVALUATION MODULE — ABLATION STUDY + BASELINE COMPARISON")
print("=" * 70)

data = pd.read_csv("dataset/fusion_data.csv")

# -------------------------------------------------------
# Leakage-safe feature definitions
# UserReputation uses historical fraud rate per card — computed
# on train split only and applied to test, so it is safe here
# as long as we recompute it per split. For this study we use
# the pre-computed column (same split as adaptive_risk) which
# is consistent across all experiments.
# -------------------------------------------------------

# Baseline: original transaction features only (no BGFAD signals)
BASELINE_FEATURES = [
    "TransactionAmt", "TransactionDT",
    "LoginFrequency", "TransactionVelocity", "TransactionRisk",
    "SessionRisk", "DeviceTrustScore", "BrowserConsistency",
    "IdentityScore", "AvgTransactionAmount"
]

# Confirm all baseline features exist
BASELINE_FEATURES = [f for f in BASELINE_FEATURES if f in data.columns]

BCS_ONLY      = ["BCS"]
GTS_ONLY      = ["GraphTrustScore"]
AAS_ONLY      = ["AAS"]
FUSION_ONLY   = ["FusionScore"]

y = data["isFraud"]

# Stratified split — same seed across all experiments for fair comparison
X_full = data[BASELINE_FEATURES + BCS_ONLY + GTS_ONLY + AAS_ONLY + FUSION_ONLY]

_, _, y_train, y_test = train_test_split(
    X_full, y, test_size=0.20, random_state=42, stratify=y
)

train_idx = y_train.index
test_idx  = y_test.index

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 2)

print(f"Train fraud rate : {pos}/{len(y_train)} ({100*pos/len(y_train):.2f}%)")
print(f"scale_pos_weight : {spw}")

# -------------------------------------------------------
# Ablation configurations
# -------------------------------------------------------

experiments = {
    "Baseline":        BASELINE_FEATURES,
    "Baseline + BCS":  BASELINE_FEATURES + BCS_ONLY,
    "Baseline + BCS + GTS":  BASELINE_FEATURES + BCS_ONLY + GTS_ONLY,
    "Baseline + BCS + GTS + AAS": BASELINE_FEATURES + BCS_ONLY + GTS_ONLY + AAS_ONLY,
    "BGFAD (Full)":    BASELINE_FEATURES + BCS_ONLY + GTS_ONLY + AAS_ONLY + FUSION_ONLY,
}

def compute_metrics(y_true, y_pred, y_prob, name, elapsed):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0
    return {
        "Experiment":    name,
        "Accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "Precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":      round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC":        round(average_precision_score(y_true, y_prob), 4),
        "FPR":           fpr,
        "FNR":           fnr,
        "Exec Time (s)": round(elapsed, 4),
    }

ablation_rows = []

print("\nRunning Ablation Study...\n")

for name, feats in experiments.items():
    X_tr = data.loc[train_idx, feats]
    X_te = data.loc[test_idx,  feats]
    y_tr = y_train
    y_te = y_test

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        scale_pos_weight=spw,
        random_state=42,
        verbose=-1
    )

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]
    elapsed = time.time() - t0

    row = compute_metrics(y_te, pred, prob, name, elapsed)
    ablation_rows.append(row)

    print(f"  {name:<35} | Acc:{row['Accuracy']} | F1:{row['F1 Score']} | "
          f"Recall:{row['Recall']} | ROC-AUC:{row['ROC-AUC']} | PR-AUC:{row['PR-AUC']}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv("results/tables/ablation_study.csv", index=False)

print("\n--- Ablation Study Table ---")
print(ablation_df.to_string(index=False))

# -------------------------------------------------------
# Table IV — Baseline Model Comparison (4 classifiers on Full BGFAD features)
# -------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("\n\nRunning Baseline Model Comparison (Table IV)...\n")

FULL_FEATURES = BASELINE_FEATURES + BCS_ONLY + GTS_ONLY + AAS_ONLY + FUSION_ONLY

X_tr_full = data.loc[train_idx, FULL_FEATURES]
X_te_full = data.loc[test_idx,  FULL_FEATURES]

classifiers = {
    "LightGBM":          LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                                         scale_pos_weight=spw, random_state=42, verbose=-1),
    "Random Forest":     RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Decision Tree":     DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
}

comparison_rows = []

for name, clf in classifiers.items():
    t0 = time.time()
    clf.fit(X_tr_full, y_train)
    pred = clf.predict(X_te_full)
    prob = clf.predict_proba(X_te_full)[:, 1]
    elapsed = time.time() - t0

    row = compute_metrics(y_test, pred, prob, name, elapsed)
    comparison_rows.append(row)

    print(f"  {name:<25} | Acc:{row['Accuracy']} | F1:{row['F1 Score']} | "
          f"Recall:{row['Recall']} | ROC-AUC:{row['ROC-AUC']}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("results/tables/model_comparison.csv", index=False)

# Save best model (LightGBM) confusion matrix
best_clf = classifiers["LightGBM"]
best_pred = best_clf.predict(X_te_full)
cm = confusion_matrix(y_test, best_pred)
cm_df = pd.DataFrame(cm,
    columns=["Pred Normal", "Pred Fraud"],
    index=["Actual Normal", "Actual Fraud"])
cm_df.to_csv("results/tables/confusion_matrix.csv")

print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))
print("\nEvaluation Completed Successfully")
