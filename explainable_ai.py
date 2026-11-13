import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

print("=" * 70)
print("EXPLAINABLE AI MODULE")
print("=" * 70)

data = pd.read_csv("dataset/fusion_data.csv")

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 2)

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    scale_pos_weight=spw,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

print("\nGenerating SHAP Values...")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values is a list [class0, class1] for binary classifiers in newer shap
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# SHAP Summary Plot (beeswarm)
plt.figure()
shap.summary_plot(sv, X_test, show=False)
plt.tight_layout()
plt.savefig("results/graphs/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# SHAP Bar Plot
plt.figure()
shap.summary_plot(sv, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/graphs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# Save importance table
importance = pd.DataFrame({
    "Feature":    X.columns,
    "SHAP_Mean":  np.abs(sv).mean(axis=0).round(6),
    "LGB_Importance": model.feature_importances_
}).sort_values("SHAP_Mean", ascending=False)

importance.to_csv("results/tables/feature_importance.csv", index=False)

print("\nFeature Importance (by mean |SHAP|)")
print(importance.to_string(index=False))
print("\nExplainable AI Completed Successfully")
