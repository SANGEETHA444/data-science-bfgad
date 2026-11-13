import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.metrics import ConfusionMatrixDisplay

print("=" * 70)
print("VISUALIZATION MODULE")
print("=" * 70)

os.makedirs("results/graphs", exist_ok=True)

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

# -------------------------------------------------------
# Helper
# -------------------------------------------------------

def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# -------------------------------------------------------
# Load data
# -------------------------------------------------------

fusion    = pd.read_csv("dataset/fusion_data.csv")
ablation  = pd.read_csv("results/tables/ablation_study.csv")
comparison= pd.read_csv("results/tables/model_comparison.csv")
cm_df     = pd.read_csv("results/tables/confusion_matrix.csv", index_col=0)

# -------------------------------------------------------
# Figure 1 — Class Distribution
# -------------------------------------------------------

counts = fusion["isFraud"].value_counts().sort_index()
labels = ["Normal", "Fraud"]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, counts.values, color=["#2196F3", "#F44336"], width=0.4)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", fontsize=10)
plt.title("Class Distribution")
plt.ylabel("Count")
save("results/graphs/class_distribution.png")

# -------------------------------------------------------
# Figure 2 — BCS Distribution
# -------------------------------------------------------

plt.figure(figsize=(7, 4))
plt.hist(fusion["BCS"], bins=40, color="#2196F3", edgecolor="white")
plt.axvline(fusion["BCS"].mean(), color="red", linestyle="--", label=f"Mean={fusion['BCS'].mean():.3f}")
plt.title("Behavioral Consistency Score (BCS) Distribution")
plt.xlabel("BCS")
plt.ylabel("Frequency")
plt.legend()
save("results/graphs/bcs_distribution.png")

# -------------------------------------------------------
# Figure 3 — GTS Distribution
# -------------------------------------------------------

plt.figure(figsize=(7, 4))
plt.hist(fusion["GraphTrustScore"], bins=40, color="#4CAF50", edgecolor="white")
plt.axvline(fusion["GraphTrustScore"].mean(), color="red", linestyle="--",
            label=f"Mean={fusion['GraphTrustScore'].mean():.3f}")
plt.title("Graph Trust Score (GTS) Distribution")
plt.xlabel("GTS")
plt.ylabel("Frequency")
plt.legend()
save("results/graphs/gts_distribution.png")

# -------------------------------------------------------
# Figure 4 — AAS Distribution
# -------------------------------------------------------

plt.figure(figsize=(7, 4))
plt.hist(fusion["AAS"], bins=40, color="#FF9800", edgecolor="white")
plt.axvline(fusion["AAS"].mean(), color="red", linestyle="--",
            label=f"Mean={fusion['AAS'].mean():.3f}")
plt.title("Adaptive Anomaly Score (AAS) Distribution")
plt.xlabel("AAS")
plt.ylabel("Frequency")
plt.legend()
save("results/graphs/aas_distribution.png")

# -------------------------------------------------------
# Figure 5 — FusionScore Distribution
# -------------------------------------------------------

plt.figure(figsize=(7, 4))
plt.hist(fusion["FusionScore"], bins=40, color="#9C27B0", edgecolor="white")
plt.axvline(fusion["FusionScore"].mean(), color="red", linestyle="--",
            label=f"Mean={fusion['FusionScore'].mean():.3f}")
plt.title("Context Fusion Score Distribution")
plt.xlabel("FusionScore")
plt.ylabel("Frequency")
plt.legend()
save("results/graphs/fusion_distribution.png")

# -------------------------------------------------------
# Figure 6 — Ablation Study (F1, Recall, ROC-AUC, PR-AUC)
# -------------------------------------------------------

metrics = ["F1 Score", "Recall", "ROC-AUC", "PR-AUC"]
x = np.arange(len(ablation))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 5))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, ablation[metric], width, label=metric, color=COLORS[i])

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(ablation["Experiment"], rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Score")
ax.set_title("Ablation Study — Incremental Component Contribution")
ax.legend()
ax.set_ylim(0, 1.05)
save("results/graphs/ablation_study.png")

# -------------------------------------------------------
# Figure 7 — Model Comparison (Accuracy, F1, ROC-AUC)
# -------------------------------------------------------

metrics_cmp = ["Accuracy", "F1 Score", "ROC-AUC"]
x2 = np.arange(len(comparison))
width2 = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
for i, metric in enumerate(metrics_cmp):
    ax.bar(x2 + i * width2, comparison[metric], width2, label=metric, color=COLORS[i])

ax.set_xticks(x2 + width2)
ax.set_xticklabels(comparison["Experiment"], rotation=10, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Baseline Model Comparison")
ax.legend()
ax.set_ylim(0, 1.05)
save("results/graphs/model_comparison.png")

# -------------------------------------------------------
# Figure 8 — Confusion Matrix
# -------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_df.values,
    display_labels=["Normal", "Fraud"]
)
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — BGFAD (LightGBM)")
save("results/graphs/confusion_matrix.png")

# -------------------------------------------------------
# Figure 9 — Precision / Recall / F1 bar (model comparison)
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric, color in zip(axes, ["Precision", "Recall", "F1 Score"], COLORS):
    ax.bar(comparison["Experiment"], comparison[metric], color=color)
    ax.set_title(metric)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15, labelsize=7)
save("results/graphs/precision_recall_f1.png")

# -------------------------------------------------------
# Figure 10 — ROC-AUC & PR-AUC comparison
# -------------------------------------------------------

x3 = np.arange(len(comparison))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x3 - 0.2, comparison["ROC-AUC"], 0.4, label="ROC-AUC", color="#2196F3")
ax.bar(x3 + 0.2, comparison["PR-AUC"],  0.4, label="PR-AUC",  color="#FF9800")
ax.set_xticks(x3)
ax.set_xticklabels(comparison["Experiment"], rotation=10, ha="right", fontsize=8)
ax.set_ylabel("Score")
ax.set_title("ROC-AUC vs PR-AUC Comparison")
ax.legend()
ax.set_ylim(0, 1.05)
save("results/graphs/roc_pr_auc.png")

# -------------------------------------------------------
# Figure 11 — FPR / FNR comparison
# -------------------------------------------------------

x4 = np.arange(len(ablation))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x4 - 0.2, ablation["FPR"], 0.4, label="FPR", color="#F44336")
ax.bar(x4 + 0.2, ablation["FNR"], 0.4, label="FNR", color="#FF9800")
ax.set_xticks(x4)
ax.set_xticklabels(ablation["Experiment"], rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Rate")
ax.set_title("False Positive Rate vs False Negative Rate — Ablation")
ax.legend()
save("results/graphs/fpr_fnr_ablation.png")
# -------------------------------------------------------
# Figure 12 — Transaction Amount Distribution
# Genuine vs Fraudulent (Log Scale)
# -------------------------------------------------------

genuine = fusion.loc[
    (fusion["isFraud"] == 0) & (fusion["TransactionAmt"] > 0),
    "TransactionAmt"
].dropna()

fraud = fusion.loc[
    (fusion["isFraud"] == 1) & (fusion["TransactionAmt"] > 0),
    "TransactionAmt"
].dropna()

# Logarithmic bins
all_amounts = pd.concat([genuine, fraud])

bins = np.logspace(
    np.log10(all_amounts.min()),
    np.log10(all_amounts.max()),
    60
)

plt.figure(figsize=(9, 5))

plt.hist(
    genuine,
    bins=bins,
    density=True,
    alpha=0.6,
    label="Genuine",
    color="#2196F3"
)

plt.hist(
    fraud,
    bins=bins,
    density=True,
    alpha=0.6,
    label="Fraudulent",
    color="#F44336"
)

plt.xscale("log")

plt.xlabel("Transaction Amount (Log Scale)")
plt.ylabel("Density")
plt.title(
    "Transaction Amount Distribution: "
    "Genuine vs Fraudulent"
)

plt.legend()
save("results/graphs/transaction_amount_distribution.png")
print("\nAll Graphs Generated Successfully")
