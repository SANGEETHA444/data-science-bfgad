# BGFAD — Behavior-Aware Graph Fusion Adaptive Detection

A Novel Explainable AI Framework for Real-Time Identity Theft and Account Takeover Detection in Fintech Ecosystems.

---

## Overview

BGFAD is a multi-layered fraud detection pipeline that combines:

- **Behavioral Consistency Scoring (BCS)** — measures how consistently a user behaves across sessions
- **Graph Trust Score (GTS)** — propagates trust through a heterogeneous transaction graph
- **Adaptive Anomaly Score (AAS)** — fuses behavioral drift, transaction risk, and graph trust
- **Context Fusion** — weighted combination of BCS, GTS, and AAS into a single fraud signal
- **Adaptive Risk Detection** — LightGBM classifier with risk-tiered decisions
- **Self-Learning** — incremental online learning with analyst feedback loop
- **Explainable AI** — SHAP values for full model transparency

---

## Dataset

IEEE-CIS Fraud Detection Dataset (Kaggle)

| File | Rows | Columns |
|---|---|---|
| train_transaction.csv | 590,540 | 394 |
| train_identity.csv | 144,233 | 41 |

Place both files in the `dataset/` folder before running.

---

## Project Structure

```
BGFAD/
├── dataset/                    # Input and intermediate datasets
├── results/
│   ├── tables/                 # CSV result tables
│   ├── graphs/                 # PNG visualizations
│   └── report/                 # Full research report
├── preprocessing.py            # Data cleaning and scaling
├── feature_engineering.py      # 10 domain-specific features
├── behavior_score.py           # Behavioral Consistency Score
├── graph_builder.py            # Heterogeneous graph construction
├── graph_trust.py              # Graph Trust Score propagation
├── anomaly_score.py            # Adaptive Anomaly Score
├── fusion.py                   # Context Fusion Score
├── adaptive_risk.py            # LightGBM risk classifier
├── evaluation.py               # Multi-model comparison
├── self_learning.py            # Incremental learning + feedback loop
├── visualization.py            # Result graphs
├── explainable_ai.py           # SHAP explanations
└── main.py                     # Full pipeline runner
```

---

## Installation

```bash
pip install pandas numpy scikit-learn lightgbm networkx shap matplotlib
```

---

## Run

```bash
python main.py
```

---

## Pipeline Steps

| Step | Module | Output |
|---|---|---|
| 1 | preprocessing.py | processed_data.csv |
| 2 | feature_engineering.py | engineered_data.csv |
| 3 | behavior_score.py | bcs_data.csv |
| 4 | graph_builder.py | graph_nodes.csv, graph_edges.csv |
| 5 | graph_trust.py | graph_trust_scores.csv |
| 6 | anomaly_score.py | anomaly_data.csv |
| 7 | fusion.py | fusion_data.csv |
| 8 | adaptive_risk.py | final_predictions.csv |
| 9 | evaluation.py | model_comparison.csv |
| 10 | self_learning.py | self_learning_results.csv |
| 11 | visualization.py | 7 PNG graphs |
| 12 | explainable_ai.py | shap_summary.png, feature_importance.csv |

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| LightGBM | 0.9891 | 0.3000 | 0.0545 | 0.0923 |
| Random Forest | 0.9874 | 0.1905 | 0.0727 | 0.1053 |
| Decision Tree | 0.9837 | 0.0976 | 0.0727 | 0.0833 |
| Logistic Regression | 0.9902 | 1.0000 | 0.0364 | 0.0702 |

Top SHAP Features: AAS > FusionScore > TransactionRisk > GraphTrustScore > BCS

---

## Risk Decision Tiers

| Risk Score | Decision |
|---|---|
| < 30 | Safe |
| 30 – 59 | OTP Verification |
| 60 – 79 | Biometric Verification |
| >= 80 | Blocked |

---

## Report

Full research report: `results/report/report.txt`
