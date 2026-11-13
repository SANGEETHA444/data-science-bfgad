import os
import time

print("=" * 80)
print("Behavior-Aware Graph Fusion Adaptive Detection (BGFAD)")
print("A Novel Explainable AI Framework for Real-Time Identity Theft")
print("and Account Takeover Detection in Fintech Ecosystems")
print("=" * 80)

start = time.time()

# -------------------------------------------------------
# Create Required Directories
# -------------------------------------------------------

folders = [
    "dataset",
    "results",
    "results/tables",
    "results/graphs",
    "results/report"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("\nRequired folders verified.")

# -------------------------------------------------------
# Execute Modules
# -------------------------------------------------------

modules = [

    "preprocessing.py",

    "feature_engineering.py",

    "behavior_score.py",

    "graph_builder.py",

    "graph_trust.py",

    "anomaly_score.py",

    "fusion.py",

    "adaptive_risk.py",

    "evaluation.py",

    "self_learning.py",

    "visualization.py",

    "explainable_ai.py"

]

print("\nStarting BGFAD Pipeline...\n")

for module in modules:

    print("-" * 60)
    print(f"Running {module}")
    print("-" * 60)

    status = os.system(f"python {module}")

    if status != 0:

        print(f"\nError occurred while executing {module}")

        print("Pipeline Stopped.")

        exit()

    print(f"{module} Completed Successfully\n")

# -------------------------------------------------------
# Execution Time
# -------------------------------------------------------

end = time.time()

execution = end - start

print("=" * 80)
print("BGFAD PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"Total Execution Time : {execution:.2f} seconds")

print("\nGenerated Outputs")

print("""
Dataset Files
-------------
processed_data.csv
engineered_data.csv
bcs_data.csv
graph_nodes.csv
graph_edges.csv
graph_trust_scores.csv
anomaly_data.csv
fusion_data.csv
final_predictions.csv

Research Tables
---------------
preprocessing_summary.csv
feature_engineering_summary.csv
behavior_score_results.csv
graph_statistics.csv
graph_trust_results.csv
anomaly_score_results.csv
fusion_results.csv
model_performance.csv
model_comparison.csv
confusion_matrix.csv
feature_importance.csv
ablation_study.csv
self_learning_curve.csv
self_learning_results.csv

Graphs
------
class_distribution.png
bcs_distribution.png
gts_distribution.png
aas_distribution.png
fusion_distribution.png
ablation_study.png
model_comparison.png
confusion_matrix.png
precision_recall_f1.png
roc_pr_auc.png
fpr_fnr_ablation.png
shap_summary.png
feature_importance.png
""")

print("\nResearch Pipeline Finished Successfully.")