import pandas as pd
import numpy as np

print("="*70)
print("ADAPTIVE ANOMALY SCORE (AAS)")
print("="*70)

# ---------------------------------------------------
# Load Files
# ---------------------------------------------------

data = pd.read_csv("dataset/bcs_data.csv")

graph = pd.read_csv("dataset/graph_trust_scores.csv")

print("Dataset Loaded")

# ---------------------------------------------------
# User Graph Trust
# ---------------------------------------------------

graph["UserID"] = graph["Node"].str.replace("USER_", "", regex=False)

user_graph = graph[
    graph["Node"].str.startswith("USER")
][["UserID", "GraphTrustScore"]]

user_graph["UserID"] = pd.to_numeric(
    user_graph["UserID"],
    errors="coerce"
)

# ---------------------------------------------------
# Merge Trust Score
# ---------------------------------------------------

if "card1" in data.columns:

    data = data.merge(

        user_graph,

        left_on="card1",

        right_on="UserID",

        how="left"

    )

else:

    data["GraphTrustScore"] = 0.7

data["GraphTrustScore"] = data["GraphTrustScore"].fillna(0.70)

# ---------------------------------------------------
# Adaptive Anomaly Score
# ---------------------------------------------------

data["AAS"] = (

    data["BehaviorDrift"]

    +

    data["TransactionRisk"]

    +

    (1-data["GraphTrustScore"])

    +

    (1-data["UserReputation"])

)/4

# ---------------------------------------------------
# Normalize
# ---------------------------------------------------

minimum = data["AAS"].min()

maximum = data["AAS"].max()

data["AAS"] = (

    data["AAS"]-minimum

)/(maximum-minimum+1e-8)

# ---------------------------------------------------
# Risk Category
# ---------------------------------------------------

conditions = [

    data["AAS"] <0.30,

    (data["AAS"]>=0.30)&(data["AAS"]<0.60),

    (data["AAS"]>=0.60)&(data["AAS"]<0.80),

    data["AAS"]>=0.80

]

choices=[

    "Normal",

    "Low Risk",

    "Medium Risk",

    "High Risk"

]

data["AnomalyCategory"]=np.select(

    conditions,

    choices,

    default="Unknown"

)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

data.to_csv(

    "dataset/anomaly_data.csv",

    index=False

)

# ---------------------------------------------------
# Research Table
# ---------------------------------------------------

summary=pd.DataFrame({

"Metric":[

"Average AAS",

"Maximum AAS",

"Minimum AAS",

"Normal",

"Low Risk",

"Medium Risk",

"High Risk"

],

"Value":[

round(data["AAS"].mean(),4),

round(data["AAS"].max(),4),

round(data["AAS"].min(),4),

len(data[data["AnomalyCategory"]=="Normal"]),

len(data[data["AnomalyCategory"]=="Low Risk"]),

len(data[data["AnomalyCategory"]=="Medium Risk"]),

len(data[data["AnomalyCategory"]=="High Risk"])

]

})

summary.to_csv(

"results/tables/anomaly_score_results.csv",

index=False

)

print(summary)

print("\nAdaptive Anomaly Score Generated Successfully")