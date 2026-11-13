import pandas as pd
import numpy as np

print("="*70)
print("CONTEXT FUSION MODULE")
print("="*70)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

data = pd.read_csv("dataset/anomaly_data.csv")

print("Dataset Loaded :", data.shape)

# -------------------------------------------------
# Check Required Columns
# -------------------------------------------------

required = ["BCS","GraphTrustScore","AAS"]

for col in required:

    if col not in data.columns:

        raise Exception(f"{col} is missing.")

# -------------------------------------------------
# Context Fusion Score
# -------------------------------------------------

data["FusionScore"] = (

    0.40 * data["BCS"]

    +

    0.35 * data["GraphTrustScore"]

    +

    0.25 * (1-data["AAS"])

)

# -------------------------------------------------
# Normalize
# -------------------------------------------------

minimum = data["FusionScore"].min()

maximum = data["FusionScore"].max()

data["FusionScore"] = (

    data["FusionScore"]-minimum

)/(maximum-minimum+1e-8)

# -------------------------------------------------
# Confidence Level
# -------------------------------------------------

conditions=[

data["FusionScore"]>=0.85,

(data["FusionScore"]>=0.70)&
(data["FusionScore"]<0.85),

(data["FusionScore"]>=0.50)&
(data["FusionScore"]<0.70),

data["FusionScore"]<0.50

]

choices=[

"Highly Trusted",

"Trusted",

"Suspicious",

"Fraudulent"

]

data["FusionCategory"]=np.select(

conditions,

choices,

default="Unknown"

)

# -------------------------------------------------
# Save
# -------------------------------------------------

data.to_csv(

"dataset/fusion_data.csv",

index=False

)

# -------------------------------------------------
# Summary Table
# -------------------------------------------------

summary=pd.DataFrame({

"Metric":[

"Average Fusion Score",

"Maximum Fusion Score",

"Minimum Fusion Score",

"Highly Trusted",

"Trusted",

"Suspicious",

"Fraudulent"

],

"Value":[

round(data["FusionScore"].mean(),4),

round(data["FusionScore"].max(),4),

round(data["FusionScore"].min(),4),

len(data[data["FusionCategory"]=="Highly Trusted"]),

len(data[data["FusionCategory"]=="Trusted"]),

len(data[data["FusionCategory"]=="Suspicious"]),

len(data[data["FusionCategory"]=="Fraudulent"])

]

})

summary.to_csv(

"results/tables/fusion_results.csv",

index=False

)

print(summary)

print("\nFusion Module Completed Successfully")