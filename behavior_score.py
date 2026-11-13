import pandas as pd
import numpy as np

print("="*70)
print("BEHAVIORAL CONSISTENCY SCORE (BCS)")
print("="*70)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

data = pd.read_csv("dataset/engineered_data.csv")

print("Dataset Loaded :", data.shape)

# -------------------------------------------------------
# Behavior Similarity
# -------------------------------------------------------

behavior = (

    data["LoginFrequency"] +

    data["AvgTransactionAmount"] +

    (1 - data["BehaviorDrift"])

) / 3

# -------------------------------------------------------
# Device Similarity
# -------------------------------------------------------

device = (

    data["DeviceTrustScore"] +

    data["BrowserConsistency"]

) / 2

# -------------------------------------------------------
# Location Similarity
# -------------------------------------------------------

location = (

    data["IdentityScore"] +

    (1 - data["SessionRisk"])

) / 2

# -------------------------------------------------------
# Normalize Function
# -------------------------------------------------------

def normalize(series):

    minimum = series.min()

    maximum = series.max()

    return (series-minimum)/(maximum-minimum+1e-8)

behavior = normalize(behavior)

device = normalize(device)

location = normalize(location)

# -------------------------------------------------------
# Behavioral Consistency Score
# -------------------------------------------------------

data["BehaviorSimilarity"] = behavior

data["DeviceSimilarity"] = device

data["LocationSimilarity"] = location

data["BCS"] = (

    behavior +

    device +

    location

)/3

# -------------------------------------------------------
# Categorize Users
# -------------------------------------------------------

conditions = [

    data["BCS"] >= 0.80,

    (data["BCS"] >= 0.60) & (data["BCS"] < 0.80),

    (data["BCS"] >= 0.40) & (data["BCS"] < 0.60),

    data["BCS"] < 0.40

]

choices = [

    "Highly Consistent",

    "Moderately Consistent",

    "Low Consistency",

    "Suspicious"

]

data["BehaviorCategory"] = np.select(

    conditions,

    choices,

    default="Unknown"

)

# -------------------------------------------------------
# Save Dataset
# -------------------------------------------------------

data.to_csv(

    "dataset/bcs_data.csv",

    index=False

)

print("\nBCS Dataset Saved")

# -------------------------------------------------------
# Generate Research Table
# -------------------------------------------------------

summary = pd.DataFrame({

    "Metric":[

        "Average BCS",

        "Maximum BCS",

        "Minimum BCS",

        "Highly Consistent",

        "Moderately Consistent",

        "Low Consistency",

        "Suspicious"

    ],

    "Value":[

        round(data["BCS"].mean(),4),

        round(data["BCS"].max(),4),

        round(data["BCS"].min(),4),

        len(data[data["BehaviorCategory"]=="Highly Consistent"]),

        len(data[data["BehaviorCategory"]=="Moderately Consistent"]),

        len(data[data["BehaviorCategory"]=="Low Consistency"]),

        len(data[data["BehaviorCategory"]=="Suspicious"])

    ]

})

summary.to_csv(

    "results/tables/behavior_score_results.csv",

    index=False

)

print(summary)

print("\nBehavior Score Module Completed Successfully")