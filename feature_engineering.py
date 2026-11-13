import pandas as pd
import numpy as np

print("="*70)
print("FEATURE ENGINEERING MODULE")
print("="*70)

# ---------------------------------------------------
# Load Processed Dataset
# ---------------------------------------------------

data = pd.read_csv("dataset/processed_data.csv")

print("Dataset Loaded Successfully")
print("Shape :", data.shape)

# ---------------------------------------------------
# 1. Login Frequency
# ---------------------------------------------------

print("\nGenerating Login Frequency...")

if "card1" in data.columns:

    data["LoginFrequency"] = data.groupby("card1")["TransactionID"].transform("count")

else:

    data["LoginFrequency"] = 1

# ---------------------------------------------------
# 2. Average Transaction Amount
# ---------------------------------------------------

print("Generating Average Transaction Amount...")

if "TransactionAmt" in data.columns and "card1" in data.columns:

    data["AvgTransactionAmount"] = data.groupby("card1")["TransactionAmt"].transform("mean")

else:

    data["AvgTransactionAmount"] = 0

# ---------------------------------------------------
# 3. Transaction Velocity
# ---------------------------------------------------

print("Generating Transaction Velocity...")

if "TransactionDT" in data.columns:

    data = data.sort_values("TransactionDT")

    data["PreviousTransaction"] = data["TransactionDT"].shift(1)

    data["TransactionVelocity"] = (
        data["TransactionDT"] -
        data["PreviousTransaction"]
    ).fillna(0)

else:

    data["TransactionVelocity"] = 0

# ---------------------------------------------------
# 4. Device Trust Score
# ---------------------------------------------------

print("Generating Device Trust Score...")

if "DeviceType" in data.columns:

    device_count = data.groupby("DeviceType")["TransactionID"].transform("count")

    data["DeviceTrustScore"] = device_count / device_count.max()

else:

    data["DeviceTrustScore"] = 0.5

# ---------------------------------------------------
# 5. Browser Consistency Score
# ---------------------------------------------------

print("Generating Browser Consistency...")

if "DeviceInfo" in data.columns:

    browser_count = data.groupby("DeviceInfo")["TransactionID"].transform("count")

    data["BrowserConsistency"] = browser_count / browser_count.max()

else:

    data["BrowserConsistency"] = 0.5

# ---------------------------------------------------
# 6. User Reputation
# ---------------------------------------------------

print("Generating User Reputation...")

if "card1" in data.columns and "isFraud" in data.columns:

    fraud_rate = data.groupby("card1")["isFraud"].transform("mean")

    data["UserReputation"] = 1 - fraud_rate

else:

    data["UserReputation"] = 1

# ---------------------------------------------------
# 7. Transaction Risk
# ---------------------------------------------------

print("Generating Transaction Risk...")

if "TransactionAmt" in data.columns:

    amount = data["TransactionAmt"]

    data["TransactionRisk"] = (
        amount - amount.min()
    ) / (
        amount.max() - amount.min()
    )

else:

    data["TransactionRisk"] = 0

# ---------------------------------------------------
# 8. Session Risk
# ---------------------------------------------------

print("Generating Session Risk...")

if "TransactionDT" in data.columns:

    hour = (data["TransactionDT"] // 3600) % 24

    data["SessionRisk"] = np.where(

        (hour >= 0) & (hour <= 5),

        1,

        0

    )

else:

    data["SessionRisk"] = 0

# ---------------------------------------------------
# 9. Identity Score
# ---------------------------------------------------

print("Generating Identity Score...")

identity_cols = [c for c in data.columns if c.startswith("id_")]

if len(identity_cols) > 0:

    data["IdentityScore"] = data[identity_cols].mean(axis=1)

else:

    data["IdentityScore"] = 0

# ---------------------------------------------------
# 10. Behavioral Drift Score
# ---------------------------------------------------

print("Generating Behavioral Drift Score...")

data["BehaviorDrift"] = (

    data["TransactionRisk"]

    + data["SessionRisk"]

    + (1 - data["UserReputation"])

) / 3

# ---------------------------------------------------
# Save Dataset
# ---------------------------------------------------

output = "dataset/engineered_data.csv"

data.to_csv(output, index=False)

print("\nEngineered Dataset Saved")

print(output)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

features = [

    "LoginFrequency",

    "AvgTransactionAmount",

    "TransactionVelocity",

    "DeviceTrustScore",

    "BrowserConsistency",

    "UserReputation",

    "TransactionRisk",

    "SessionRisk",

    "IdentityScore",

    "BehaviorDrift"

]

summary = pd.DataFrame({

    "Feature": features,

    "Description": [

        "Number of user logins",

        "Average amount spent",

        "Time between transactions",

        "Trust score of device",

        "Browser consistency",

        "Historical user reputation",

        "Risk based on transaction amount",

        "Risk based on login time",

        "Identity confidence",

        "Behavioral deviation"

    ]

})

summary.to_csv(

    "results/tables/feature_engineering_summary.csv",

    index=False

)

print("\nFeature Engineering Completed Successfully")