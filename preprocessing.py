import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------------------------------
# STEP 1 : Load Dataset
# -----------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

transaction = pd.read_csv("dataset/train_transaction.csv")
identity = pd.read_csv("dataset/train_identity.csv")

print("Transaction Dataset :", transaction.shape)
print("Identity Dataset    :", identity.shape)

# -----------------------------------------------------
# STEP 2 : Merge Dataset
# -----------------------------------------------------

print("\nMerging datasets...")

data = transaction.merge(identity, on="TransactionID", how="left")

print("Merged Dataset Shape :", data.shape)

# -----------------------------------------------------
# STEP 3 : Dataset Statistics
# -----------------------------------------------------

print("\nDataset Information")

print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

# -----------------------------------------------------
# STEP 4 : Duplicate Removal
# -----------------------------------------------------

duplicate_rows = data.duplicated().sum()

print("\nDuplicate Rows :", duplicate_rows)

data.drop_duplicates(inplace=True)

print("Rows after removing duplicates :", len(data))

# -----------------------------------------------------
# STEP 5 : Missing Value Handling
# -----------------------------------------------------

print("\nHandling Missing Values...")

missing_before = data.isnull().sum().sum()

print("Missing Values Before :", missing_before)

for col in data.columns:

    if data[col].dtype == "object":

        data[col].fillna("Unknown", inplace=True)

    else:

        data[col].fillna(data[col].median(), inplace=True)

missing_after = data.isnull().sum().sum()

print("Missing Values After :", missing_after)

# -----------------------------------------------------
# STEP 6 : Remove Outliers (IQR)
# -----------------------------------------------------

print("\nRemoving Outliers...")

numeric_columns = data.select_dtypes(include=np.number).columns

rows_before = len(data)

for col in numeric_columns:

    if data[col].nunique() < 10:
        continue

    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data = data[
        (data[col] >= lower) &
        (data[col] <= upper)
    ]

rows_after = len(data)

print("Rows Before :", rows_before)
print("Rows After  :", rows_after)
print("Outliers Removed :", rows_before - rows_after)

# -----------------------------------------------------
# STEP 7 : Label Encoding
# -----------------------------------------------------

print("\nEncoding Categorical Features...")

categorical_columns = data.select_dtypes(include="object").columns

encoder = LabelEncoder()

for col in categorical_columns:

    data[col] = encoder.fit_transform(data[col].astype(str))

print("Encoded Columns :", len(categorical_columns))

# -----------------------------------------------------
# STEP 8 : Feature Scaling
# -----------------------------------------------------

print("\nApplying Standard Scaling...")

scaler = StandardScaler()

numeric_columns = data.select_dtypes(include=np.number).columns

numeric_columns = [c for c in numeric_columns if c != "isFraud"]

data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print("Scaling Completed")

# -----------------------------------------------------
# STEP 9 : Save Dataset
# -----------------------------------------------------

output_path = "dataset/processed_data.csv"

data.to_csv(output_path, index=False)

print("\nProcessed Dataset Saved")

print(output_path)

# -----------------------------------------------------
# STEP 10 : Summary Table
# -----------------------------------------------------

summary = pd.DataFrame({

    "Metric": [

        "Original Rows",
        "Final Rows",
        "Original Columns",
        "Final Columns",
        "Duplicate Rows Removed",
        "Missing Values Filled",
        "Outliers Removed",
        "Encoded Features"

    ],

    "Value": [

        transaction.shape[0],
        data.shape[0],
        transaction.shape[1] + identity.shape[1] - 1,
        data.shape[1],
        duplicate_rows,
        missing_before,
        rows_before - rows_after,
        len(categorical_columns)

    ]

})

print("\n")
print("=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

print(summary)

summary.to_csv("results/tables/preprocessing_summary.csv", index=False)

print("\nSummary Table Saved")

print("results/tables/preprocessing_summary.csv")

print("\nPreprocessing Completed Successfully")