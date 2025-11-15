# -----------------------------------------
# TASK 1: DATA LOADING AND INTEGRATION
# Clean + Merge + Feature Engineering
# -----------------------------------------

import pandas as pd
import numpy as np

# -----------------------------------------
# 1. LOAD DATASETS
# -----------------------------------------

df_primary = pd.read_csv("load.zoo.csv")
df_class   = pd.read_csv("load.class.csv")
df_aux     = pd.read_json("load.auxiliary_metadata.json")

print("Primary:", df_primary.shape)
print("Class:  ", df_class.shape)
print("Aux:    ", df_aux.shape)


# -----------------------------------------
# 2. NAME NORMALIZATION
# -----------------------------------------

def normalize_names(df):
    # rename any animal name column
    for c in df.columns:
        if c.lower().strip() in ["animal", "animal_name", "name"]:
            df = df.rename(columns={c: "animal_name"})
    df["animal_name"] = df["animal_name"].astype(str).str.upper().str.strip()
    return df

df_primary = normalize_names(df_primary)
df_class   = normalize_names(df_class)
df_aux     = normalize_names(df_aux)


# -----------------------------------------
# 3. STANDARDIZE FIELD NAMES + CLEAN TEXT
# -----------------------------------------

def standardize_fields(df):
    rename_map = {
        "conservation": "conservation_status",
        "status": "conservation_status",
        "habitat": "habitat_type",
        "habitats": "habitat_type",
        "diet_type": "diet",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # fix text
    if "conservation_status" in df.columns:
        df["conservation_status"] = (
            df["conservation_status"].astype(str).str.lower().str.strip()
        )
        df["conservation_status"] = df["conservation_status"].replace({
            "least": "least concern"
        })

    if "diet" in df.columns:
        df["diet"] = df["diet"].astype(str).str.lower().str.strip().replace({
            "omnivor": "omnivore",
            "herbivor": "herbivore"
        })

    if "habitat_type" in df.columns:
        df["habitat_type"] = df["habitat_type"].astype(str).str.lower().str.strip()
        df["habitat_type"] = df["habitat_type"].replace({
            "fresh water": "freshwater",
            "fresh water ": "freshwater",
            "freshwater ": "freshwater"
        })

    return df

df_primary = standardize_fields(df_primary)
df_class   = standardize_fields(df_class)
df_aux     = standardize_fields(df_aux)


# -----------------------------------------
# 4. MERGE ALL DATASETS ON animal_name
# -----------------------------------------

merged = df_primary.merge(df_class, on="animal_name", how="left", suffixes=("", "_class"))
merged = merged.merge(df_aux, on="animal_name", how="left", suffixes=("", "_aux"))

# Consolidate fields (primary → class → aux)
def consolidate(col):
    for c in [col, col + "_class", col + "_aux"]:
        if c in merged.columns:
            return merged[c]
    return np.nan

merged["conservation_status"] = consolidate("conservation_status")
merged["habitat_type"]        = consolidate("habitat_type")
merged["diet"]                = consolidate("diet")


# -----------------------------------------
# 5. HANDLE MISSING VALUES
# -----------------------------------------

# Categorical → "unknown"
merged["conservation_status"] = merged["conservation_status"].fillna("unknown")
merged["diet"] = merged["diet"].fillna("unknown")
merged["habitat_type"] = merged["habitat_type"].fillna("unknown")

# Numerical → median
for col in merged.select_dtypes(include=[np.number]).columns:
    merged[col] = merged[col].fillna(merged[col].median())


# -----------------------------------------
# 6. FEATURE ENGINEERING
# -----------------------------------------

priority_map = {
    "endangered": 5,
    "vulnerable": 4,
    "near threatened": 3,
    "least concern": 1,
    "unknown": 0
}

merged["conservation_priority"] = (
    merged["conservation_status"]
    .str.lower()
    .map(priority_map)
    .fillna(0)
    .astype(int)
)

merged["aquatic_flag"] = merged["habitat_type"].str.contains("water|marine", case=False).astype(int)


# -----------------------------------------
# 7. FINAL OUTPUT
# -----------------------------------------

print("\nFinal dataset shape:", merged.shape)
merged.head()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

col4 = merged.columns[3]

# A1 Bar plot with error bars (count + standard deviation)
plt.figure(figsize=(8,5))
counts = merged[col4].value_counts()
errors = np.sqrt(counts)
plt.bar(counts.index, counts.values, yerr=errors)
plt.xticks(rotation=45)
plt.title(f"Class Counts with Error Bars ({col4})")
plt.show()

# A2 Hexbin plot of first two numeric features
num_cols = merged.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(7,5))
plt.hexbin(merged[num_cols[0]], merged[num_cols[1]], gridsize=20)
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])
plt.title("Hexbin Plot of Two Numeric Features")
plt.show()

# A3 Swarmplot of conservation_status across column 4
plt.figure(figsize=(10,5))
sns.swarmplot(x=col4, y="conservation_priority", data=merged)
plt.xticks(rotation=45)
plt.title("Conservation Status Across Classes")
plt.show()

# A4 Clustermap of numerical features
sns.clustermap(merged[num_cols].corr(), annot=True, cmap="viridis")
plt.title("Clustermap of Numerical Features")
plt.show()

# B1 Class imbalance ratio
class_sizes = merged[col4].value_counts()
imbalance_ratio = class_sizes.max() / class_sizes.min()
print("Class imbalance ratio:", imbalance_ratio)

# B2 Low variance features (<0.01)
variances = merged.var(numeric_only=True)
low_var_features = variances[variances < 0.01]
print("\nLow variance features:\n", low_var_features)

# B3 Highly correlated pairs (> 0.8)
corr_matrix = merged[num_cols].corr().abs()
high_corr_pairs = [(i, j, corr_matrix.loc[i, j])
                   for i in num_cols
                   for j in num_cols
                   if i != j and corr_matrix.loc[i, j] > 0.8]
print("\nHighly correlated pairs > 0.8:")
for p in high_corr_pairs:
    print(p)
