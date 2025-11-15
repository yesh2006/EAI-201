import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ensure column 4 exists
if len(merged.columns) < 4:
    raise ValueError("Dataset does not have at least 4 columns")

col4 = merged.columns[3]

# ---------------------------------------------------------
# SAFE: convert col4 to string (avoids seaborn category errors)
merged[col4] = merged[col4].astype(str)

# ---------------------------------------------------------
# Numeric column detection
num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols) < 2:
    raise ValueError("Not enough numeric columns for hexbin plot")

# ---------------------------------------------------------
# A1 Bar plot
plt.figure(figsize=(8,5))
counts = merged[col4].value_counts()
errors = np.sqrt(counts)
plt.bar(counts.index, counts.values, yerr=errors)
plt.xticks(rotation=45)
plt.title(f"Class Counts with Error Bars ({col4})")
plt.show()

# ---------------------------------------------------------
# A2 Hexbin plot (first two numeric features)
plt.figure(figsize=(7,5))
plt.hexbin(merged[num_cols[0]], merged[num_cols[1]], gridsize=20)
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])
plt.title("Hexbin Plot of Two Numeric Features")
plt.show()

# ---------------------------------------------------------
# A3 Swarmplot
plt.figure(figsize=(10,5))
sns.swarmplot(x=col4, y="conservation_priority", data=merged)
plt.xticks(rotation=45)
plt.title("Conservation Status Across Column 4")
plt.show()

# ---------------------------------------------------------
# A4 Clustermap (fill NA to avoid errors)
numeric_data = merged[num_cols].fillna(0)

sns.clustermap(numeric_data.corr(), annot=True, cmap="viridis")
plt.show()

# ---------------------------------------------------------
# B1 Class imbalance ratio
class_sizes = merged[col4].value_counts()
imbalance_ratio = class_sizes.max() / max(1, class_sizes.min())
print("Class imbalance ratio:", imbalance_ratio)

# ---------------------------------------------------------
# B2 Low variance features
variances = merged[num_cols].var()
low_var_features = variances[variances < 0.01]
print("\nLow variance features (<0.01):\n", low_var_features)

# ---------------------------------------------------------
# B3 Highly correlated pairs (>0.8)
corr_matrix = numeric_data.corr().abs()

high_corr_pairs = []
for i in num_cols:
    for j in num_cols:
        if i < j and corr_matrix.loc[i, j] > 0.8:
            high_corr_pairs.append((i, j, corr_matrix.loc[i, j]))

print("\nHighly correlated pairs (>0.8):")
for p in high_corr_pairs:
    print(p)
