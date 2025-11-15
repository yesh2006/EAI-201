from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# A — Prepare Data & Split
X = merged.drop(columns=["conservation_status"])
y = merged["conservation_status"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, test_size=0.30, random_state=42
)

# B — Configure & Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)

# C — Training Performance
train_score = rf.score(X_train, y_train)
print("Training Accuracy:", train_score)

# D — Classification Report
y_pred = rf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# E — Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix (100 trees, depth=10)")
plt.show()

# F — Feature Importance Plot (top 12)
importances = rf.feature_importances_
feature_names = X.columns
fi = pd.DataFrame({"feature": feature_names, "importance": importances})
fi = fi.sort_values(by="importance", ascending=False).head(12)

colors = ["red" if f in ["conservation_priority","aquatic_flag"] else "blue" 
          for f in fi["feature"]]

plt.figure(figsize=(8,6))
plt.barh(fi["feature"], fi["importance"], color=colors)
plt.gca().invert_yaxis()
plt.title("Top 12 Feature Importances")
plt.show()

# G — K-Nearest Neighbors Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print("\nKNN Test Accuracy:", knn_acc)

# H — Critical Analysis Output
print("\n--- Critical Analysis ---")
print("Random Forest Training Accuracy:", train_score)
print("Random Forest Test Accuracy:", rf.score(X_test, y_test))
print("KNN Test Accuracy:", knn_acc)

if rf.score(X_test, y_test) > knn_acc:
    print("Random Forest performed better due to handling nonlinear patterns and mixed features.")
else:
    print("KNN performed better, likely due to simpler decision boundaries in the dataset.")
