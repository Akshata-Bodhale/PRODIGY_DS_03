"""
PRODIGY_DS_03 - Data Science Internship @ Prodigy InfoTech
Task 03: Decision tree classifier to predict whether a customer will subscribe
(purchase) a term deposit, using the UCI Bank Marketing dataset.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)

sns.set_theme(style="whitegrid")

# ======================= 1. LOAD =======================
df = pd.read_csv("bank-full.csv", sep=";")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("\nTarget distribution:\n", df["y"].value_counts(normalize=True).round(3))

# ======================= 2. QUICK EDA =======================
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
sns.countplot(x="y", data=df, ax=ax[0], hue="y", palette="Set2", legend=False)
ax[0].set_title("Term Deposit Subscription")
job = df.groupby("job")["y"].apply(lambda s: (s == "yes").mean()).sort_values()
job.plot(kind="barh", ax=ax[1], color="#4C72B0")
ax[1].set_title("Subscription Rate by Job"); ax[1].set_xlabel("Rate")
sns.histplot(data=df, x="age", hue="y", bins=30, ax=ax[2], multiple="stack")
ax[2].set_title("Age vs Subscription")
plt.tight_layout(); plt.savefig("01_eda.png", dpi=150); plt.close()

# ======================= 3. PREPROCESSING =======================
# NOTE: 'duration' (last call length) is only known AFTER the call, so it is
# dropped for a realistic model, as recommended in the UCI documentation.
data = df.drop(columns=["duration"])
data["y"] = data["y"].map({"no": 0, "yes": 1})
X = pd.get_dummies(data.drop(columns="y"), drop_first=True)
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ======================= 4. MODEL =======================
# Tune depth / leaf size with cross-validation to avoid over-fitting
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    param_grid={"max_depth": [3, 4, 5, 6, 8, 10],
                "min_samples_leaf": [20, 50, 100],
                "criterion": ["gini", "entropy"]},
    cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("\nBest parameters:", grid.best_params_)

# ======================= 5. EVALUATION =======================
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(f"\nAccuracy : {accuracy_score(y_test, pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, proba):.4f}")
print("\nClassification report:\n",
      classification_report(y_test, pred, target_names=["No", "Yes"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["No", "Yes"]).plot(cmap="Blues")
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig("02_confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
plt.savefig("03_roc_curve.png", dpi=150); plt.close()

# ======================= 6. INTERPRETATION =======================
imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
plt.figure(figsize=(9, 6))
imp.plot(kind="barh", color="#55A868")
plt.title("Top 10 Feature Importances"); plt.tight_layout()
plt.savefig("04_feature_importance.png", dpi=150); plt.close()
print("Top features:\n", imp.sort_values(ascending=False).round(3))

plt.figure(figsize=(22, 10))
plot_tree(model, max_depth=3, feature_names=X.columns, class_names=["No", "Yes"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (top 3 levels)"); plt.tight_layout()
plt.savefig("05_decision_tree.png", dpi=120); plt.close()

with open("tree_rules.txt", "w") as f:
    f.write(export_text(model, feature_names=list(X.columns), max_depth=4))

print("\nDone. Figures and tree_rules.txt saved.")
