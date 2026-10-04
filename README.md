# PRODIGY_DS_03 – Decision Tree: Bank Marketing Prediction

**Prodigy InfoTech – Data Science Internship | Task 03**

## Objective
Build a decision tree classifier to predict whether a customer will subscribe to a term deposit, using demographic and behavioural data.

## Dataset
UCI Bank Marketing (`bank-full.csv`) – 45,211 rows, 16 features. Target `y` = subscribed (yes/no). Only 11.7% said yes (imbalanced).

## Approach
1. Load and explore the data
2. One-hot encode categorical features
3. **Dropped `duration`** – call length is only known after the call, so using it would leak the answer
4. 80/20 stratified train-test split
5. `GridSearchCV` (5-fold, ROC-AUC) to tune `max_depth`, `min_samples_leaf`, `criterion`
6. `class_weight="balanced"` to handle the imbalance

## Results (test set)
| Metric | Value |
|---|---|
| Accuracy | 81.6% |
| ROC AUC | 0.777 |
| Recall (Yes) | 0.59 |
| Precision (Yes) | 0.34 |

Best parameters: entropy, max_depth = 10, min_samples_leaf = 50.

**Top predictors:** previous campaign outcome = success, contact type unknown, day of month, housing loan, balance, month (Jul/Aug), pdays, campaign count, age.

## Run
```bash
pip install -r requirements.txt
python PRODIGY_DS_03.py
```

## Output
`01_eda.png`, `02_confusion_matrix.png`, `03_roc_curve.png`, `04_feature_importance.png`, `05_decision_tree.png`, `tree_rules.txt`
