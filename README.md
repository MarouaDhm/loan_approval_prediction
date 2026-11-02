# Loan Approval Prediction

Predicting whether a loan application will be approved, with a focus on **precision, recall and F1** on imbalanced data.

Part of the Elevvo Machine Learning Internship (Task 4).

## Dataset
Loan Prediction training data: 614 applications, 12 features, target `Loan_Status` (68.7% approved, 31.3% rejected). Seven columns contain missing values.

## Approach
1. Dropped the identifier `Loan_ID` and encoded the target (Y = 1, N = 0).
2. **Stratified 80/20 split before any imputation**, so that fill values are learned from the training set only (no data leakage).
3. One scikit-learn `Pipeline` with a `ColumnTransformer`:
   - numeric columns: median imputation and scaling;
   - categorical columns (including `Credit_History`): most-frequent imputation and one-hot encoding.
4. Models: Logistic Regression (baseline), Logistic Regression with `class_weight="balanced"`, Logistic Regression with **SMOTE** (applied inside the training folds only), and a Decision Tree.
5. Evaluation on the test split plus 5-fold stratified cross-validation (macro-F1).

## Results (test set: 123 rows, 38 rejected loans)
| Model | Rejected loans approved by mistake | Good loans rejected | Recall (rejected) | Recall (approved) | Accuracy | CV macro-F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 16 | 1 | 0.58 | 0.99 | 0.86 | 0.725 ± 0.034 |
| Logistic Regression, balanced | 11 | 10 | 0.71 | 0.88 | 0.83 | 0.690 ± 0.034 |
| Logistic Regression + SMOTE | 12 | 8 | 0.68 | 0.91 | 0.84 | 0.701 ± 0.036 |
| Decision Tree (depth 4, balanced) | 16 | 8 | 0.58 | 0.91 | 0.80 | 0.673 ± 0.040 |

## Key findings
- The plain model looks strong (accuracy 0.86) but misses 42% of rejected loans, because it favors the majority class.
- Class weighting and SMOTE catch more risky applications (16 down to 11-12 wrongly approved) at the price of rejecting more good applicants (1 up to 8-10). The right choice depends on the relative cost of the two errors.
- Logistic Regression beat the Decision Tree on every cross-validated and test metric.
- Differences between the three Logistic Regression variants are within one standard deviation of the CV score, so none is clearly best.

## Limitations
Small dataset (614 rows) and only 38 rejected loans in the test split, so single-split numbers are noisy; the cross-validated scores are more reliable. The default 0.5 decision threshold was used; tuning it is a possible next step.

## Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, imbalanced-learn.

## Files
- `loan_approval_prediction.ipynb`: full analysis (code and outputs)
