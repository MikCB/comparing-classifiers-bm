# Comparing Classifiers for Bank Marketing

This project compares Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines for predicting whether a bank client will subscribe to a term deposit.

## Business question

How can the bank identify promising clients before outreach, reduce unnecessary calls, and retain a meaningful share of successful subscriptions?

## Data

The analysis uses the UCI Bank Marketing `bank-additional-full.csv` dataset supplied with the assignment. It contains 41,188 records and 20 input features before cleaning. Twelve duplicate rows are removed. The target is imbalanced: approximately 11.3% of clients subscribe.

Call `duration` is excluded because it is unknown before a call and would leak post-contact information into a pre-call targeting model.

## Methods

- Data-quality and exploratory analysis
- Descriptive and inferential statistics
- Reproducible stratified train/test split
- One-hot encoding and numeric standardization inside scikit-learn pipelines
- Majority-class baseline
- Default comparison of four required classifiers
- Stratified three-fold cross-validation
- Grid-search hyperparameter tuning
- Logistic Regression coefficient interpretation
- Probability-threshold analysis using precision, recall, F1, and ROC-AUC

## Main findings

- Logistic Regression has the strongest test ROC-AUC at approximately 0.802 and is the most interpretable model.
- Accuracy alone is misleading because predicting no subscription for every client is already about 88.7% accurate.
- The default Logistic Regression threshold identifies only about 21% of subscribers.
- Lowering its threshold to approximately 0.216 identifies about 55% of subscribers while flagging roughly 14% of test clients for contact.
- A class-balanced SVM reaches approximately 63% recall, but with lower precision and a larger call list.
- Prior campaign success is the clearest observed positive segment signal. Student and retired clients also show higher observed subscription rates, while telephone contact and May contacts are negative predictive signals.

## Recommendation

Use Logistic Regression to rank clients, then select the probability threshold according to weekly call-center capacity. A threshold near 0.216 is a reasonable starting point when the goal is to create a compact call list while capturing substantially more subscribers than the default threshold. Validate the rule on a later campaign before deployment.

## Repository contents

- [`comparing_classifiers.ipynb`](comparing_classifiers.ipynb): complete analysis, visualizations, modeling, findings, and recommendations
- `data/bank-additional-full.csv`: assignment dataset
- `requirements.txt`: Python dependencies used to reproduce the notebook

## Run locally

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open `comparing_classifiers.ipynb` from the repository root.
4. Restart the kernel and run all cells.

The notebook uses relative paths and a fixed random seed for reproducibility.
