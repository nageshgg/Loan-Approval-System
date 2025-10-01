# 🏦 Loan Approval Prediction: A Machine Learning Approach

## 1. Project Overview

This project focuses on building a machine learning classifier to predict whether a loan application will be **Approved** or **Rejected**. The primary goal is to use historical applicant data, including financial assets, income, and CIBIL score, to create an automated system that can reliably screen applications.

The analysis involves extensive Exploratory Data Analysis (EDA) to understand feature importance (particularly **CIBIL Score**), followed by a comparison of four different classification algorithms to determine the optimal predictive model.

## 2. Dataset and Features

The dataset, sourced from Kaggle, contains information on various applicant attributes.

| Feature Name | Description | Type | 
 | ----- | ----- | ----- | 
| `no_of_dependents` | Number of dependents. | Numerical | 
| `education` | Applicant's education level (Encoded). | Categorical | 
| `self_employed` | Whether the applicant is self-employed (Encoded). | Categorical | 
| `income_annum` | Annual income of the applicant. | Numerical | 
| `loan_amount` | The loan amount requested. | Numerical | 
| `loan_term` | The term of the loan in years. | Numerical | 
| **`cibil_score`** | The credit score of the applicant (Crucial Feature). | Numerical | 
| `residential_assets_value` | Value of residential assets. | Numerical | 
| `commercial_assets_value` | Value of commercial assets. | Numerical | 
| `luxury_assets_value` | Value of luxury assets. | Numerical | 
| `bank_asset_value` | Value of bank assets. | Numerical | 
| **`loan_status`** | Target variable: Approved (0) or Rejected (1). | Categorical | 

**Data Quality:** The dataset was found to have **no missing (null) values**. Column names were cleaned by removing unnecessary leading spaces.

## 3. Exploratory Data Analysis (EDA) Highlights

The EDA revealed several strong factors driving loan decisions:

### A. CIBIL Score is the Dominant Factor

* A new feature, `CIBIL_rating`, was created to categorize scores (Poor, Fair, Good, Very Good, Excellent).

* The analysis showed a clear threshold: **almost all applications with a Poor CIBIL Score (300-579) were rejected**, regardless of income or asset value.

* Conversely, applicants with scores in the **Good, Very Good, or Excellent** ranges had a very high chance of approval.

### B. Income and Assets Relationship

* **Positive Correlation:** A strong positive correlation was observed between `income_annum`, `loan_amount`, and `luxury_assets_value`. As expected, higher income leads to higher requested loan amounts and higher luxury/bank assets.

* **Approval Variance:** Even among high-income applicants, both approved and rejected cases were present, suggesting that other factors (like the CIBIL score) override income in critical denial cases.

* **Asset Patterns:** For high CIBIL scores (above 579), the presence of **Residential and Commercial assets** appeared to be a differentiating factor between approved and rejected applications, although the correlation was less pronounced than the CIBIL score itself.

## 4. Modeling and Evaluation

### A. Preprocessing

1. **Feature Dropping:** The unique identifier `loan_id` and the derived feature `CIBIL_rating` were dropped from the feature set (`X`).

2. **Encoding:** Categorical variables (`education`, `self_employed`, `loan_status`) were converted to numerical using **Label Encoding**.

3. **Scaling:** Features for the Logistic Regression and Decision Tree models were scaled using `StandardScaler`.

4. **Splitting:** The dataset was split into training and testing sets with a `test_size` of **30%**.

### B. Algorithms Compared

Four robust classification models were trained and evaluated on the test set, using **Accuracy** as the primary metric.

1. **Logistic Regression (LR):** A linear model providing a baseline.

2. **Random Forest Classifier (RF):** An ensemble tree model known for high accuracy and robustness.

3. **Decision Tree Classifier (DT):** A simple, interpretable tree model (used the scaled data incorrectly, resulting in low performance).

4. **XGBoost Classifier (XGBoost):** A highly optimized gradient boosting framework.

## 5. Results and Conclusion

The ensemble and boosting methods significantly outperformed the simpler models, demonstrating the need for non-linear decision boundaries to capture the complex relationship between CIBIL score, income, and assets.

| Algorithm | Accuracy | F1 Score | 
 | ----- | ----- | ----- | 
| **XGBoost Classifier** | **0.9813** | **0.9813** | 
| Random Forest Classifier | 0.9797 | 0.9797 | 
| Logistic Regression | 0.9126 | 0.9124 | 
| Decision Tree Classifier | 0.6058 | 0.4571 | 

### Best Model

The **XGBoost Classifier** achieved the highest accuracy of **98.13%**, making it the recommended model for the automated loan approval system. Its ability to correctly classify both approved and rejected cases, as shown by the confusion matrix, is excellent.

## 6. Dependencies

To run this analysis, the following Python libraries are required:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
