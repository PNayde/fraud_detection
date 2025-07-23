# Credit Card Fraud Detection – End-to-End Date Science Project

![Fraud Detection Banner](https://img.shields.io/badge/machine%20learning-fraud%20detection-blue) 

## Overview

This project demonstrates an end-to-end machine learning workflow for credit card fraud detection, using a real-world anonymized dataset of European card transactions. The solution prioritizes catching fraudulent activity (high recall) while minimizing disruption to genuine customers (high precision), aligning technical modeling with practical business needs.

---

## Project Goals

- **Explore and visualize transaction data**
- **Address severe class imbalance with robust metrics and sampling**
- **Compare multiple supervised and unsupervised models**
- **Optimize thresholds to maximize business value**
- **Interpret results for operational and executive audiences**
- **Follow reproducible, best-practice data science workflow**

---

## Data

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Shape:** 284,807 transactions, 492 frauds (0.172%)
- **Features:** Principal components (V1–V28) from PCA, plus `Amount`, `Time`, and `Class` (fraud label)

---

## Approach

1. **Exploratory Data Analysis**
   - Visualized class imbalance and transaction distributions
2. **Preprocessing**
   - Scaling, PCA, and careful prevention of data leakage
3. **Modeling**
   - Supervised: Logistic Regression, Random Forest, LightGBM (with cross-validation)
   - Unsupervised: Isolation Forest, Local Outlier Factor (LOF)
4. **Threshold Optimization**
   - Tuned classification thresholds for best business balance (F1-score for fraud)
5. **Evaluation & Interpretation**
   - Confusion matrices, ROC-AUC, F1/precision/recall, feature importance
6. **Robust Validation**
   - Stratified K-fold cross-validation and a final holdout test set

---

## Model Comparison

| Model               | F1-score (Fraud) | Precision (Fraud) | Recall (Fraud) | ROC-AUC | Notes                                            |
|---------------------|:----------------:|:-----------------:|:--------------:|:-------:|:------------------------------------------------|
| Logistic Regression |      0.70*       |      0.60*        |     0.82*      |  0.98   | *Threshold tuned to maximize F1 (0.99)*         |
| Random Forest       |      0.83        |      0.97         |     0.73       |  0.94   | High precision, low false positive rate         |
| LightGBM            |      0.85        |      0.87         |     0.84       |  0.98   | Best overall fraud detection performance        |
| Isolation Forest    |      0.28        |      0.28         |     0.27       |    —    | Unsupervised; detects limited fraud             |
| LOF                 |      0.00        |      0.00         |     0.00       |    —    | Unsupervised; ineffective on this dataset       |

\*Logistic Regression metrics shown for optimal threshold, not the default 0.5.

---

## Business Impact

- **Fraud Detection:**  
  LightGBM and Random Forest models deliver industry-standard detection rates with low false positives, making them suitable for deployment.
- **Threshold Tuning:**  
  Improved actionable alert rates from 6% to 60% without sacrificing recall.
- **Transparency:**  
  Model interpretability supports compliance and executive confidence.
- **Reproducibility:**  
  All steps use best-practice validation to ensure reliable, real-world-ready results.

---

## How to Use This Notebook

1. **Clone the repo and install requirements**
2. **Download the dataset** (Kaggle link above) to the project folder
3. **Run the notebook step-by-step**, following code and markdown explanations
4. **Modify threshold tuning and business metrics as needed** for your use case

---

## Next Steps

- Deploy as a batch or streaming prediction service
- Integrate new features (merchant, behavioral, device info)
- Experiment with ensemble and deep learning methods
- Monitor and recalibrate with new data regularly

---

## Author

Dr. Plamena Naydenova 
Data Scientist, Phd | www.linkedin.com/in/plamena-naydenova

---

## License

This project is for educational and portfolio use.  
Original data license: [Kaggle terms of use](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

