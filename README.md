# Credit Card Fraud Detection – End-to-End Data Science Project
![CI](https://github.com/PNayde/fraud_detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-informational)
![API](https://img.shields.io/badge/API-FastAPI-informational)
![Container](https://img.shields.io/badge/Container-Docker-informational)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PNayde/fraud_detection/blob/main/fraud_detection_notebook.ipynb)
![Fraud Detection Banner](https://img.shields.io/badge/machine%20learning-fraud%20detection-blue)

## Overview
This project demonstrates an **end-to-end** machine-learning workflow for credit-card fraud detection using a real-world anonymised dataset of European transactions. The solution prioritises **catching fraud (recall)** while minimising disruption to genuine customers (**precision**), aligning technical modelling with practical business needs.

---

## Project Goals
- Explore and visualise transaction data  
- Address severe class imbalance with robust metrics and sampling  
- Compare multiple supervised and unsupervised models  
- Tune decision thresholds to maximise business value  
- Provide explainability and stakeholder-friendly reporting  
- Ship reproducible code with tests, API stub, Docker, and CI

---

## Data
- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Shape:** 284,807 transactions, 492 frauds (**0.172%**)  
- **Features:** PCA components `V1–V28` + `Amount`, `Time`, and label `Class` (fraud)

---

## Approach
1. **EDA** – class imbalance, feature distributions, leakage checks  
2. **Preprocessing** – scaling; care to avoid leakage across CV folds  
3. **Modelling**  
   - *Supervised:* Logistic Regression, Random Forest, **LightGBM** (with cross-validation)  
   - *Unsupervised:* Isolation Forest, LOF  
4. **Threshold optimisation** – tuned the probability cutoff to maximise **F1 (fraud)** and evaluated trade-offs  
5. **Evaluation & interpretation** – ROC-AUC, precision/recall/F1, confusion matrices, feature importance  
6. **Robust validation** – stratified K-fold CV and a final hold-out test set

---

## Model Comparison
| Model               | F1-score (Fraud) | Precision (Fraud) | Recall (Fraud) | ROC-AUC | Notes                                            |
|---------------------|:----------------:|:-----------------:|:--------------:|:-------:|:------------------------------------------------|
| Logistic Regression |      0.70*       |      0.60*        |     0.82*      |  0.98   | *Threshold tuned to maximize F1 (0.99)*         |
| Random Forest       |      0.87*       |      0.93*        |     0.81*      |  0.94   | *Threshold tuned to maximize F1 (0.26)*         |
| LightGBM            |      0.85        |      0.87         |     0.84       |  0.98   | Best overall fraud detection performance (default 0.5 cutoff) |
| Isolation Forest    |      0.28        |      0.28         |     0.27       |    —    | Unsupervised; detects limited fraud             |
| LOF                 |      0.00        |      0.00         |     0.00       |    —    | Unsupervised; ineffective on this dataset       |

\* Metrics shown at optimal threshold, not default 0.5.

### ✅ Recommended operating point (LightGBM)
| T (probability cutoff) | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC |
|:----------------------:|:-----------------:|:--------------:|:----------:|:------:|
| **0.50 (default)**     | **0.87**          | **0.84**       | **0.85**   | **0.98** |

### 🔎 Validation protocol (reproducibility)
- Split: stratified train/test (e.g., **80/20**), fixed seed (e.g., **42**)  
- CV/HPO: **Stratified K-fold** with consistent preprocessing per fold  
- Class imbalance: handled via weighting/thresholding  
- Thresholding: selected on validation to maximise **F1 (fraud)**, then **fixed** for test  
- Metrics reported on the **test** set only

---

## Business Impact
- **Detection with minimal friction:** LightGBM/Random Forest deliver strong recall with low false positives → fewer customer disruptions  
- **Threshold tuning:** increased actionable alert rate from **6% → 60%** without sacrificing recall  
- **Transparency:** interpretable outputs support compliance and executive confidence  
- **Reproducibility:** validated workflow, pinned env, tests, CI, and container to ease deployment

---

## Run locally (API + tests)
> The API serves two endpoints: `GET /health` and `POST /predict` (rows of features → predictions).  
> If `models/pipeline.joblib` exists, it’s loaded automatically; otherwise a tiny fallback model is used so the API always responds.

~~~bash
# 1) Install
pip install -r requirements.txt

# 2) Serve API
uvicorn api.main:app --reload --port 8000

# 3) Quick check
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows":[{"a":-1.0,"b":0.2},{"a":1.5,"b":0.3}]}'

# 4) Tests (used in CI)
pytest -q
~~~

## Docker
~~~bash
docker build -t fraud-api:latest .
docker run -p 8000:8000 -e MODEL_PATH=models/pipeline.joblib fraud-api:latest
~~~

---

## Notebook usage
1. Clone the repo & install requirements  
2. Download the dataset (Kaggle link above) into the project folder  
3. Run `fraud_detection_notebook.ipynb` step-by-step  
4. Adjust threshold tuning and business metrics for your use case

---

## Repository layout
~~~text
api/
  └── main.py           # FastAPI app (/health, /predict)
tests/
  └── test_api.py       # minimal tests for CI
.github/workflows/
  └── ci.yml            # GitHub Actions workflow (pytest)
Dockerfile
requirements.txt
README.md
fraud_detection_notebook.ipynb
~~~

---

## Next steps
- Promote to a batch/streaming prediction service (feature store, model registry)  
- Add domain features (merchant, device, behavioural)  
- Try ensembles / calibrated probabilities  
- Monitor drift and recalibrate thresholds periodically

---

## Author
**Dr. Plamena Naydenova** — Data Scientist, PhD  
[linkedin.com/in/plamena-naydenova](https://www.linkedin.com/in/plamena-naydenova)

---

## License
This project is for educational and portfolio use.  
Original data license: [Kaggle terms of use](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
