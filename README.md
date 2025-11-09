# Customer Churn Prediction - Model Development, Validation, and Deployment

This project develops a comprehensive customer churn prediction system for the telecommunications industry. It implements multiple machine learning models, validates their performance, and provides a complete deployment framework.

##  Objectives

- Develop accurate churn prediction models using statistical and machine learning techniques
- Extract interpretable decision rules using CHAID algorithm
- Compare multiple models using comprehensive evaluation metrics
- Design a production-ready deployment framework
- Implement model updating and monitoring strategies

##  Dataset

**Source:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

**Description:** Contains customer data from a telecommunications company including:
- Customer demographics (gender, senior citizen status, partners, dependents)
- Service subscriptions (phone, internet, streaming services)
- Account information (tenure, contract type, payment method)
- Billing information (monthly charges, total charges)
- Target variable: Churn (Yes/No)

**Dataset Statistics:**
- Total Records: 7,043 customers
- Features: 20 predictor variables
- Target Distribution: ~27% churn rate
- Missing Values: Minimal (handled during preprocessing)

##  Repository Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── cleaned_churn_data.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_chaid_modeling.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_deployment_setup.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── deployment.py
│
├── models/
│   ├── churn_model.pkl
│   ├── churn_model.joblib
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── deployment_metadata.pkl
│
├── visualizations/
│   ├── eda_visualizations.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── lift_charts.png
│
├── reports/
│   └── Customer_Churn_Prediction_Report.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/[Yashnaidu17]/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Libraries

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.3.1
jupyter==1.0.0
CHAID==5.4.1
imbalanced-learn==0.10.1
```

##  Usage

### 1. Data Preprocessing

```python
from src.data_preprocessing import preprocess_data

# Load and clean data
df_clean = preprocess_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

### 2. Train Models

```python
from src.model_training import train_chaid_model, train_logistic_regression

# Train CHAID model
chaid_model = train_chaid_model(X_train, y_train)

# Train Logistic Regression
lr_model = train_logistic_regression(X_train, y_train)
```

### 3. Evaluate Models

```python
from src.model_evaluation import evaluate_models

# Compare models
results = evaluate_models(models_dict, X_test, y_test)
```

### 4. Make Predictions

```python
import pickle

# Load trained model
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict churn
customer_data = {...}  # Customer features
prediction = model.predict(customer_data)
probability = model.predict_proba(customer_data)[:, 1]
```

## 📈 Methodology

### 1. Data Preparation
- **Data Cleaning:** Handled missing values, removed duplicates, treated outliers
- **Feature Engineering:** Encoded categorical variables, standardized numerical features
- **EDA:** Comprehensive visualization of patterns and relationships
- **Target Definition:** Binary classification (Churn: Yes/No)

### 2. CHAID Model Development 
- **Algorithm:** Chi-squared Automatic Interaction Detection
- **Implementation:** Decision tree with chi-square splitting criterion
- **Rule Extraction:** Interpretable decision rules for business insights
- **Key Findings:**
  - Contract type is the most influential factor
  - Tenure strongly correlates with churn probability
  - Monthly charges impact retention significantly

### 3. Model Comparison 

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Decision Tree (CHAID) | 78.5% | 0.82 | 0.76 | 0.72 |
| Logistic Regression | 80.2% | 0.85 | 0.79 | 0.75 |

**Evaluation Metrics:**
- ✅ Accuracy Score
- ✅ ROC-AUC Score
- ✅ Lift Charts
- ✅ Gains Charts
- ✅ Confusion Matrix
- ✅ Precision-Recall Curves

### 4. Model Deployment 

**Deployment Framework:**
- Serialization using Pickle and Joblib
- REST API development (Flask/FastAPI)
- Docker containerization
- Cloud deployment (AWS/Azure/GCP)

**Model Updating Strategy:**
- Scheduled retraining (monthly)
- Performance monitoring
- Data drift detection
- A/B testing framework
- Version control with MLflow

##  Key Results

### Business Insights

1. **High-Risk Customer Profile:**
   - Month-to-month contract customers
   - High monthly charges (>$70)
   - Low tenure (<12 months)
   - No tech support or online security

2. **Retention Strategies:**
   - Promote longer-term contracts
   - Offer bundled services with discounts
   - Provide proactive customer support
   - Implement loyalty programs for long-tenure customers

3. **Cost-Benefit Analysis:**
   - Model identifies 75% of churners in top 3 deciles
   - Potential cost savings: 30-40% reduction in churn
   - ROI: 3:1 on retention campaigns

### Feature Importance Rankings

1. Contract Type (23.5%)
2. Tenure (18.2%)
3. Monthly Charges (15.8%)
4. Total Charges (12.3%)
5. Internet Service Type (10.7%)
6. Tech Support (7.4%)
7. Online Security (6.1%)
8. Payment Method (3.8%)
9. Paperless Billing (2.2%)

##  Model Updating Framework

### Automated Retraining Pipeline

```python
def retrain_pipeline():
    # 1. Load new data
    new_data = load_new_customer_data()
    
    # 2. Data validation
    validate_data_quality(new_data)
    
    # 3. Monitor data drift
    drift_detected = check_data_drift(new_data)
    
    # 4. Retrain if necessary
    if drift_detected or scheduled_retrain:
        new_model = train_model(new_data)
        
        # 5. Evaluate new model
        if new_model.performance > threshold:
            deploy_model(new_model)
            
    # 6. Log results
    log_pipeline_execution()
```

### Monitoring Metrics
- Prediction accuracy over time
- Feature distribution changes
- False positive/negative rates
- Model latency and throughput


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request
