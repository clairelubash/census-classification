# Income Prediction Project

## 📌 Project Overview
This project predicts whether an individual earns more or less than **$50,000 per year** using census data.  

The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling with machine learning.

---

## 📂 Project Structure
```bash
.
├── data/                # Raw and processed datasets
├── outputs/             # Model outputs, plots, and artifacts
├── notebooks/           # Jupyter notebooks (e.g., EDA)
│   └── eda.ipynb
├── src/                 # Source code
│   ├── data_prep.py
│   ├── model_pipeline.py
│   └── xgboost_retrained.py
├── requirements.txt
├── README.md
```


## ⚙️ Setup & Installation
1. Clone the repository
```bash
git clone https://github.com/clairelubash/census-classification.git
cd census-classification
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```


## ▶️ Usage
1. Data preprocessing
Run preprocessing to clean and prepare the dataset for modeling:
```bash
python src/data_preprocessing.py
```
2. Baseline Model Training
Train classification models on processed data:
```bash
python src/model_pipeline.py
```
3. Model Tuning & Retraining
Hyperparameter tuning and SHAP feature selection for XGBoost:
```bash
python src/xgboost_retrained.py
```
Outputs (confusion matrices, SHAP plots, etc.) will be saved in the `outputs/` folder.


## 📊 Exploratory Data Analysis (EDA)
Data analysis is performed and visualizations are created in a Jupyter notebook: 
`notebooks/eda.ipynb`


## 📈 Models Implemented
- Logistic Regression
- XGBoost

Evaluation metrics include:
- **ROC-AUC** as the primary metric due to class imblanace
- Accuracy
- Precision, Recall, F1-Score
- SHAP for interpretability


## ✅ Key Findings
- **Age** is the strongest predictor of income.
- **Tax filing status** and ** eeks worked per year** are also highly influential.
- **Gender**, **Occupation**, **Industry**, and **Education** have smaller but non-negligible impacts.


## 🛠️ Tech Stack

- Python 3.10+
- pandas, numpy, scikit-learn
- XGBoost
- matplotlib, seaborn, SHAP
- Jupyter Notebook





