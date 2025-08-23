from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# -------------------------------
# Setup paths
# -------------------------------
OUTPUT_DIR = Path('outputs')
DATA_DIR = Path('data')

# -------------------------------
# Load Data
# -------------------------------
train = pd.read_csv(DATA_DIR / 'train_clean.csv')
test = pd.read_csv(DATA_DIR / 'test_clean.csv')

X_train = train.drop(columns=['label', 'instance_weight'])
y_train = train['label']
X_test = test.drop(columns=['label', 'instance_weight'])
y_test = test['label']

# Class imbalance for XGBoost
num_majority = sum(y_train == 0)
num_minority = sum(y_train == 1)
scale_pos_weight = num_majority / num_minority

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(model_name, y_true, y_pred, y_pred_proba, save_path_prefix=None):
    print(f'\n--- {model_name} ---')
    print('Confusion Matrix:')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print('\nClassification Report:')
    cr = classification_report(y_true, y_pred, digits=4)
    print(cr)
    
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f'ROC-AUC Score: {roc_auc:.4f}')
    
    # Save report
    if save_path_prefix:
        with open(f'{save_path_prefix}_report.txt', 'w') as f:
            f.write(f'Confusion Matrix:\n{cm}\n\n')
            f.write(f'Classification Report:\n{cr}\n')
            f.write(f'ROC-AUC Score: {roc_auc:.4f}\n')

# -------------------------------
# Logistic Regression with L1
# -------------------------------
print('\nTraining Logistic Regression with L1 Regularization...')

logreg = LogisticRegression(
    penalty='l1',
    solver='liblinear', 
    class_weight='balanced',
    C=1.0,
    random_state=42
)

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', logreg)
])

pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]

evaluate_model(
    'LogisticRegression_L1', 
    y_test, y_pred_lr, y_pred_proba_lr,
    save_path_prefix=OUTPUT_DIR / 'logreg_l1'
)

# -------------------------------
# Feature Importance via SHAP
# -------------------------------
print('\nGenerating SHAP values for Logistic Regression...')

# StandardScaler is inside the pipeline; transform features first
X_train_scaled = pipeline_lr.named_steps['scaler'].transform(X_train)
X_test_scaled = pipeline_lr.named_steps['scaler'].transform(X_test)

explainer = shap.LinearExplainer(pipeline_lr.named_steps['clf'], X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'logreg_l1_shap_summary.png')
plt.close()

# SHAP bar plot
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X_train.columns, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'logreg_l1_shap_bar.png')
plt.close()

# Selected features for logistic regression
logreg_coef = pipeline_lr.named_steps['clf'].coef_[0]
selected_features_lr = X_train.columns[logreg_coef != 0]
print("\nFeatures selected by Logistic Regression (non-zero coefficients):")
print(list(selected_features_lr))

# -------------------------------
# XGBoost Model
# -------------------------------
print('\nTraining XGBoost...')

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    random_state=42
)

pipeline_xgb = Pipeline([('clf', xgb)])
pipeline_xgb.fit(X_train, y_train)
y_pred_xgb = pipeline_xgb.predict(X_test)
y_pred_proba_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]

evaluate_model(
    'XGBoost', 
    y_test, y_pred_xgb, y_pred_proba_xgb,
    save_path_prefix=OUTPUT_DIR / "xgboost"
)

# SHAP for XGBoost
print('\nGenerating SHAP values for XGBoost...')
explainer_xgb = shap.Explainer(pipeline_xgb.named_steps['clf'], X_train)
shap_values_xgb = explainer_xgb(X_test)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values_xgb, X_test, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'xgboost_shap_summary.png')
plt.close()

plt.figure()
shap.summary_plot(shap_values_xgb, X_test, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'xgboost_shap_bar.png')
plt.close()

# Selected features for XGBoost
xgb_importances = pipeline_xgb.named_steps['clf'].feature_importances_
selected_features_xgb = X_train.columns[xgb_importances > 0]
print("\nFeatures selected by XGBoost (non-zero importance):")
print(list(selected_features_xgb))

print(f'\nAll reports and figures saved to {OUTPUT_DIR}/')
