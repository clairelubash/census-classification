from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from model_pipeline import (
    evaluate_model, 
    load_data, 
    get_scale_pos_weight,
    save_shap_plot
)

# -------------------------------
# Setup paths
# -------------------------------
OUTPUT_DIR = Path('outputs')
DATA_DIR = Path('data')
train_suffix = 'train_clean.csv'
test_suffix = 'test_clean.csv'

# -------------------------------
# Load Data
# -------------------------------
X_train, y_train, X_test, y_test = load_data(DATA_DIR, train_suffix, test_suffix)

# -------------------------------
# Hyperparameter Tuning for XGBoost
# -------------------------------
xgb = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=get_scale_pos_weight(y_train),
    random_state=42,
    n_jobs=1
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print('\nStarting hyperparameter tuning for XGBoost...')
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
print(f'Best parameters: {search.best_params_}')

# -------------------------------
# Initial Evaluation
# -------------------------------
# y_pred = best_xgb.predict(X_test)
# y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
# evaluate_model('XGBoost_Tuned', y_test, y_pred, y_pred_proba, OUTPUT_DIR / 'xgb_tuned')

# -------------------------------
# SHAP Feature Selection
# -------------------------------
print('\nComputing SHAP values for feature selection...')
explainer = shap.Explainer(best_xgb, X_train)
shap_values = explainer(X_train)

# Calculate mean absolute SHAP values
shap_importances = np.abs(shap_values.values).mean(axis=0)
feature_importances = pd.Series(shap_importances, index=X_train.columns)
selected_features = feature_importances[feature_importances > 0.01].index 

# -------------------------------
# Retrain XGBoost on Selected Features
# -------------------------------
print('\nRetraining XGBoost with selected features...')
best_xgb.fit(X_train[selected_features], y_train)

y_pred_sel = best_xgb.predict(X_test[selected_features])
y_pred_proba_sel = best_xgb.predict_proba(X_test[selected_features])[:, 1]

evaluate_model('XGBoost_SelectedFeatures', y_test, y_pred_sel, y_pred_proba_sel, OUTPUT_DIR / 'xgb_selected')

# SHAP summary plot for final model
save_shap_plot(explainer(X_test[selected_features]), X_test[selected_features], X_train, 'xgboost', 'xgb_selected_shap_summary.png', 'summary')

# # SHAP summary plot for final model
# plt.figure()
# shap.summary_plot(explainer(X_test[selected_features]), X_test[selected_features], show=False)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / 'xgb_selected_shap_summary.png')
# plt.close()
