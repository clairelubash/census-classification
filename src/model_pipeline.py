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
train_suffix = 'train_clean.csv'
test_suffix = 'test_clean.csv'


# -------------------------------
# Load Data
# -------------------------------
def load_data(DATA_DIR, train_suffix: str, test_suffix: str):
    train = pd.read_csv(DATA_DIR / train_suffix)
    test = pd.read_csv(DATA_DIR / test_suffix)

    X_train = train.drop(columns=['label', 'instance_weight'])
    y_train = train['label']
    X_test = test.drop(columns=['label', 'instance_weight'])
    y_test = test['label']

    return X_train, y_train, X_test, y_test


# -------------------------------
# Class imbalance for XGBoost
# -------------------------------
def get_scale_pos_weight(y_train):
    num_majority = sum(y_train == 0)
    num_minority = sum(y_train == 1)
    scale_pos_weight = num_majority / num_minority
    return scale_pos_weight


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(model_name, y_true, y_pred, y_pred_proba, save_path_prefix=None):
    print(f'\n--- Evaluating {model_name} ---')
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    if save_path_prefix:
        with open(f'{save_path_prefix}_report.txt', 'w') as f:
            f.write(f'Confusion Matrix:\n{cm}\n\n')
            f.write(f'Classification Report:\n{cr}\n')
            f.write(f'ROC-AUC Score: {roc_auc:.4f}\n')


# -------------------------------
# Logistic Regression with L1
# -------------------------------
def log_reg_l1(X_train, y_train, X_test):
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

    X_train_scaled = pipeline_lr.named_steps['scaler'].transform(X_train)
    X_test_scaled = pipeline_lr.named_steps['scaler'].transform(X_test)

    return pipeline_lr, X_train_scaled, X_test_scaled


# -------------------------------
# Feature Importance via SHAP
# -------------------------------
def get_shap_values(model_name, pipeline, shap_explainer, X_train, X_test):
    print(f'\nGenerating SHAP values for {model_name}...')
    explainer = shap.shap_explainer(pipeline.named_steps['clf'], X_train)
    shap_values = explainer.shap_values(X_test)
    return shap_values


# -------------------------------
# XGBoost Model
# -------------------------------
def xgboost_fit(X_train, y_train, X_test):
    print('\nTraining XGBoost...')

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=get_scale_pos_weight(y_train),
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
        save_path_prefix=OUTPUT_DIR / 'xgboost'
    )

    return pipeline_xgb, y_pred_xgb, y_pred_proba_xgb


# -------------------------------
# Save Plots
# -------------------------------
def save_shap_plot(shap_values, X_test, X_train, model_name, output_suffix, plot_type):

    plt.figure()
    if model_name == 'xgboost':
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
    else:
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, plot_type='bar', show=False)
        else:
            shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_suffix)
    plt.close()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(DATA_DIR, train_suffix, test_suffix)

    # logistic regression
    pipeline_lr, X_train_scaled, X_test_scaled = log_reg_l1(X_train, y_train, X_test)

    # logistic regression shap values
    lr_shap_values = get_shap_values('Logistic Regression', pipeline_lr, X_train_scaled, X_test_scaled)

    # logistic regression shap summary plot
    save_shap_plot(lr_shap_values, X_test_scaled, X_train, 'logistic regression', 'logreg_l1_shap_summary.png', 'summary')

    # logistic regression shap bar plot
    save_shap_plot(lr_shap_values, X_test_scaled, X_train, 'logistic regression', 'logreg_l1_shap_bar.png', 'bar')

    # xgboost
    pipeline_xgb, y_pred_xgb, y_pred_proba_xgb = xgboost_fit(X_train, y_train, X_test)

    # xgboost shap values
    shap_values_xgb = get_shap_values('XGBoost', pipeline_xgb, X_train, X_test)

    # xgboost shap bar plot
    save_shap_plot(shap_values_xgb, X_test, X_train, 'xgboost', 'xgboost_shap_bar.png', 'bar')

    print(f'\nAll reports and figures saved to {OUTPUT_DIR}/')

