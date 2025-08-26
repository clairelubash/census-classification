from pathlib import Path
import logging
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score
)
from xgboost import XGBClassifier


# -------------------------------
# Configuration
# -------------------------------
OUTPUT_DIR = Path('outputs')
DATA_DIR = Path('data')
TRAIN_SUFFIX = 'train_clean.csv'
TEST_SUFFIX = 'test_clean.csv'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)


# -------------------------------
# Data Loading
# -------------------------------
def load_data(data_dir: Path, train_suffix: str, test_suffix: str):
    '''Load train/test data from CSVs and return feature/label splits.'''
    train = pd.read_csv(data_dir / train_suffix)
    test = pd.read_csv(data_dir / test_suffix)

    X_train = train.drop(columns=['label', 'instance_weight'])
    y_train = train['label']
    X_test = test.drop(columns=['label', 'instance_weight'])
    y_test = test['label']

    return X_train, y_train, X_test, y_test


# -------------------------------
# Class imbalance for XGBoost
# -------------------------------
def get_scale_pos_weight(y_train):
    '''Calculate scale_pos_weight for imbalanced datasets.'''
    num_majority = sum(y_train == 0)
    num_minority = sum(y_train == 1)
    return num_majority / num_minority


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(model_name, y_true, y_pred, y_pred_proba, save_path_prefix=None):
    '''Evaluate model predictions and save metrics to a report file.'''
    logging.info(f'Evaluating {model_name}...')
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    if save_path_prefix:
        report_path = Path(f'{save_path_prefix}_report.txt')
        with open(report_path, 'w') as f:
            f.write(f'Confusion Matrix:\n{cm}\n\n')
            f.write(f'Classification Report:\n{cr}\n')
            f.write(f'ROC-AUC Score: {roc_auc:.4f}\n')
        logging.info(f'Saved evaluation report to {report_path}')


# -------------------------------
# Logistic Regression with L1
# -------------------------------
def log_reg_l1(X_train, y_train, X_test, y_test):
    '''Train Logistic Regression with L1 regularization and evaluate.'''
    logging.info('Training Logistic Regression with L1 Regularization...')

    logreg = LogisticRegression(
        penalty='l1',
        solver='liblinear', 
        class_weight='balanced',
        C=1.0,
        random_state=42,
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
def get_shap_values(model, X_train, X_test):
    '''Compute SHAP values for a trained model.'''
    logging.info(f'Generating SHAP values for {type(model).__name__}...')
    explainer = shap.Explainer(model, X_train)
    return explainer(X_test)


# -------------------------------
# XGBoost Model
# -------------------------------
def xgboost_fit(X_train, y_train, X_test, y_test):
    '''Train an XGBoost model with default settings and evaluate.'''
    logging.info('Training XGBoost...')

    xgb = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=get_scale_pos_weight(y_train),
        n_estimators=200,
        random_state=42,
        use_label_encoder=False
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
# Save SHAP Plots
# -------------------------------
def save_shap_plot(shap_values, X_test, X_train, model_name, output_suffix, plot_type):
    '''Generate and save SHAP summary plots.'''
    plt.figure()

    if plot_type == 'bar':
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=None if model_name == 'xgboost' else X_train.columns,
            plot_type='bar',
            show=False,
        )
    else:
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=None if model_name == 'xgboost' else X_train.columns,
            show=False,
        )

    plt.tight_layout()
    output_path = OUTPUT_DIR / output_suffix
    plt.savefig(output_path)
    plt.close()
    logging.info(f'Saved SHAP {plot_type} plot to {output_path}')


# -------------------------------
# Script Entry Point
# -------------------------------
def main():
    '''Run Logistic Regression and XGBoost pipelines with SHAP analysis.'''
    X_train, y_train, X_test, y_test = load_data(DATA_DIR, TRAIN_SUFFIX, TEST_SUFFIX)

    # Logistic Regression
    pipeline_lr, X_train_scaled, X_test_scaled = log_reg_l1(X_train, y_train, X_test, y_test)
    lr_shap_values = get_shap_values(pipeline_lr.named_steps['clf'], X_train_scaled, X_test_scaled)
    save_shap_plot(lr_shap_values, X_test_scaled, X_train, 'logistic_regression', 'logreg_l1_shap_summary.png', 'summary')
    save_shap_plot(lr_shap_values, X_test_scaled, X_train, 'logistic_regression', 'logreg_l1_shap_bar.png', 'bar')

    # XGBoost
    pipeline_xgb, _, _ = xgboost_fit(X_train, y_train, X_test, y_test)
    xgb_shap_values = get_shap_values(pipeline_xgb.named_steps['clf'], X_train, X_test)
    save_shap_plot(xgb_shap_values, X_test, X_train, 'xgboost', 'xgboost_shap_bar.png', 'bar')

    logging.info(f'All reports and figures saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
