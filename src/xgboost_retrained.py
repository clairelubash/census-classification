from pathlib import Path
import logging
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
# Configuration
# -------------------------------
OUTPUT_DIR = Path('outputs')
DATA_DIR = Path('data')
TRAIN_SUFFIX = 'train_clean.csv'
TEST_SUFFIX = 'test_clean.csv'

PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

N_ITER_SEARCH = 20
RANDOM_STATE = 42


# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)


def get_xgb_model(y_train):
    '''Initialize an XGBoost model with class imbalance handling.'''
    return XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=get_scale_pos_weight(y_train),
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def tune_hyperparameters(model, X_train, y_train):
    '''Perform hyperparameter tuning with RandomizedSearchCV.'''
    logging.info('Starting hyperparameter tuning for XGBoost...')
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER_SEARCH,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    logging.info(f'Best parameters: {search.best_params_}')
    return search.best_estimator_


def compute_shap_feature_importance(model, X_train, threshold=0.01):
    '''Compute SHAP values and select important features above a threshold.'''
    logging.info('Computing SHAP values for feature selection...')
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    shap_importances = np.abs(shap_values.values).mean(axis=0)
    feature_importances = pd.Series(shap_importances, index=X_train.columns)

    selected_features = feature_importances[feature_importances > threshold].index
    logging.info(f'Selected {len(selected_features)} features using SHAP.')
    return explainer, selected_features


def retrain_with_selected_features(model, explainer, X_train, y_train, X_test, y_test, selected_features):
    '''Retrain XGBoost with SHAP-selected features and evaluate.'''
    logging.info('Retraining XGBoost with selected features...')
    model.fit(X_train[selected_features], y_train)

    y_pred = model.predict(X_test[selected_features])
    y_pred_proba = model.predict_proba(X_test[selected_features])[:, 1]

    evaluate_model(
        'XGBoost_SelectedFeatures',
        y_test,
        y_pred,
        y_pred_proba,
        OUTPUT_DIR / 'xgb_selected',
    )

    # Save SHAP summary plots
    shap_values = explainer(X_test[selected_features])
    save_shap_plot(
        shap_values,
        X_test[selected_features],
        X_train,
        'xgboost',
        'xgb_selected_shap_summary.png',
        'summary',
    )

    plt.figure()
    shap.summary_plot(shap_values, X_test[selected_features], show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'xgb_selected_shap_summary.png')
    plt.close()


def main():
    '''Main retraining pipeline.'''
    # Load data
    X_train, y_train, X_test, y_test = load_data(DATA_DIR, TRAIN_SUFFIX, TEST_SUFFIX)

    # Initial training & tuning
    xgb_model = get_xgb_model(y_train)
    best_xgb = tune_hyperparameters(xgb_model, X_train, y_train)

    # Initial evaluation
    y_pred = best_xgb.predict(X_test)
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    evaluate_model('XGBoost_Tuned', y_test, y_pred, y_pred_proba, OUTPUT_DIR / 'xgb_tuned')

    # SHAP feature selection
    explainer, selected_features = compute_shap_feature_importance(best_xgb, X_train)

    # Retrain with selected features
    retrain_with_selected_features(best_xgb, explainer, X_train, y_train, X_test, y_test, selected_features)


if __name__ == '__main__':
    main()
