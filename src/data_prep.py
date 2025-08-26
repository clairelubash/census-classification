from pathlib import Path
from typing import Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# -------------------------------
# Config
# -------------------------------
DATA_PATH = Path('data')
TRAIN_FILE = DATA_PATH / 'train.csv'
TEST_FILE = DATA_PATH / 'test.csv'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

NAN_REPLACEMENTS = [' ?', ' NA', ' Do not know']
NAN_STARTSWITH = ' Not in universe'

DROP_COLS = ['year', 'veterans_benefits']

BINARY_MAPS = {
    'sex': {' Male': 1, ' Female': 0},
    'label': {' 50000+.': 1, ' - 50000.': 0},
}

ONE_HOT_FEATURES = [
    'citizenship',
    'race',
    'full_or_part_time_employment_stat',
    'tax_filer_status',
    'detailed_household_summary_in_household',
    'hispanic_origin',
    'education',
    'marital_status',
    'country_of_birth_father',
    'country_of_birth_mother',
    'country_of_birth_self',
    'detailed_household_and_family_stat',
]

EDUCATION_GROUPS = {
    ' No schooling': 'No schooling',
    ' Less than 1st grade': 'No schooling',
    ' 1st 2nd 3rd or 4th grade': 'Elementary school',
    ' 5th or 6th grade': 'Elementary school',
    ' 7th and 8th grade': 'Middle school',
    ' 9th grade': 'High school',
    ' 10th grade': 'High school',
    ' 11th grade': 'High school',
    ' 12th grade no diploma': 'High school',
    ' High school graduate': 'High school grad',
    ' Some college but no degree': 'Some college',
    ' Associates degree-academic program': 'Associates',
    ' Associates degree-occup /vocational': 'Associates',
    ' Bachelors degree(BA AB BS)': 'Bachelors',
    ' Masters degree(MA MS MEng MEd MSW MBA)': 'Masters',
    ' Prof school degree (MD DDS DVM LLB JD)': 'Professional',
    ' Doctorate degree(PhD EdD)': 'Doctorate',
    ' Children': 'Other',
}


# -------------------------------
# Helper Functions
# -------------------------------
def group_household(val: str) -> str:
    '''Group household and family relationship values into broader categories.'''
    val = val.strip()
    if 'Householder' in val:
        return 'Householder'
    if 'Spouse' in val:
        return 'Spouse'
    if 'Child' in val or 'Grandchild' in val:
        return 'Child/Grandchild'
    if 'Other Rel' in val:
        return 'Other relative'
    if 'Secondary individual' in val or 'Nonfamily' in val:
        return 'Nonfamily'
    if 'group quarters' in val:
        return 'Group quarters'
    return 'Other'


def load_data(train_file: Path, test_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Load raw train and test CSV files.'''
    return pd.read_csv(train_file), pd.read_csv(test_file)


def replace_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    '''Replace placeholder strings with NaN values.'''
    df = df.replace(NAN_REPLACEMENTS, np.nan)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(
            lambda x: np.nan if isinstance(x, str) and x.startswith(NAN_STARTSWITH) else x
        )
    return df


def drop_high_nan_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    '''Drop columns with proportion of NaN values above threshold.'''
    return df.loc[:, df.isnull().mean() <= threshold]


def encode_binary(df: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    '''Encode binary categorical columns using a mapping dictionary.'''
    df[col] = df[col].map(mapping)
    return df


# -------------------------------
# Preprocessing Pipeline
# -------------------------------
def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Full preprocessing pipeline for train and test datasets.'''

    # Replace placeholder values with NaN
    train = replace_with_nan(train)
    test = replace_with_nan(test)

    # Drop high-NaN columns (apply train mask to test)
    train = drop_high_nan_cols(train)
    test = test[train.columns]

    # Binary encoding
    for col, mapping in BINARY_MAPS.items():
        if col in train.columns:
            train = encode_binary(train, col, mapping)
            test = encode_binary(test, col, mapping)

    # Education grouping
    train['education'] = train['education'].map(EDUCATION_GROUPS)
    test['education'] = test['education'].map(EDUCATION_GROUPS)

    # Household grouping
    train['detailed_household_and_family_stat'] = train[
        'detailed_household_and_family_stat'
    ].map(group_household)
    test['detailed_household_and_family_stat'] = test[
        'detailed_household_and_family_stat'
    ].map(group_household)

    # Fill NaN for specific categorical
    train['hispanic_origin'] = train['hispanic_origin'].fillna(' All other')
    test['hispanic_origin'] = test['hispanic_origin'].fillna(' All other')

    # One-hot encoding
    ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    train_ohe = pd.DataFrame(
        ohe.fit_transform(train[ONE_HOT_FEATURES]),
        columns=ohe.get_feature_names_out(ONE_HOT_FEATURES),
        index=train.index,
    )
    test_ohe = pd.DataFrame(
        ohe.transform(test[ONE_HOT_FEATURES]),
        columns=ohe.get_feature_names_out(ONE_HOT_FEATURES),
        index=test.index,
    )

    train = train.drop(columns=ONE_HOT_FEATURES).join(train_ohe)
    test = test.drop(columns=ONE_HOT_FEATURES).join(test_ohe)

    # Drop unnecessary columns
    train = train.drop(columns=DROP_COLS)
    test = test.drop(columns=DROP_COLS)

    # Fill any remaining missing values
    train = train.fillna('unknown')
    test = test.fillna('unknown')

    return train, test


# -------------------------------
# Main Execution
# -------------------------------
def main() -> None:
    '''Run preprocessing and save cleaned datasets.'''
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    train_clean, test_clean = preprocess(train_df, test_df)

    logging.info(f'Training data shape: {train_clean.shape}')
    logging.info(f'Testing data shape: {test_clean.shape}')

    train_clean.to_csv(DATA_PATH / 'train_clean.csv', index=False)
    test_clean.to_csv(DATA_PATH / 'test_clean.csv', index=False)


if __name__ == '__main__':
    main()

