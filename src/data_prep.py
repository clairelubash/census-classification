import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------
# Config
# -------------------------------
DATA_PATH = Path('data')
TRAIN_FILE = DATA_PATH / 'train.csv'
TEST_FILE = DATA_PATH / 'test.csv'

NAN_REPLACEMENTS = [' ?', ' NA', ' Do not know']
NAN_STARTSWITH = ' Not in universe'

DROP_COLS = ['year', 'veterans_benefits']

BINARY_MAPS = {
    'sex': {' Male': 1, ' Female': 0},
    'label': {' 50000+.': 1, ' - 50000.': 0}
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
    'detailed_household_and_family_stat'
]

# -------------------------------
# Custom Groupings
# -------------------------------
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
    ' Children': 'Other'
}

def group_household(val: str) -> str:
    val = val.strip()
    if 'Householder' in val:
        return 'Householder'
    elif 'Spouse' in val:
        return 'Spouse'
    elif 'Child' in val or 'Grandchild' in val:
        return 'Child/Grandchild'
    elif 'Other Rel' in val:
        return 'Other relative'
    elif 'Secondary individual' in val or 'Nonfamily' in val:
        return 'Nonfamily'
    elif 'group quarters' in val:
        return 'Group quarters'
    else:
        return 'Other'

# -------------------------------
# Data Preparation Functions
# -------------------------------
def load_data(train_file: Path, test_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train, test

def replace_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(NAN_REPLACEMENTS, np.nan)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(lambda x: np.nan if isinstance(x, str) and x.startswith(NAN_STARTSWITH) else x)
    return df

def drop_high_nan_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    return df.loc[:, df.isnull().mean() <= threshold]

def encode_binary(df: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    df[col] = df[col].map(mapping)
    return df

def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Replace with NaN
    train = replace_with_nan(train)
    test = replace_with_nan(test)

    # Drop high-NaN cols
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
    train['detailed_household_and_family_stat'] = train['detailed_household_and_family_stat'].map(group_household)
    test['detailed_household_and_family_stat'] = test['detailed_household_and_family_stat'].map(group_household)

    # Impute hispanic_origin NaN with 'All other'
    train['hispanic_origin'] = train['hispanic_origin'].fillna(' All other')
    test['hispanic_origin'] = test['hispanic_origin'].fillna(' All other')

    # One-hot encoding
    ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    train_ohe = pd.DataFrame(
        ohe.fit_transform(train[ONE_HOT_FEATURES]),
        columns=ohe.get_feature_names_out(ONE_HOT_FEATURES),
        index=train.index
    )

    test_ohe = pd.DataFrame(
        ohe.transform(test[ONE_HOT_FEATURES]),
        columns=ohe.get_feature_names_out(ONE_HOT_FEATURES),
        index=test.index
    )

    # Drop original categorical cols & concat OHE
    train = train.drop(columns=ONE_HOT_FEATURES).join(train_ohe)
    test = test.drop(columns=ONE_HOT_FEATURES).join(test_ohe)

    # Drop unnecessary columns based on EDA
    train = train.drop(columns=DROP_COLS)
    test = test.drop(columns=DROP_COLS)

    train = train.fillna('unknown')
    test = test.fillna('unknown')

    return train, test

# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    train_clean, test_clean = preprocess(train_df, test_df)

    print('Training data shape:', train_clean.shape)
    print('Testing data shape:', test_clean.shape)

    # Save cleaned versions
    train_clean.to_csv(DATA_PATH / 'train_clean.csv', index=False)
    test_clean.to_csv(DATA_PATH / 'test_clean.csv', index=False)
