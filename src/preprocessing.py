import numpy as np
import pandas as pd

from src.data_loader import load_train_data


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid 0 values with NaN in screen dimensions columns."""
    df_clean = df.copy()
    df_clean['screen_width_cm'] = df_clean['screen_width_cm'].replace(0, np.nan)
    df_clean['screen_pixel_height'] = df_clean['screen_pixel_height'].replace(0, np.nan)
    return df_clean


def calculate_median_map(df: pd.DataFrame, columns: list) -> dict:
    """ """
    median_map = {}
    for col in columns:
        median_map[col] = df[col].median()
    return median_map


def impute_missing_values_with_map(df: pd.DataFrame, fill_map: dict) -> pd.DataFrame:
    """Impute missing values in specified columns using a provided mapping dictionary(median)"""
    df_imputed = df.copy()
    for col, median_value in fill_map.items():
        df_imputed[col] = df_imputed[col].fillna(median_value)
    return df_imputed



# Exploratory data analysis and local verification of data anomalies.
if __name__ == "__main__":

    df_train = load_train_data()

    print("Initial NaN values count:")
    print(df_train.isna().sum())
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("*" * 100)
    print("DataFrame Summary:")
    print(df_train.describe(include='all').T)

# Check NaN values count after replacing invalid 0s with NaN
    raw_df = df_train
    clean_df = replace_zeros_with_nan(raw_df)
    print("*" * 100)
    print("NaN values count after cleaning:")
    print(clean_df.isna().sum())

# Calculate median map and impute missing values
    target_cols = ['screen_width_cm', 'screen_pixel_height']
    medians_map = calculate_median_map(clean_df, target_cols)
    imputed_df = impute_missing_values_with_map(clean_df, medians_map)

# Check NaN values count after median imputation for full DataFrame
    print("*" * 100)
    print("NaN values count after median imputation:")
    print(imputed_df.isna().sum())