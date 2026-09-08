import numpy as np
import pandas as pd
from src.data_loader import load_train_data


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid 0 values with NaN in screen dimensions columns."""
    df_clean = df.copy()
    df_clean['screen_width_cm'] = df_clean['screen_width_cm'].replace(0, np.nan)
    df_clean['screen_pixel_height'] = df_clean['screen_pixel_height'].replace(0, np.nan)
    return df_clean


# Exploratory Data Analysis (EDA) and local verification of data anomalies.
if __name__ == "__main__":
    df_train = load_train_data()

    print("Initial NaN values count:")
    print(df_train.isna().sum())
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("*" * 100)
    print("DataFrame Summary:")
    print(df_train.describe(include='all').T)

    raw_df = df_train
    clean_df = replace_zeros_with_nan(raw_df)
    print("*" * 100)
    print("NaN values count after cleaning:")
    print(clean_df.isna().sum())