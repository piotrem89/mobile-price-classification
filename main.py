from src.data_loader import load_train_data, load_test_data
from src.preprocessing import replace_zeros_with_nan, calculate_median_map, impute_missing_values_with_map

df_train = load_train_data()
df_test = load_test_data()

clean_train = replace_zeros_with_nan(df_train)
clean_test = replace_zeros_with_nan(df_test)

median_map = calculate_median_map(clean_train, ['screen_width_cm', 'screen_pixel_height'])

imputed_train = impute_missing_values_with_map(clean_train, median_map)
imputed_test = impute_missing_values_with_map(clean_test, median_map)


print("NaN count in train dataset after imputation:")
print(imputed_train[['screen_width_cm', 'screen_pixel_height']].isna().sum())

print("\nNaN count in test dataset after imputation:")
print(imputed_test[['screen_width_cm', 'screen_pixel_height']].isna().sum())