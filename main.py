from src.data_loader import load_train_data, load_test_data
from src.preprocessing import replace_zeros_with_nan, calculate_median_map, impute_missing_values_with_map
from src.models import prepare_training_data, evaluate_models, get_feature_importance
from sklearn.metrics import classification_report
import pandas as pd
from src.visualization import merge_df, plot_test_predictions

if __name__ == '__main__':
    # Load raw datasets
    df_train = load_train_data()
    df_test = load_test_data()

    # Replace invalid 0 values with NaN
    clean_train = replace_zeros_with_nan(df_train)
    clean_test = replace_zeros_with_nan(df_test)

    # Calculate medians ONLY on training set to prevent data leakage
    median_map = calculate_median_map(clean_train, ['screen_width_cm', 'screen_pixel_height'])

    # Impute missing values in both sets using train medians
    imputed_train = impute_missing_values_with_map(clean_train, median_map)
    imputed_test = impute_missing_values_with_map(clean_test, median_map)

    # Verify imputation results
    print("NaN count in train dataset after imputation:")
    print(imputed_train[['screen_width_cm', 'screen_pixel_height']].isna().sum())

    print("\nNaN count in test dataset after imputation:")
    print(imputed_test[['screen_width_cm', 'screen_pixel_height']].isna().sum())

    # Prepare features and target with stratified split
    X_tr, X_val, y_tr, y_val = prepare_training_data(imputed_train)

    # Evaluate classifiers and find the best model
    (best_name, best_f1, best_y_pred, best_clf), results_df = evaluate_models(X_tr, X_val, y_tr, y_val)

    # Display evaluation results
    print("\n--- Model Evaluation Results ---")
    print(results_df.to_string(index=False))
    print("--------------------------------\n")

    print(f"\nBest Classifier: {best_name}")
    print(f"Validation Macro F1-Score: {best_f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_val, best_y_pred))

    # Calculate and display top feature importances
    importances = get_feature_importance(best_clf, X_val, y_val)
    print("\nTop 5 Most Important Features:")
    print(importances.head(5))

    # Test prediction
    print("\nGenerating predictions for test dataset")
    test_id = imputed_test['phone_id']
    X_test = imputed_test.drop(columns=['phone_id'])
    test_predictions = best_clf.predict(X_test)

    # Create DataFrame with results
    results = pd.DataFrame({
        'phone_id': test_id,
        'predicted_price_range': test_predictions
    })

    results.to_csv('data/test_predictions.csv', index=False)
    print("Predictions successfully saved to data/test_predictions.csv")

    print(results.head())

    # Merge test set features with predictions for reporting
    prediction_summary_df = merge_df(imputed_test,results)

    # Generate and save diagnostic plot to data directory
    plot_test_predictions(prediction_summary_df)