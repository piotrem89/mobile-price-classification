import pandas as pd
import matplotlib.pyplot as plt



def merge_df(df_test: pd.DataFrame, df_result: pd.DataFrame) -> pd.DataFrame:
    """Merge test dataset features with model predictions on phone_id."""
    merged_df = pd.merge(df_test, df_result, on= 'phone_id')
    return merged_df

def plot_test_predictions(merged_df: pd.DataFrame) -> None:
    """Generate and save plots for test prediction results."""
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count prediction instances per class
    counts = merged_df['predicted_price_range'].value_counts().sort_index()
    labels = ['Low Cost','Medium Cost', 'High Cost', 'Premium']

    # 1. Bar plot: Predicted class distribution
    axes[0].bar(counts.index, counts.values, tick_label=labels)
    axes[0].set_title('Predicted Price Range Distribution')
    axes[0].set_xlabel('Price Range')
    axes[0].set_ylabel('Count')

    # Extract RAM capacity data grouped by predicted price range
    classes = sorted(merged_df['predicted_price_range'].unique())
    ram_data = [
        merged_df[merged_df['predicted_price_range'] == c]['ram_capacity_mb']
        for c in classes
    ]

    # 2. Box plot: RAM capacity distribution by predicted class
    axes[1].boxplot(ram_data, tick_labels=labels)
    axes[1].set_title('RAM Capacity by Predicted Price Range')
    axes[1].set_xlabel('Price Range')
    axes[1].set_ylabel('RAM (MB)')

    # Adjust layout and save figure to file
    plt.tight_layout()
    plt.savefig("data/test_predictions_plot.png")
    plt.close()
