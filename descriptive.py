import pandas as pd

# ------------------------------------
# Load Dataset
# ------------------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


# ------------------------------------
# Descriptive Statistics Function
# ------------------------------------
def descriptive_statistics(df):
    mean_value = df.mean(numeric_only=True)
    median_value = df.median(numeric_only=True)
    mode_value = df.mode().iloc[0]
    std_value = df.std(numeric_only=True)
    summary = df.describe()

    print("Mean:\n", mean_value)
    print("\nMedian:\n", median_value)
    print("\nMode:\n", mode_value)
    print("\nStandard Deviation:\n", std_value)
    print("\nSummary Statistics:\n", summary)

    return {
        "mean": mean_value,
        "median": median_value,
        "mode": mode_value,
        "std": std_value,
        "summary": summary
    }
