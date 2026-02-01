import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def standardize_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include="number"))
    print("\nStandardized Data (mean=0, std=1):")
    print(scaled_data)
    return scaled_data


# ------------------------------------
# Normalization Function
def normalize_data(df):
    minmax = MinMaxScaler()
    normalized_data = minmax.fit_transform(df.select_dtypes(include="number"))
    print("\nNormalized Data (0 to 1 range):")
    print(normalized_data)
    return normalized_data



def descriptive_statistics(df):
    mean_value = df.mean(numeric_only=True)
    median_value = df.median(numeric_only=True)
    mode_value = df.mode().iloc[0]
    std_value = df.std(numeric_only=True)
    summary = df.describe()

    print("\nMean:\n", mean_value)
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

