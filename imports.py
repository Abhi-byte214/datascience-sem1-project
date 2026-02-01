import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# Load Dataset
# ------------------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Shape:")
    print(df.shape)
    return df


# ------------------------------------
# Save Dataset
# ------------------------------------
def save_dataset(df):
    df.to_csv("cleaned_data.csv", index=False)
    df.to_excel("cleaned_data.xlsx", index=False)
    print("Cleaned dataset saved as CSV and Excel")


# ------------------------------------
# Handle Missing Values
# ------------------------------------
def handle_missing_values(df):
    print("\nMissing Values Before:")
    print(df.isnull().sum())

    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("\nMissing Values After:")
    print(df.isnull().sum())
    return df


# ------------------------------------
# Remove Duplicates
# ------------------------------------
def remove_duplicates(df):
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]

    print(f"\nDuplicates Removed: {before - after}")
    return df


# ------------------------------------
# Handle Outliers using IQR
# ------------------------------------
def handle_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | 
                      (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    print("\nOutliers removed")
    return df_cleaned


# ------------------------------------
# Boxplot Visualization
# ------------------------------------
def boxplot_visualization(df):
    df.boxplot()
    plt.title("Boxplot for Outlier Detection")
    plt.show()
