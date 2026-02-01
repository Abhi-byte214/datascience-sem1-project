import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------
# Load Dataset Function
# ------------------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(df.head())
    print("\nColumns:")
    print(df.columns)
    return df


# ------------------------------------
# Pair Plot Function
# ------------------------------------
def pair_plot(df):
    sns.pairplot(df, hue="Outcome")
    plt.suptitle("Pair Plot of Diabetes Dataset", y=1.02)
    plt.show()


# ------------------------------------
# Correlation Heatmap Function
# ------------------------------------
def correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


# ------------------------------------
# Violin Plot Function
# ------------------------------------
def violin_plot(df):
    plt.figure()
    sns.violinplot(x="Outcome", y="BMI", data=df)
    plt.title("Violin Plot of BMI vs Outcome")
    plt.xlabel("Outcome")
    plt.ylabel("BMI")
    plt.show()


# ------------------------------------
# Correlation & Covariance Function
# ------------------------------------
def correlation_covariance(df):
    print("\nCorrelation Matrix:")
    print(df.corr())

    print("\nCovariance Matrix:")
    print(df.cov())
