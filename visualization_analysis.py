import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(df):
    sns.pairplot(df)
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def violin_plot(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df)
    plt.title("Violin Plot")
    plt.xticks(rotation=45)
    plt.show()
