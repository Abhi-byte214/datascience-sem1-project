from imports import load_dataset
from visualization_analysis import pair_plot, correlation_heatmap, violin_plot
from descriptive import descriptive_statistics
from modelling import knn_classification, kmeans_clustering
from minmax import standardize_data,normalize_data,descriptive_statistics
from plotydash import interactive_scatter
print("Dataset Preview:")
df = load_dataset("diabetes.csv")

descriptive_statistics(df)

pair_plot(df)
correlation_heatmap(df)
violin_plot(df)

standardize_data(df)
normalize_data(df)


knn_classification(df)
kmeans_clustering(df)
interactive_scatter(df)
