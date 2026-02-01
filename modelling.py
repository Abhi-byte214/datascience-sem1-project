import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans

# ------------------------------------
# Load Dataset
# ------------------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(df.head())
    return df


# ------------------------------------
# k-NN Classification Function
# ------------------------------------
def knn_classification(df, k=5):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train k-NN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prediction
    y_pred = knn.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- k-NN Classification Results ---")
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return accuracy


# ------------------------------------
# k-Means Clustering Function
# ------------------------------------
def kmeans_clustering(df, n_clusters=3):
    X = df.drop("Outcome", axis=1)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply k-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster labels
    df["Cluster"] = clusters

    # Visualization
    plt.figure()
    plt.scatter(
        df["Glucose"],
        df["BMI"],
        c=df["Cluster"]
    )
    plt.title("k-Means Clustering (Glucose vs BMI)")
    plt.xlabel("Glucose Level")
    plt.ylabel("Body Mass Index (BMI)")
    plt.show()

    print("\n--- k-Means Clustering Completed ---")
    return df

