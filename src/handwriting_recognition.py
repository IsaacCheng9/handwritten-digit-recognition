import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def show_data_sample_details():
    data, labels = load_digits(return_X_y=True)
    (num_samples, num_features) = data.shape
    num_digits = np.unique(labels).size
    print(("Number of Samples: {}\nNumber of Features: {}\nNumber of "
           "Digits: {} ").format(num_samples, num_features, num_digits))
    return data


def k_means_clustering(data):
    # Reduces the data to points in 2D space.
    reduced_data = PCA(n_components=2).fit_transform(data)
    # Creates a k-means object with 10 clusters, each representing a digit.
    k_means = KMeans(n_clusters=10)
    k_means.fit(reduced_data)

    # Plots the resultant k-means clusters onto a graph.
    centroids = k_means.cluster_centers_
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=k_means.labels_,
                cmap="rainbow", s=8)
    plt.scatter(centroids[:, 0], centroids[:, 1], color="red")
    plt.title("K-Means Clustering on the Digits Data Set Using PCA-Reduced "
              "Data\n(centroids marked with red dots)")
    plt.show()


def main():
    data = show_data_sample_details()
    k_means_clustering(data)


if __name__ == "__main__":
    main()
