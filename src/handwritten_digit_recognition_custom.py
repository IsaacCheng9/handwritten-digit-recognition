"""
A program to perform machine learning using k-means clustering over a sample of
handwritten digits. It creates the clusters, and plots them in a graph for the
user to see.

This uses an implementation of k-means which has been written from scratch (as
opposed to a Python library).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def show_data_sample_details():
    """
    Shows details of the sample of handwritten digits used for k-means.

    Returns:
        The sample data used for handwritten digit recognition.
    """
    data, labels = load_digits(return_X_y=True)
    (num_samples, _) = data.shape
    num_digits = np.unique(labels).size
    print("Number of Samples: {}\nNumber of Digits: {}\n".
          format(num_samples, num_digits))
    return data


def k_means_clustering(data):
    """
    Performs k-means clustering on the sample data.

    Args:
        data: The sample data used for handwritten digit recognition.

    Returns:
        The ten clusters representing each digit, and reduced sample data.
    """
    # Reduces the data to points in 2D space.
    reduced_data = PCA(n_components=2).fit_transform(data)
    # Creates a k-means object with 10 clusters, each representing a digit.
    k_means = k_means_algorithm(reduced_data, 10, 1000)
    return k_means, reduced_data


def k_means_algorithm(x, k, iterations):
    # Creates random initial centroids.
    index = np.random.choice(len(x), k, replace=False)
    centroids = x[index, :]
    # Calculates the distances between centroids and data points using the
    # Euclidean distance metric.
    distances = cdist(x, centroids, "euclidean")
    # Assigns the point to the centroid with the least distance.
    points = np.array([np.argmin(i) for i in distances])

    for _ in range(iterations):
        centroids = []
        for idx in range(k):
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points


def plot_cluster_graph(k_means, reduced_data):
    """
    Plots the cluster graph and shows it.

    Args:
        k_means: The ten clusters representing each digit.
        reduced_data: Sample data reduced to 2D space.
    """
    for unique in np.unique(k_means):
        plt.scatter(reduced_data[k_means == unique, 0],
                    reduced_data[k_means == unique, 1],
                    label=unique, s=5)
    plt.title("K-Means Clustering on the Digits Data Set Using PCA-Reduced "
              "Data\n(centroids marked with red dots)")
    plt.show()


def main():
    """
    Performs the handwritten digit recognition using k-means clustering.
    """
    data = show_data_sample_details()
    k_means, reduced_data = k_means_clustering(data)
    plot_cluster_graph(k_means, reduced_data)


if __name__ == "__main__":
    main()
