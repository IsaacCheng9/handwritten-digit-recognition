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
    points = k_means_algorithm(reduced_data, 10, 1000)
    return points, reduced_data


def k_means_algorithm(data_points, number_of_clusters, iterations):
    index = np.random.choice(len(data_points), number_of_clusters,
                             replace=False)
    # Creates random initial centroids and calculates the distances between
    # centroids and data points using the Euclidean distance metric.
    centroids = data_points[index, :]
    distances = cdist(data_points, centroids, "euclidean")
    # Assigns the point to the centroid with the least distance.
    points = np.array([np.argmin(distance) for distance in distances])

    # Updates position of clusters iteratively.
    for _ in range(iterations):
        centroids = []
        # Updates position of centroids.
        for idx in range(number_of_clusters):
            centroids.append(data_points[points == idx].mean(axis=0))
        distances = cdist(data_points, np.array(centroids), "euclidean")
        points = np.array([np.argmin(distance) for distance in distances])

    return points


def plot_cluster_graph(points, reduced_data):
    """
    Plots the cluster graph and shows it.

    Args:
        points: The ten clusters representing each digit.
        reduced_data: Sample data reduced to 2D space.
    """
    for unique in np.unique(points):
        plt.scatter(reduced_data[points == unique, 0],
                    reduced_data[points == unique, 1],
                    label=unique, s=5)
    plt.title("K-Means Clustering on the Digits Data Set Using PCA-Reduced "
              "Data\n(centroids marked with red dots)")
    plt.show()


def main():
    """
    Performs the handwritten digit recognition using k-means clustering.
    """
    data = show_data_sample_details()
    points, reduced_data = k_means_clustering(data)
    plot_cluster_graph(points, reduced_data)


if __name__ == "__main__":
    main()
