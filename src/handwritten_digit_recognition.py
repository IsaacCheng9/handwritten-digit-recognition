"""
A program to perform machine learning using k-means clustering over a sample of
handwritten digits. It creates the clusters, and plots them in a graph for the
user to see.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
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
    points = KMeans(n_clusters=10)
    cluster_numbers = points.fit_predict(reduced_data)

    return points, reduced_data, cluster_numbers


def evaluate_accuracy(data):
    """
    Evaluates accuracy of the k-means clustering analysis.

    Args:
        data: The sample data used for handwritten digit recognition.
    """
    digits = load_digits()

    correct_predictions = 0
    for run in range(1, 11):
        _, _, cluster_numbers = k_means_clustering(data)
        cluster_groups = determine_cluster_representations(cluster_numbers,
                                                           digits)
        cluster_representations = list(cluster_groups.values())
        print("Cluster Representations for Run #{}: {}".format(
            run, cluster_representations))
        for number, cluster_index in enumerate(cluster_numbers):
            real_value = digits.target[number]
            if real_value == cluster_groups[cluster_index]:
                correct_predictions += 1
    print("Accuracy (%): {}".format(correct_predictions / 1797 * 10))


def determine_cluster_representations(cluster_numbers, digits):
    """
    Determines the digit each cluster represents.

    Args:
        cluster_numbers: A list containing the cluster index of each number.
        digits: The sample data used for handwritten digit recognition.

    Returns:
        A dictionary showing the digit representation of each cluster group.
    """
    # Creates a dictionary to group the real values within each cluster.
    cluster_groups = {}
    for number, cluster_index in enumerate(cluster_numbers):
        real_value = digits.target[number]
        try:
            cluster_groups[cluster_index].append(real_value)
        except KeyError:
            cluster_groups[cluster_index] = [real_value]
    # Identifies each cluster representation based on majority vote.
    for key in cluster_groups:
        cluster_representation = Counter(cluster_groups[key]).most_common(1)
        cluster_groups[key] = cluster_representation[0][0]
    return cluster_groups


def create_decision_boundaries(points, reduced_data):
    """
    Creates the decision boundaries to show where clusters will pick up points.

    Args:
        points: The ten clusters representing each digit.
        reduced_data: Sample data reduced to 2D space.
    """
    # Sets visual quality of the boundaries (lower is better).
    quality = 0.05
    # Plots decision boundaries for clusters.
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, quality),
                                 np.arange(y_min, y_max, quality))
    # Obtains labels for each point in mesh.
    mesh_labels = points.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])
    # Plots resultant mesh clusters.
    mesh_labels = mesh_labels.reshape(mesh_x.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(mesh_labels, interpolation="nearest",
               extent=(mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()),
               cmap="tab10", aspect="auto", origin="lower")


def plot_cluster_graph(points, reduced_data):
    """
    Plots the cluster graph and shows it.

    Args:
        points: The ten clusters representing each digit.
        reduced_data: Sample data reduced to 2D space.
    """
    # Plots the resultant k-means clusters onto a graph.
    centroids = points.cluster_centers_
    # Adds the clusters of the data to the graph in different colours.
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=points.labels_,
                cmap="turbo", s=5)
    # Marks centroids with white crosses.
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100,
                color="white")
    plt.title("K-Means Clustering on the Digits Data Set Using PCA-Reduced "
              "Data\n(centroids marked with red dots)")
    plt.show()


def main():
    """
    Performs the handwritten digit recognition using k-means clustering.
    """
    data = show_data_sample_details()
    points, reduced_data, _ = k_means_clustering(data)
    create_decision_boundaries(points, reduced_data)
    plot_cluster_graph(points, reduced_data)
    evaluate_accuracy(data)


if __name__ == "__main__":
    main()
