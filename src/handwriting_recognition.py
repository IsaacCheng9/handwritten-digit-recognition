"""
A program to perform machine learning using k-means clustering over a sample of
handwritten digits. It creates the clusters, and plots them in a graph for the
user to see.
"""
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
    (num_samples, num_features) = data.shape
    num_digits = np.unique(labels).size
    print(("Number of Samples: {}\nNumber of Features: {}\nNumber of "
           "Digits: {} ").format(num_samples, num_features, num_digits))
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
    k_means = KMeans(n_clusters=10)
    k_means.fit(reduced_data)
    return k_means, reduced_data


def create_decision_boundaries(k_means, reduced_data):
    """
    Creates the decision boundaries to show where clusters will pick up points.

    Args:
        k_means: The ten clusters representing each digit.
        reduced_data: Sample data reduced to 2D space.
    """
    # Sets visual quality of the boundaries (lower is better).
    quality = .01
    # Plots decision boundaries for clusters.
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, quality),
                                 np.arange(y_min, y_max, quality))
    # Obtains labels for each point in mesh.
    mesh_labels = k_means.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])
    # Plots resultant mesh clusters.
    mesh_labels = mesh_labels.reshape(mesh_x.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(mesh_labels, interpolation="nearest",
               extent=(mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()),
               cmap="tab10", aspect="auto", origin="lower")


def plot_cluster_graph(k_means, reduced_data):
    """
    Plots the cluster graph and shows it.

    Args:
        k_means: The ten clusters representing each digit.
        reduced_data: Sample data reduced to 2D space.
    """
    # Plots the resultant k-means clusters onto a graph.
    centroids = k_means.cluster_centers_
    # Adds the clusters of the data to the graph in different colours.
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=k_means.labels_,
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
    k_means, reduced_data = k_means_clustering(data)
    create_decision_boundaries(k_means, reduced_data)
    plot_cluster_graph(k_means, reduced_data)


if __name__ == "__main__":
    main()
