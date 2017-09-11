"""Assignment 1: UTSA CS 6243/4593 Machine Learning Fall 2017"""

import random
from statistics import mean

__author__ = 'Rohit Mehra'


class KMeans:
    """Basic KMeans Cluster Class"""

    def __init__(self, num_clusters=3, max_iterations=900, random_seed=None):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.labels_ = None

    @staticmethod
    def sim(x1, x2):
        """calculates distanc/similarity between two feature vectors

        :param x1: first vector
        :param x2: second vector

        :returns dist: manhattan dist between two vectors"""

        if not len(x1) == len(x2):
            raise Exception("Vectors don't have same number of features")
        dist = 0
        for i, j in zip(x1, x2):
            dist += abs(i - j)
        return dist

    def get_labels(self, X, centroids):
        """returns asssigned centroids to each vector in X
        :param X: set of feature vectors
        :param centroids: set of centroids

        :return labels: set of size len(X), where label[i] is the centroid of i'th element in set X
        """
        labels = []
        for x in X:
            distances = [(c, self.sim(x, c)) for c in centroids]
            labels.append(min(distances, key=lambda t: t[1])[0])
        return labels

    @staticmethod
    def get_centroids(X, labels):
        """get new centroids for the clusters

        :param X: data points array
        :param labels: corresponding labels for each data point, array

        :return new_centroids: new centroid of the clusters
        """
        xy = list(zip(X, labels))  # (x, x's centroid)
        old_centroids = set(labels)  # set of centroids

        new_centroids = []
        for c in old_centroids:
            # list of similar features [(all xs), (all ys)]
            feats = list(zip(*[x for x, l in xy if l == c]))
            # tuple of features shape = (1, num_features) = (x1, x2)
            centroid = tuple([mean(f) for f in feats])
            assert len(centroid) == len(c)
            new_centroids.append(centroid)

        assert len(new_centroids) == len(old_centroids)
        return new_centroids

    def k_means(self, X, num_clusters=3, max_iterations=9000, random_seed=None, verbose=False):
        """ Basic K-means clustering algorithm

        :param X: set of feature vectors
        :optional param num_clusters: number of cluster required
        :optional param max_iterations: to cap max_iterations in optimization step
        :optional param random_seed: to seed random init

        :return centroids: set of centroids of clusters
        :return labels: set of assigned centroids in same sequence as set of input data,
                       label[i] gives label for i'th data point in the input
        :return iterations: iterations took to reach minima
        """
        random.seed(random_seed)

        # random vectors as centroids
        centroids = random.sample(X, num_clusters)

        # init labelling each vector to the cluster
        labels = self.get_labels(X, centroids)

        iterations = 0

        # breaks when clusters stop changing
        while True:
            # to avoid infinite loop
            if iterations > max_iterations:
                break

            if verbose:
                print(self.transform(X, labels))
            # counter
            iterations += 1
            # get new centroids based of clusters
            new_centroids = self.get_centroids(X, labels)
            # reassign data points to new centroids
            new_labels = self.get_labels(X, new_centroids)
            # check if new_labels(cluster assignment) are same to the previous assignment or not
            if new_labels != labels:
                labels = new_labels
                centroids = new_centroids
            else:
                break

        return centroids, labels, iterations

    def fit(self, X, verbose=False):
        """execute kmeans algo on X, iteratively learn cluster centers from data

        :param X: data for learning
        :optional param verbose: True if you want to print clusters after each iteration of learning
        """
        self.cluster_centers_, self.labels_, self.n_iter_ = self.k_means(
            X, verbose=verbose)
        return self

    def transform(self, X, labels=None):
        """assign input points to their cluster, transform the data in center: [datapoints] format
        :param X: data points to be transformed
        :param labels: assigned cluster centers to the given points

        :return transformed_data: format = {center: [list of datapoints in this cluster],...}
        """
        if self.labels_:  # if clusters set i.e. non intermediate stage
            labels = self.labels_

        if labels:
            xy = list(zip(X, labels))  # (x, centroid)
            centroids = set(labels)  # set of centroids
            transformed_data = dict()
            for c in centroids:
                transformed_data[c] = [x for x, l in xy if c == l]
            return transformed_data
        else:
            raise Exception(
                'Please use fit(X) to make the object learn the clusters first, then call transform(X)')


if __name__ == '__main__':
    data_points = [(2, 10), (2, 5), (8, 4), (5, 8),
                   (7, 5), (6, 4), (1, 2), (4, 9)]
    names = ['A' + str(i) for i in range(1, 9)]
    km = KMeans().fit(data_points, verbose=False)
    print(km.transform(names))

    """Output: {(1.5, 3.5): ['A2', 'A7'], (7, 4.333333333333333): ['A3', 'A5', 'A6'], (3.6666666666666665, 9): ['A1', 'A4', 'A8']}"""
