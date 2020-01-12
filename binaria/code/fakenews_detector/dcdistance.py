from __future__ import annotations
from typing import Callable

from scipy.spatial.distance import euclidean, cosine

import numpy as np


class DCDistance(object):
    """DCDistance algorithm for dimensionality reduction.

    Parameters
    ----------
    distance : Callable
        A function that takes two numbers and returns their distance (e.g. scipy.spatial.distance.cosine)

    Attributes
    ----------
    X_training_reduced : array-like, shape = [n_instances]
        Distance betweetn class_vector and ith sample in training data.

    distance : function
        Function that computes the distance between two vectors.

    class_vectors : array-like, shape = [n_classes, n_features]
        Sum (along axis=0) of all vectors of given class in X_training.
    """

    def __init__(self, distance: Callable):
        self.X_training_reduced = None
        self.distance = distance
        self.class_vectors = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DCDistance:
        """Reduces dimensionality of X.
        :param X: np.ndarray, shape=(n_instances, n_features)
        :param y: labels, shape=(n_instances, 1)
        :return: the instance itself.
        """
        if type(X).__name__ == 'csr_matrix':
            X = X.toarray()
        if len(y.shape) < 2:  # i.e. shape=(n,)
            y = np.reshape(y, (y.shape[0], 1))
        Xy = np.append(X, y, axis=1)
        labels = np.unique(y)
        self.class_vectors = np.empty(shape=(0, X.shape[1]))
        for label in labels:
            class_vector = np.array(list(map(lambda v: v[0:-1], list(filter(lambda v: v[-1] == label, Xy))))).sum(axis=0)
            self.class_vectors = np.vstack((self.class_vectors, class_vector))
        self.X_training_reduced = self.transform(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces dimensionality of X down to the
        number of class vectors, which is the same as the
        number of classes fitted.
        :param X: np.ndarray, shape=(n_instances, n_features)
        :return: np.array, shape(n_instances, self.class_vectors.shape[0])
        """
        if type(X).__name__ == 'csr_matrix':
            X = X.toarray()
        X_transformed = np.empty(shape=(0, self.class_vectors.shape[0]))
        for x in X:
            x_reduced = np.array([])
            for cv in self.class_vectors:
                x_reduced = np.append(x_reduced, self.distance(cv, x))
            X_transformed = np.vstack((X_transformed, x_reduced))
        return X_transformed

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Same as fit, followed by transform,
        but with slighlty better performance.
        :param X: np.ndarray, shape=(n_instances, n_features)
        :param y: labels, shape=(n_instances, 1)
        :return: np.array, shape(n_instances, self.class_vectors.shape[0])
        """
        self.fit(X, y)
        return self.X_training_reduced


#X = np.array([ [1,0,0,1,0], [0,1,1,0,0], [1,0,0,1,0], [0,1,0,0,1], [1,1,0,1,0] ])
#y = np.array([ [1], [1],[1],[2],[2] ])
#d = DCDistance(distance=cosine)
#d.fit(X, y)
#d.X_training_reduced
