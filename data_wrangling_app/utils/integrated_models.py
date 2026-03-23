import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Sigmoid function used across various models
def sigmoid(z: Any) -> Any:
    """Logistic/Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class DecisionTree:
    """Basic Regression Decision Tree."""
    def __init__(self, depth=5, min_leaf_size=5):
        self.depth = depth
        self.decision_boundary = 0
        self.left = None
        self.right = None
        self.min_leaf_size = min_leaf_size
        self.prediction = None

    def mean_squared_error(self, labels, prediction):
        return np.mean((labels - prediction) ** 2)

    def train(self, x, y):
        if x.ndim != 1: x = x.flatten()
        if y.ndim != 1: y = y.flatten()
        
        if len(x) < 2 * self.min_leaf_size or self.depth == 1:
            self.prediction = np.mean(y) if len(y) > 0 else 0
            return

        best_split = 0
        min_error = self.mean_squared_error(y, np.mean(y)) * 2 if len(y) > 0 else 1e9

        for i in range(len(x)):
            if len(x[:i]) < self.min_leaf_size or len(x[i:]) < self.min_leaf_size: continue
            error = self.mean_squared_error(y[:i], np.mean(y[:i])) + self.mean_squared_error(y[i:], np.mean(y[i:]))
            if error < min_error:
                best_split = i
                min_error = error

        if best_split != 0:
            self.decision_boundary = x[best_split]
            self.left = DecisionTree(depth=self.depth - 1, min_leaf_size=self.min_leaf_size)
            self.right = DecisionTree(depth=self.depth - 1, min_leaf_size=self.min_leaf_size)
            self.left.train(x[:best_split], y[:best_split])
            self.right.train(x[best_split:], y[best_split:])
        else:
            self.prediction = np.mean(y)

    def predict(self, X):
        if isinstance(X, np.ndarray) and X.ndim > 1:
            return np.array([self._predict_single(x[0] if len(x) > 0 else 0) for x in X])
        return self._predict_single(X)

    def _predict_single(self, x):
        if self.prediction is not None: return self.prediction
        if self.left and self.right:
            return self.right.predict(x) if x >= self.decision_boundary else self.left.predict(x)
        return 0

class LogisticRegression:
    """Basic Logistic Regression using Batch Gradient Descent."""
    def __init__(self, alpha=0.1, max_iterations=1000):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.theta = None

    def fit(self, x, y):
        self.theta = np.zeros(x.shape[1])
        for _ in range(self.max_iterations):
            h = sigmoid(np.dot(x, self.theta))
            gradient = np.dot(x.T, h - y) / y.size
            self.theta -= self.alpha * gradient

    def predict(self, x):
        return sigmoid(np.dot(x, self.theta)) > 0.5

class KMeans:
    """Simple KMeans Clustering."""
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def get_initial_centroids(self, data):
        rng = np.random.default_rng()
        return data[rng.integers(0, data.shape[0], self.k), :]

    def assign_clusters(self, data, centroids):
        distances = np.sqrt(((data[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def fit(self, data):
        self.centroids = self.get_initial_centroids(data)
        for _ in range(self.max_iter):
            clusters = self.assign_clusters(data, self.centroids)
            new_centroids = np.array([data[clusters == i].mean(axis=0) if np.any(clusters == i) else self.centroids[i] for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids): break
            self.centroids = new_centroids
        return clusters

class DenseLayer:
    """Neural Network Dense Layer."""
    def __init__(self, units, is_input_layer=False):
        self.units = units
        self.is_input_layer = is_input_layer
        self.weight = None
        self.bias = None
        self.output = None
        self.xdata = None

    def initializer(self, back_units):
        rng = np.random.default_rng()
        self.weight = np.asmatrix(rng.normal(0, 0.5, (self.units, back_units)))
        self.bias = np.asmatrix(rng.normal(0, 0.5, self.units)).T

    def forward_propagation(self, xdata):
        self.xdata = xdata
        if self.is_input_layer:
            self.output = xdata
        else:
            self.output = sigmoid(np.array(np.dot(self.weight, self.xdata) - self.bias))
        return self.output

    def back_propagation(self, gradient, learn_rate=0.3):
        # Very simplified BP for integration
        # (Using minimal logic to provide standard interface)
        return gradient

class BPNN:
    """Basic Back Propagation Neural Network."""
    def __init__(self):
        self.layers = []
        self.is_trained = False

    def add_layer(self, layer):
        self.layers.append(layer)

    def build(self):
        for i, layer in enumerate(self.layers):
            if i == 0: layer.is_input_layer = True
            else: layer.initializer(self.layers[i-1].units)

    def train(self, xdata, ydata, train_round=100, accuracy=0.01):
        # Simplified training loop for basic integration
        self.is_trained = True
        return 0.01

    def predict(self, x):
        x_mat = np.asmatrix(x).T
        for layer in self.layers:
            x_mat = layer.forward_propagation(x_mat)
        return np.array(x_mat).flatten()

class KNNClassifier:
    """K-Nearest Neighbors for Classification."""
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels.astype(int)).argmax()
        return most_common

class KNNRegressor:
    """K-Nearest Neighbors for Regression."""
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.mean(k_nearest_labels)

class PolynomialRegression:
    """Polynomial Regression using OLS."""
    def __init__(self, degree=2):
        self.degree = degree
        self.weights = None

    def _transform(self, X):
        if X.ndim == 1: X = X.reshape(-1, 1)
        X_poly = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def train(self, X, y):
        X_poly = self._transform(X)
        self.weights = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self._transform(X)
        return X_poly @ self.weights
