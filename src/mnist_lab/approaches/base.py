"""Common interface that every approach in MNIST Lab must implement."""

from abc import ABC, abstractmethod

import numpy as np


class Approach(ABC):
    """Base class for all approaches (MLP, CNN, NumPy MLP, baselines, ...).

    Data contract shared by all approaches:
        x: images, np.ndarray of shape (n, 28, 28), dtype float32, values in [0, 1]
        y: labels, np.ndarray of shape (n,), dtype int64, values 0-9
    """

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the approach on the training data.

        Args:
            x_train: training images, shape (n, 28, 28).
            y_train: training labels, shape (n,).
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a batch of images.

        Args:
            x: images, shape (n, 28, 28).

        Returns:
            Probabilities, shape (n, 10). Each row sums to 1.
        """
