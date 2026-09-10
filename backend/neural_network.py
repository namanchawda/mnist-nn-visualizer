"""NumPy-only feedforward neural network for MNIST classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


class NeuralNetwork:
    """784 -> 128 -> 64 -> 10 network with ReLU and softmax activations."""

    def __init__(self, input_size: int = 784, hidden1_size: int = 128,
                 hidden2_size: int = 64, output_size: int = 10) -> None:
        """Initialize He-scaled weights and zero biases for every layer."""
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        # He initialization is suited to ReLU layers and uses fan-in scaling.
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """Apply ReLU elementwise, preserving the input shape."""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Return 1 where z is positive and 0 everywhere else."""
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using a stable row-wise softmax."""
        # Subtracting the row maximum prevents exp overflow without changing probabilities.
        shifted = z - np.max(z, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Run forward propagation and return intermediates needed by backward."""
        z1 = x @ self.W1 + self.b1  # x (batch, 784) -> z1 (batch, 128)
        a1 = self.relu(z1)          # (batch, 128)
        z2 = a1 @ self.W2 + self.b2 # (batch, 64)
        a2 = self.relu(z2)          # (batch, 64)
        z3 = a2 @ self.W3 + self.b3 # (batch, 10), logits
        a3 = self.softmax(z3)       # (batch, 10), probabilities
        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}

    @staticmethod
    def compute_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Return mean one-hot cross-entropy, with epsilon avoiding log(0)."""
        return float(-np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1)))

    def backward(self, x: np.ndarray, y_true: np.ndarray,
                 cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute all six parameter gradients manually via the chain rule."""
        batch_size = x.shape[0]
        z1, a1 = cache["z1"], cache["a1"]
        z2, a2, a3 = cache["z2"], cache["a2"], cache["a3"]
        # Softmax plus cross-entropy simplifies to probabilities minus labels.
        dz3 = a3 - y_true                         # (batch, 10)
        dW3 = (a2.T @ dz3) / batch_size            # (64, 10)
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size  # (1, 10)
        # Chain rule through W3 and ReLU layer 2; ReLU gates inactive units.
        da2 = dz3 @ self.W3.T                     # (batch, 64)
        dz2 = da2 * self.relu_derivative(z2)       # (batch, 64)
        dW2 = (a1.T @ dz2) / batch_size            # (128, 64)
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size  # (1, 64)
        # Chain rule through W2 and ReLU layer 1.
        da1 = dz2 @ self.W2.T                     # (batch, 128)
        dz1 = da1 * self.relu_derivative(z1)       # (batch, 128)
        dW1 = (x.T @ dz1) / batch_size             # (784, 128)
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size  # (1, 128)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def update(self, gradients: Dict[str, np.ndarray], learning_rate: float) -> None:
        """Apply one standard gradient-descent update to each parameter."""
        self.W1 -= learning_rate * gradients["dW1"]
        self.b1 -= learning_rate * gradients["db1"]
        self.W2 -= learning_rate * gradients["dW2"]
        self.b2 -= learning_rate * gradients["db2"]
        self.W3 -= learning_rate * gradients["dW3"]
        self.b3 -= learning_rate * gradients["db3"]

    def save_weights(self, path: str | Path) -> None:
        """Save all six parameter arrays in a NumPy .npz archive."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load_weights(self, path: str | Path) -> None:
        """Restore all six parameter arrays from a NumPy .npz archive."""
        with np.load(path) as weights:
            self.W1, self.b1 = weights["W1"], weights["b1"]
            self.W2, self.b2 = weights["W2"], weights["b2"]
            self.W3, self.b3 = weights["W3"], weights["b3"]
