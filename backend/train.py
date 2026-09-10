"""Train and evaluate the NumPy-only MNIST neural network."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.data_loader import load_mnist_data
from backend.neural_network import NeuralNetwork


def evaluate_model(model: NeuralNetwork, images: np.ndarray, labels: np.ndarray) -> float:
    """Return classification accuracy for a dataset of images and one-hot labels."""
    probabilities = model.forward(images)["a3"]
    predictions = np.argmax(probabilities, axis=1)
    expected = np.argmax(labels, axis=1)
    return float(np.mean(predictions == expected))


def train_model(
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.1,
    data_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> NeuralNetwork:
    """Train on MNIST with mini-batch gradient descent and save the model."""
    loader_args = {} if data_dir is None else {"data_dir": data_dir}
    x_train, y_train, x_test, y_test = load_mnist_data(**loader_args)
    model = NeuralNetwork()

    # 0.1 is a useful starting point for this network; tune it if loss stalls.
    rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(x_train.shape[0])
        x_train, y_train = x_train[order], y_train[order]
        losses = []

        for start in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            cache = model.forward(x_batch)
            losses.append(model.compute_loss(cache["a3"], y_batch))
            gradients = model.backward(x_batch, y_batch, cache)
            model.update(gradients, learning_rate)

        accuracy = evaluate_model(model, x_test, y_test)
        print(f"Epoch {epoch}/{epochs} - loss: {np.mean(losses):.4f} - test accuracy: {accuracy:.4f}")

    if weights_path is None:
        weights_path = Path(__file__).resolve().parent / "weights" / "mnist_weights.npz"
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)
    print(f"Saved weights to {weights_path}")
    return model


def main() -> None:
    """Run the default end-to-end MNIST training job."""
    train_model()


if __name__ == "__main__":
    main()
