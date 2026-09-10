"""Download, cache, and preprocess the MNIST dataset using NumPy."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(filename: str, data_dir: Path) -> Path:
    """Download one official MNIST archive unless it is already cached."""
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / filename
    if not destination.exists():
        urlretrieve(MNIST_BASE_URL + filename, destination)
    return destination


def _read_images(path: Path) -> np.ndarray:
    """Read an IDX image archive into an (count, rows, columns) uint8 array."""
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    return data.reshape(count, rows, columns)


def _read_labels(path: Path) -> np.ndarray:
    """Read an IDX label archive into a (count,) uint8 array."""
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    return data.reshape(count)


def load_mnist_data(
    data_dir: str | Path = Path(__file__).resolve().parent / "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST if needed and return normalized images and one-hot labels.

    The official training and test splits are retained. Images are flattened
    to 784 features and scaled to [0, 1]; labels have shape (count, 10).
    """
    data_dir = Path(data_dir)
    paths = {key: _download_if_missing(filename, data_dir) for key, filename in FILES.items()}
    return (
        preprocess_images(_read_images(paths["train_images"])),
        preprocess_labels(_read_labels(paths["train_labels"])),
        preprocess_images(_read_images(paths["test_images"])),
        preprocess_labels(_read_labels(paths["test_labels"])),
    )


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Flatten images and normalize uint8 pixels from [0, 255] to [0, 1]."""
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0


def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    """Convert integer digit labels to ten-column one-hot float arrays."""
    encoded = np.zeros((labels.shape[0], 10), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return encoded
