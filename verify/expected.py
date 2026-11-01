"""Generate reference outputs (NumPy) for the Node-side TS parity check.

Usage (from the repo root):
    python verify/expected.py

Writes expected.json to a temp dir; the absolute path is printed and stored in
verify/expected_path.txt for verify/verify.cjs to pick up.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from backend.data_loader import load_mnist_data
from backend.neural_network import NeuralNetwork
from backend.preprocessing import center_digit

SAMPLE_COUNT = 40


def main() -> None:
    weights_path = Path("backend/weights/mnist_weights.npz")
    model = NeuralNetwork()
    model.load_weights(weights_path)

    _, _, x_test, y_test = load_mnist_data()
    # float64 gives exact parity with JavaScript doubles.
    images = x_test[:SAMPLE_COUNT].astype(np.float64)
    labels = np.argmax(y_test[:SAMPLE_COUNT], axis=1).tolist()

    cache = model.forward(images)

    # Preprocessing reference on the same 28x28 images (these mimic what the
    # browser's drawImage downscale produces before center_digit runs).
    centered = np.stack([center_digit(image) for image in images])

    expected = {
        "images": images.tolist(),
        "labels": labels,
        "probabilities": cache["a3"].astype(np.float64).tolist(),
        "hidden1": cache["a1"].astype(np.float64).tolist(),
        "hidden2": cache["a2"].astype(np.float64).tolist(),
        "centered": centered.tolist(),
    }

    output_path = Path(tempfile.gettempdir()) / "mnist_ts_verify_expected.json"
    output_path.write_text(json.dumps(expected))
    Path(__file__).parent.joinpath("expected_path.txt").write_text(str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
