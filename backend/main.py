"""FastAPI application serving predictions from the trained NumPy model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models import PredictRequest, PredictResponse
from backend.neural_network import NeuralNetwork
from backend.preprocessing import center_digit


WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "mnist_weights.npz"

# Load the model and weights once when the application module starts. Inference
# requests reuse this in-memory model instead of repeatedly reading the archive.
model = NeuralNetwork()
model.load_weights(WEIGHTS_PATH)

app = FastAPI(title="MNIST Digit Recognizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mnist-neural-network-from-scratch.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple service health response."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict a digit from 784 frontend-normalized grayscale pixels."""
    pixels = center_digit(np.asarray(payload.pixels, dtype=np.float32)).reshape(1, 784)
    cache = model.forward(pixels)
    probabilities = cache["a3"][0]
    predicted_digit = int(np.argmax(probabilities))

    return PredictResponse(
        probabilities=probabilities.tolist(),
        predicted_digit=predicted_digit,
        hidden1_activations=cache["a1"][0].tolist(),
        hidden2_activations=cache["a2"][0].tolist(),
    )
