"""Pydantic request and response schemas for the prediction API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request containing one normalized, flattened 28x28 image."""

    pixels: List[float] = Field(
        ...,
        min_length=784,
        max_length=784,
        description="Exactly 784 grayscale pixel values normalized to [0, 1].",
    )


class PredictResponse(BaseModel):
    """Response schema for digit prediction."""

    probabilities: List[float] = Field(
        ..., description="Ten class probabilities for digits 0 through 9."
    )
    predicted_digit: int = Field(..., description="Predicted digit class.")
    hidden1_activations: List[float] = Field(
        ..., description="128 activations from the first hidden layer."
    )
    hidden2_activations: List[float] = Field(
        ..., description="64 activations from the second hidden layer."
    )
