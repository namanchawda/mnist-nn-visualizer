"""MNIST-style preprocessing for freehand digit images."""

from __future__ import annotations

import numpy as np


def center_digit(pixels: np.ndarray) -> np.ndarray:
    """Crop, resize, and center a normalized flat 28x28 digit image.

    MNIST digits are centered and sized consistently, while freehand canvas
    input may be small or drawn near an edge. Matching that convention helps
    the trained classifier see a more familiar input distribution.
    """
    original = np.asarray(pixels)
    image = original.reshape(28, 28)

    # Ignore faint canvas noise when locating the actual digit.
    rows, columns = np.where(image > 0.05)
    if rows.size == 0:
        # A blank canvas should remain blank and must not enter crop math.
        return original.copy()

    # Crop away empty margins so the available 20x20 area is used efficiently.
    cropped = image[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    crop_height, crop_width = cropped.shape

    # Scale the longest side to 20 while preserving the digit's aspect ratio.
    scale = 20.0 / max(crop_height, crop_width)
    resized_height = max(1, int(round(crop_height * scale)))
    resized_width = max(1, int(round(crop_width * scale)))

    # NumPy-only nearest-neighbor resize: map each output coordinate to the
    # corresponding source coordinate. This avoids adding a resizing library.
    row_indices = np.minimum(
        (np.arange(resized_height) / scale).astype(int), crop_height - 1
    )
    column_indices = np.minimum(
        (np.arange(resized_width) / scale).astype(int), crop_width - 1
    )
    resized = cropped[np.ix_(row_indices, column_indices)]

    # Calculate the intensity-weighted center of mass in resized-image coords.
    # This centers the visual ink, rather than merely centering its rectangle.
    total_intensity = float(resized.sum())
    if total_intensity == 0:
        return original.copy()
    y_grid, x_grid = np.indices(resized.shape)
    centroid_y = float((y_grid * resized).sum() / total_intensity)
    centroid_x = float((x_grid * resized).sum() / total_intensity)

    output = np.zeros((28, 28), dtype=resized.dtype)
    # Position the resized digit so its centroid is nearest to (14, 14).
    top = int(round(14.0 - centroid_y))
    left = int(round(14.0 - centroid_x))

    # Clip source and destination windows if rounding puts an edge outside the
    # canvas. Normally the 20x20 maximum makes this unnecessary, but it keeps
    # the function safe for every valid crop shape.
    dst_top = max(0, top)
    dst_left = max(0, left)
    src_top = max(0, -top)
    src_left = max(0, -left)
    height = min(resized_height - src_top, 28 - dst_top)
    width = min(resized_width - src_left, 28 - dst_left)
    if height > 0 and width > 0:
        output[dst_top:dst_top + height, dst_left:dst_left + width] = resized[
            src_top:src_top + height, src_left:src_left + width
        ]

    return output.reshape(784)
