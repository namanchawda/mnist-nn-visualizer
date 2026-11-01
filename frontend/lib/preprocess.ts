/**
 * Client-side replication of backend/preprocessing.py (center_digit) plus the
 * browser-side 280x280 -> 28x28 downscale the old backend relied on.
 *
 * Pipeline:
 *  1. drawImage the 280x280 drawing canvas into an offscreen 28x28 canvas
 *     (identical to the previous frontend sampling code), reading red/255.
 *  2. bounding box where pixel > 0.05 -> crop
 *  3. resize longest side to 20px, nearest-neighbor, preserving aspect ratio
 *  4. place in a zeroed 28x28 frame shifted so the intensity center of mass
 *     sits nearest to (14, 14)
 * Returns null when the canvas has no strokes.
 */

const MNIST_SIZE = 28;
const STROKE_THRESHOLD = 0.05;
const TARGET_REGION = 20;
const FRAME_CENTER = 14;

/** Python's round(): ties round half to even, unlike Math.round. */
function roundTiesToEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Exact nearest-neighbor resize port: (crop x scale).astype(trunc) clip. */
function resizeNearest(
  crop: Float64Array,
  cropHeight: number,
  cropWidth: number,
  resizedHeight: number,
  resizedWidth: number,
  scale: number,
): Float64Array {
  const resized = new Float64Array(resizedHeight * resizedWidth);
  for (let y = 0; y < resizedHeight; y += 1) {
    const sourceRow = Math.min(Math.trunc(y / scale), cropHeight - 1);
    for (let x = 0; x < resizedWidth; x += 1) {
      const sourceColumn = Math.min(Math.trunc(x / scale), cropWidth - 1);
      resized[y * resizedWidth + x] = crop[sourceRow * cropWidth + sourceColumn];
    }
  }
  return resized;
}

/** Exact port of backend/preprocessing.py:center_digit for a 28x28 flat image. */
export function centerDigit28(pixels: Float64Array): Float64Array {
  const output = new Float64Array(MNIST_SIZE * MNIST_SIZE);

  let rowMin = -1;
  let rowMax = -1;
  let columnMin = -1;
  let columnMax = -1;
  for (let index = 0; index < pixels.length; index += 1) {
    if (pixels[index] <= STROKE_THRESHOLD) continue;
    const row = Math.trunc(index / MNIST_SIZE);
    const column = index % MNIST_SIZE;
    if (rowMin === -1 || row < rowMin) rowMin = row;
    if (rowMax < row) rowMax = row;
    if (columnMin === -1 || column < columnMin) columnMin = column;
    if (columnMax < column) columnMax = column;
  }
  if (rowMin === -1) {
    return new Float64Array(pixels);
  }

  const cropHeight = rowMax - rowMin + 1;
  const cropWidth = columnMax - columnMin + 1;
  const crop = new Float64Array(cropHeight * cropWidth);
  for (let row = 0; row < cropHeight; row += 1) {
    for (let column = 0; column < cropWidth; column += 1) {
      crop[row * cropWidth + column] =
        pixels[(rowMin + row) * MNIST_SIZE + (columnMin + column)];
    }
  }

  const scale = TARGET_REGION / Math.max(cropHeight, cropWidth);
  const resizedHeight = Math.max(1, roundTiesToEven(cropHeight * scale));
  const resizedWidth = Math.max(1, roundTiesToEven(cropWidth * scale));
  const resized = resizeNearest(crop, cropHeight, cropWidth, resizedHeight, resizedWidth, scale);

  let totalIntensity = 0;
  let weightedRow = 0;
  let weightedColumn = 0;
  for (let row = 0; row < resizedHeight; row += 1) {
    for (let column = 0; column < resizedWidth; column += 1) {
      const intensity = resized[row * resizedWidth + column];
      totalIntensity += intensity;
      weightedRow += row * intensity;
      weightedColumn += column * intensity;
    }
  }
  if (totalIntensity === 0) {
    return new Float64Array(pixels);
  }

  const centroidRow = weightedRow / totalIntensity;
  const centroidColumn = weightedColumn / totalIntensity;
  const top = roundTiesToEven(FRAME_CENTER - centroidRow);
  const left = roundTiesToEven(FRAME_CENTER - centroidColumn);

  const destinationTop = Math.max(0, top);
  const destinationLeft = Math.max(0, left);
  const sourceTop = Math.max(0, -top);
  const sourceLeft = Math.max(0, -left);
  const height = Math.min(resizedHeight - sourceTop, MNIST_SIZE - destinationTop);
  const width = Math.min(resizedWidth - sourceLeft, MNIST_SIZE - destinationLeft);
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      output[(destinationTop + row) * MNIST_SIZE + (destinationLeft + column)] =
        resized[(sourceTop + row) * resizedWidth + (sourceLeft + column)];
    }
  }

  return output;
}

/** Sample the drawing canvas to a normalized 28x28 flat image (red / 255). */
function sampleMnist28(canvas: HTMLCanvasElement): Float64Array {
  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = MNIST_SIZE;
  smallCanvas.height = MNIST_SIZE;
  const smallContext = smallCanvas.getContext("2d");
  if (!smallContext) throw new Error("2D canvas context unavailable");
  smallContext.drawImage(canvas, 0, 0, MNIST_SIZE, MNIST_SIZE);

  const { data } = smallContext.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE);
  const pixels = new Float64Array(MNIST_SIZE * MNIST_SIZE);
  for (let index = 0; index < pixels.length; index += 1) {
    pixels[index] = data[index * 4] / 255;
  }
  return pixels;
}

/** Full pipeline: canvas -> 784-value MNIST input, or null for a blank canvas. */
export function preprocess(canvas: HTMLCanvasElement): Float64Array | null {
  const sampled = sampleMnist28(canvas);
  const hasStrokes = sampled.some((value) => value > STROKE_THRESHOLD);
  if (!hasStrokes) return null;
  return centerDigit28(sampled);
}
