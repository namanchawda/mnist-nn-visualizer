/** Core forward pass using explicit weights, kept importable without caches. */

export type Weights = {
  W1: Float64Array; // (784, 128) row-major
  b1: Float64Array; // (128)
  W2: Float64Array; // (128, 64) row-major
  b2: Float64Array; // (64)
  W3: Float64Array; // (64, 10) row-major
  b3: Float64Array; // (10)
};

export type ForwardResult = {
  probabilities: number[];
  predicted: number;
  hidden1: number[];
  hidden2: number[];
};

const INPUT_SIZE = 784;
const HIDDEN1_SIZE = 128;
const HIDDEN2_SIZE = 64;
const OUTPUT_SIZE = 10;

/** Plain-loop dense layer: y = ReLU(x @ W + b) or the logits form when relu=false. */
function dense(
  input: Float64Array,
  weights: Float64Array,
  biases: Float64Array,
  outputSize: number,
  appliesRelu: boolean,
): { activations: Float64Array; logits: Float64Array } {
  const output = new Float64Array(outputSize);
  // weights are row-major (in, out): output[j] = sum_i(input[i] * W[i][j]) + b[j]
  for (let i = 0; i < input.length; i += 1) {
    const value = input[i];
    if (value === 0) continue;
    const rowOffset = i * outputSize;
    for (let j = 0; j < outputSize; j += 1) {
      output[j] += value * weights[rowOffset + j];
    }
  }
  for (let j = 0; j < outputSize; j += 1) {
    output[j] += biases[j];
    if (appliesRelu && output[j] < 0) {
      output[j] = 0;
    }
  }
  return { activations: output, logits: output };
}

/** Numerically stable softmax (subtract the row maximum, like the backend). */
function softmax(logits: Float64Array): number[] {
  let max = logits[0];
  for (let index = 1; index < logits.length; index += 1) {
    if (logits[index] > max) max = logits[index];
  }
  const shifted = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    shifted[index] = Math.exp(logits[index] - max);
    total += shifted[index];
  }
  return Array.from(shifted, (value) => value / total);
}

/**
 * Exact client-side replication of NeuralNetwork.forward for a single sample.
 * Returns probabilities, the argmax digit, and both hidden ReLU layers.
 */
export function forwardWithWeights(input: Float64Array, weights: Weights): ForwardResult {
  if (input.length !== INPUT_SIZE) {
    throw new Error(`Expected an input of ${INPUT_SIZE} pixels, got ${input.length}`);
  }

  const { activations: hidden1 } = dense(input, weights.W1, weights.b1, HIDDEN1_SIZE, true);
  const { activations: hidden2 } = dense(hidden1, weights.W2, weights.b2, HIDDEN2_SIZE, true);
  const { activations: logits } = dense(hidden2, weights.W3, weights.b3, OUTPUT_SIZE, false);

  const probabilities = softmax(logits);
  let predicted = 0;
  for (let digit = 1; digit < OUTPUT_SIZE; digit += 1) {
    if (probabilities[digit] > probabilities[predicted]) predicted = digit;
  }

  return {
    probabilities,
    predicted,
    hidden1: Array.from(hidden1),
    hidden2: Array.from(hidden2),
  };
}
