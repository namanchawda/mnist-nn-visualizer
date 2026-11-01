/**
 * Node-side parity check between the original NumPy forward pass and this
 * codebase's TypeScript implementation.
 *
 * Usage (after `python verify/expected.py` generated expected_path.txt):
 *   1. Compile frontend TS to CommonJS for Node:
 *        npx tsc frontend/lib/network.ts frontend/lib/preprocess.ts
 *            --outDir verify/tmp --module commonjs --target es2020
 *            --moduleResolution node --skipLibCheck
 *        (relative-command: run from frontend/ with ../verify paths)
 *   2. node verify/verify.cjs
 */

const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const expectedPath = fs
  .readFileSync(path.join(__dirname, "expected_path.txt"), "utf8")
  .trim();
const expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));

const forwardModule = require("./tmp/network.js");
const preprocessModule = require("./tmp/preprocess.js");

const weights = forwardModule.parseWeights(
  JSON.parse(fs.readFileSync(path.join(repoRoot, "frontend", "public", "weights.json"), "utf8")),
);
const { forwardWithWeights } = forwardModule;
const { centerDigit28 } = preprocessModule;

const PROBABILITY_TOLERANCE = 1e-4;
const HIDDEN_TOLERANCE = 1e-4;

let worstProbability = 0;
let worstHidden = 0;
let failures = 0;
const labelMismatches = [];

expected.images.forEach((image, index) => {
  const result = forwardWithWeights(Float64Array.from(image), weights);

  const label = expected.labels[index];
  if (result.predicted !== label) {
    labelMismatches.push(`#${index}: TS=${result.predicted} MNIST label=${label}`);
    failures += 1;
  }

  expected.probabilities[index].forEach((reference, digit) => {
    worstProbability = Math.max(worstProbability, Math.abs(reference - result.probabilities[digit]));
  });
  result.hidden1.forEach((_, neuron) => {
    worstHidden = Math.max(worstHidden, Math.abs(expected.hidden1[index][neuron] - result.hidden1[neuron]));
  });
  result.hidden2.forEach((_, neuron) => {
    worstHidden = Math.max(worstHidden, Math.abs(expected.hidden2[index][neuron] - result.hidden2[neuron]));
  });
});

// Preprocessing parity: TS centerDigit28 vs python center_digit on 28x28
// inputs. Same doubles in, so any nonzero delta is a porting bug.
let preprocessingDeltas = 0;
let maxPreprocessDelta = 0;
expected.centered.forEach((centered, index) => {
  const ts = centerDigit28(Float64Array.from(expected.images[index]));
  ts.forEach((value, pixel) => {
    const delta = Math.abs(value - centered[pixel]);
    if (delta !== 0) {
      preprocessingDeltas += 1;
      maxPreprocessDelta = Math.max(maxPreprocessDelta, delta);
    }
  });
});

console.log(`samples: ${expected.images.length}`);
console.log(`max |Δprobability| vs NumPy: ${worstProbability}`);
console.log(`max |Δhidden| vs NumPy:      ${worstHidden}`);
console.log(`predicted-digit mismatches:  ${failures}`);
console.log(
  `preprocessing pixels differing from Python center_digit: ${preprocessingDeltas}/784*40 (max Δ ${maxPreprocessDelta})`,
);

const probabilityOk = worstProbability < PROBABILITY_TOLERANCE;
const hiddenOk = worstHidden < HIDDEN_TOLERANCE;
const preprocessOk = preprocessingDeltas === 0;

if (!(probabilityOk && hiddenOk && preprocessOk)) {
  console.error("PARITY CHECK FAILED");
  process.exit(1);
}
console.log("PARITY CHECK PASSED (probs < 1e-4, hidden < 1e-4, preprocessing exact)");
