# MNIST Digit Recognizer — Neural Network Built From Scratch

A handwritten digit classifier where every part of the neural network — forward propagation, backpropagation, and gradient descent — is implemented by hand using only NumPy. No PyTorch, TensorFlow, or scikit-learn is used for the model itself. Paired with a live drawing canvas that samples predictions every 300ms while you draw, and a real-time visualization of the network's internal neuron activations. Inference and preprocessing run entirely in the browser — there is no prediction server at runtime.

**Live demo:** https://mnist-neural-network-from-scratch.vercel.app/

## Why build this

Frameworks like PyTorch hide the actual mechanics of how a neural network learns behind `.backward()` and `optimizer.step()`. This project exists to implement that mechanism directly — the matrix multiplications, the chain rule, the gradient updates — to demonstrate a working understanding of what's actually happening underneath every high-level ML framework.

## Architecture

```text
Next.js frontend (all inference runs here, fully client-side)
- Drawable canvas (280x280)
- Every 300ms while drawing: samples canvas, downsamples to 28x28
- Replicates MNIST-style centering in TypeScript (lib/preprocess.ts)
- Runs the forward pass in hand-written TypeScript (lib/network.ts)
- Live-updating probability bars (digits 0-9)
- Live neural network activation visualization (all 128+64+10 neurons)
- /weights.json: exported trained weights, loaded once at page mount

Training + export (offline, once)
- Downloads and parses the original MNIST binary format directly
- Trains a 784 -> 128 -> 64 -> 10 network with hand-written NumPy backprop
- export_weights.py converts the .npz into frontend/public/weights.json

Optional FastAPI backend (not needed at runtime)
- backend/main.py still wraps the NumPy model in a /predict endpoint
- useful for development/verification; the deployed app never calls it
```

## The network, precisely

- **Input:** 784 neurons (each 28x28 image flattened, normalized to [0,1])
- **Hidden layer 1:** 128 neurons, ReLU activation
- **Hidden layer 2:** 64 neurons, ReLU activation
- **Output:** 10 neurons, Softmax activation (probability distribution over digits 0-9)
- **Loss:** cross-entropy
- **Optimizer:** mini-batch gradient descent, implemented by hand

Trained for 15 epochs on the full 60,000-image MNIST training set, reaching **97.8% accuracy** on the 10,000-image held-out test set.

## A real bug: freehand drawings predicted poorly despite 97.8% test accuracy

Early testing showed the model performing noticeably worse on hand-drawn digits in the browser than its test-set accuracy suggested it should.

**Diagnosis:** MNIST's original images are centered by the digit's center of mass and sized to fit a specific 20x20 region within the 28x28 frame — a specific preprocessing convention baked into how the dataset was created. A digit drawn freehand on a canvas and simply downscaled to 28x28 doesn't match this convention at all — it could be off-center, a different size, or positioned anywhere in the frame. The network was never trained on that variation.

**Fix:** added a preprocessing step that finds the bounding box of the drawn strokes, crops to it, resizes to fit a 20x20 region preserving aspect ratio, and re-centers the result within a 28x28 frame using the pixel intensity center of mass — replicating MNIST's actual original preprocessing convention.

**Result:** noticeably more consistent and accurate predictions on hand-drawn digits, confirmed through direct testing before/after the fix.

## Tech stack

- **Model:** Python, NumPy only (no ML frameworks) — training and export
- **Client inference:** plain TypeScript with typed-array loops (no TensorFlow.js, ONNX, or other ML libraries — the from-scratch spirit continues in the browser)
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Optional dev server:** FastAPI (backend/main.py)
- **Data:** original MNIST binary format, parsed directly (IDX file format)

## Running locally

The app needs no prediction server — only the Next.js frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 and start drawing. The frontend loads the trained weights once from `/weights.json` (see `frontend/public/weights.json`), then runs preprocessing and inference entirely in your browser.

## Retraining and re-exporting weights

```bash
# 1. Train (downloads MNIST on first run, takes a few minutes)
python -m backend.train
# 2. Export the trained .npz to frontend/public/weights.json
python backend/export_weights.py
```

Then redeploy (or just refresh a local dev server) — the new weights are picked up on the next page load. Training requires `pip install -r backend/requirements.txt`.

## Verifying the client-side model matches NumPy

A parity check compares the TypeScript forward pass and preprocessing against the original NumPy implementation on MNIST test images:

```bash
python verify/expected.py        # generates NumPy reference outputs
# compile the frontend TS for Node (from frontend/):
npx tsc lib/network.ts lib/preprocess.ts --outDir ../verify/tmp \
  --module commonjs --target es2020 --moduleResolution node --skipLibCheck --types node
node verify/verify.cjs
```

Currently: probability differences < 1e-6, hidden activations < 1e-5, and the preprocessing port is bit-exact against `backend/preprocessing.py`.

## Known limitations

- The exported weights are rounded to 6 significant digits to keep `weights.json` small; verified prediction differences vs the NumPy model stay below 1e-6.
- The network operates on flattened pixel vectors with no convolutional structure, so it has no built-in translation invariance — this is why the centering preprocessing step matters as much as it does. A convolutional architecture would be more naturally robust to this, at the cost of being a meaningfully more complex from-scratch implementation.
- Training data and weights are fixed; there's no online learning from user-drawn digits.
