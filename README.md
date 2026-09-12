# MNIST Digit Recognizer — Neural Network Built From Scratch

A handwritten digit classifier where every part of the neural network — forward propagation, backpropagation, and gradient descent — is implemented by hand using only NumPy. No PyTorch, TensorFlow, or scikit-learn is used for the model itself. Paired with a live drawing canvas that samples predictions every 300ms while you draw, and a real-time visualization of the network's internal neuron activations.

**Live demo:** https://mnist-nn-visualizer-seven.vercel.app/

## Why build this

Frameworks like PyTorch hide the actual mechanics of how a neural network learns behind `.backward()` and `optimizer.step()`. This project exists to implement that mechanism directly — the matrix multiplications, the chain rule, the gradient updates — to demonstrate a working understanding of what's actually happening underneath every high-level ML framework.

## Architecture

```text
Next.js frontend
- Drawable canvas (280x280, downscaled to 28x28)
- Every 300ms while drawing: samples canvas, sends to backend
- Live-updating probability bars (digits 0-9)
- Live neural network activation visualization (all 128+64+10 neurons)
↓ HTTP
FastAPI backend
- Loads trained weights (.npz) once at startup
- Centers/normalizes the incoming digit to match MNIST's training convention
- Runs inference through the from-scratch forward pass
- Returns probabilities, predicted digit, and hidden-layer activations
↑
Training (offline, once)
- Downloads and parses the original MNIST binary format directly
- Trains a 784 -> 128 -> 64 -> 10 network with hand-written backprop
- Saves the learned weights
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

- **Model:** Python, NumPy only (no ML frameworks)
- **Backend:** FastAPI
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Data:** original MNIST binary format, parsed directly (IDX file format)

## Running locally

**Backend:**

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Train the model (downloads MNIST on first run, takes a few minutes):

```bash
cd ..
python -m backend.train
```

Start the API:

```bash
uvicorn backend.main:app --reload
```

**Frontend** (in a separate terminal):

```bash
cd frontend
npm install
cp .env.local.example .env.local
# edit .env.local if your backend runs somewhere other than http://127.0.0.1:8000
npm run dev
```

Open http://localhost:3000 and start drawing.

## Known limitations

- The deployed backend runs on a free-tier hosting service that spins down after inactivity — the first prediction after idle time may take up to a minute while it wakes up.
- The network operates on flattened pixel vectors with no convolutional structure, so it has no built-in translation invariance — this is why the centering preprocessing step matters as much as it does. A convolutional architecture would be more naturally robust to this, at the cost of being a meaningfully more complex from-scratch implementation.
- Training data and weights are fixed; there's no online learning from user-drawn digits.
