"use client";

import { RefObject, useCallback, useEffect, useRef, useState } from "react";

import { forward, loadWeights } from "../lib/network";
import { preprocess } from "../lib/preprocess";

const HIDDEN1_SIZE = 128;
const HIDDEN2_SIZE = 64;

export type PredictionSnapshot = {
  /** Preprocessed 784-value MNIST input, or null when nothing is drawn. */
  input28x28: Float64Array | null;
  hidden1: number[];
  hidden2: number[];
  probabilities: number[];
  predicted: number | null;
};

const EMPTY_SNAPSHOT: PredictionSnapshot = {
  input28x28: null,
  hidden1: Array(HIDDEN1_SIZE).fill(0),
  hidden2: Array(HIDDEN2_SIZE).fill(0),
  probabilities: Array(10).fill(0),
  predicted: null,
};

export type UsePredictionResult = PredictionSnapshot & {
  /** True once weights.json has loaded and sample() runs the network. */
  ready: boolean;
  /** True when loading the weights failed. */
  error: boolean;
  /** Run preprocess + forward on the referenced canvas; true when a new prediction was produced. */
  sample: () => boolean;
  /** Clear all activations back to the idle state. */
  reset: () => void;
};

/**
 * Loads the client-side model once and runs the sample -> preprocess ->
 * forward loop against a drawing canvas.
 */
export function usePrediction(
  canvasRef: RefObject<HTMLCanvasElement | null>,
): UsePredictionResult {
  const readyRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(false);
  const [snapshot, setSnapshot] = useState<PredictionSnapshot>(EMPTY_SNAPSHOT);

  useEffect(() => {
    let cancelled = false;
    loadWeights()
      .then(() => {
        if (cancelled) return;
        readyRef.current = true;
        setReady(true);
      })
      .catch(() => {
        if (cancelled) return;
        setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const sample = useCallback((): boolean => {
    if (!readyRef.current) return false;
    const canvas = canvasRef.current;
    if (!canvas) return false;
    const input = preprocess(canvas);
    if (!input) return false; // blank canvas, nothing drawn yet

    const result = forward(input);
    setSnapshot({
      input28x28: input,
      hidden1: result.hidden1,
      hidden2: result.hidden2,
      probabilities: result.probabilities,
      predicted: result.predicted,
    });
    return true;
  }, [canvasRef]);

  const reset = useCallback(() => setSnapshot(EMPTY_SNAPSHOT), []);

  return { ...snapshot, ready, error, sample, reset };
}
