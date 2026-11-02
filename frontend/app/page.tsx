"use client";

import Link from "next/link";
import { useRef } from "react";

import NetworkVisualization from "./NetworkVisualization";
import DrawCanvas, { DrawCanvasHandle } from "../components/DrawCanvas";
import { usePrediction } from "../hooks/usePrediction";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawCanvasRef = useRef<DrawCanvasHandle>(null);
  const {
    ready,
    error,
    hidden1: hidden1Activations,
    hidden2: hidden2Activations,
    probabilities,
    predicted: predictedDigit,
    sample,
    reset,
  } = usePrediction(canvasRef);

  const clearCanvas = () => {
    drawCanvasRef.current?.clear();
    reset();
  };

  return (
    <main className="min-h-screen bg-[#11110f] px-5 py-12 text-[#f4f0e8]">
      <div className="mx-auto max-w-2xl">
        <p className="mb-3 font-mono text-xs uppercase tracking-[0.3em] text-orange-400">MNIST / LIVE INFERENCE</p>
        <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">Draw a digit.</h1>
        <p className="mt-4 max-w-lg text-[#a8a49c]">A handwritten digit is sampled every 300ms and passed through a neural network built from scratch.</p>

        <section className="mt-10 flex flex-col items-center rounded-3xl border border-white/10 bg-[#1b1b18] p-6 shadow-2xl sm:p-9">
          <DrawCanvas ref={drawCanvasRef} canvasRef={canvasRef} onSample={sample} />
          <button onClick={clearCanvas} className="mt-5 rounded-full border border-white/20 px-6 py-2 text-sm font-semibold transition hover:border-orange-400 hover:text-orange-300">Clear</button>
          {ready && <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.2em] text-[#8e8b84]">Model loaded — running in your browser</p>}
        </section>

        <section className="mt-10">
          <NetworkVisualization
            hidden1Activations={hidden1Activations}
            hidden2Activations={hidden2Activations}
            probabilities={probabilities}
            predictedDigit={predictedDigit}
          />
        </section>

        <section className="mt-10">
          <div className="mb-7 flex items-end justify-between border-b border-white/10 pb-5">
            <span className="text-sm uppercase tracking-widest text-[#8e8b84]">Recognized digit</span>
            <span className="text-8xl font-bold leading-none text-orange-400">{predictedDigit ?? "-"}</span>
          </div>
          <div className="space-y-3">
            {probabilities.map((probability, digit) => (
              <div key={digit} className="flex items-center gap-4 font-mono text-sm">
                <span className="w-4 text-[#bdb8ae]">{digit}</span>
                <div className="h-3 flex-1 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-orange-400 transition-[width] duration-300 ease-out" style={{ width: `${probability * 100}%` }} />
                </div>
                <span className="w-16 text-right text-[#8e8b84]">{(probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          {error && (
            <p className="mt-5 text-sm text-red-400">
              Failed to load the model weights (weights.json). Try refreshing the page.
            </p>
          )}
          {!ready && !error && (
            <p className="mt-5 text-sm text-[#8e8b84]">Loading model…</p>
          )}
        </section>

        <p className="mt-10 text-center">
          <Link href="/3d" className="font-mono text-xs uppercase tracking-[0.25em] text-[#a8a49c] transition hover:text-orange-300">
            View the network in 3D →
          </Link>
        </p>
      </div>
    </main>
  );
}
