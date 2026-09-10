"use client";

import { useEffect, useRef, useState } from "react";

import NetworkVisualization from "./NetworkVisualization";

const CANVAS_SIZE = 280;
const MNIST_SIZE = 28;
const API_URL = `${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/predict`;

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const drawingRef = useRef(false);
  const [probabilities, setProbabilities] = useState<number[]>(Array(10).fill(0));
  const [hidden1Activations, setHidden1Activations] = useState<number[]>(Array(128).fill(0));
  const [hidden2Activations, setHidden2Activations] = useState<number[]>(Array(64).fill(0));
  const [predictedDigit, setPredictedDigit] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = "#050505";
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const predict = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // The small canvas lets the browser perform the MNIST-sized downsampling.
    const smallCanvas = document.createElement("canvas");
    smallCanvas.width = MNIST_SIZE;
    smallCanvas.height = MNIST_SIZE;
    const smallContext = smallCanvas.getContext("2d");
    if (!smallContext) return;
    smallContext.drawImage(canvas, 0, 0, MNIST_SIZE, MNIST_SIZE);

    const pixels = smallContext.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE).data;
    const normalized = Array.from({ length: MNIST_SIZE * MNIST_SIZE }, (_, index) =>
      pixels[index * 4] / 255,
    );

    fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: normalized }),
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Backend returned ${response.status}`);
        return response.json() as Promise<{
          probabilities: number[];
          predicted_digit: number;
          hidden1_activations: number[];
          hidden2_activations: number[];
        }>;
      })
      .then((result) => {
        setProbabilities(result.probabilities);
        setHidden1Activations(result.hidden1_activations);
        setHidden2Activations(result.hidden2_activations);
        setPredictedDigit(result.predicted_digit);
        setError("");
      })
      .catch(() => setError("Could not reach the prediction server. Is FastAPI running?"));
  };

  const stopDrawing = () => {
    drawingRef.current = false;
    lastPointRef.current = null;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const pointFromEvent = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const bounds = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * CANVAS_SIZE,
      y: ((event.clientY - bounds.top) / bounds.height) * CANVAS_SIZE,
    };
  };

  const startDrawing = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    canvas.setPointerCapture(event.pointerId);
    const point = pointFromEvent(event);
    drawingRef.current = true;
    lastPointRef.current = point;
    context.fillStyle = "white";
    context.beginPath();
    context.arc(point.x, point.y, 9, 0, Math.PI * 2);
    context.fill();
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(predict, 300);
  };

  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    const currentPoint = pointFromEvent(event);
    if (!canvas || !context || !lastPointRef.current) return;
    context.strokeStyle = "white";
    context.lineWidth = 18;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(lastPointRef.current.x, lastPointRef.current.y);
    context.lineTo(currentPoint.x, currentPoint.y);
    context.stroke();
    lastPointRef.current = currentPoint;
  };

  const clearCanvas = () => {
    stopDrawing();
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) {
      context.fillStyle = "#050505";
      context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
    setProbabilities(Array(10).fill(0));
    setHidden1Activations(Array(128).fill(0));
    setHidden2Activations(Array(64).fill(0));
    setPredictedDigit(null);
    setError("");
  };

  return (
    <main className="min-h-screen bg-[#11110f] px-5 py-12 text-[#f4f0e8]">
      <div className="mx-auto max-w-2xl">
        <p className="mb-3 font-mono text-xs uppercase tracking-[0.3em] text-orange-400">MNIST / LIVE INFERENCE</p>
        <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">Draw a digit.</h1>
        <p className="mt-4 max-w-lg text-[#a8a49c]">A handwritten digit is sampled every 300ms and passed through a neural network built from scratch.</p>

        <section className="mt-10 flex flex-col items-center rounded-3xl border border-white/10 bg-[#1b1b18] p-6 shadow-2xl sm:p-9">
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            onPointerDown={startDrawing}
            onPointerMove={draw}
            onPointerUp={stopDrawing}
            onPointerLeave={stopDrawing}
            className="h-[280px] w-[280px] touch-none cursor-crosshair rounded-lg border border-white/10"
            aria-label="Drawing canvas"
          />
          <button onClick={clearCanvas} className="mt-5 rounded-full border border-white/20 px-6 py-2 text-sm font-semibold transition hover:border-orange-400 hover:text-orange-300">Clear</button>
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
          {error && <p className="mt-5 text-sm text-red-400">{error}</p>}
        </section>
      </div>
    </main>
  );
}
