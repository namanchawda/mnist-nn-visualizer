"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import DrawCanvas, { DrawCanvasHandle } from "../../components/DrawCanvas";
import { usePrediction } from "../../hooks/usePrediction";
import type { SceneNetworkHandle } from "./scene";

// Three.js must never enter the main page's bundle: load the client-only scene
// lazily for this route alone.
const SceneNetwork = dynamic(() => import("./scene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center font-mono text-xs uppercase tracking-[0.2em] text-[#8e8b84]">
      Loading 3D scene…
    </div>
  ),
});

export default function ThreeDView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawCanvasRef = useRef<DrawCanvasHandle>(null);
  const sceneRef = useRef<SceneNetworkHandle>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [generation, setGeneration] = useState(0);
  const [traceMode, setTraceMode] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const inference = usePrediction(canvasRef);

  // Land at the top of the page, and track fullscreen state of the stage.
  useEffect(() => {
    window.scrollTo(0, 0);
    const update = () => setIsFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", update);
    update();
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
      return;
    }
    void stageRef.current?.requestFullscreen();
  };

  const clearCanvas = () => {
    drawCanvasRef.current?.clear();
    inference.reset();
    setGeneration(0);
  };

  // DrawCanvas calls this every 300ms while drawing; the generation counter
  // tells the 3D scene when a fresh prediction (pulse + edge recompute) landed.
  const onSample = () => {
    if (inference.sample()) {
      setGeneration((value) => value + 1);
    }
  };

  const predicted = inference.predicted;
  const idle = predicted === null;

  const overlayButton = (active: boolean) =>
    active
      ? "z-[60] rounded-full border border-orange-400 bg-orange-400/15 px-4 py-1.5 font-mono text-xs uppercase tracking-[0.15em] text-orange-300 transition hover:bg-orange-400/25"
      : "z-[60] rounded-full border border-white/20 bg-[#11110f]/70 px-4 py-1.5 font-mono text-xs uppercase tracking-[0.15em] text-[#bdb8ae] transition hover:border-orange-400 hover:text-orange-300";

  return (
    <main className="min-h-screen bg-[#11110f] px-5 py-12 text-[#f4f0e8]">
      <div className="mx-auto max-w-7xl">
        <p className="mb-3 font-mono text-xs uppercase tracking-[0.3em] text-orange-400">MNIST / 3D VIEW</p>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">The network, in three dimensions.</h1>
          <Link href="/" className="text-sm text-[#a8a49c] transition hover:text-orange-300">← Back to 2D view</Link>
        </div>

        <section className="mt-10 grid items-start gap-8 lg:grid-cols-[300px_minmax(0,1fr)]">
          <div className="flex flex-col items-center rounded-3xl border border-white/10 bg-[#1b1b18] p-6">
            <DrawCanvas ref={drawCanvasRef} canvasRef={canvasRef} onSample={onSample} />
            <button onClick={clearCanvas} className="mt-5 rounded-full border border-white/20 px-6 py-2 text-sm font-semibold transition hover:border-orange-400 hover:text-orange-300">Clear</button>
            {!inference.ready && !inference.error && <p className="mt-3 text-sm text-[#8e8b84]">Loading model…</p>}
            {inference.error && <p className="mt-3 text-sm text-red-400">Failed to load the model weights (weights.json).</p>}
            {inference.ready && (
              <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.2em] text-[#8e8b84]">Model loaded</p>
            )}
          </div>

          <div
            ref={stageRef}
            className={
              isFullscreen
                ? "fixed inset-0 z-50 flex flex-col gap-0 bg-[#11110f] p-6"
                : "flex flex-col rounded-3xl border border-white/10 bg-[#1b1b18]"
            }
          >
            {isFullscreen && (
              <div className="flex items-end justify-between pb-3">
                <div>
                  <p className="font-mono text-xs uppercase tracking-[0.3em] text-orange-400">MNIST / 3D VIEW</p>
                  <p className="text-lg font-semibold">The network, in three dimensions.</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-4xl font-bold leading-none text-orange-400">{predicted ?? "-"}</span>
                  <Link href="/" className="text-sm text-[#a8a49c] transition hover:text-orange-300">← 2D</Link>
                </div>
              </div>
            )}

            <div
              className={
                isFullscreen
                  ? "relative min-h-0 flex-1 overflow-hidden rounded-2xl border border-white/10"
                  : "relative overflow-hidden rounded-2xl border border-white/10"
              }
            >
              <div className={isFullscreen ? "absolute inset-0" : "h-[58vh] min-h-[420px] lg:h-[640px]"}>
                <SceneNetwork
                  ref={sceneRef}
                  input={inference.input28x28}
                  hidden1={inference.hidden1}
                  hidden2={inference.hidden2}
                  probabilities={inference.probabilities}
                  predicted={predicted}
                  generation={generation}
                  idle={idle}
                  traceMode={traceMode}
                />
              </div>

              <div className="absolute right-3 top-3 z-[60] flex flex-wrap justify-end gap-2">
                <button onClick={() => setTraceMode((value) => !value)} className={overlayButton(traceMode)}>
                  Trace: {traceMode ? "on" : "off"}
                </button>
                <button onClick={() => sceneRef.current?.resetView()} className={overlayButton(false)}>
                  Reset view
                </button>
                <button onClick={toggleFullscreen} className={overlayButton(isFullscreen)}>
                  {isFullscreen ? "Exit full" : "Fullscreen"}
                </button>
              </div>

              <p
                className="pointer-events-none absolute bottom-2 left-0 right-0 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-[#8e8b84]"
                style={{ zIndex: 5 }}
              >
                drag to rotate · pinch or scroll to zoom · hover a neuron · click to pin its edges
              </p>
            </div>

            <div className={(isFullscreen ? "pt-3" : "mt-7 px-6 pb-6") + " grid grid-cols-5 gap-x-4 gap-y-1 font-mono text-xs"}>
              {inference.probabilities.map((probability, digit) => (
                <div key={digit} className="flex items-center gap-2">
                  <span className="w-3 text-[#bdb8ae]">{digit}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full rounded-full bg-orange-400 transition-[width] duration-300 ease-out" style={{ width: `${probability * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>

            {!isFullscreen && (
              <div className="flex items-end justify-between border-t border-white/10 px-6 pb-6 pt-4">
                <span className="text-sm uppercase tracking-widest text-[#8e8b84]">Recognized digit</span>
                <span className="text-6xl font-bold leading-none text-orange-400">{predicted ?? "-"}</span>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
