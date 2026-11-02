"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";

import DrawCanvas, { DrawCanvasHandle } from "../components/DrawCanvas";
import { usePrediction } from "../hooks/usePrediction";
import type { SceneNetworkHandle } from "../components/SceneNetwork";

// Three.js must stay out of the initial bundle: the scene is loaded lazily,
// client-side only, into a container with a fixed responsive height (so the
// layout does not jump while the chunk loads).
const SceneNetwork = dynamic(() => import("../components/SceneNetwork"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center font-mono text-xs uppercase tracking-[0.2em] text-[#8e8b84]">
      <span className="mt-2 inline-block h-1.5 w-32 origin-left overflow-hidden rounded-full bg-white/10" role="status">
        <span className="block h-full w-1/3 animate-pulse rounded-full bg-orange-400/70" />
      </span>
      Loading 3D view…
    </div>
  ),
});

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawCanvasRef = useRef<DrawCanvasHandle>(null);
  const sceneRef = useRef<SceneNetworkHandle>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [generation, setGeneration] = useState(0);
  const [traceMode, setTraceMode] = useState(true);
  const isFullscreenSupported =
    typeof document !== "undefined" && typeof document.documentElement.requestFullscreen === "function";
  const [nativeFullscreen, setNativeFullscreen] = useState(false);
  const [fallbackFullscreen, setFallbackFullscreen] = useState(false);
  const inference = usePrediction(canvasRef);

  const isFullscreen = nativeFullscreen || fallbackFullscreen;

  // Track native fullscreen state of the stage (so Esc / OS gestures escape
  // cleanly); the fallback overlay has no document-level state to track.
  useEffect(() => {
    const update = () => setNativeFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", update);
    update();
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);

  // Escape closes the fallback overlay too.
  useEffect(() => {
    if (!fallbackFullscreen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFallbackFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fallbackFullscreen]);

  const exitFullscreen = useCallback(() => {
    setFallbackFullscreen(false);
    if (document.fullscreenElement) void document.exitFullscreen();
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement || fallbackFullscreen) {
      exitFullscreen();
      return;
    }
    // Fullscreen API is missing/blocked (e.g. iOS Safari on divs): use the
    // fixed-viewport overlay fallback instead of silently failing.
    const stage = stageRef.current;
    if (!isFullscreenSupported || !stage) {
      setFallbackFullscreen(true);
      return;
    }
    void stage.requestFullscreen().catch(() => setFallbackFullscreen(true));
  }, [exitFullscreen, fallbackFullscreen, isFullscreenSupported]);

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
  const confidence = predicted !== null ? Math.max(...inference.probabilities) : 0;

  const overlayButton = (active: boolean) =>
    active
      ? "min-h-[44px] z-[60] rounded-full border border-orange-400 bg-orange-400/15 px-4 py-2 font-mono text-xs uppercase tracking-[0.15em] text-orange-300 transition hover:bg-orange-400/25"
      : "min-h-[44px] z-[60] rounded-full border border-white/20 bg-[#11110f]/70 px-4 py-2 font-mono text-xs uppercase tracking-[0.15em] text-[#bdb8ae] transition hover:border-orange-400 hover:text-orange-300";

  return (
    <main
      className="min-h-dvh bg-[#11110f] px-4 py-5 text-[#f4f0e8] sm:px-6 sm:py-7"
      style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}
    >
      <div className="mx-auto max-w-7xl">
        <header>
          <p className="mb-2 font-mono text-xs uppercase tracking-[0.3em] text-orange-400">MNIST / LIVE INFERENCE</p>
          <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
            <h1 className="text-[clamp(2rem,1.2rem+2.2vw,3.5rem)] font-bold leading-[1.05] tracking-tight">
              Draw a digit. Watch the network think.
            </h1>
          </div>
          <p className="mt-2 max-w-2xl text-sm text-[#a8a49c]">
            Every 300ms while you draw, the strokes are preprocessed exactly like the training images and pushed
            through the network whose weights are shown to the right. All inference runs in your browser.
          </p>
        </header>

        {/* Desktop: canvas card (left) + scene (right); bars land under the
         * canvas card via grid auto-placement. Mobile/tablet: stacked. */}
        <section className="mt-4 grid grid-cols-1 gap-4 sm:gap-5 lg:grid-cols-[340px_minmax(0,1fr)] lg:gap-4">
          {/* Canvas card. Desktop (lg): canvas first, filling the column,
           * then digit + confidence in one row, then Clear + model status.
           * Tablet/mobile: canvas and readout side by side (readout stacks
           * under the canvas on phones narrower than 380px). */}
          <div className="flex flex-col rounded-3xl border border-white/10 bg-[#1b1b18] p-4 sm:p-5">
            <div className="flex flex-col gap-4 min-[380px]:flex-row min-[380px]:items-center lg:flex-col lg:items-stretch">
              <div className="mx-auto w-full max-w-[280px] min-[380px]:mx-0 min-[380px]:min-w-[160px] min-[380px]:w-[55%] min-[380px]:max-w-[280px] lg:w-full lg:max-w-none">
                <DrawCanvas ref={drawCanvasRef} canvasRef={canvasRef} onSample={onSample} />
              </div>
              <div className="w-full min-[380px]:flex-1">
                <div className="flex flex-wrap items-end gap-x-4 gap-y-1">
                  <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-[#8e8b84] lg:hidden">
                    Recognized digit
                  </p>
                  <span
                    className={
                      "ml-auto text-[clamp(2.75rem,1.6rem+3.5vw,4rem)] font-bold leading-none transition-colors lg:order-first lg:ml-0 " +
                      (idle ? "text-[#4a4842]" : "text-orange-400")
                    }
                  >
                    <span key={predicted ?? -1} className={idle ? "" : "digit-pop"}>
                      {predicted ?? "-"}
                    </span>
                  </span>
                  <p className="ml-auto font-mono text-xs uppercase tracking-[0.2em] text-[#8e8b84]">
                    {idle ? "start drawing" : `confidence ${(confidence * 100).toFixed(1)}%`}
                  </p>
                </div>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <button
                onClick={clearCanvas}
                className="min-h-[44px] rounded-full border border-white/20 px-6 py-2 text-sm font-semibold transition hover:border-orange-400 hover:text-orange-300"
              >
                Clear
              </button>
              {!inference.ready && !inference.error && <p className="text-xs text-[#8e8b84]">Loading model…</p>}
              {inference.error && (
                <p className="text-xs text-red-400">Failed to load the model weights (weights.json).</p>
              )}
              {inference.ready && (
                <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#8e8b84]">Model loaded</p>
              )}
            </div>
          </div>

          {/* 3D scene stage. Fixed responsive height keeps the skeleton from
           * jumping; fullscreen promotes this same node to a fixed overlay. */}
          <div
            ref={stageRef}
            className={
              isFullscreen
                ? "fixed inset-0 z-50 flex min-h-0 flex-col bg-[#11110f] p-4 sm:p-6"
                : "relative flex flex-col overflow-hidden rounded-3xl border border-white/10 bg-[#1b1b18] p-2 sm:p-3"
            }
          >
            {isFullscreen && (
              <div className="flex items-center justify-between gap-4 pb-3">
                <div className="min-w-0">
                  <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-orange-400">MNIST / LIVE INFERENCE</p>
                  <p className="truncate text-lg font-semibold">Draw a digit. Watch the network think.</p>
                </div>
                <div className="flex items-center gap-4">
                  <span className={"text-4xl font-bold leading-none " + (idle ? "text-[#4a4842]" : "text-orange-400")}>
                    <span key={predicted ?? -1} className={idle ? "" : "digit-pop"}>
                      {predicted ?? "-"}
                    </span>
                  </span>
                  <button
                    onClick={toggleFullscreen}
                    className="min-h-[44px] rounded-full border border-white/20 px-4 py-2 text-sm font-semibold text-[#e6e1d7] transition hover:border-orange-400 hover:text-orange-300"
                  >
                    Exit
                  </button>
                </div>
              </div>
            )}

            <div
              className={
                isFullscreen
                  ? "relative min-h-0 flex-1"
                  : "relative h-[55dvh] min-h-[340px] sm:h-[60vh] lg:h-[min(64vh,560px)]"
              }
            >
              <div className={isFullscreen ? "absolute inset-0" : "h-full"}>
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

              <div className="absolute right-2 top-2 z-[60] flex flex-wrap justify-end gap-2 sm:right-3 sm:top-3">
                <button onClick={() => setTraceMode((value) => !value)} className={overlayButton(traceMode)}>
                  Trace: {traceMode ? "on" : "off"}
                </button>
                <button
                  onClick={() => sceneRef.current?.resetView()}
                  className={overlayButton(false)}
                >
                  Reset view
                </button>
                {!isFullscreen && (
                  <button onClick={toggleFullscreen} className={overlayButton(false)}>
                    Fullscreen
                  </button>
                )}
              </div>

              <p
                className="pointer-events-none absolute bottom-2 left-0 right-0 text-center font-mono text-[11px] uppercase tracking-[0.18em] text-[#8e8b84]"
                style={{ zIndex: 5 }}
              >
                drag to rotate · pinch or scroll to zoom · click a neuron to pin its edges
              </p>
            </div>
          </div>

          {/* Compact probability grid — under the canvas card on desktop,
           * after the scene on mobile/tablet (grid auto-placement order). */}
          <div className="grid grid-cols-2 gap-x-5 gap-y-1.5 rounded-3xl border border-white/10 bg-[#1b1b18] p-4 font-mono text-[11px]">
            {inference.probabilities.map((probability, digit) => (
              <div key={digit} className="flex items-center gap-2">
                <span className={"w-3 " + (predicted === digit ? "text-orange-300" : "text-[#bdb8ae]")}>{digit}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full transition-[width] duration-300 ease-out"
                    style={{ width: `${probability * 100}%`, background: predicted === digit ? "#ffa64d" : "#8e8b84" }}
                  />
                </div>
                <span className="w-10 text-right text-[#8e8b84]">{(probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
