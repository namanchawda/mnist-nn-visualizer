"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";

export type DrawCanvasHandle = {
  clear: () => void;
};

type DrawCanvasProps = {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  /** Called every 300ms while a stroke is in progress (may be a no-op). */
  onSample: () => void;
};

const CANVAS_SIZE = 280;
const BRUSH_RADIUS = 9;
const STROKE_WIDTH = 18;
const SAMPLE_INTERVAL_MS = 300;

function paintBackground(context: CanvasRenderingContext2D) {
  context.fillStyle = "#050505";
  context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

/**
 * The 280x280 drawing surface. Owns stroke rendering and the 300ms sampling
 * interval; inference state lives in the parent via usePrediction.
 */
const DrawCanvas = forwardRef<DrawCanvasHandle, DrawCanvasProps>(function DrawCanvas(
  { canvasRef, onSample },
  ref,
) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const drawingRef = useRef(false);
  const onSampleRef = useRef(onSample);
  onSampleRef.current = onSample;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    paintBackground(context);
  }, [canvasRef]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

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
    context.arc(point.x, point.y, BRUSH_RADIUS, 0, Math.PI * 2);
    context.fill();
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => onSampleRef.current(), SAMPLE_INTERVAL_MS);
  };

  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    const currentPoint = pointFromEvent(event);
    if (!canvas || !context || !lastPointRef.current) return;
    context.strokeStyle = "white";
    context.lineWidth = STROKE_WIDTH;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(lastPointRef.current.x, lastPointRef.current.y);
    context.lineTo(currentPoint.x, currentPoint.y);
    context.stroke();
    lastPointRef.current = currentPoint;
  };

  const clear = useCallback(() => {
    stopDrawing();
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) {
      paintBackground(context);
    }
  }, [canvasRef]);

  useImperativeHandle(ref, () => ({ clear }), [clear]);

  return (
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
  );
});

export default DrawCanvas;
