"use client";

import { useEffect, useRef } from "react";

type NetworkVisualizationProps = {
  hidden1Activations: number[];
  hidden2Activations: number[];
  probabilities: number[];
  predictedDigit: number | null;
};

const WIDTH = 900;
const HEIGHT = 500;
const HIDDEN1_COUNT = 128;
const HIDDEN2_COUNT = 64;
const OUTPUT_COUNT = 10;

function normalize(values: number[]): number[] {
  const max = Math.max(...values, 0);
  return values.map((value) => (max > 0 ? Math.max(0, value) / max : 0));
}

export default function NetworkVisualization({
  hidden1Activations,
  hidden2Activations,
  probabilities,
  predictedDigit,
}: NetworkVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, WIDTH, HEIGHT);
    context.fillStyle = "#11110f";
    context.fillRect(0, 0, WIDTH, HEIGHT);

    const firstLayer = normalize(hidden1Activations);
    const secondLayer = normalize(hidden2Activations);
    const outputLayer = normalize(probabilities);

    // Two compact sub-columns keep all 128 first-layer neurons on the canvas.
    const firstNodes = Array.from({ length: HIDDEN1_COUNT }, (_, index) => ({
      x: index < 64 ? 130 : 180,
      y: 35 + (index % 64) * 6.8,
    }));
    const secondNodes = Array.from({ length: HIDDEN2_COUNT }, (_, index) => ({
      x: 430,
      y: 35 + index * 6.8,
    }));
    const outputNodes = Array.from({ length: OUTPUT_COUNT }, (_, index) => ({
      x: 745,
      y: 65 + index * 40,
    }));

    // Draw structural edges first, so neurons remain visually prominent.
    context.lineWidth = 1;
    context.strokeStyle = "rgba(116, 155, 190, 0.035)";
    for (const firstNode of firstNodes) {
      for (const secondNode of secondNodes) {
        context.beginPath();
        context.moveTo(firstNode.x, firstNode.y);
        context.lineTo(secondNode.x, secondNode.y);
        context.stroke();
      }
    }
    context.strokeStyle = "rgba(255, 166, 77, 0.05)";
    for (const secondNode of secondNodes) {
      for (const outputNode of outputNodes) {
        context.beginPath();
        context.moveTo(secondNode.x, secondNode.y);
        context.lineTo(outputNode.x, outputNode.y);
        context.stroke();
      }
    }

    const drawNode = (
      node: { x: number; y: number },
      intensity: number,
      radius: number,
      label?: string,
      highlighted = false,
    ) => {
      const fillAlpha = 0.12 + intensity * 0.88;

      if (highlighted) {
        context.shadowColor = "#ffad5c";
        context.shadowBlur = 18;
      }
      context.beginPath();
      context.arc(node.x, node.y, radius, 0, Math.PI * 2);
      context.fillStyle = label
        ? `rgba(255, 166, 77, ${fillAlpha})`
        : `rgba(86, 174, 255, ${fillAlpha})`;
      context.fill();
      context.shadowBlur = 0;

      context.lineWidth = highlighted ? 2.5 : 1;
      context.strokeStyle = highlighted
        ? "#ffd39a"
        : `rgba(190, 220, 245, ${0.2 + intensity * 0.55})`;
      context.stroke();

      if (label) {
        context.fillStyle = "#f4f0e8";
        context.font = "600 14px monospace";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(label, node.x, node.y);
      }
    };

    firstNodes.forEach((node, index) =>
      drawNode(node, firstLayer[index] ?? 0, 3),
    );
    secondNodes.forEach((node, index) =>
      drawNode(node, secondLayer[index] ?? 0, 4),
    );
    outputNodes.forEach((node, index) =>
      drawNode(
        node,
        outputLayer[index] ?? 0,
        15,
        String(index),
        predictedDigit === index,
      ),
    );

    context.textAlign = "center";
    context.textBaseline = "alphabetic";
    context.font = "11px monospace";
    context.fillStyle = "rgba(244, 240, 232, 0.48)";
    context.fillText("HIDDEN LAYER 1", 155, 490);
    context.fillText("HIDDEN LAYER 2", 430, 490);
    context.fillText("OUTPUT", 745, 490);
  }, [hidden1Activations, hidden2Activations, probabilities, predictedDigit]);

  return (
    <figure className="w-full overflow-hidden rounded-2xl border border-white/10 bg-[#11110f]">
      <canvas
        ref={canvasRef}
        width={WIDTH}
        height={HEIGHT}
        className="h-auto w-full"
        aria-label="Neural network activation visualization"
      />
      <figcaption className="border-t border-white/10 px-4 py-3 text-center font-mono text-xs uppercase tracking-[0.18em] text-[#8e8b84]">
        Hidden Layer 1 (128) → Hidden Layer 2 (64) → Output (10)
      </figcaption>
    </figure>
  );
}
