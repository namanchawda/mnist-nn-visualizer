"use client";

/**
 * Self-contained 3D neural-network visualization (Three.js via react-three-fiber).
 * Rendering only: all inference/preprocess math stays in lib/; the weights for
 * edge contributions come from the already-cached loadWeights() result.
 */

import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { ComponentRef, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { Weights } from "../lib/internal/forward";
import { loadWeights } from "../lib/network";

/* ------------------------------------------------------------------ */
/* Tuning constants                                                    */
/* ------------------------------------------------------------------ */

const INPUT_COLS = 28;          // 28x28 grid of instanced tiles
const INPUT_PITCH = 0.36;
const INPUT_TILE = 0.25;
const H1_COLS = 16;             // 16x8 grid for the 128 hidden neurons
const H2_COLS = 8;              // 8x8 grid for the 64 hidden neurons
const HIDDEN_PITCH = 1.0;
const OUT_PITCH = 1.2;          // single column of 10 output spheres

const LAYER_Z = { input: 0, hidden1: -12, hidden2: -22, output: -32 };
/** Strongest edges drawn per layer pair in "flow" mode — the knob to play with. */
const EDGE_TOP_K_PER_PAIR = 300;
const EDGE_TOP_K_SMALL_SCREEN = 110;

/* Decision trace (default mode): a narrow beam from the winning output back
 * through the layers: top N hidden2 neurons per output, top M hidden1 neurons
 * per hidden2 neuron, top M input pixels per hidden1 neuron. */
const TRACE_OUT_NEURONS = 6;
const TRACE_BRANCH = 4;
const TRACE_NEURON_DIM = 0.16;   // brightness of neurons outside the trace
const TRACE_INPUT_DIM = 0.3;     // input pixels are kept a bit brighter
const TRACE_OUTPUT_DIM = 0.25;   // non-winner output neurons in trace mode
/** Edges isolated when a neuron is clicked (incoming + outgoing each). */
const FOCUS_K = 16;
const LAYER_NAMES = ["INPUT", "HIDDEN 1", "HIDDEN 2", "OUTPUT"];
/* Edge material: additive blending at reduced opacity so strong edges stay
 * saturated with color (orange/blue) instead of washing out to white. */
const EDGE_OPACITY = 0.5;
const EDGE_MAX_BRIGHTNESS = 0.7;

const COLOR_POSITIVE = new THREE.Color("#ffa64d"); // orange, positive contributions
const COLOR_NEGATIVE = new THREE.Color("#56aeff"); // blue, negative contributions
const COLOR_BASE = new THREE.Color("#56aeff");     // hidden neuron base
const COLOR_INPUT = new THREE.Color("#9ecfff");    // input tile base
const COLOR_OUTPUT = new THREE.Color("#ffa64d");   // output neuron base
const COLOR_IDLE = new THREE.Color("#5a5850");     // dim idle glow

const LERP_RATE = 12;          // smoothing speed for brightness lerps
const PULSE_LAYER_DELAY = [0, 0.16, 0.32, 0.48];
const PULSE_EDGE_DELAY = [0.08, 0.24, 0.4];
const PULSE_WIDTH = 0.1;

type SceneQuality = {
  topK: number;
  segments: [number, number];
  dpr: [number, number];
  antialias: boolean;
  /** Skip bloom on small/low-power devices — keeps the frame budget for edges. */
  bloom: boolean;
};

export type SceneNetworkProps = {
  /** Preprocessed 784-value MNIST input (null until first stroke / after clear). */
  input: Float64Array | null;
  hidden1: number[];
  hidden2: number[];
  probabilities: number[];
  predicted: number | null;
  /** Bumped on every fresh prediction — triggers pulse + edge recompute. */
  generation: number;
  /** True when there is no prediction (start or cleared canvas). */
  idle: boolean;
  /** Decision-trace edge mode (default); false = global top-K flow. */
  traceMode: boolean;
};

/** Shared decision-trace state. Module-level mutable objects: written only
 * from effects/updates, read per-frame inside useFrame loops — never read or
 * written during render (react-hooks/immutability). */
export type TraceState = {
  mode: boolean;
  predicted: number;
};

const TRACE: TraceState = { mode: false, predicted: -1 };
const TRACE_INPUT_FLAGS = new Uint8Array(784);
const TRACE_H1_FLAGS = new Uint8Array(128);
const TRACE_H2_FLAGS = new Uint8Array(64);
const TRACE_OUTPUT_FLAGS = new Uint8Array(10);

/** Clicked-neuron focus: layer index 0=input, 1=h1, 2=h2, 3=output. */
export type NeuronFocus = { layer: number; index: number } | null;

/** Hover tooltip content: pre-formatted text + world position. */
export type InspectEntry = { text: string; position: [number, number, number] } | null;

/* ------------------------------------------------------------------ */
/* Layout + math helpers (pure, no DOM)                                */
/* ------------------------------------------------------------------ */

function gridPositions(count: number, columns: number, pitch: number, z: number): Float32Array {
  const rows = count / columns;
  const positions = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const row = Math.trunc(index / columns);
    const column = index % columns;
    positions[index * 3] = (column - (columns - 1) / 2) * pitch;
    // Image row 0 is the top of the canvas.
    positions[index * 3 + 1] = ((rows - 1) / 2 - row) * pitch;
    positions[index * 3 + 2] = z;
  }
  return positions;
}

const INPUT_POSITIONS = gridPositions(784, INPUT_COLS, INPUT_PITCH, LAYER_Z.input);
const H1_POSITIONS = gridPositions(128, H1_COLS, HIDDEN_PITCH, LAYER_Z.hidden1);
const H2_POSITIONS = gridPositions(64, H2_COLS, HIDDEN_PITCH, LAYER_Z.hidden2);
const OUT_POSITIONS = gridPositions(10, 1, OUT_PITCH, LAYER_Z.output);

/* Translucent backdrop panels, one per layer, sitting just behind the neurons
 * (smaller z = farther from the camera) with thin edge outlines. */
const PANEL_PAD = 1.5;
const PANELS = [
  { w: 27 * INPUT_PITCH + INPUT_TILE + PANEL_PAD, h: 27 * INPUT_PITCH + INPUT_TILE + PANEL_PAD, z: LAYER_Z.input - 0.8 },
  { w: 15 * HIDDEN_PITCH + 0.7 + PANEL_PAD, h: 7 * HIDDEN_PITCH + 0.7 + PANEL_PAD, z: LAYER_Z.hidden1 - 0.8 },
  { w: 7 * HIDDEN_PITCH + 0.84 + PANEL_PAD, h: 7 * HIDDEN_PITCH + 0.84 + PANEL_PAD, z: LAYER_Z.hidden2 - 0.8 },
  { w: 1.2 + PANEL_PAD, h: 9 * OUT_PITCH + 1.1 + PANEL_PAD, z: LAYER_Z.output - 0.8 },
];
// Shared source geometry for the panel outlines (R3F disposes the derived
// edgesGeometry on unmount, but the source planes stay as module constants).
const PANEL_PLANES = PANELS.map((panel) => new THREE.PlaneGeometry(panel.w, panel.h));

/* ------------------------------------------------------------------ */
/* Layout framing: bounding box, centered offset, camera auto-fit       */
/* ------------------------------------------------------------------ */

const OUT_LABEL_OFFSET = 1.0;

/* Layer captions. Each title is anchored above its own panel and horizontally
 * staggered (HIDDEN 1 left-of-center, HIDDEN 2 right-of-center) so the two
 * never collide on screen at any orbit angle; Html renders them at fixed
 * screen size. */
const CAPTIONS = [
  { text: "INPUT 28×28", x: 0, y: 6.4, z: LAYER_Z.input },
  { text: "HIDDEN 1 · 128", x: -3.4, y: 5.8, z: LAYER_Z.hidden1 },
  { text: "HIDDEN 2 · 64", x: 3.4, y: 5.8, z: LAYER_Z.hidden2 },
  { text: "OUTPUT · 10", x: 0, y: 7.6, z: LAYER_Z.output },
];

/** Exact world-space bounding box of the whole network — neurons, panels,
 * caption anchors and digit-label anchors — derived from the layout
 * constants above (instanced meshes cannot be measured from geometry). */
const NETWORK = (() => {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  const grow = (x0: number, x1: number, y0: number, y1: number, z0: number, z1: number) => {
    minX = Math.min(minX, x0);
    maxX = Math.max(maxX, x1);
    minY = Math.min(minY, y0);
    maxY = Math.max(maxY, y1);
    minZ = Math.min(minZ, z0);
    maxZ = Math.max(maxZ, z1);
  };

  const inputHalf = ((INPUT_COLS - 1) * INPUT_PITCH + INPUT_TILE) / 2;
  grow(-inputHalf, inputHalf, -inputHalf, inputHalf, LAYER_Z.input - 0.3, LAYER_Z.input + 0.3);
  const h1HalfX = (15 * HIDDEN_PITCH) / 2 + 0.7;
  const h1HalfY = (7 * HIDDEN_PITCH) / 2 + 0.7;
  grow(-h1HalfX, h1HalfX, -h1HalfY, h1HalfY, LAYER_Z.hidden1 - 0.6, LAYER_Z.hidden1 + 0.6);
  const h2Half = (7 * HIDDEN_PITCH) / 2 + 0.9;
  grow(-h2Half, h2Half, -h2Half, h2Half, LAYER_Z.hidden2 - 0.6, LAYER_Z.hidden2 + 0.6);
  const outHalfY = (9 * OUT_PITCH) / 2 + 0.7;
  grow(-0.5, OUT_LABEL_OFFSET + 0.6, -outHalfY, outHalfY, LAYER_Z.output - 0.6, LAYER_Z.output + 0.6);
  PANELS.forEach((panel) => grow(-panel.w / 2, panel.w / 2, -panel.h / 2, panel.h / 2, panel.z - 0.05, panel.z + 0.05));
  CAPTIONS.forEach((caption) => grow(caption.x - 0.5, caption.x + 0.5, caption.y, caption.y + 0.7, caption.z - 0.05, caption.z + 0.05));

  return {
    /** Center of the box — the whole scene content is shifted by -center so
     * the orbit pivot sits exactly on the network's mid-point. */
    center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2] as [number, number, number],
    /** Full size of the box along each world axis. */
    extent: [maxX - minX, maxY - minY, maxZ - minZ] as [number, number, number],
  };
})();

const SCENE_OFFSET: [number, number, number] = [-NETWORK.center[0], -NETWORK.center[1], -NETWORK.center[2]];

const NETWORK_FOV = 42;
/** margin over the projected content size; 1/margin ≈ screen-space fill of
 * the limiting axis. 1.08 keeps the tall vertical span fully covered while
 * leaving ≈ 65–83% of the panel width to the content depending on aspect. */
const NETWORK_FIT_MARGIN = 1.08;
const INITIAL_ASPECT = 1.4;

/** Default camera direction: azimuth ~32°, elevation ~18°, looking at the
 * centered origin. */
const CAM_DIR = new THREE.Vector3(0.62, 0.34, 1).normalize();
const CAM_RIGHT = new THREE.Vector3(0, 1, 0).cross(CAM_DIR.clone().negate()).normalize();
const CAM_UP = CAM_DIR.clone().negate().cross(CAM_RIGHT).normalize();

/* Pane-accurate fit: the network is four THIN slabs (plus captions), so
 * projecting the full AABB corners charges every rotation for diagonal
 * space that is mostly empty and yanks the camera way out. Instead each
 * slab's (x, y) extent at its own depth is projected onto the camera's
 * right/up axes and unioned — this is the exact default-view extent the
 * frustum must cover. */
const INPUT_PANE_HALF = ((INPUT_COLS - 1) * INPUT_PITCH + INPUT_TILE) / 2;
const H1_PANE_HALF_X = (15 * HIDDEN_PITCH) / 2 + 0.7;
const H1_PANE_HALF_Y = (7 * HIDDEN_PITCH) / 2 + 0.7;
const H2_PANE_HALF = (7 * HIDDEN_PITCH) / 2 + 0.9;
const OUT_PANE_HALF_Y = (9 * OUT_PITCH) / 2 + 0.7;

const PANES_LOCAL: { x0: number; x1: number; y0: number; y1: number; z: number }[] = [
  { x0: -INPUT_PANE_HALF, x1: INPUT_PANE_HALF, y0: -INPUT_PANE_HALF, y1: CAPTIONS[0].y + 0.7, z: LAYER_Z.input },
  { x0: -H1_PANE_HALF_X, x1: H1_PANE_HALF_X, y0: -H1_PANE_HALF_Y, y1: CAPTIONS[1].y + 0.7, z: LAYER_Z.hidden1 },
  { x0: -H2_PANE_HALF, x1: H2_PANE_HALF, y0: -H2_PANE_HALF, y1: CAPTIONS[2].y + 0.7, z: LAYER_Z.hidden2 },
  { x0: -PANELS[3].w / 2, x1: OUT_LABEL_OFFSET + 1.2, y0: -OUT_PANE_HALF_Y, y1: CAPTIONS[3].y + 0.7, z: LAYER_Z.output },
];

const PANE_BOUNDS = (() => {
  const projected = (basis: THREE.Vector3) => {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const pane of PANES_LOCAL) {
      const zTerm = (pane.z - NETWORK.center[2]) * basis.z;
      const xHi = basis.x >= 0 ? pane.x1 * basis.x : pane.x0 * basis.x;
      const xLo = basis.x >= 0 ? pane.x0 * basis.x : pane.x1 * basis.x;
      const yHi = basis.y >= 0 ? pane.y1 * basis.y : pane.y0 * basis.y;
      const yLo = basis.y >= 0 ? pane.y0 * basis.y : pane.y1 * basis.y;
      min = Math.min(min, xLo + yLo + zTerm);
      max = Math.max(max, xHi + yHi + zTerm);
    }
    return { min, max };
  };
  return { right: projected(CAM_RIGHT), up: projected(CAM_UP) };
})();

/** Distance from the centered origin that puts the projected network slabs
 * inside the viewport for the given panel aspect ratio (horizontal fov
 * grows with the aspect; vertical fov is fixed). */
function fitDistance(aspect: number): number {
  const safeAspect = aspect > 0 ? aspect : 1;
  const tanV = Math.tan(((NETWORK_FOV / 2) * Math.PI) / 180);
  const tanH = tanV * safeAspect;
  const coverX = Math.max(Math.abs(PANE_BOUNDS.right.min), Math.abs(PANE_BOUNDS.right.max));
  const coverY = Math.max(Math.abs(PANE_BOUNDS.up.min), Math.abs(PANE_BOUNDS.up.max));
  return Math.max(coverX / tanH, coverY / tanV) * NETWORK_FIT_MARGIN;
}

interface ContributionPair {
  sourcePositions: Float32Array;
  targetPositions: Float32Array;
  sourceCount: number;
  targetCount: number;
  /** Float64Array weight matrix, row-major (source, target). */
  getWeights: (weights: Weights) => Float64Array;
  pulseDelay: number;
}

const PAIRS: ContributionPair[] = [
  { sourcePositions: INPUT_POSITIONS, targetPositions: H1_POSITIONS, sourceCount: 784, targetCount: 128, getWeights: (w) => w.W1, pulseDelay: PULSE_EDGE_DELAY[0] },
  { sourcePositions: H1_POSITIONS, targetPositions: H2_POSITIONS, sourceCount: 128, targetCount: 64, getWeights: (w) => w.W2, pulseDelay: PULSE_EDGE_DELAY[1] },
  { sourcePositions: H2_POSITIONS, targetPositions: OUT_POSITIONS, sourceCount: 64, targetCount: 10, getWeights: (w) => w.W3, pulseDelay: PULSE_EDGE_DELAY[2] },
];

function pulseFactor(age: number, delay: number): number {
  const x = age - delay;
  if (x <= 0) return 0;
  return Math.exp(-(x * x) / (PULSE_WIDTH * PULSE_WIDTH));
}

/**
 * Indices of the k largest values among the first `n`, written into
 * `indexOut` (not sorted). Iterative quickselect over a reusable index
 * buffer — no per-update allocation.
 */
function selectTopK(values: Float64Array, n: number, k: number, indexOut: Int32Array): number {
  for (let i = 0; i < n; i += 1) indexOut[i] = i;

  let low = 0;
  let high = n - 1;
  while (low < high) {
    // Lomuto partition around the last element.
    const pivot = values[indexOut[high]];
    let store = low;
    for (let i = low; i < high; i += 1) {
      if (values[indexOut[i]] > pivot) {
        const tmp = indexOut[i];
        indexOut[i] = indexOut[store];
        indexOut[store] = tmp;
        store += 1;
      }
    }
    const tmp = indexOut[high];
    indexOut[high] = indexOut[store];
    indexOut[store] = tmp;
    if (store === k) break;
    if (store < k) low = store + 1;
    else high = store - 1;
  }
  return Math.min(k, n);
}

function createHaloTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d")!;
  const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
  gradient.addColorStop(0.25, "rgba(255, 255, 255, 0.55)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/* ------------------------------------------------------------------ */
/* Environment hooks                                                   */
/* ------------------------------------------------------------------ */

function useDocumentHidden(): boolean {
  const [hidden, setHidden] = useState(false);
  useEffect(() => {
    const update = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", update);
    update();
    return () => document.removeEventListener("visibilitychange", update);
  }, []);
  return hidden;
}

function detectQuality(): SceneQuality {
  const small =
    typeof window !== "undefined" &&
    (window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 640);
  const lowPower =
    typeof window !== "undefined" &&
    (navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency < 4 : true);
  return small || lowPower
    ? { topK: EDGE_TOP_K_SMALL_SCREEN, segments: [8, 6], dpr: [1, 1.3], antialias: false, bloom: false }
    : { topK: EDGE_TOP_K_PER_PAIR, segments: [14, 10], dpr: [1, 2], antialias: true, bloom: true };
}

/* ------------------------------------------------------------------ */
/* Instanced neuron layers                                             */
/* ------------------------------------------------------------------ */

type NeuronLayerProps = {
  count: number;
  positions: Float32Array;
  /** Normalized target brightness heuristic per neuron, written by the parent. */
  targets: React.RefObject<Float32Array>;
  /** Optional per-neuron raw values (0..1) that drive size, e.g. probabilities. */
  scaleTargets?: React.RefObject<Float32Array>;
  baseColor: THREE.Color;
  baseScale: number;
  growScale: number;
  /** Extra scale for the highlighted output neuron (0 = none). */
  highlightIndex: () => number;
  /** Layer code for click-focus (0..3) and hover/tooltip text. */
  layerCode: number;
  describe: (index: number) => string;
  onInspect: (entry: InspectEntry) => void;
  onSelect: (layer: number, index: number) => void;
  /** Trace membership flags: 1 = on the decision trace; null = no dimming. */
  memberFlags?: Uint8Array | null;
  /** Brightness for off-trace neurons while trace mode is active. */
  memberDim: number;
  traceState: TraceState;
  pulseRef: React.RefObject<number>;
  pulseDelay: number;
  variant: "tile" | "sphere";
  segments: [number, number];
};

const IDENTITY_QUATERNION = new THREE.Quaternion();
const COMPOSE_POSITION = new THREE.Vector3();
const COMPOSE_SCALE = new THREE.Vector3();
const COMPOSE_MATRIX = new THREE.Matrix4();
const COLOR_WORK = new THREE.Color();

function NeuronLayer({
  count,
  positions,
  targets,
  scaleTargets,
  baseColor,
  baseScale,
  growScale,
  highlightIndex,
  layerCode,
  describe,
  onInspect,
  onSelect,
  memberFlags,
  memberDim,
  traceState,
  pulseRef,
  pulseDelay,
  variant,
  segments,
}: NeuronLayerProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  // Per-neuron displayed brightness, lerped toward targets inside useFrame
  // (never touched during render, per react-hooks/immutability).
  const currentRef = useRef<Float32Array | null>(null);
  // Optional second lerp channel for scale-only values (e.g. raw probability).
  const currentScaleRef = useRef<Float32Array | null>(null);

  /** World position of neuron `index` (screen-tooltip anchor). */
  const positionAt = (index: number): [number, number, number] => [
    positions[index * 3],
    positions[index * 3 + 1],
    positions[index * 3 + 2],
  ];

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    const instanceId = event.instanceId;
    if (instanceId == null) return;
    event.stopPropagation();
    onInspect({ text: describe(instanceId), position: positionAt(instanceId) });
  };

  const handlePointerDown = (event: ThreeEvent<PointerEvent>) => {
    const instanceId = event.instanceId;
    if (instanceId == null) return;
    event.stopPropagation();
    onInspect({ text: describe(instanceId), position: positionAt(instanceId) });
  };

  const handlePointerOut = () => {
    onInspect(null);
  };

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    const instanceId = event.instanceId;
    if (instanceId == null) return;
    event.stopPropagation();
    onSelect(layerCode, instanceId);
  };

  // Seed matrices + an instanceColor buffer before the first painted frame.
  useLayoutEffectOnce(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    COMPOSE_SCALE.setScalar(baseScale);
    for (let i = 0; i < count; i += 1) {
      COMPOSE_POSITION.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      COMPOSE_MATRIX.compose(COMPOSE_POSITION, IDENTITY_QUATERNION, COMPOSE_SCALE);
      mesh.setMatrixAt(i, COMPOSE_MATRIX);
      mesh.setColorAt(i, COLOR_IDLE);
    }
    if (mesh.instanceMatrix) mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const target = targets.current;
    if (!target) return;
    const scaleValues = scaleTargets?.current;
    if (currentRef.current === null) currentRef.current = new Float32Array(count);
    if (currentScaleRef.current === null) currentScaleRef.current = new Float32Array(count);
    const current = currentRef.current;
    const currentScale = currentScaleRef.current;
    const alpha = 1 - Math.exp(-LERP_RATE * Math.min(delta, 0.1));
    const age = (performance.now() - (pulseRef.current ?? -1e9)) / 1000;
    const pulse = age < 1.5 ? pulseFactor(age, pulseDelay) : 0;
    const highlight = highlightIndex();
    const trace = traceState;

    for (let i = 0; i < count; i += 1) {
      let goal = target[i] + pulse * 0.45;
      if (i === highlight) goal = 1;
      else if (trace.mode && memberFlags && memberFlags[i] === 0) goal *= memberDim;
      if (goal > 1) goal = 1;

      let value = current[i] + (goal - current[i]) * alpha;
      // Snap so converged neurons stop receiving instanced-attribute writes.
      if (Math.abs(goal - value) < 0.001 && pulse === 0) value = goal;

      // When scale values come from a separate channel (raw probabilities),
      // lerp them independently from the brightness value.
      let displayed = value;
      if (scaleValues) {
        const scaleGoal = scaleValues[i];
        let scaleValue = currentScale[i] + (scaleGoal - currentScale[i]) * alpha;
        if (Math.abs(scaleGoal - scaleValue) < 0.001 && pulse === 0) scaleValue = scaleGoal;
        if (scaleValue !== currentScale[i]) {
          currentScale[i] = scaleValue;
        }
        displayed = scaleValue;
      }

      if (value === current[i] && displayed === value) continue;
      current[i] = value;

      const scale = baseScale + displayed * growScale + pulse * (baseScale * 0.35);
      COMPOSE_POSITION.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      COMPOSE_SCALE.setScalar(i === highlight ? scale * 1.5 : scale);
      COMPOSE_MATRIX.compose(COMPOSE_POSITION, IDENTITY_QUATERNION, COMPOSE_SCALE);
      mesh.setMatrixAt(i, COMPOSE_MATRIX);

      COLOR_WORK.copy(COLOR_IDLE).lerp(baseColor, 0.18 + value * 0.82);
      mesh.setColorAt(i, COLOR_WORK);
    }
    if (mesh.instanceMatrix) mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      frustumCulled={false}
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerDown}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    >
      {variant === "tile" ? (
        <boxGeometry args={[INPUT_TILE, INPUT_TILE, INPUT_TILE * 0.25]} />
      ) : (
        // Unit radius: the world radius comes entirely from the matrix scale.
        <sphereGeometry args={[1, segments[0], segments[1]]} />
      )}
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

function useLayoutEffectOnce(effect: () => void) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(effect, []);
}

/* ------------------------------------------------------------------ */
/* Input grid (28x28 preprocessed image)                               */
/* ------------------------------------------------------------------ */

function InputGrid({
  input,
  idle,
  pulseRef,
  onInspect,
  onSelect,
}: {
  input: Float64Array | null;
  idle: boolean;
  pulseRef: React.RefObject<number>;
  onInspect: (entry: InspectEntry) => void;
  onSelect: (layer: number, index: number) => void;
}) {
  const targetsRef = useRef(new Float32Array(784));
  const describe = (index: number) => `${LAYER_NAMES[0]} · ${Math.round((input?.[index] ?? 0) * 100)}%`;

  useEffect(() => {
    const targets = targetsRef.current;
    if (idle || !input) {
      targets.fill(0);
      return;
    }
    for (let i = 0; i < targets.length; i += 1) {
      targets[i] = input[i];
    }
  }, [input, idle]);

  return (
    <NeuronLayer
      count={784}
      positions={INPUT_POSITIONS}
      targets={targetsRef}
      baseColor={COLOR_INPUT}
      baseScale={1}
      growScale={0}
      highlightIndex={() => -1}
      layerCode={0}
      describe={describe}
      onInspect={onInspect}
      onSelect={onSelect}
      memberFlags={TRACE_INPUT_FLAGS}
      memberDim={TRACE_INPUT_DIM}
      traceState={TRACE}
      pulseRef={pulseRef}
      pulseDelay={PULSE_LAYER_DELAY[0]}
      variant="tile"
      segments={[8, 6]}
    />
  );
}

/* ------------------------------------------------------------------ */
/* Hidden layers (16x8 / 8x8 grids)                                    */
/* ------------------------------------------------------------------ */

function HiddenGrid({
  activations,
  idle,
  pulseRef,
  pulseDelay,
  positions,
  radius,
  segments,
  memberFlags,
  layerCode,
  onInspect,
  onSelect,
}: {
  activations: number[];
  idle: boolean;
  pulseRef: React.RefObject<number>;
  pulseDelay: number;
  positions: Float32Array;
  radius: number;
  segments: [number, number];
  memberFlags: Uint8Array;
  layerCode: number;
  onInspect: (entry: InspectEntry) => void;
  onSelect: (layer: number, index: number) => void;
}) {
  const count = positions.length / 3;
  const targetsRef = useRef(new Float32Array(count));
  const describe = (index: number) => `${LAYER_NAMES[layerCode]} · ${(activations[index] ?? 0).toFixed(3)}`;

  useEffect(() => {
    const targets = targetsRef.current;
    if (idle || activations.length !== count) {
      targets.fill(0);
      return;
    }
    let max = 0;
    for (const value of activations) if (value > max) max = value;
    if (max > 0) {
      for (let i = 0; i < count; i += 1) targets[i] = activations[i] / max;
    } else {
      targets.fill(0);
    }
  }, [activations, idle, count]);

  return (
    <NeuronLayer
      count={count}
      positions={positions}
      targets={targetsRef}
      baseColor={COLOR_BASE}
      baseScale={radius}
      growScale={radius * 0.55}
      highlightIndex={() => -1}
      layerCode={layerCode}
      describe={describe}
      onInspect={onInspect}
      onSelect={onSelect}
      memberFlags={memberFlags}
      memberDim={TRACE_NEURON_DIM}
      traceState={TRACE}
      pulseRef={pulseRef}
      pulseDelay={pulseDelay}
      variant="sphere"
      segments={segments}
    />
  );
}

/* ------------------------------------------------------------------ */
/* Output column + digit labels + predicted halo                       */
/* ------------------------------------------------------------------ */

function OutputColumn({
  probabilities,
  predicted,
  idle,
  pulseRef,
  segments,
  onInspect,
  onSelect,
}: {
  probabilities: number[];
  predicted: number | null;
  idle: boolean;
  pulseRef: React.RefObject<number>;
  segments: [number, number];
  onInspect: (entry: InspectEntry) => void;
  onSelect: (layer: number, index: number) => void;
}) {
  const targetsRef = useRef(new Float32Array(10));
  const scaleRef = useRef(new Float32Array(10));
  const predictedRef = useRef(-1);
  const describe = (index: number) =>
    `${LAYER_NAMES[3]} ${index} · ${((probabilities[index] ?? 0) * 100).toFixed(1)}%`;

  useEffect(() => {
    const targets = targetsRef.current;
    const scale = scaleRef.current;
    if (idle || probabilities.length !== 10) {
      targets.fill(0);
      scale.fill(0);
      predictedRef.current = -1;
      TRACE_OUTPUT_FLAGS.fill(0);
      return;
    }
    let max = 0;
    for (const value of probabilities) if (value > max) max = value;
    for (let i = 0; i < 10; i += 1) scale[i] = probabilities[i];
    if (max > 0) {
      for (let i = 0; i < 10; i += 1) targets[i] = probabilities[i] / max;
    } else {
      targets.fill(0);
    }
    predictedRef.current = predicted ?? -1;
    for (let i = 0; i < 10; i += 1) TRACE_OUTPUT_FLAGS[i] = i === predicted ? 1 : 0;
  }, [probabilities, predicted, idle]);

  return (
    <group>
      <NeuronLayer
        count={10}
        positions={OUT_POSITIONS}
        targets={targetsRef}
        scaleTargets={scaleRef}
        baseColor={COLOR_OUTPUT}
        baseScale={0.42}
        growScale={0.6}
        highlightIndex={() => predictedRef.current}
        layerCode={3}
        describe={describe}
        onInspect={onInspect}
        onSelect={onSelect}
        memberFlags={TRACE_OUTPUT_FLAGS}
        memberDim={TRACE_OUTPUT_DIM}
        traceState={TRACE}
        pulseRef={pulseRef}
        pulseDelay={PULSE_LAYER_DELAY[3]}
        variant="sphere"
        segments={segments}
      />
      <DigitLabels probabilities={probabilities} predicted={predicted} idle={idle} />
      <Halo position={getOutputPosition(predicted)} predicted={predicted} pulseRef={pulseRef} />
    </group>
  );
}

function getOutputPosition(digit: number | null): [number, number, number] {
  if (digit === null) return [OUT_POSITIONS[0], OUT_POSITIONS[1], OUT_POSITIONS[2]];
  return [OUT_POSITIONS[digit * 3], OUT_POSITIONS[digit * 3 + 1], OUT_POSITIONS[digit * 3 + 2]];
}

function DigitLabels({
  probabilities,
  predicted,
  idle,
}: {
  probabilities: number[];
  predicted: number | null;
  idle: boolean;
}) {
  return (
    <group>
      {Array.from({ length: 10 }, (_, digit) => (
        <Html
          key={digit}
          position={[OUT_POSITIONS[digit * 3] + OUT_LABEL_OFFSET, OUT_POSITIONS[digit * 3 + 1], OUT_POSITIONS[digit * 3 + 2]]}
          center
          zIndexRange={[10, 0]}
          style={{ pointerEvents: "none" }}
        >
          <span
            className={
              "whitespace-nowrap font-mono text-sm tracking-wide transition-colors " +
              (predicted === digit && !idle ? "text-orange-300" : "text-[#6a675f]")
            }
          >
            {digit} · {((probabilities[digit] ?? 0) * 100).toFixed(1)}%
          </span>
        </Html>
      ))}
    </group>
  );
}

function Halo({
  position,
  predicted,
  pulseRef,
}: {
  position: [number, number, number];
  predicted: number | null;
  pulseRef: React.RefObject<number>;
}) {
  const texture = useMemo(() => createHaloTexture(), []);
  const materialRef = useRef<THREE.SpriteMaterial>(null);

  useEffect(() => () => texture.dispose(), [texture]);

  useFrame((_, delta) => {
    const material = materialRef.current;
    if (!material) return;
    const alpha = 1 - Math.exp(-8 * Math.min(delta, 0.1));
    const age = (performance.now() - (pulseRef.current ?? -1e9)) / 1000;
    const pulse = age < 1.5 ? pulseFactor(age, PULSE_LAYER_DELAY[3]) : 0;
    const goal = 0.45 + pulse * 0.55;
    material.opacity += (goal - material.opacity) * alpha;
  });

  if (predicted === null) return null;

  return (
    <sprite position={position} scale={[2.1, 2.1, 1]}>
      <spriteMaterial
        ref={materialRef}
        map={texture}
        transparent
        opacity={0}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        color="#ffad5c"
        toneMapped={false}
      />
    </sprite>
  );
}

/* ------------------------------------------------------------------ */
/* Layer caption labels                                                */
/* ------------------------------------------------------------------ */

function LayerCaptions() {
  return (
    <group>
      {CAPTIONS.map((entry) => (
        <Html
          key={entry.text}
          position={[entry.x, entry.y, entry.z]}
          center
          zIndexRange={[10, 0]}
          style={{ pointerEvents: "none" }}
        >
          <span className="whitespace-nowrap font-mono text-xs uppercase tracking-[0.3em] text-[#9a968c]">
            {entry.text}
          </span>
        </Html>
      ))}
    </group>
  );
}

/* ------------------------------------------------------------------ */
/* Translucent layer panels                                            */
/* ------------------------------------------------------------------ */

function LayerPanels() {
  return (
    <group>
      {PANELS.map((panel, index) => (
        <group key={index} position={[0, 0, panel.z]}>
          <mesh>
            <planeGeometry args={[panel.w, panel.h]} />
            <meshBasicMaterial
              color="#17171a"
              transparent
              opacity={0.07}
              depthWrite={false}
              side={THREE.DoubleSide}
              toneMapped={false}
            />
          </mesh>
          <lineSegments>
            <edgesGeometry args={[PANEL_PLANES[index]]} />
            <lineBasicMaterial color="#54504a" transparent opacity={0.55} depthWrite={false} toneMapped={false} />
          </lineSegments>
        </group>
      ))}
    </group>
  );
}

/* ------------------------------------------------------------------ */
/* Contribution edges                                                  */
/* ------------------------------------------------------------------ */

type EdgeBuffers = {
  position: THREE.BufferAttribute;
  color: THREE.BufferAttribute;
  colorTarget: Float32Array;
  abs: Float64Array;
  index: Int32Array;
  values: Float64Array;
  sourceIds: Int32Array;
  targetIds: Int32Array;
  /** Compact per-column candidate buffer (784 = largest source count). */
  cand: Float64Array;
};

function ContributionEdges({
  input,
  hidden1,
  hidden2,
  predicted,
  idle,
  generation,
  pulseRef,
  topK,
  traceMode,
  focus,
}: {
  input: Float64Array | null;
  hidden1: number[];
  hidden2: number[];
  predicted: number | null;
  idle: boolean;
  generation: number;
  pulseRef: React.RefObject<number>;
  topK: number;
  traceMode: boolean;
  focus: NeuronFocus;
}) {
  const [weights, setWeights] = useState<Weights | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry>(null);
  const lineRef = useRef<THREE.LineSegments>(null);
  // Mutable buffers (positions/colors/scratch) — created in an effect, only
  // touched from effects and useFrame afterwards.
  const edgeRef = useRef<EdgeBuffers | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadWeights()
      .then((loaded) => {
        if (!cancelled) setWeights(loaded);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  // Edge lines must never intercept hover/click raycasts.
  useEffect(() => {
    const lines = lineRef.current;
    if (!lines) return;
    lines.raycast = () => {};
  }, []);

  // Attach pooled attributes exactly once; the geometry, material and their
  // attributes are disposed with the declarative <lineSegments> on unmount.
  useEffect(() => {
    const geometry = geometryRef.current;
    if (!geometry) return;
    if (edgeRef.current) return;
    const capacity = topK * PAIRS.length;
    const position = new THREE.BufferAttribute(new Float32Array(capacity * 2 * 3), 3);
    const color = new THREE.BufferAttribute(new Float32Array(capacity * 2 * 3), 3);
    geometry.setAttribute("position", position);
    geometry.setAttribute("color", color);
    edgeRef.current = {
      position,
      color,
      colorTarget: new Float32Array(capacity * 2 * 3),
      abs: new Float64Array(784 * 128),
      index: new Int32Array(784 * 128),
      values: new Float64Array(topK),
      sourceIds: new Int32Array(topK),
      targetIds: new Int32Array(topK),
      cand: new Float64Array(784),
    };
  }, [topK]);

  useEffect(() => {
    const buffers = edgeRef.current;
    if (!weights || !buffers) return;
    const { colorTarget, abs, index, values, sourceIds, targetIds, cand } = buffers;

    if (idle || !input) {
      // Fade all edges out on clear; the per-frame lerp handles the transition.
      colorTarget.fill(0);
      TRACE.mode = false;
      TRACE.predicted = -1;
      TRACE_INPUT_FLAGS.fill(0);
      TRACE_H1_FLAGS.fill(0);
      TRACE_H2_FLAGS.fill(0);
      buffers.position.needsUpdate = true;
      return;
    }

    const layerActivations = [input, Float64Array.from(hidden1), Float64Array.from(hidden2)];
    TRACE.predicted = predicted ?? -1;
    TRACE.mode = traceMode && predicted !== null && !focus;

    /** Re-color + position `count` prepared edges (values/sourceIds/targetIds)
     * into this pair's slot region; stale slots are zeroed. */
    const emitPair = (pairIndex: number, count: number, activations: Float64Array) => {
      const pair = PAIRS[pairIndex];
      const weight = pair.getWeights(weights);
      let pairMax = 0;
      for (let i = 0; i < count; i += 1) if (values[i] > pairMax) pairMax = values[i];
      if (pairMax === 0) pairMax = 1;

      const pairSlotStart = pairIndex * topK * 2;
      // Stale slots left by a previous, wider prediction must stay invisible.
      colorTarget.fill(0, pairSlotStart * 3, (pairIndex + 1) * topK * 2 * 3);
      for (let i = 0; i < count; i += 1) {
        const source = pair.sourcePositions;
        const target = pair.targetPositions;
        const v0 = pairSlotStart + i * 2;
        const p0 = v0 * 3;
        const p1 = (v0 + 1) * 3;

        buffers.position.array[p0] = source[sourceIds[i] * 3];
        buffers.position.array[p0 + 1] = source[sourceIds[i] * 3 + 1];
        buffers.position.array[p0 + 2] = source[sourceIds[i] * 3 + 2];
        buffers.position.array[p1] = target[targetIds[i] * 3];
        buffers.position.array[p1 + 1] = target[targetIds[i] * 3 + 1];
        buffers.position.array[p1 + 2] = target[targetIds[i] * 3 + 2];

        // Re-derive the sign (magnitudes only were kept for the selection pass).
        const contribution =
          activations[sourceIds[i]] * weight[sourceIds[i] * pair.targetCount + targetIds[i]];
        const normalized = Math.pow(values[i] / pairMax, 0.6);
        const tintColor = contribution >= 0 ? COLOR_POSITIVE : COLOR_NEGATIVE;
        COLOR_WORK.copy(tintColor).multiplyScalar(EDGE_MAX_BRIGHTNESS * (0.25 + normalized * 0.75));
        colorTarget[p0] = COLOR_WORK.r;
        colorTarget[p0 + 1] = COLOR_WORK.g;
        colorTarget[p0 + 2] = COLOR_WORK.b;
        colorTarget[p1] = COLOR_WORK.r;
        colorTarget[p1 + 1] = COLOR_WORK.g;
        colorTarget[p1 + 2] = COLOR_WORK.b;
      }
    };

    /** Isolate a clicked neuron: incoming (pair = layer-1) + outgoing
     * (pair = layer) edges only; other pairs are zeroed. */
    if (focus) {
      TRACE_INPUT_FLAGS.fill(0);
      TRACE_H1_FLAGS.fill(0);
      TRACE_H2_FLAGS.fill(0);

      PAIRS.forEach((pair, pairIndex) => {
        const weight = pair.getWeights(weights);
        const activations = layerActivations[pairIndex];
        let count = 0;

        if (pairIndex === focus.layer - 1) {
          // Incoming: strongest sources feeding the focused neuron.
          const targetId = focus.index;
          for (let s = 0; s < pair.sourceCount; s += 1) {
            cand[s] = Math.abs(activations[s] * weight[s * pair.targetCount + targetId]);
          }
          const selected = selectTopK(cand, pair.sourceCount, FOCUS_K, index);
          for (let i = 0; i < selected && count < topK; i += 1) {
            const s = index[i];
            if (cand[s] <= 0) break;
            values[count] = cand[s];
            sourceIds[count] = s;
            targetIds[count] = targetId;
            count += 1;
          }
        } else if (pairIndex === focus.layer) {
          // Outgoing: strongest targets fed by the focused neuron.
          const sourceId = focus.index;
          const rowOffset = sourceId * pair.targetCount;
          for (let t = 0; t < pair.targetCount; t += 1) {
            cand[t] = Math.abs(activations[sourceId] * weight[rowOffset + t]);
          }
          const selected = selectTopK(cand, pair.targetCount, FOCUS_K, index);
          for (let i = 0; i < selected && count < topK; i += 1) {
            const t = index[i];
            if (cand[t] <= 0) break;
            values[count] = cand[t];
            sourceIds[count] = sourceId;
            targetIds[count] = t;
            count += 1;
          }
        }
        emitPair(pairIndex, count, activations);
      });

      buffers.position.needsUpdate = true;
      return;
    }

    if (TRACE.mode) {
      // Decision trace: a narrow beam traced backward from the winning output.
      const o = predicted as number;

      // h2 → output: strongest hidden2 neurons feeding the winner.
      let count = 0;
      for (let j = 0; j < 64; j += 1) {
        cand[j] = Math.abs(layerActivations[2][j] * weights.W3[j * 10 + o]);
      }
      const selected2 = selectTopK(cand, 64, TRACE_OUT_NEURONS, index);
      for (let i = 0; i < selected2 && count < topK; i += 1) {
        const j = index[i];
        if (cand[j] <= 0) break;
        values[count] = cand[j];
        sourceIds[count] = j;
        targetIds[count] = o;
        TRACE_H2_FLAGS[j] = 1;
        count += 1;
      }
      emitPair(2, count, layerActivations[2]);

      // h1 → h2: per selected hidden2 neuron, its strongest hidden1 inputs.
      count = 0;
      for (let j = 0; j < 64 && count < topK; j += 1) {
        if (TRACE_H2_FLAGS[j] === 0) continue;
        for (let i = 0; i < 128; i += 1) {
          cand[i] = Math.abs(layerActivations[1][i] * weights.W2[i * 64 + j]);
        }
        const selected1 = selectTopK(cand, 128, TRACE_BRANCH, index);
        for (let e = 0; e < selected1 && count < topK; e += 1) {
          const i = index[e];
          if (cand[i] <= 0) break;
          values[count] = cand[i];
          sourceIds[count] = i;
          targetIds[count] = j;
          TRACE_H1_FLAGS[i] = 1;
          count += 1;
        }
      }
      emitPair(1, count, layerActivations[1]);

      // input → h1: per selected hidden1 neuron, its four strongest pixels.
      count = 0;
      for (let i = 0; i < 128 && count < topK; i += 1) {
        if (TRACE_H1_FLAGS[i] === 0) continue;
        for (let p = 0; p < 784; p += 1) {
          cand[p] = Math.abs(layerActivations[0][p] * weights.W1[p * 128 + i]);
        }
        const selected0 = selectTopK(cand, 784, TRACE_BRANCH, index);
        for (let e = 0; e < selected0 && count < topK; e += 1) {
          const p = index[e];
          if (cand[p] <= 0) break;
          values[count] = cand[p];
          sourceIds[count] = p;
          targetIds[count] = i;
          TRACE_INPUT_FLAGS[p] = 1;
          count += 1;
        }
      }
      emitPair(0, count, layerActivations[0]);

      buffers.position.needsUpdate = true;
      return;
    }

    // Global flow mode: strongest |activation * weight| edges across each pair.
    TRACE_INPUT_FLAGS.fill(0);
    TRACE_H1_FLAGS.fill(0);
    TRACE_H2_FLAGS.fill(0);

    PAIRS.forEach((pair, pairIndex) => {
      const weight = pair.getWeights(weights);
      const activations = layerActivations[pairIndex];

      // contribution = activation_prev[i] * W[i][j], magnitudes only.
      const n = pair.sourceCount * pair.targetCount;
      for (let i = 0; i < pair.sourceCount; i += 1) {
        const activation = activations[i];
        const rowOffset = i * pair.targetCount;
        for (let j = 0; j < pair.targetCount; j += 1) {
          abs[rowOffset + j] = Math.abs(activation * weight[rowOffset + j]);
        }
      }

      const selected = selectTopK(abs, n, topK, index);
      let count = 0;
      for (let i = 0; i < selected && count < topK; i += 1) {
        const flat = index[i];
        if (abs[flat] <= 0) break;
        values[count] = abs[flat];
        sourceIds[count] = Math.trunc(flat / pair.targetCount);
        targetIds[count] = flat % pair.targetCount;
        count += 1;
      }
      emitPair(pairIndex, count, activations);
    });

    buffers.position.needsUpdate = true;
  }, [weights, idle, generation, input, hidden1, hidden2, predicted, topK, traceMode, focus]);

  // Smooth edges + fade to idle via a per-frame color lerp with pulse accent.
  useFrame((_, delta) => {
    const buffers = edgeRef.current;
    if (!buffers) return;
    const alpha = 1 - Math.exp(-LERP_RATE * Math.min(delta, 0.1));
    const age = (performance.now() - (pulseRef.current ?? -1e9)) / 1000;
    const colors = buffers.color.array as Float32Array;
    const target = buffers.colorTarget;
    let changed = false;

    for (let pairIndex = 0; pairIndex < PAIRS.length; pairIndex += 1) {
      const boost = age < 1.5 ? 1 + pulseFactor(age, PULSE_EDGE_DELAY[pairIndex]) * 0.9 : 1;
      const start = pairIndex * topK * 2 * 3;
      const end = (pairIndex + 1) * topK * 2 * 3;
      for (let i = start; i < end; i += 3) {
        const goalR = target[i] * boost;
        const goalG = target[i + 1] * boost;
        const goalB = target[i + 2] * boost;
        const r = colors[i] + (goalR - colors[i]) * alpha;
        const g = colors[i + 1] + (goalG - colors[i + 1]) * alpha;
        const b = colors[i + 2] + (goalB - colors[i + 2]) * alpha;
        if (r !== colors[i] || g !== colors[i + 1] || b !== colors[i + 2]) {
          colors[i] = r;
          colors[i + 1] = g;
          colors[i + 2] = b;
          changed = true;
        }
      }
    }
    if (changed) buffers.color.needsUpdate = true;
  });

  return (
    <lineSegments ref={lineRef} frustumCulled={false}>
      <bufferGeometry ref={geometryRef} />
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={EDGE_OPACITY}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        toneMapped={false}
      />
    </lineSegments>
  );
}

/* ------------------------------------------------------------------ */
/* Root                                                                */
/* ------------------------------------------------------------------ */

type OrbitControlsImpl = ComponentRef<typeof OrbitControls>;

/** Owns the OrbitControls instance and its frame-loop auto-rotate policy:
 * spins after 4s of no user dragging, slower while a prediction is live. The
 * parent-shared ref doubles as the imperative reset target. */
function OrbitRig({
  idle,
  controlsRef,
  fitDist,
}: {
  idle: boolean;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
  /** Current fitted default distance; dolly range derives from it. */
  fitDist: number;
}) {
  const lastInteractRef = useRef(0);

  useFrame(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    controls.autoRotate = performance.now() - lastInteractRef.current > 4000;
    controls.autoRotateSpeed = idle ? 0.45 : 0.18;
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enablePan={false}
      autoRotateSpeed={0.45}
      /* Orbit pivot = origin = center of the offset content group. */
      minDistance={fitDist * 0.4}
      maxDistance={fitDist * 2.5}
      minPolarAngle={0.05}
      maxPolarAngle={Math.PI * 0.62}
      onStart={() => {
        lastInteractRef.current = performance.now();
      }}
    />
  );
}

function NetworkRig(
  props: SceneNetworkProps & {
    quality: SceneQuality;
    inspect: InspectEntry;
    onInspect: (entry: InspectEntry) => void;
    onSelect: (layer: number, index: number) => void;
    focus: NeuronFocus;
  },
) {
  const pulseRef = useRef(-1e9);

  // Start a new traveling pulse on every fresh prediction.
  useEffect(() => {
    if (!props.idle && props.generation > 0) {
      pulseRef.current = performance.now();
    }
  }, [props.generation, props.idle]);

  // The entire content is wrapped in a group shifted by -center of the
  // network's bounding box, so the orbit pivot starts at (0,0,0).
  return (
    <group position={SCENE_OFFSET}>
      <InputGrid
        input={props.input}
        idle={props.idle}
        pulseRef={pulseRef}
        onInspect={props.onInspect}
        onSelect={props.onSelect}
      />
      <HiddenGrid
        activations={props.hidden1}
        idle={props.idle}
        pulseRef={pulseRef}
        pulseDelay={PULSE_LAYER_DELAY[1]}
        positions={H1_POSITIONS}
        radius={0.34}
        segments={props.quality.segments}
        memberFlags={TRACE_H1_FLAGS}
        layerCode={1}
        onInspect={props.onInspect}
        onSelect={props.onSelect}
      />
      <HiddenGrid
        activations={props.hidden2}
        idle={props.idle}
        pulseRef={pulseRef}
        pulseDelay={PULSE_LAYER_DELAY[2]}
        positions={H2_POSITIONS}
        radius={0.42}
        segments={props.quality.segments}
        memberFlags={TRACE_H2_FLAGS}
        layerCode={2}
        onInspect={props.onInspect}
        onSelect={props.onSelect}
      />
      <OutputColumn
        probabilities={props.probabilities}
        predicted={props.predicted}
        idle={props.idle}
        pulseRef={pulseRef}
        segments={props.quality.segments}
        onInspect={props.onInspect}
        onSelect={props.onSelect}
      />
      <ContributionEdges
        input={props.input}
        hidden1={props.hidden1}
        hidden2={props.hidden2}
        predicted={props.predicted}
        idle={props.idle}
        generation={props.generation}
        pulseRef={pulseRef}
        topK={props.quality.topK}
        traceMode={props.traceMode}
        focus={props.focus}
      />
      {props.inspect && (
        <Html
          position={props.inspect.position}
          center
          zIndexRange={[40, 0]}
          style={{ pointerEvents: "none" }}
        >
          <div className="whitespace-nowrap rounded-lg border border-white/15 bg-[#11110f]/85 px-2.5 py-1 font-mono text-[11px] text-[#eee9df]">
            {props.inspect.text}
          </div>
        </Html>
      )}
      <LayerPanels />
      <LayerCaptions />
    </group>
  );
}

export type SceneNetworkHandle = {
  /** Return the camera + orbit target to the initial framed view. */
  resetView: () => void;
};

export default forwardRef<SceneNetworkHandle, SceneNetworkProps>(function SceneNetwork(props, ref) {
  const hidden = useDocumentHidden();
  const [quality] = useState(detectQuality);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [inspect, setInspect] = useState<InspectEntry>(null);
  const [focus, setFocus] = useState<NeuronFocus>(null);

  // Camera auto-fit: the fitted distance depends on the panel aspect ratio
  // and is recomputed whenever the panel resizes (includes fullscreen).
  const [fitDist, setFitDist] = useState(() => fitDistance(INITIAL_ASPECT));
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      const aspect = rect.height > 0 ? rect.width / rect.height : INITIAL_ASPECT;
      const dist = fitDistance(aspect);
      setFitDist((prev) => (Math.abs(prev - dist) < 0.02 ? prev : dist));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    window.addEventListener("resize", update);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);

  // Re-seat the camera at the fitted default view whenever the fitted distance
  // changes — on mount (real panel size, replacing the INITIAL_ASPECT guess)
  // and on every resize, including entering/leaving fullscreen.
  useEffect(() => {
    const controls = controlsRef.current;
    const camera = (controls?.object ?? cameraRef.current) as THREE.PerspectiveCamera | null;
    if (!camera || !camera.isPerspectiveCamera) return;
    camera.position.set(
      CAM_DIR.x * fitDist,
      CAM_DIR.y * fitDist,
      CAM_DIR.z * fitDist,
    );
    const maxHalf = Math.max(...NETWORK.extent) / 2;
    camera.near = Math.max(0.1, fitDist - maxHalf);
    camera.far = fitDist + maxHalf * 2;
    camera.updateProjectionMatrix();
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [fitDist]);

  useImperativeHandle(
    ref,
    () => ({
      resetView: () => {
        const camera = cameraRef.current;
        const controls = controlsRef.current;
        camera?.position.set(CAM_DIR.x * fitDist, CAM_DIR.y * fitDist, CAM_DIR.z * fitDist);
        if (controls) {
          controls.target.set(0, 0, 0);
          controls.update();
        }
      },
    }),
    [fitDist],
  );

  // Hover tooltips: only re-render when the hovered neuron actually changes.
  const handleInspect = useCallback((entry: InspectEntry) => {
    setInspect((prev) => {
      if (prev === null && entry === null) return prev;
      if (prev && entry && prev.text === entry.text) return prev;
      return entry;
    });
  }, []);

  // Click-focus: pin a neuron's edges; clicking it again unpins.
  const handleSelect = useCallback((layer: number, index: number) => {
    setFocus((prev) => (prev && prev.layer === layer && prev.index === index ? null : { layer, index }));
  }, []);

  const handlePointerMissed = useCallback(() => {
    setInspect(null);
    setFocus(null);
  }, []);

  return (
    <div ref={wrapperRef} className="h-full w-full">
      <Canvas
        frameloop={hidden ? "never" : "always"}
        dpr={quality.dpr}
        camera={{
          position: [CAM_DIR.x * fitDist, CAM_DIR.y * fitDist, CAM_DIR.z * fitDist],
          fov: NETWORK_FOV,
          near: 0.1,
          far: fitDist * 12,
        }}
        gl={{ antialias: quality.antialias }}
        style={{ borderRadius: "1rem" }}
        onCreated={({ gl, camera }) => {
          cameraRef.current = camera as THREE.PerspectiveCamera;
          // Keep vertical page scrolling available on touch screens; rotate
          // only on horizontal drags inside the scene.
          gl.domElement.style.touchAction = "pan-y";
        }}
        onPointerMissed={handlePointerMissed}
      >
        <color attach="background" args={["#11110f"]} />
        {/* Depth-mixing mist so layers farther back sink into the dark. */}
        <fogExp2 attach="fog" args={["#11110f", 0.017]} />
        <NetworkRig
          {...props}
          quality={quality}
          inspect={inspect}
          onInspect={handleInspect}
          onSelect={handleSelect}
          focus={focus}
        />
        {quality.bloom && (
          <EffectComposer multisampling={quality.antialias ? 4 : 0}>
            <Bloom intensity={0.4} luminanceThreshold={0.5} luminanceSmoothing={0.3} mipmapBlur />
          </EffectComposer>
        )}
        <OrbitRig idle={props.idle} controlsRef={controlsRef} fitDist={fitDist} />
      </Canvas>
    </div>
  );
});
