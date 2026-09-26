// The ways the indicator can draw the sound. Every one is fed the same
// numbers: `history` is the last rows of frequency bands, newest first, each
// value 0 to 1; `mic` is the newest level; `frame` counts the frames drawn.

export type MicLevel = {
  peak: number;
  rms: number;
  seconds: number;
  pitch: number;
  bands: number[];
  pause: number;
  typing: boolean;
  sentences: number;
  armed: boolean;
};

export type Visualisation = {
  id: string;
  name: string;
  description: string;
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][], mic: MicLevel, frame: number) => void;
};

const BINS = 64;
const ROWS = 30;

// The isometric waterfall: the newest row in front, older rows receding.
function waterfall(ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][]) {
  const FILL_START = 8;
  const originY = h * 0.9;
  const colW = (w * 0.72) / BINS;
  const stepX = (w * 0.24) / ROWS;
  const stepY = (h * 0.55) / ROWS;
  const heightScale = h * 0.24;
  const rowSpan = colW * (BINS - 1);
  const originX = (w - rowSpan) / 2;
  const skew = (j: number) => (j - (ROWS - 1) / 2) * stepX;
  const px = (i: number, j: number) => originX + i * colW + skew(j);
  const py = (j: number, mag: number) => originY - j * stepY - mag * heightScale;

  const crestPath = (j: number) => {
    const row = history[j];
    const p = new Path2D();
    p.moveTo(px(0, j), py(j, row[0]));
    for (let i = 1; i < BINS; i++) p.lineTo(px(i, j), py(j, row[i]));
    return p;
  };

  for (let j = ROWS - 1; j >= 0; j--) {
    const row = history[j];
    const depth = 1 - j / ROWS;
    const baselineY = originY - j * stepY;
    const crest = crestPath(j);

    if (j >= FILL_START) {
      const body = new Path2D(crest);
      body.lineTo(px(BINS - 1, j), baselineY);
      body.lineTo(px(0, j), baselineY);
      body.closePath();
      const t = (j - FILL_START) / (ROWS - 1 - FILL_START);
      const r = Math.round(56 + (236 - 56) * t);
      const g = Math.round(132 + (248 - 132) * t);
      const b = Math.round(250 + (255 - 250) * t);
      ctx.globalAlpha = 0.32 + t * 0.3;
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fill(body);
    }

    ctx.globalAlpha = 0.1 + depth * 0.18;
    ctx.strokeStyle = "rgb(90, 205, 255)";
    ctx.lineWidth = 4;
    ctx.stroke(crest);

    ctx.globalAlpha = 0.45 + depth * 0.55;
    ctx.strokeStyle = "rgb(226, 248, 255)";
    ctx.lineWidth = 1.1;
    ctx.stroke(crest);

    ctx.globalAlpha = 0.3 + depth * 0.6;
    ctx.strokeStyle = "rgb(240, 252, 255)";
    ctx.lineWidth = 1.4;
    const capW = colW * 1.6;
    ctx.beginPath();
    for (let i = 2; i < BINS - 2; i++) {
      const m = row[i];
      if (m > 0.32 && m >= row[i - 1] && m > row[i + 1]) {
        const cx = px(i, j);
        const cy = py(j, m);
        ctx.moveTo(cx - capW / 2, cy - 4);
        ctx.lineTo(cx + capW / 2, cy - 4);
      }
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// Blue for quiet, white for loud, cyan between.
function heat(level: number): string {
  let r: number, g: number, b: number;
  if (level < 0.5) {
    const k = level * 2;
    r = 37 + (56 - 37) * k;
    g = 99 + (189 - 99) * k;
    b = 235 + (248 - 235) * k;
  } else {
    const k = (level - 0.5) * 2;
    r = 56 + (255 - 56) * k;
    g = 189 + (255 - 189) * k;
    b = 248 + (255 - 248) * k;
  }
  return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
}

// Upright spectrum bars with a falling peak cap on each.
const peaks = new Array(BINS).fill(0);
function bars(ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][]) {
  const row = history[0];
  const gap = 2;
  const barW = (w * 0.9 - gap * (BINS - 1)) / BINS;
  const left = (w - (barW * BINS + gap * (BINS - 1))) / 2;
  const base = h * 0.82;
  const reach = h * 0.62;
  for (let i = 0; i < BINS; i++) {
    const level = row[i];
    peaks[i] = Math.max(level, peaks[i] - 0.012);
    const x = left + i * (barW + gap);
    const barH = Math.max(2, level * reach);
    const color = heat(level);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 4 + level * 10;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x, base - barH, barW, barH);
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "rgb(240, 252, 255)";
    ctx.fillRect(x, base - peaks[i] * reach - 3, barW, 2);
  }
  ctx.globalAlpha = 1;
}

// A wave mirrored above and below the middle line, in the style of a voice
// assistant. Low bands in the middle, high bands out at the sides.
function mirror(ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][], _mic: MicLevel, frame: number) {
  const row = history[0];
  const cy = h * 0.55;
  const reach = h * 0.34;
  const half = BINS / 2;
  const points: { x: number; y: number }[] = [];
  for (let k = 0; k <= BINS; k++) {
    // k runs left to right; the band index runs out from the centre.
    const i = Math.min(BINS - 1, Math.abs(k - half) * 2);
    const wobble = 0.6 + 0.4 * Math.sin(frame * 0.08 + k * 0.35);
    points.push({ x: (k / BINS) * w * 0.9 + w * 0.05, y: row[i] * reach * wobble });
  }
  for (const [sign, alpha, width] of [[1, 0.35, 5], [-1, 0.35, 5], [1, 0.95, 1.6], [-1, 0.95, 1.6]] as const) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, cy);
    for (let k = 1; k < points.length - 1; k++) {
      const a = points[k];
      const b = points[k + 1];
      const mx = (a.x + b.x) / 2;
      const my = cy - sign * (a.y + b.y) / 2;
      ctx.quadraticCurveTo(a.x, cy - sign * a.y, mx, my);
    }
    ctx.lineTo(points[points.length - 1].x, cy);
    ctx.strokeStyle = width > 3 ? "rgb(56, 189, 248)" : "rgb(226, 248, 255)";
    ctx.globalAlpha = alpha;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.stroke();
  }
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = "rgb(56, 189, 248)";
  ctx.beginPath();
  ctx.moveTo(points[0].x, cy);
  for (const p of points) ctx.lineTo(p.x, cy - p.y);
  for (let k = points.length - 1; k >= 0; k--) ctx.lineTo(points[k].x, cy + points[k].y);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

// Bars standing on a circle, turning slowly, with a core that swells with
// the overall level.
function ring(ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][], mic: MicLevel, frame: number) {
  const row = history[0];
  const cx = w / 2;
  const cy = h / 2;
  const size = Math.min(w, h);
  const inner = size * 0.2;
  const reach = size * 0.24;
  const loud = Math.min(1, row.reduce((a, b) => a + b, 0) / BINS * 2.5);

  ctx.beginPath();
  ctx.arc(cx, cy, inner * (0.55 + loud * 0.3), 0, Math.PI * 2);
  ctx.fillStyle = "rgba(56, 189, 248, 0.16)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, inner - 3, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(120, 200, 255, 0.18)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.lineCap = "round";
  const BARS = BINS * 2;
  for (let k = 0; k < BARS; k++) {
    // Each band twice, mirrored, so the ring closes on itself.
    const i = k < BINS ? k : BARS - 1 - k;
    const level = row[i];
    const angle = (k / BARS) * Math.PI * 2 + frame * 0.004 - Math.PI / 2;
    const len = 3 + reach * level;
    const color = heat(level);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(cx + cos * inner, cy + sin * inner);
    ctx.lineTo(cx + cos * (inner + len), cy + sin * (inner + len));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 4 + level * 8;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  void mic;
}

// A grid of dots, one column per band, lit from the bottom up.
function dots(ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][]) {
  const row = history[0];
  const COLS = 32;
  const LEVELS = 9;
  const cellW = (w * 0.9) / COLS;
  const cellH = (h * 0.7) / LEVELS;
  const left = w * 0.05;
  const bottom = h * 0.85;
  const radius = Math.min(cellW, cellH) * 0.3;
  for (let c = 0; c < COLS; c++) {
    const a = row[c * 2];
    const b = row[c * 2 + 1];
    const level = Math.max(a, b);
    const lit = Math.round(level * LEVELS);
    for (let r = 0; r < LEVELS; r++) {
      const on = r < lit;
      const x = left + c * cellW + cellW / 2;
      const y = bottom - r * cellH - cellH / 2;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (on) {
        const color = heat(r / (LEVELS - 1));
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.globalAlpha = 0.95;
      } else {
        ctx.fillStyle = "rgb(120, 170, 220)";
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 0.12;
      }
      ctx.fill();
    }
  }
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// One smooth glowing curve, with the last few rows fading behind it.
function curve(ctx: CanvasRenderingContext2D, w: number, h: number, history: number[][]) {
  const base = h * 0.85;
  const reach = h * 0.65;
  const left = w * 0.05;
  const span = w * 0.9;
  const TRAIL = 6;
  for (let j = TRAIL - 1; j >= 0; j--) {
    const row = history[j * 2];
    const age = j / TRAIL;
    ctx.beginPath();
    ctx.moveTo(left, base);
    for (let i = 0; i < BINS; i++) {
      const x = left + (i / (BINS - 1)) * span;
      const y = base - row[i] * reach * (1 - age * 0.25);
      if (i === 0) ctx.lineTo(x, y);
      else {
        const px = left + ((i - 1) / (BINS - 1)) * span;
        const py = base - row[i - 1] * reach * (1 - age * 0.25);
        ctx.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2);
      }
    }
    ctx.lineTo(left + span, base);
    ctx.closePath();
    if (j === 0) {
      const grad = ctx.createLinearGradient(0, base - reach, 0, base);
      grad.addColorStop(0, "rgba(226, 248, 255, 0.55)");
      grad.addColorStop(1, "rgba(56, 132, 250, 0.08)");
      ctx.fillStyle = grad;
      ctx.globalAlpha = 1;
      ctx.fill();
      ctx.strokeStyle = "rgb(226, 248, 255)";
      ctx.lineWidth = 1.8;
      ctx.shadowColor = "rgb(90, 205, 255)";
      ctx.shadowBlur = 10;
      ctx.stroke();
      ctx.shadowBlur = 0;
    } else {
      ctx.strokeStyle = "rgb(90, 205, 255)";
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.35 * (1 - age);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

export const VISUALISATIONS: Visualisation[] = [
  { id: "waterfall", name: "Waterfall", description: "The spectrum flowing away into the distance", draw: waterfall },
  { id: "bars", name: "Bars", description: "Classic spectrum bars with falling peaks", draw: bars },
  { id: "mirror", name: "Mirror", description: "A wave mirrored around the middle, low notes in the centre", draw: mirror },
  { id: "ring", name: "Ring", description: "Bars around a circle that swells with your voice", draw: ring },
  { id: "dots", name: "Dots", description: "A grid of dots lit from the bottom up", draw: dots },
  { id: "curve", name: "Curve", description: "One glowing line, with the last moments fading behind it", draw: curve },
];

export const DEFAULT_VISUALISATION = "waterfall";

export function visualisation(id: string): Visualisation {
  return VISUALISATIONS.find((v) => v.id === id) ?? VISUALISATIONS[0];
}
