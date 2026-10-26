// What a prediction gives: a parquet with zephyr predict's columns, and two plots, the
// breathing rate over the whole video and the trace in a window you move through it.
import { parquetWriteBuffer } from "hyparquet-writer";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Result } from "./run";

export function toParquet(r: Result, meta: Record<string, string>): ArrayBuffer {
  return parquetWriteBuffer({
    columnData: [
      { name: "time", data: r.time, type: "DOUBLE" },
      { name: "frame", data: Array.from(r.frame, BigInt), type: "INT64" },
      { name: "signal", data: r.signal, type: "FLOAT" },
      { name: "onset", data: r.onset, type: "FLOAT" },
      { name: "inhale", data: Array.from(r.inhale, (v) => v === 1), type: "BOOLEAN" },
    ],
    kvMetadata: Object.entries(meta).map(([key, value]) => ({ key, value })),
  });
}

export function download(buffer: ArrayBuffer, name: string) {
  const url = URL.createObjectURL(new Blob([buffer], { type: "application/vnd.apache.parquet" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Inhale times and the rate (1 / interval, Hz) each one starts. */
export function rates(r: Result): { t: number[]; hz: number[] } {
  const t: number[] = [];
  const hz: number[] = [];
  let last = -1;
  r.inhale.forEach((v, k) => {
    if (!v) return;
    if (last >= 0) {
      t.push(r.time[k]);
      hz.push(1 / (r.time[k] - r.time[last]));
    }
    last = k;
  });
  return { t, hz };
}

function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class Plots {
  private detail: uPlot | null = null;
  private overview: uPlot | null = null;
  private result: Result | null = null;
  span = 20;

  constructor(private overviewEl: HTMLElement, private detailEl: HTMLElement) {
    new ResizeObserver(() => this.resize()).observe(detailEl);
  }

  show(r: Result) {
    this.result = r;
    this.detail?.destroy();
    this.overview?.destroy();
    const width = this.detailEl.clientWidth;
    const ink = css("--ink");
    const grid = css("--grid");
    const axes = [
      { stroke: ink, grid: { stroke: grid }, ticks: { stroke: grid } },
      { stroke: ink, grid: { stroke: grid }, ticks: { stroke: grid } },
    ];

    // Breathing rate: the median over 10 s bins of each breath's rate
    const { t, hz } = rates(r);
    const end = r.time[r.time.length - 1];
    const bins: number[] = [];
    const med: (number | null)[] = [];
    for (let b = 0; b < end; b += 10) {
      const inside = hz.filter((_, k) => t[k] >= b && t[k] < b + 10).sort((x, y) => x - y);
      bins.push((b + 5) / 60);
      med.push(inside.length > 2 ? inside[Math.floor(inside.length / 2)] : null);
    }
    this.overview = new uPlot(
      {
        width,
        height: 180,
        cursor: { drag: { x: false, y: false } },
        scales: { x: { time: false } },
        axes: [{ ...axes[0], label: "minutes" }, { ...axes[1], label: "breaths / s" }],
        legend: { show: false },
        series: [{}, { stroke: css("--series-1"), width: 1.5, spanGaps: false }],
        hooks: {
          drawSeries: [() => {}],
          draw: [(u) => this.drawWindow(u)],
        },
      },
      [bins, med],
      this.overviewEl,
    );
    this.overview.over.addEventListener("click", (e) => {
      const minutes = this.overview!.posToVal((e as MouseEvent).offsetX, "x");
      this.goTo(minutes * 60 - this.span / 2);
    });

    const inhaleY = Array.from(r.inhale, (v, k) => (v ? r.signal[k] : null));
    const onsetY = Array.from(r.onset, (v) => v * 2 - 4.5); // under the trace
    this.detail = new uPlot(
      {
        width,
        height: 320,
        scales: { x: { time: false }, y: { range: [-5, 4] } },
        axes: [{ ...axes[0], label: "seconds" }, { ...axes[1], label: "signal (z)" }],
        series: [
          { label: "time (s)", value: (_u, v) => (v == null ? "" : v.toFixed(2)) },
          { label: "signal", stroke: css("--ink"), width: 1 },
          { label: "onset probability (lower band)", stroke: css("--series-2"), width: 1 },
          {
            label: "inhale",
            stroke: css("--series-1"),
            width: 0,
            points: { show: true, size: 6, fill: css("--series-1") },
          },
        ],
        hooks: { setScale: [() => this.overview?.redraw(false)] },
      },
      [Array.from(r.time), Array.from(r.signal), onsetY, inhaleY],
      this.detailEl,
    );
    this.goTo(0);
  }

  goTo(start: number) {
    if (!this.detail || !this.result) return;
    const end = this.result.time[this.result.time.length - 1];
    const lo = Math.max(0, Math.min(start, end - this.span));
    this.detail.setScale("x", { min: lo, max: lo + this.span });
  }

  private drawWindow(u: uPlot) {
    if (!this.detail) return;
    const { min, max } = this.detail.scales.x;
    if (min == null || max == null) return;
    const x0 = u.valToPos(min / 60, "x", true);
    const x1 = u.valToPos(max / 60, "x", true);
    const ctx = u.ctx;
    ctx.save();
    ctx.fillStyle = css("--window");
    ctx.fillRect(x0, u.bbox.top, Math.max(2, x1 - x0), u.bbox.height);
    ctx.restore();
  }

  private resize() {
    const width = this.detailEl.clientWidth;
    this.detail?.setSize({ width, height: 320 });
    this.overview?.setSize({ width, height: 180 });
  }
}
