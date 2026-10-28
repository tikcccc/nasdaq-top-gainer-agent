import type { Analysis, Bar, Series } from "@/lib/api";
import type { ChartOption } from "./EChart";

export const COLORS = { gain: "#37735b", loss: "#ae4843", brand: "#1f5653", sp: "#839087", ndx: "#b49563", grid: "#eef0e9", axis: "#839087", line: "#e3e7df" };

const axisLabel = { color: COLORS.axis, fontSize: 11 };
const tooltip = {
  trigger: "axis",
  backgroundColor: "#fff",
  borderColor: "#e3e7df",
  borderWidth: 1,
  padding: [10, 14],
  textStyle: { color: "#1d3030", fontSize: 12 },
  extraCssText: "box-shadow:0 12px 32px -8px rgba(20,21,28,.18);border-radius:6px;",
};

const compact = (v: number) => new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
const signedPct = (v: number, digits = 2) => `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;

const shortDate = (d: string) => new Date(`${d}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

export function stockSymbol(series: Series, preferred?: string): string | undefined {
  if (preferred && series[preferred]) return preferred;
  return Object.keys(series).find((k) => k !== "GSPC" && k !== "IXIC");
}

/** Candlesticks with volume underneath. */
export function priceOption(bars: Bar[]): ChartOption {
  const dates = bars.map((b) => shortDate(b.date));
  const up = (b: Bar) => b.close >= b.open;
  return {
    animationDuration: 400,
    tooltip: {
      ...tooltip,
      axisPointer: { type: "cross", lineStyle: { color: "#cbd4ca" }, crossStyle: { color: "#cbd4ca" }, label: { backgroundColor: "#1d3030" } },
      formatter: (params: { dataIndex: number }[]) => {
        const b = bars[params[0].dataIndex];
        const row = (k: string, v: string) => `<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:#64748b">${k}</span><b>${v}</b></div>`;
        return `<div style="font-weight:600;margin-bottom:4px">${shortDate(b.date)}</div>`
          + row("Open", `$${b.open.toFixed(2)}`) + row("High", `$${b.high.toFixed(2)}`) + row("Low", `$${b.low.toFixed(2)}`)
          + row("Close", `$${b.close.toFixed(2)}`) + row("Volume", (b.volume ?? 0).toLocaleString("en-US"));
      },
    },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    grid: [
      { left: 56, right: 16, top: 16, height: "58%" },
      { left: 56, right: 16, top: "76%", bottom: 28 },
    ],
    xAxis: [
      { type: "category", data: dates, boundaryGap: true, axisLine: { lineStyle: { color: COLORS.line } }, axisTick: { show: false }, axisLabel: { show: false } },
      { type: "category", gridIndex: 1, data: dates, boundaryGap: true, axisLine: { lineStyle: { color: COLORS.line } }, axisTick: { show: false }, axisLabel },
    ],
    yAxis: [
      {
        scale: true, splitNumber: 4, splitLine: { lineStyle: { color: COLORS.grid } },
        axisLabel: { ...axisLabel, formatter: (v: number) => `$${Number.isInteger(v) ? v : v.toFixed(2)}` },
        axisPointer: { label: { formatter: (p: { value: number }) => `$${p.value.toFixed(2)}` } },
      },
      {
        gridIndex: 1, splitNumber: 2, splitLine: { show: false },
        axisLabel: { ...axisLabel, formatter: compact },
        axisPointer: { label: { formatter: (p: { value: number }) => compact(p.value) } },
      },
    ],
    series: [
      {
        type: "candlestick",
        name: "Price",
        barMaxWidth: 28,
        data: bars.map((b) => [b.open, b.close, b.low, b.high]),
        itemStyle: { color: COLORS.gain, color0: COLORS.loss, borderColor: COLORS.gain, borderColor0: COLORS.loss },
      },
      {
        type: "bar",
        name: "Volume",
        xAxisIndex: 1,
        yAxisIndex: 1,
        barMaxWidth: 28,
        data: bars.map((b) => ({ value: b.volume ?? 0, itemStyle: { color: up(b) ? "rgba(55,115,91,.25)" : "rgba(174,72,67,.25)" } })),
      },
    ],
  };
}

/** Cumulative % change of each series since the start of the window. */
export function performanceOption(series: Series, symbol: string): ChartOption {
  const lines = [
    { key: symbol, name: symbol, color: COLORS.brand, width: 2.5 },
    { key: "GSPC", name: "S&P 500", color: COLORS.sp, width: 1.75 },
    { key: "IXIC", name: "NASDAQ", color: COLORS.ndx, width: 1.75 },
  ].filter((l) => series[l.key]?.length);
  const dates = series[symbol].map((b) => b.date);
  return {
    animationDuration: 400,
    color: lines.map((l) => l.color),
    legend: { top: 0, right: 0, icon: "roundRect", itemWidth: 10, itemHeight: 4, itemGap: 12, textStyle: { color: "#6c7a74", fontSize: 11 } },
    tooltip: { ...tooltip, valueFormatter: (v: number) => signedPct(v) },
    grid: { left: 52, right: 12, top: 36, bottom: 28 },
    xAxis: { type: "category", data: dates.map(shortDate), boundaryGap: false, axisLine: { lineStyle: { color: COLORS.line } }, axisTick: { show: false }, axisLabel: { ...axisLabel, interval: 0 } },
    yAxis: { type: "value", axisLabel: { ...axisLabel, formatter: (v: number) => signedPct(v, 0) }, splitLine: { lineStyle: { color: COLORS.grid } } },
    series: lines.map((l) => {
      const bars = series[l.key];
      const base = bars[0].close;
      const byDate = new Map(bars.map((b) => [b.date, +((b.close / base - 1) * 100).toFixed(2)]));
      const main = l.key === symbol;
      return {
        type: "line",
        name: l.name,
        symbol: "circle",
        symbolSize: main ? 6 : 4,
        z: main ? 3 : 2,
        lineStyle: { width: l.width },
        data: dates.map((d) => byDate.get(d) ?? null),
        ...(main ? {
          areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: "rgba(31,86,83,.12)" }, { offset: 1, color: "rgba(31,86,83,0)" }] } },
          markLine: { silent: true, symbol: "none", lineStyle: { color: "#cbd4ca", type: "dashed" }, label: { show: false }, data: [{ yAxis: 0 }] },
        } : {}),
      };
    }),
  };
}

export function returnsOption(analysis: Analysis, bars: Bar[]): ChartOption {
  const dates = bars.slice(-analysis.daily_returns.length).map((b) => shortDate(b.date));
  return {
    animationDuration: 400,
    tooltip: { ...tooltip, axisPointer: { type: "shadow", shadowStyle: { color: "rgba(148,163,184,.08)" } }, valueFormatter: (v: number) => signedPct(v) },
    grid: { left: 52, right: 12, top: 24, bottom: 28 },
    xAxis: { type: "category", data: dates, axisLine: { lineStyle: { color: COLORS.line } }, axisTick: { show: false }, axisLabel: { ...axisLabel, interval: 0 } },
    yAxis: { type: "value", axisLabel: { ...axisLabel, formatter: (v: number) => `${v}%` }, splitLine: { lineStyle: { color: COLORS.grid } } },
    series: [{
      type: "bar",
      name: "Daily return",
      barMaxWidth: 36,
      data: analysis.daily_returns.map((r) => ({
        value: +(r * 100).toFixed(2),
        itemStyle: { color: r >= 0 ? COLORS.gain : COLORS.loss, borderRadius: r >= 0 ? [4, 4, 0, 0] : [0, 0, 4, 4] },
      })),
      label: { show: true, position: "top", color: "#6c7a74", fontSize: 11, formatter: (p: { value: number }) => signedPct(p.value, 1) },
    }],
  };
}

/** Sentiment score on the 0–4 scale. */
export function sentimentOption(score: number): ChartOption {
  return {
    series: [{
      type: "gauge",
      min: 0,
      max: 4,
      startAngle: 200,
      endAngle: -20,
      radius: "100%",
      center: ["50%", "68%"],
      splitNumber: 4,
      axisLine: { lineStyle: { width: 14, color: [[0.25, COLORS.loss], [0.5, "#b49563"], [0.75, "#ccd5c4"], [1, COLORS.gain]] } },
      pointer: { length: "58%", width: 5, itemStyle: { color: "#1d3030" } },
      anchor: { show: true, size: 10, itemStyle: { color: "#1d3030" } },
      axisTick: { show: false },
      splitLine: { length: 14, lineStyle: { color: "#fff", width: 3 } },
      axisLabel: { show: false },
      detail: { valueAnimation: true, offsetCenter: [0, "32%"], fontSize: 22, fontWeight: 600, color: "#1d3030", formatter: (v: number) => `${v.toFixed(1)} / 4` },
      data: [{ value: score }],
    }],
  };
}

/** Minimal closing-price line for the overview hero. */
export function sparkOption(bars: Bar[]): ChartOption {
  const up = bars[bars.length - 1].close >= bars[0].close;
  const color = up ? COLORS.gain : COLORS.loss;
  return {
    animationDuration: 500,
    grid: { left: 2, right: 2, top: 10, bottom: 6 },
    tooltip: { ...tooltip, formatter: (p: { dataIndex: number }[]) => `${shortDate(bars[p[0].dataIndex].date)} · <b>$${bars[p[0].dataIndex].close.toFixed(2)}</b>` },
    xAxis: { type: "category", show: false, boundaryGap: false, data: bars.map((b) => b.date) },
    yAxis: { type: "value", show: false, scale: true },
    series: [{
      type: "line",
      data: bars.map((b) => b.close),
      symbol: "circle",
      symbolSize: 5,
      showSymbol: false,
      lineStyle: { width: 2.5, color },
      itemStyle: { color },
      areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: up ? "rgba(55,115,91,.12)" : "rgba(174,72,67,.12)" }, { offset: 1, color: "rgba(255,255,255,0)" }] } },
    }],
  };
}
