/**
 * Chart primitives.
 *
 * Every figure on this site is drawn from d3 scales evaluated at build time.
 * Nothing here runs in the browser: the pages ship as static SVG, so the
 * charts cost no javascript, cannot fail to hydrate, and are readable with
 * scripting off. d3 is a devDependency of the build in effect, not a runtime.
 *
 * The split is deliberate and worth keeping:
 *
 *   geometry  — this module. Scales, ticks, layout, number formatting.
 *   colour    — CSS custom properties, applied through classes.
 *
 * Colour stays in CSS because the palette has to respond to
 * `prefers-color-scheme`, and a hex value baked into a `fill` attribute at
 * build time cannot. Anything here that returned a colour would quietly
 * break dark mode.
 */
import { scaleLinear, scaleBand, scalePoint } from "d3-scale";
import { format } from "d3-format";
import { line, area, curveMonotoneX } from "d3-shape";
import { extent, max, min, range as d3range } from "d3-array";

export { scaleLinear, scaleBand, scalePoint, line, area, curveMonotoneX,
         extent, max, min, d3range };

/** Padding around a plot. Named rather than positional, because a four-number
 *  tuple in source order is the classic way to silently swap top and bottom. */
export interface Pad { top: number; right: number; bottom: number; left: number }

export interface Frame {
  width: number; height: number; pad: Pad;
  /** Inner drawing area. */
  x0: number; x1: number; y0: number; y1: number;
  innerW: number; innerH: number;
  viewBox: string;
}

/** A plot frame. Every chart starts here so that "the drawing area" means the
 *  same thing in all of them and no figure re-derives it slightly differently. */
export const frame = (width: number, height: number, pad: Partial<Pad> = {}): Frame => {
  const p: Pad = { top: 24, right: 24, bottom: 40, left: 48, ...pad };
  return {
    width, height, pad: p,
    x0: p.left, x1: width - p.right,
    y0: p.top,  y1: height - p.bottom,
    innerW: width - p.left - p.right,
    innerH: height - p.top - p.bottom,
    viewBox: `0 0 ${width} ${height}`,
  };
};

export interface Tick { value: number; pos: number; label: string }

/** Ticks from the scale itself, so the labels are the round numbers d3 picks
 *  rather than a hand-written array that drifts when the data moves. */
export const ticks = (
  scale: { (v: number): number; ticks: (n?: number) => number[] },
  count = 6,
  fmt: (v: number) => string = format("~g"),
): Tick[] =>
  scale.ticks(count).map((value) => ({ value, pos: scale(value), label: fmt(value) }));

/* Formatters. The site quotes recall to four places in prose, but four places
 * on every gridline is noise, so axes get their own shorter form. */
export const f2 = format(".2f");
export const f1 = format(".1f");
export const f3 = format(".3f");
export const f4 = format(".4f");
export const fPct = format(".0%");
export const fPct1 = format(".1%");
export const fInt = format(",d");

/** Nudges a domain outward to the next round number so the largest mark is not
 *  drawn on the frame edge. `nice()` on the scale does this for ticks; this
 *  does it for the domain when the data max is what sets the extent. */
export const niceMax = (v: number, step: number) => Math.ceil(v / step) * step;
