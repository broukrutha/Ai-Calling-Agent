/**
 * Small, dependency-free SVG charts.
 *
 * Every mark, tick and label is placed with one scale per axis, and every tick
 * label names a value the axis actually reaches. Colours come from CSS classes
 * bound to theme tokens, so charts follow light and dark mode. Missing values
 * break a line instead of being drawn as zero.
 */

import { el, svg } from './dom.js';

const HEIGHT = 180;
const PAD = { top: 12, right: 12, bottom: 26, left: 52 };

function niceStep(rough) {
  if (rough <= 0) return 1;
  const exponent = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
  return nice * exponent;
}

/** Axis from 0 to a round number at or above ``max``, with about four ticks. */
export function niceAxis(max) {
  const step = niceStep((max > 0 ? max : 1) / 4);
  const top = Math.max(step, Math.ceil((max > 0 ? max : 1) / step - 1e-9) * step);
  const ticks = [];
  for (let value = 0; value <= top + step / 2; value += step) ticks.push(Number(value.toFixed(6)));
  return { top, ticks };
}

function frame(container) {
  const width = Math.max(container.clientWidth || 600, 240);
  const plot = { x: PAD.left, y: PAD.top, w: width - PAD.left - PAD.right, h: HEIGHT - PAD.top - PAD.bottom };
  const root = svg('svg', { class: 'chart', viewBox: `0 0 ${width} ${HEIGHT}`, width: '100%', height: HEIGHT, role: 'img' });
  return { width, plot, root };
}

function yAxis(root, plot, axis, format) {
  for (const tick of axis.ticks) {
    const y = plot.y + plot.h - (tick / axis.top) * plot.h;
    root.append(svg('line', { class: 'chart-grid', x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y }));
    root.append(svg('text', { class: 'chart-label', x: plot.x - 8, y: y + 4, 'text-anchor': 'end' }, document.createTextNode(format(tick))));
  }
}

function xLabels(root, plot, points, xFormat) {
  const count = Math.min(points.length, plot.w > 520 ? 6 : 4);
  if (count < 2) return;
  for (let i = 0; i < count; i += 1) {
    const index = Math.round((i * (points.length - 1)) / (count - 1));
    const x = plot.x + (points.length === 1 ? plot.w / 2 : (index / (points.length - 1)) * plot.w);
    const anchor = i === 0 ? 'start' : i === count - 1 ? 'end' : 'middle';
    root.append(svg('text', { class: 'chart-label', x, y: HEIGHT - 6, 'text-anchor': anchor }, document.createTextNode(xFormat(points[index].t))));
  }
}

function attachTooltip(container, root, plot, points, positionOf, describe) {
  const tip = el('div', { class: 'chart-tip', hidden: true });
  const cross = svg('line', { class: 'chart-cross', y1: plot.y, y2: plot.y + plot.h, x1: 0, x2: 0, visibility: 'hidden' });
  root.append(cross);
  container.append(tip);
  const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); };
  root.addEventListener('pointerleave', hide);
  root.addEventListener('pointermove', (event) => {
    const box = root.getBoundingClientRect();
    const scale = box.width / Number(root.viewBox.baseVal.width);
    const x = (event.clientX - box.left) / scale;
    if (x < plot.x - 4 || x > plot.x + plot.w + 4) { hide(); return; }
    const index = points.length === 1 ? 0 : Math.round(((x - plot.x) / plot.w) * (points.length - 1));
    const point = points[Math.max(0, Math.min(points.length - 1, index))];
    const px = positionOf(Math.max(0, Math.min(points.length - 1, index)));
    cross.setAttribute('x1', px);
    cross.setAttribute('x2', px);
    cross.setAttribute('visibility', 'visible');
    tip.replaceChildren(...describe(point));
    tip.hidden = false;
    const left = px * scale;
    tip.style.left = `${Math.min(Math.max(left, 70), box.width - 70)}px`;
  });
}

/**
 * Lines over time. ``series``: [{key, label, cls}] where cls picks the colour.
 * ``budget`` draws a dashed reference line.
 */
export function lineChart(container, { points, series, format, xFormat, budget, empty = 'No data in this period.' }) {
  const values = points.flatMap((p) => series.map((s) => p[s.key])).filter((v) => typeof v === 'number');
  if (!values.length) {
    container.replaceChildren(el('div', { class: 'chart-empty', text: empty }));
    return;
  }
  const { plot, root } = frame(container);
  const axis = niceAxis(Math.max(...values, budget || 0));
  yAxis(root, plot, axis, format);
  const xOf = (i) => plot.x + (points.length === 1 ? plot.w / 2 : (i / (points.length - 1)) * plot.w);
  const yOf = (v) => plot.y + plot.h - (v / axis.top) * plot.h;

  if (budget) {
    root.append(svg('line', { class: 'chart-budget', x1: plot.x, x2: plot.x + plot.w, y1: yOf(budget), y2: yOf(budget) }));
    root.append(svg('text', { class: 'chart-label chart-budget-label', x: plot.x + plot.w, y: yOf(budget) - 5, 'text-anchor': 'end' },
      document.createTextNode(`budget ${format(budget)}`)));
  }

  for (const s of series) {
    let path = '';
    let pen = false;
    let lastX = 0;
    let lastY = 0;
    points.forEach((p, i) => {
      const v = p[s.key];
      if (typeof v !== 'number') { pen = false; return; }
      const x = xOf(i);
      const y = yOf(v);
      path += `${pen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
      pen = true;
      lastX = x;
      lastY = y;
    });
    root.append(svg('path', { class: `chart-line ${s.cls}`, d: path }));
    if (path) root.append(svg('circle', { class: `chart-dot ${s.cls}`, cx: lastX, cy: lastY, r: 3.5 }));
  }
  xLabels(root, plot, points, xFormat);
  container.replaceChildren(root);
  attachTooltip(container, root, plot, points, xOf, (point) => [
    el('div', { class: 'chart-tip-title', text: xFormat(point.t, true) }),
    ...series.map((s) => el('div', { class: 'chart-tip-row' },
      el('span', { class: `chart-swatch ${s.cls}` }), el('span', { text: s.label }),
      el('strong', { text: typeof point[s.key] === 'number' ? format(point[s.key]) : '–' }))),
  ]);
}

/** Bars over time, with an optional overlaid count (errors) in a second colour. */
export function barChart(container, { points, key, overlayKey, format, xFormat, label, overlayLabel }) {
  const max = Math.max(0, ...points.map((p) => p[key] || 0));
  if (!max) {
    container.replaceChildren(el('div', { class: 'chart-empty', text: 'No conversations in this period.' }));
    return;
  }
  const { plot, root } = frame(container);
  const axis = niceAxis(max);
  yAxis(root, plot, axis, format);
  const slot = plot.w / points.length;
  const barWidth = Math.max(Math.min(slot * 0.66, 22), 2);
  const xOf = (i) => plot.x + slot * i + slot / 2;
  points.forEach((p, i) => {
    const value = p[key] || 0;
    if (!value) return;
    const h = (value / axis.top) * plot.h;
    root.append(svg('rect', { class: 'chart-bar', x: xOf(i) - barWidth / 2, y: plot.y + plot.h - h, width: barWidth, height: h, rx: 2 }));
    const overlay = overlayKey ? p[overlayKey] || 0 : 0;
    if (overlay) {
      const oh = (overlay / axis.top) * plot.h;
      root.append(svg('rect', { class: 'chart-bar chart-bar-bad', x: xOf(i) - barWidth / 2, y: plot.y + plot.h - oh, width: barWidth, height: oh, rx: 2 }));
    }
  });
  root.append(svg('line', { class: 'chart-axis', x1: plot.x, x2: plot.x + plot.w, y1: plot.y + plot.h, y2: plot.y + plot.h }));
  xLabels(root, plot, points.map((p) => p), xFormat);
  container.replaceChildren(root);
  attachTooltip(container, root, plot, points, xOf, (point) => [
    el('div', { class: 'chart-tip-title', text: xFormat(point.t, true) }),
    el('div', { class: 'chart-tip-row' }, el('span', { class: 'chart-swatch bar' }), el('span', { text: label }), el('strong', { text: format(point[key] || 0) })),
    overlayKey ? el('div', { class: 'chart-tip-row' }, el('span', { class: 'chart-swatch bad' }), el('span', { text: overlayLabel }), el('strong', { text: format(point[overlayKey] || 0) })) : null,
  ].filter(Boolean));
}

/** Tiny trend line for a tile: area fill, line and an emphasised last point. */
export function sparkline(values, { width = 120, height = 32 } = {}) {
  const numbers = values.map((v) => (typeof v === 'number' ? v : null));
  const present = numbers.filter((v) => v !== null);
  const root = svg('svg', { class: 'spark', viewBox: `0 0 ${width} ${height}`, width, height, 'aria-hidden': 'true' });
  if (present.length < 2) return root;
  const max = Math.max(...present) || 1;
  const xOf = (i) => (i / (numbers.length - 1)) * (width - 4) + 2;
  const yOf = (v) => height - 3 - (v / max) * (height - 7);
  let line = '';
  let first = null;
  let last = null;
  numbers.forEach((v, i) => {
    if (v === null) return;
    line += `${line ? 'L' : 'M'}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`;
    first = first ?? i;
    last = i;
  });
  root.append(svg('path', { class: 'spark-area', d: `${line}L${xOf(last)},${height}L${xOf(first)},${height}Z` }));
  root.append(svg('path', { class: 'spark-line', d: line }));
  root.append(svg('circle', { class: 'spark-dot', cx: xOf(last), cy: yOf(numbers[last]), r: 2.5 }));
  return root;
}
