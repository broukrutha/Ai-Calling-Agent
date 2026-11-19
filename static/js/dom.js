/**
 * Shared helpers: DOM building, formatting, the API client, toasts and dialogs.
 *
 * Everything from the server is treated as untrusted text: nodes are built
 * with textContent, never by concatenating HTML.
 */

export const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';

/** el('div', {class: 'x', text: 'hi', dataset: {id: '1'}}, child, …) */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  applyProps(node, props);
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

/** The same for SVG elements. */
export function svg(tag, props = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  applyProps(node, props);
  for (const child of children.flat()) {
    if (child) node.append(child);
  }
  return node;
}

function applyProps(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.setAttribute('class', value);
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
}

export function icon(name, extraClass = '') {
  return svg('svg', { class: `icon ${extraClass}`.trim(), 'aria-hidden': 'true' }, svg('use', { href: `#i-${name}` }));
}

export const show = (node, visible = true) => { if (node) node.hidden = !visible; };
export const setContent = (container, ...nodes) => container.replaceChildren(...nodes.flat().filter(Boolean));
/** A status label in sentence case ("quota exhausted" → "Quota exhausted"). */
export const pill = (text, tone = '') => {
  const label = String(text ?? '');
  return el('span', { class: `pill ${tone}`.trim(), text: label.charAt(0).toUpperCase() + label.slice(1) });
};
export const emptyNote = (text) => el('div', { class: 'empty-inline', text });
export const plural = (count, word, many = `${word}s`) => `${count} ${count === 1 ? word : many}`;

export function skeleton(lines = 3) {
  return el('div', { class: 'skeleton-group', 'aria-hidden': 'true' },
    Array.from({ length: lines }, (_, i) => el('div', { class: `skeleton w${(i % 3) + 1}` })));
}

// ── Formatting ───────────────────────────────────────────────────────────────

export function formatTime(slot) {
  const [hourText, minute = '00'] = String(slot || '').split(':');
  const hour = Number(hourText);
  if (!Number.isFinite(hour)) return slot || '';
  return `${hour % 12 || 12}:${minute} ${hour >= 12 ? 'PM' : 'AM'}`;
}

export function formatDate(iso, options = { weekday: 'short', day: 'numeric', month: 'short' }) {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString('en-IN', options);
}

export function formatDuration(seconds) {
  const total = Math.max(Math.round(Number(seconds) || 0), 0);
  const pad = (n) => String(n).padStart(2, '0');
  const hours = Math.floor(total / 3600);
  const rest = `${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
  return hours ? `${hours}:${rest}` : rest.replace(/^0/, '');
}

export function parseTimestamp(value) {
  if (!value) return null;
  const text = String(value).replace(' ', 'T');
  const date = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(text) ? text : `${text}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatTimestamp(value) {
  const date = parseTimestamp(value);
  return date ? date.toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' }) : '';
}

export function relativeTime(value) {
  const date = parseTimestamp(value);
  if (!date) return '';
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 45) return 'just now';
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return formatTimestamp(value);
}

export function formatMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return '–';
  const value = Number(ms);
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function pct(value, digits = 1) {
  return value === null || value === undefined ? '–' : `${(value * 100).toFixed(digits)}%`;
}

export function inr(value) {
  const amount = Number(value) || 0;
  return amount >= 1 ? `₹${amount.toFixed(2)}` : `₹${amount.toFixed(4)}`;
}

export function localISODate(date) {
  // toISOString() is UTC and gives yesterday's date in the evening east of Greenwich.
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

// ── Storage ──────────────────────────────────────────────────────────────────

export const storage = {
  get(store, key) { try { return store.getItem(key) || ''; } catch (_) { return ''; } },
  set(store, key, value) {
    try { if (value) store.setItem(key, value); else store.removeItem(key); } catch (_) { /* storage blocked */ }
  },
};

const TOKEN_KEY = 'adminApiToken';
export const getToken = () => storage.get(sessionStorage, TOKEN_KEY);
export const setToken = (value) => storage.set(sessionStorage, TOKEN_KEY, value);

// ── API ──────────────────────────────────────────────────────────────────────

const listeners = new Set();
/** Called with true/false whenever an admin request is refused or accepted. */
export const onLockChange = (callback) => listeners.add(callback);

/** JSON request → {ok, status, data}. Never throws for HTTP errors. */
export async function api(path, { method = 'GET', body, form, admin = false, raw = false } = {}) {
  const headers = {};
  if (admin && getToken()) headers['X-Admin-Token'] = getToken();
  const init = { method, headers };
  if (form) init.body = form;
  else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(path, init);
  } catch (_) {
    return { ok: false, status: 0, data: { detail: 'The server could not be reached.' } };
  }
  if (admin) listeners.forEach((callback) => callback(response.status === 401));
  if (raw) return { ok: response.ok, status: response.status, data: await response.text() };
  let data = {};
  try { data = await response.json(); } catch (_) { /* empty or non-JSON body */ }
  return { ok: response.ok, status: response.status, data };
}

export function errorText(result, fallback) {
  if (result.status === 401) return 'Enter the admin token in Settings to do this.';
  if (result.status === 429) return 'Too many requests. Wait a moment and try again.';
  const detail = result.data && result.data.detail;
  return typeof detail === 'string' && detail ? detail : fallback;
}

// ── Feedback ─────────────────────────────────────────────────────────────────

export function toast(message, tone = '') {
  const container = $('toast-container');
  while (container.children.length >= 3) container.firstChild.remove();
  const node = el('div', { class: `toast ${tone}`.trim() }, icon(tone === 'bad' ? 'alert' : 'check'), el('span', { text: message }));
  container.append(node);
  setTimeout(() => {
    node.classList.add('leaving');
    setTimeout(() => node.remove(), 220);
  }, 4200);
}

/** In-page confirmation. Resolves true only when the user confirms. */
export function confirmAction(title, text, confirmLabel = 'Confirm', danger = true) {
  const dialog = $('confirm-dialog');
  $('confirm-title').textContent = title;
  $('confirm-text').textContent = text;
  const ok = $('confirm-ok');
  ok.textContent = confirmLabel;
  ok.className = `btn ${danger ? 'btn-danger' : 'btn-primary'}`;
  return new Promise((resolve) => {
    const finish = (value) => {
      ok.removeEventListener('click', onOk);
      $('confirm-cancel').removeEventListener('click', onCancel);
      dialog.removeEventListener('close', onClose);
      if (dialog.open) dialog.close();
      resolve(value);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onClose = () => finish(false);
    ok.addEventListener('click', onOk);
    $('confirm-cancel').addEventListener('click', onCancel);
    dialog.addEventListener('close', onClose);
    dialog.showModal();
  });
}

export function download(filename, content, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  el('a', { href: url, download: filename }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Animate a number from its current value (respects reduced motion). */
export function countTo(node, value, format = (v) => String(Math.round(v))) {
  const target = Number(value);
  if (!Number.isFinite(target) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    node.textContent = Number.isFinite(target) ? format(target) : String(value ?? '–');
    return;
  }
  const start = Number(node.dataset.value || 0);
  node.dataset.value = String(target);
  const began = performance.now();
  const step = (now) => {
    const t = Math.min((now - began) / 450, 1);
    const eased = 1 - (1 - t) ** 3;
    node.textContent = format(start + (target - start) * eased);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
