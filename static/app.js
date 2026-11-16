/**
 * Calling Agent Console.
 *
 * Everything the server sends is treated as untrusted text: the page is built
 * from DOM nodes and textContent, never by concatenating HTML, and there are
 * no inline handlers or styles, so it runs under a strict Content-Security-Policy.
 */

'use strict';

// ── DOM helpers ──────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';

/** el('div', {class: 'x', text: 'hi', dataset: {uid: '1'}}, child, …) */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

function icon(name, extraClass = '') {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', `icon ${extraClass}`.trim());
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

const show = (node, visible = true) => { if (node) node.hidden = !visible; };
const setContent = (container, ...nodes) => container.replaceChildren(...nodes);
const pill = (text, tone = '') => el('span', { class: `pill ${tone}`.trim(), text });
const emptyNote = (text) => el('div', { class: 'empty-inline', text });
const skeleton = () => el('div', {}, el('div', { class: 'skeleton' }), el('div', { class: 'skeleton' }));
const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;

// ── State ────────────────────────────────────────────────────────────────────

const state = {
  view: 'overview',
  health: null,
  budgetMs: 3000,
  locked: false,
  ws: null,
  wsRetryMs: 1000,
  wsTimer: null,
  recorder: null,
  recording: false,
  recordStartedAt: 0,
  chunks: [],
  audioContext: null,
  meterFrame: 0,
  streamingEl: null,
  streamingText: '',
  audioQueue: [],
  currentAudio: null,
  slot: '',
  bookingFilter: '',
  benchmark: null,
};

const TOKEN_KEY = 'adminApiToken';
const storage = {
  get(store, key) { try { return store.getItem(key) || ''; } catch (_) { return ''; } },
  set(store, key, value) {
    try { if (value) store.setItem(key, value); else store.removeItem(key); } catch (_) { /* storage blocked */ }
  },
};
const getToken = () => storage.get(sessionStorage, TOKEN_KEY);

// ── HTTP ─────────────────────────────────────────────────────────────────────

/** JSON request → {ok, status, data}. Never throws for HTTP errors. */
async function api(path, { method = 'GET', body, form, admin = false } = {}) {
  const headers = {};
  if (admin && getToken()) headers['X-Admin-Token'] = getToken();
  const init = { method, headers };
  if (form) {
    init.body = form;
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(path, init);
  } catch (_) {
    return { ok: false, status: 0, data: { detail: 'The server could not be reached.' } };
  }
  let data = {};
  try { data = await response.json(); } catch (_) { /* empty or non-JSON body */ }
  if (admin) setLocked(response.status === 401);
  return { ok: response.ok, status: response.status, data };
}

function errorText(result, fallback) {
  if (result.status === 401) return 'Enter the admin token in Settings to do this.';
  if (result.status === 429) return 'Too many requests. Wait a moment and try again.';
  const detail = result.data && result.data.detail;
  return typeof detail === 'string' && detail ? detail : fallback;
}

function setLocked(locked) {
  state.locked = locked;
  show($('lock-banner'), locked);
}

// ── Feedback ─────────────────────────────────────────────────────────────────

function toast(message, tone = '') {
  const container = $('toast-container');
  while (container.children.length >= 3) container.firstChild.remove();
  const node = el('div', { class: `toast ${tone}`.trim(), text: message });
  container.append(node);
  setTimeout(() => {
    node.classList.add('leaving');
    setTimeout(() => node.remove(), 260);
  }, 4200);
}

/** In-page confirmation. Resolves true only when the user confirms. */
function confirmAction(title, text, confirmLabel = 'Confirm') {
  const dialog = $('confirm-dialog');
  $('confirm-title').textContent = title;
  $('confirm-text').textContent = text;
  $('confirm-ok').textContent = confirmLabel;
  return new Promise((resolve) => {
    const finish = (value) => {
      $('confirm-ok').removeEventListener('click', onOk);
      $('confirm-cancel').removeEventListener('click', onCancel);
      dialog.removeEventListener('close', onClose);
      if (dialog.open) dialog.close();
      resolve(value);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onClose = () => finish(false);
    $('confirm-ok').addEventListener('click', onOk);
    $('confirm-cancel').addEventListener('click', onCancel);
    dialog.addEventListener('close', onClose);
    dialog.showModal();
  });
}

// ── Formatting ───────────────────────────────────────────────────────────────

function formatTime(slot) {
  const [hourText, minute = '00'] = String(slot || '').split(':');
  const hour = Number(hourText);
  if (!Number.isFinite(hour)) return slot || '';
  return `${hour % 12 || 12}:${minute} ${hour >= 12 ? 'PM' : 'AM'}`;
}

function formatDate(iso, options = { weekday: 'short', day: 'numeric', month: 'short' }) {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString('en-IN', options);
}

function formatDuration(seconds) {
  const total = Math.max(Math.round(Number(seconds) || 0), 0);
  const pad = (n) => String(n).padStart(2, '0');
  const hours = Math.floor(total / 3600);
  const rest = `${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
  return hours ? `${hours}:${rest}` : rest.replace(/^0/, '');
}

function formatTimestamp(utc) {
  if (!utc) return '';
  const date = new Date(`${String(utc).replace(' ', 'T')}Z`);
  return Number.isNaN(date.getTime())
    ? String(utc)
    : date.toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
}

const formatMs = (ms) => {
  const value = Number(ms) || 0;
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(0)} ms`;
};

function localISODate(date) {
  // toISOString() is UTC and yields yesterday's date in the evening east of Greenwich.
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

// ── Navigation ───────────────────────────────────────────────────────────────

const VIEW_LOADERS = {
  overview: () => loadOverview(),
  playground: () => checkHealth(),
  knowledge: () => loadSources(),
  appointments: () => { loadBookingStats(); loadBookings(); loadSlots(); },
  calls: () => loadCalls(),
  evaluation: () => {},
};

function showView(name) {
  const view = VIEW_LOADERS[name] ? name : 'overview';
  state.view = view;
  document.querySelectorAll('.view').forEach((section) => { section.hidden = section.id !== `view-${view}`; });
  document.querySelectorAll('.nav-item[data-view]').forEach((link) => {
    const active = link.dataset.view === view;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  const section = $(`view-${view}`);
  $('view-title').textContent = section.dataset.title;
  $('view-subtitle').textContent = section.dataset.subtitle;
  document.title = `${section.dataset.title} · Calling Agent Console`;
  closeNav();
  VIEW_LOADERS[view]();
}

function openNav() { $('sidebar').classList.add('open'); show($('scrim'), true); }
function closeNav() { $('sidebar').classList.remove('open'); show($('scrim'), false); }

// ── Theme ────────────────────────────────────────────────────────────────────

const THEMES = ['system', 'light', 'dark'];

function currentTheme() {
  const saved = storage.get(localStorage, 'theme');
  return THEMES.includes(saved) ? saved : 'system';
}

function applyTheme(theme) {
  if (theme === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
  storage.set(localStorage, 'theme', theme === 'system' ? '' : theme);
  $('theme-label').textContent = `Theme: ${theme[0].toUpperCase()}${theme.slice(1)}`;
}

function cycleTheme() {
  applyTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);
}

// ── Health and system status ─────────────────────────────────────────────────

function setConnection(tone, text) {
  $('conn-dot').className = `dot ${tone}`;
  $('conn-text').textContent = text;
}

async function checkHealth() {
  const result = await api('/health');
  if (!result.ok) {
    state.health = null;
    setConnection('bad', 'Server unreachable');
    return null;
  }
  const health = result.data;
  state.health = health;
  if (Number.isFinite(health.latency_budget_ms)) state.budgetMs = health.latency_budget_ms;

  const model = health.api || {};
  if (health.status === 'ok') setConnection('good', 'All systems operational');
  else if (model.status === 'api_key_missing') setConnection('warn', 'Setup incomplete');
  else if (model.status === 'error') setConnection('bad', 'Language model unreachable');
  else setConnection('warn', 'Running with limits');

  $('app-version').textContent = health.version ? `v${health.version}` : '';
  $('about-version').textContent = health.version || '–';
  $('about-env').textContent = health.environment || '–';
  $('session-llm').textContent = model.llm_model ? `${model.llm_provider} · ${model.llm_model}` : 'Not configured';
  $('session-speech').textContent = model.stt_model ? `${model.stt_model} · ${model.tts_model}` : 'Not configured';
  renderTokenState();
  return health;
}

function statusRow(name, detail, label, tone) {
  return el('li', {},
    el('div', { class: 'row-main' },
      el('div', { class: 'row-title', text: name }),
      el('div', { class: 'row-sub', text: detail, title: detail }),
    ),
    pill(label, tone),
  );
}

function renderSystemStatus(health) {
  const model = health.api || {};
  const kb = health.knowledge_base || {};
  const telephony = health.telephony || {};
  const rows = [];

  if (model.status === 'ok') {
    const latency = Number.isFinite(model.latency_ms) ? ` · ${model.latency_ms.toFixed(0)} ms` : '';
    rows.push(statusRow('Language model', `${model.llm_provider} · ${model.llm_model}${latency}`, 'Operational', 'good'));
  } else if (model.status === 'api_key_missing') {
    rows.push(statusRow('Language model', 'Add provider keys to the server environment', 'Not configured', 'warn'));
  } else {
    rows.push(statusRow('Language model', model.message || 'The provider did not respond', 'Unreachable', 'bad'));
  }

  rows.push(model.status === 'api_key_missing'
    ? statusRow('Speech', 'Set SARVAM_API_KEY to enable voice', 'Not configured', 'warn')
    : statusRow('Speech', `${model.stt_model || 'speech-to-text'} · ${model.tts_model || 'text-to-speech'}`, 'Ready', 'good'));

  if (kb.available === false) rows.push(statusRow('Knowledge base', 'The embedding model could not be loaded', 'Unavailable', 'bad'));
  else if (!kb.total_chunks) rows.push(statusRow('Knowledge base', 'No documents indexed yet', 'Empty', 'warn'));
  else rows.push(statusRow('Knowledge base', `${plural(kb.total_chunks, 'passage')} indexed`, 'Ready', 'good'));

  rows.push((health.bookings || {}).mode === 'mock'
    ? statusRow('Calendar', 'Bookings are saved locally, not to Cal.com', 'Mock mode', 'warn')
    : statusRow('Calendar', 'Cal.com', 'Connected', 'good'));

  if (telephony.configured && telephony.outbound) rows.push(statusRow('Telephony', 'LiveKit + SIP trunk', 'Ready', 'good'));
  else if (telephony.configured) rows.push(statusRow('Telephony', 'Inbound only: no outbound trunk configured', 'Partial', 'warn'));
  else rows.push(statusRow('Telephony', 'LiveKit and SIP settings are missing', 'Not configured', ''));

  rows.push(health.auth_required
    ? statusRow('Access control', 'Admin token required for changes', 'Protected', 'good')
    : statusRow('Access control', 'No admin token set on the server', 'Open', 'warn'));

  setContent($('status-list'), ...rows);
  setContent($('overview-env'), document.createTextNode(health.environment || ''));
  $('overview-env').className = `pill plain ${health.environment === 'production' ? 'accent' : ''}`.trim();

  $('tile-kb').textContent = kb.available === false ? '–' : String(kb.total_chunks || 0);
  $('tile-kb-foot').textContent = kb.available === false ? 'Unavailable' : 'Indexed passages';
}

// ── Overview ─────────────────────────────────────────────────────────────────

function callRow(call) {
  const outbound = call.direction === 'outbound';
  const tones = { completed: 'good', in_progress: 'accent', failed: 'bad', ringing: 'accent' };
  return el('div', { class: 'row' },
    el('span', { class: `dir ${outbound ? 'out' : ''}`.trim(), title: outbound ? 'Outbound' : 'Inbound' }, icon(outbound ? 'out' : 'in')),
    el('div', { class: 'row-main' },
      el('div', { class: 'row-title mono', text: call.phone || 'Unknown number' }),
      el('div', { class: 'row-sub', text: [formatTimestamp(call.created_at), call.detail].filter(Boolean).join(' · ') }),
    ),
    el('div', { class: 'row-aside' },
      el('span', { text: formatDuration(call.duration_sec) }),
      pill(String(call.status || '').replace('_', ' '), tones[call.status] || ''),
    ),
  );
}

async function loadOverview() {
  for (const id of ['overview-calls', 'overview-appointments']) {
    if (!$(id).children.length) setContent($(id), skeleton());
  }
  const [health, calls, bookings] = await Promise.all([
    checkHealth(),
    api('/calls?limit=6', { admin: true }),
    api('/booking/stats', { admin: true }),
  ]);
  if (health) renderSystemStatus(health);
  else setContent($('status-list'), el('li', {}, el('div', { class: 'row-sub', text: 'The server is not responding.' })));

  if (calls.ok) {
    const stats = calls.data.stats || {};
    $('tile-calls').textContent = String(stats.total || 0);
    $('tile-calls-foot').textContent = `${stats.inbound || 0} inbound · ${stats.outbound || 0} outbound`;
    $('tile-talk').textContent = formatDuration(stats.total_duration_sec);
    $('tile-talk-foot').textContent = stats.failed ? `${plural(stats.failed, 'call')} not connected` : 'Across all calls';
    const list = calls.data.calls || [];
    setContent($('overview-calls'), ...(list.length
      ? list.map(callRow)
      : [emptyNote('No calls yet. Calls appear here as soon as the telephony worker handles one.')]));
  } else {
    setContent($('overview-calls'), emptyNote(errorText(calls, 'Calls could not be loaded.')));
  }

  if (bookings.ok) {
    const stats = bookings.data.stats || {};
    $('tile-upcoming').textContent = String(stats.upcoming || 0);
    $('tile-upcoming-foot').textContent = `${stats.total || 0} booked in total · ${stats.cancelled || 0} cancelled`;
    const next = bookings.data.next_appointments || [];
    setContent($('overview-appointments'), ...(next.length
      ? next.map((booking) => el('div', { class: 'row' },
        el('div', { class: 'row-main' },
          el('div', { class: 'row-title', text: `${formatDate(booking.date)} · ${formatTime(booking.time_ist)}` }),
          el('div', { class: 'row-sub', text: `${booking.attendee_name} · ${booking.attendee_email}` }),
        ),
        el('div', { class: 'row-aside' }, pill(booking.source || 'manual', 'plain')),
      ))
      : [emptyNote('Nothing scheduled. New bookings from calls and chat show up here.')]));
  } else {
    setContent($('overview-appointments'), emptyNote(errorText(bookings, 'Appointments could not be loaded.')));
  }
}

// ── Settings ─────────────────────────────────────────────────────────────────

function renderTokenState() {
  const note = $('admin-token-state');
  if (getToken()) note.textContent = 'A token is saved for this browser tab.';
  else if (state.health && state.health.auth_required) note.textContent = 'Required to manage documents, bookings and calls.';
  else note.textContent = 'This server has no admin token set, so no token is needed (development mode).';
}

function openSettings() {
  renderTokenState();
  $('settings-dialog').showModal();
}

function saveToken(event) {
  event.preventDefault();
  const value = $('admin-token-input').value.trim();
  if (value) storage.set(sessionStorage, TOKEN_KEY, value);
  $('admin-token-input').value = '';
  $('settings-dialog').close();
  if (value) toast('Admin token saved for this tab.');
  showView(state.view);
}

function clearToken() {
  storage.set(sessionStorage, TOKEN_KEY, '');
  renderTokenState();
  toast('Admin token removed.');
}

// ── WebSocket ────────────────────────────────────────────────────────────────

function connectWebSocket() {
  clearTimeout(state.wsTimer);
  if (state.ws && state.ws.readyState <= WebSocket.OPEN) return;

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/voice`);
  state.ws = socket;

  socket.addEventListener('open', () => {
    state.wsRetryMs = 1000;
    socket.send(JSON.stringify({ type: 'ping' }));
  });
  socket.addEventListener('message', (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch (_) { return; }
    handleSocketMessage(message);
  });
  socket.addEventListener('close', () => {
    finishTurn();
    state.wsTimer = setTimeout(connectWebSocket, state.wsRetryMs);
    state.wsRetryMs = Math.min(state.wsRetryMs * 2, 15000);     // back off, up to 15 s
  });
}

const socketReady = () => Boolean(state.ws) && state.ws.readyState === WebSocket.OPEN;

// Keeps idle connections open through proxies and tunnels.
setInterval(() => { if (socketReady()) state.ws.send(JSON.stringify({ type: 'ping' })); }, 25000);

function handleSocketMessage(message) {
  switch (message.type) {
    case 'processing':
      setBusy(true, 'Transcribing…');
      break;
    case 'transcript':
      addMessage('user', message.text);
      setBusy(true, 'Thinking…');
      break;
    case 'noise':
      finishTurn();
      toast(message.message || 'That was hard to make out. Please try again.');
      break;
    case 'text_chunk':
      state.streamingText += message.text || '';
      if (!state.streamingEl) state.streamingEl = addMessage('assistant', state.streamingText);
      else state.streamingEl.querySelector('.msg-text').textContent = state.streamingText;
      scrollTranscript();
      break;
    case 'audio_chunk':
      if (message.audio) enqueueAudio(message.audio);
      break;
    case 'done':
      if (state.streamingEl) annotateAnswer(state.streamingEl, message);
      finishTurn();
      renderTurn(message, 'Voice');
      break;
    case 'text_result':
      annotateAnswer(addMessage('assistant', message.answer || 'No answer was returned.'), message);
      finishTurn();
      renderTurn(message, 'Text');
      break;
    case 'error':
      finishTurn();
      toast(message.message || 'Something went wrong.', 'bad');
      break;
    default:
      break;
  }
}

function setBusy(busy, text = '') {
  show($('query-status'), busy);
  if (text) $('query-status-text').textContent = text;
  $('send-btn').disabled = busy;
}

function finishTurn() {
  setBusy(false);
  state.streamingEl = null;
  state.streamingText = '';
}

// ── Audio playback ───────────────────────────────────────────────────────────

function base64ToBlob(base64, type) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type });
}

function enqueueAudio(base64) {
  state.audioQueue.push(base64);
  if (!state.currentAudio) playNext();
}

function playNext() {
  const next = state.audioQueue.shift();
  if (!next) { state.currentAudio = null; return; }
  let url;
  try {
    url = URL.createObjectURL(base64ToBlob(next, 'audio/wav'));
  } catch (_) {
    playNext();
    return;
  }
  const audio = new Audio(url);
  state.currentAudio = audio;
  const advance = () => { URL.revokeObjectURL(url); playNext(); };
  audio.addEventListener('ended', advance, { once: true });
  audio.addEventListener('error', advance, { once: true });
  audio.play().catch(advance);
}

/** Stop the agent speaking: used when the user starts to talk (barge-in). */
function stopPlayback() {
  state.audioQueue = [];
  if (state.currentAudio) {
    state.currentAudio.pause();
    state.currentAudio = null;
  }
}

// ── Recording ────────────────────────────────────────────────────────────────

const METER_BARS = 18;

function buildMeter() {
  const bars = [];
  for (let i = 0; i < METER_BARS; i += 1) bars.push(el('i'));
  setContent($('meter'), ...bars);
}

/** Drive the level meter from the live microphone signal. */
function startMeter(stream) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  const context = new AudioContextClass();
  const analyser = context.createAnalyser();
  analyser.fftSize = 64;
  context.createMediaStreamSource(stream).connect(analyser);
  state.audioContext = context;

  const levels = new Uint8Array(analyser.frequencyBinCount);
  const bars = $('meter').children;
  const draw = () => {
    analyser.getByteFrequencyData(levels);
    for (let i = 0; i < bars.length; i += 1) {
      bars[i].style.height = `${3 + Math.round((levels[i + 1] / 255) * 19)}px`;
    }
    const seconds = Math.floor((performance.now() - state.recordStartedAt) / 1000);
    $('rec-timer').textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    state.meterFrame = requestAnimationFrame(draw);
  };
  draw();
}

function stopMeter() {
  cancelAnimationFrame(state.meterFrame);
  if (state.audioContext) { state.audioContext.close().catch(() => {}); state.audioContext = null; }
}

function setRecordingUI(recording) {
  state.recording = recording;
  const button = $('mic-btn');
  button.classList.toggle('recording', recording);
  button.setAttribute('aria-pressed', String(recording));
  button.setAttribute('aria-label', recording ? 'Stop and send' : 'Start recording');
  show($('composer-live'), recording);
  show($('text-input'), !recording);
  show($('send-btn'), !recording);
}

function supportedMime() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  return candidates.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || '';
}

async function toggleRecording() {
  if (state.recording) {
    if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop();
    return;
  }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    toast('This browser cannot record audio. Type your message instead.', 'bad');
    return;
  }
  if (!socketReady()) {
    toast('Reconnecting to the server. Try again in a moment.');
    connectWebSocket();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (_) {
    toast('Microphone access is blocked. Allow it in your browser settings.', 'bad');
    return;
  }

  stopPlayback();
  const mimeType = supportedMime();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  state.recorder = recorder;
  state.chunks = [];
  state.recordStartedAt = performance.now();

  recorder.addEventListener('dataavailable', (event) => {
    if (event.data.size > 0) state.chunks.push(event.data);
  });
  recorder.addEventListener('stop', () => {
    const duration = performance.now() - state.recordStartedAt;
    stopMeter();
    stream.getTracks().forEach((track) => track.stop());
    setRecordingUI(false);
    sendRecording(recorder.mimeType || mimeType, duration);
  }, { once: true });

  recorder.start(250);
  setRecordingUI(true);
  startMeter(stream);
}

function blobToBase64(blob) {
  // FileReader copes with recordings of any length; String.fromCharCode(...bytes)
  // overflows the call stack once a recording passes roughly 100 KB.
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] || '');
    reader.readAsDataURL(blob);
  });
}

async function sendRecording(mimeType, durationMs) {
  if (durationMs < 300 || state.chunks.length === 0) {
    toast('That was too short. Keep recording while you speak.');
    return;
  }
  if (!socketReady()) {
    toast('The connection dropped. Please try again.', 'bad');
    return;
  }
  let data;
  try {
    data = await blobToBase64(new Blob(state.chunks, { type: mimeType || 'audio/webm' }));
  } catch (_) {
    toast('The recording could not be read.', 'bad');
    return;
  }
  setBusy(true, 'Sending audio…');
  state.ws.send(JSON.stringify({
    type: 'audio',
    data,
    format: (mimeType || 'audio/webm').split(';')[0].replace('audio/', ''),
    language: $('language-select').value,
    duration_ms: Math.round(durationMs),
  }));
}

// ── Conversation ─────────────────────────────────────────────────────────────

function scrollTranscript() {
  const transcript = $('conversation');
  transcript.scrollTop = transcript.scrollHeight;
}

function addMessage(role, text) {
  show($('chat-empty'), false);
  const node = el('div', { class: `msg ${role}` },
    el('div', { class: 'msg-who', text: role === 'user' ? 'You' : 'Agent' }),
    el('div', { class: 'msg-text', text }),
  );
  $('conversation').append(node);
  scrollTranscript();
  return node;
}

/** Sources, reference number and cache marker under a finished answer. */
function annotateAnswer(node, result) {
  // Booking replies come from the calendar, not from documents: no source tags.
  const sources = result.is_booking ? [] : result.sources || [];
  const tags = sources.map((source) => el('span', { class: 'tag', text: source, title: source }));
  if (result.ref_code) tags.unshift(el('span', { class: 'tag ref', text: `Ref ${result.ref_code}` }));
  if (result.cached) tags.push(el('span', { class: 'tag', text: 'From cache' }));
  if (tags.length) node.append(el('div', { class: 'msg-meta' }, ...tags));
  scrollTranscript();
}

function clearConversation() {
  const empty = $('chat-empty');
  setContent($('conversation'), empty);
  show(empty, true);
}

async function sendText(query) {
  if (!query) return;
  addMessage('user', query);
  setBusy(true, 'Thinking…');
  stopPlayback();

  if (socketReady()) {
    state.ws.send(JSON.stringify({ type: 'query', query }));
    return;
  }
  // No socket: answer over REST. Works, but without conversation memory.
  const result = await api('/query', { method: 'POST', body: { query } });
  finishTurn();
  if (!result.ok) { toast(errorText(result, 'The question could not be answered.'), 'bad'); return; }
  annotateAnswer(addMessage('assistant', result.data.answer), result.data);
  renderTurn(result.data, 'Text');
}

function submitChat(event) {
  event.preventDefault();
  const input = $('text-input');
  const query = input.value.trim();
  input.value = '';
  autoGrow(input);
  sendText(query);
}

function autoGrow(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
}

// ── Turn inspector ───────────────────────────────────────────────────────────

/**
 * Latency waterfall: each stage is drawn where it happened on the turn's
 * timeline, so overlap (speech synthesis starting before the model finishes)
 * is visible. A dashed marker shows when the caller first heard audio.
 */
function renderTurn(result, kind) {
  const latency = result.latency;
  if (!latency) return;
  show($('turn-empty'), false);
  show($('turn-detail'), true);
  show($('turn-kind'), true);
  $('turn-kind').textContent = result.is_booking ? `${kind} · booking` : kind;
  $('turn-kind').className = 'pill plain';

  const total = Math.max(Number(latency.total_ms) || 0, 1);
  const stt = Number(latency.stt_ms) || 0;
  const search = Number(latency.retrieval_ms) || 0;
  const llm = Number(latency.llm_ms) || 0;
  const tts = Number(latency.tts_ms) || 0;
  const first = Number(latency.first_audio_ms) || 0;

  const stages = [
    ['Speech in', 'stt', 0, stt],
    ['Search', 'search', stt, search],
    ['Model', 'llm', stt + search, llm],
    ['Speech out', 'tts', Math.max(total - tts, 0), tts],
  ].filter(([, , , duration]) => duration > 0);

  const rows = stages.map(([name, key, start, duration]) => {
    const bar = el('div', { class: `wf-bar ${key}` });
    bar.style.left = `${Math.min((start / total) * 100, 99)}%`;
    bar.style.width = `${Math.min((duration / total) * 100, 100 - (start / total) * 100)}%`;
    return el('div', { class: 'wf-row' },
      el('span', { class: 'wf-name', text: name }),
      el('div', { class: 'wf-lane' }, bar),
      el('span', { class: 'wf-ms', text: formatMs(duration) }),
    );
  });
  if (first > 0) {
    const position = Math.min((first / total) * 100, 100);
    const marker = el('div', { class: `wf-marker ${position > 70 ? 'flip' : ''}`.trim() }, el('span', { text: 'first audio' }));
    marker.style.left = `${position}%`;
    rows.push(el('div', { class: 'wf-axis' }, marker));
  }
  setContent($('waterfall'), ...(rows.length ? rows : [el('div', { class: 'help', text: 'Answered without calling the model.' })]));

  const grade = (ms) => (ms <= state.budgetMs ? 'good' : ms <= state.budgetMs * 1.3 ? 'warn' : 'bad');
  $('val-first').textContent = first ? formatMs(first) : kind === 'Text' ? 'Text only' : 'No audio';
  $('val-first').className = `kv-big ${first ? grade(first) : ''}`.trim();
  $('val-total').textContent = formatMs(total);
  $('val-cost').textContent = result.cost_inr !== undefined ? `₹${Number(result.cost_inr).toFixed(5)}` : '–';
  $('val-budget').textContent = `${formatMs(state.budgetMs)} to first audio`;
}

// ── Knowledge base ───────────────────────────────────────────────────────────

function selectSourceKind(kind) {
  show($('source-file'), kind === 'file');
  show($('url-form'), kind === 'url');
  show($('text-form'), kind === 'text');
  document.querySelectorAll('[data-action="source-kind"]').forEach((button) => {
    const active = button.dataset.kind === kind;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
}

function setIngesting(busy, text = '') {
  show($('ingest-status'), busy);
  if (text) $('ingest-status-text').textContent = text;
}

function finishIngest(result, label) {
  setIngesting(false);
  if (!result.ok) { toast(errorText(result, 'That source could not be added.'), 'bad'); return false; }
  toast(`${label} added: ${plural(result.data.chunks_added, 'passage')} indexed.`);
  loadSources();
  return true;
}

async function uploadPDF(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) { toast('Choose a PDF file.', 'bad'); return; }
  setIngesting(true, `Uploading ${file.name}…`);
  const form = new FormData();
  form.append('file', file);
  finishIngest(await api('/ingest/pdf', { method: 'POST', form, admin: true }), 'PDF');
  $('pdf-file-input').value = '';
}

async function ingestURL(event) {
  event.preventDefault();
  const url = $('url-input').value.trim();
  if (!url) return;
  setIngesting(true, 'Fetching the page…');
  if (finishIngest(await api('/ingest/url', { method: 'POST', body: { url }, admin: true }), 'Page')) $('url-input').value = '';
}

async function ingestText(event) {
  event.preventDefault();
  const text = $('text-area-input').value.trim();
  if (!text) return;
  const name = $('text-name-input').value.trim() || 'Pasted text';
  setIngesting(true, 'Indexing…');
  if (finishIngest(await api('/ingest/text', { method: 'POST', body: { text, name }, admin: true }), 'Text')) {
    $('text-area-input').value = '';
    $('text-name-input').value = '';
  }
}

async function loadSources() {
  const list = $('kb-sources');
  if (!list.children.length) setContent(list, skeleton());
  const result = await api('/kb/sources', { admin: true });
  if (!result.ok) {
    $('sources-count').textContent = '';
    setContent(list, emptyNote(errorText(result, 'Sources could not be loaded.')));
    return;
  }
  const sources = result.data.sources || [];
  $('sources-count').textContent = sources.length ? `(${sources.length})` : '';
  if (sources.length === 0) {
    setContent(list, emptyNote('No sources yet. Add a PDF, a web page or some text and the agent can answer from it straight away.'));
    return;
  }
  setContent(list, ...sources.map((item) => {
    const isLink = /^https?:\/\//i.test(item.source);
    return el('div', { class: 'row' },
      el('div', { class: 'row-main' },
        el('div', { class: 'row-title', text: item.source, title: item.source }),
        el('div', { class: 'row-sub', text: `${isLink ? 'Web page' : item.source.toLowerCase().endsWith('.pdf') ? 'PDF' : 'Text'} · ${plural(item.chunks, 'passage')}` }),
      ),
      el('button', {
        class: 'btn btn-sm btn-danger-ghost', type: 'button', text: 'Remove',
        dataset: { action: 'delete-source', source: item.source },
      }),
    );
  }));
}

async function deleteSource(source) {
  if (!await confirmAction('Remove this source?', `“${source}” will no longer be used to answer callers.`, 'Remove')) return;
  const result = await api(`/kb/sources?source=${encodeURIComponent(source)}`, { method: 'DELETE', admin: true });
  if (!result.ok) { toast(errorText(result, 'That source could not be removed.'), 'bad'); return; }
  toast('Source removed.');
  loadSources();
}

async function clearKB() {
  if (!await confirmAction('Remove every source?', 'The agent will have nothing to answer from until you add documents again. This cannot be undone.', 'Remove all')) return;
  const result = await api('/kb/clear', { method: 'DELETE', admin: true });
  if (!result.ok) { toast(errorText(result, 'The knowledge base could not be cleared.'), 'bad'); return; }
  toast('Knowledge base cleared.');
  loadSources();
}

// ── Appointments ─────────────────────────────────────────────────────────────

function tile(label, value, tone = '') {
  return el('div', { class: 'tile' },
    el('div', { class: 'tile-label', text: label }),
    el('div', { class: `tile-value ${tone}`.trim(), text: String(value) }),
  );
}

function initBookingDate() {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  $('book-date').min = localISODate(today);
  $('book-date').value = localISODate(tomorrow);
}

async function loadSlots() {
  const day = $('book-date').value;
  const box = $('slots');
  state.slot = '';
  if (!day) { setContent(box, el('span', { class: 'help', text: 'Choose a date to see open times.' })); return; }
  const result = await api(`/booking/slots?date=${encodeURIComponent(day)}`);
  if (day !== $('book-date').value) return;                     // the date changed while loading
  if (!result.ok) { setContent(box, el('span', { class: 'help', text: errorText(result, 'Open times could not be loaded.') })); return; }
  const slots = result.data.available_slots || [];
  if (slots.length === 0) { setContent(box, el('span', { class: 'help', text: 'No open times on this date.' })); return; }
  setContent(box, ...slots.map((slot) => el('button', {
    class: 'slot', type: 'button', role: 'radio', 'aria-checked': 'false', text: formatTime(slot),
    dataset: { action: 'pick-slot', slot },
  })));
}

function pickSlot(button) {
  state.slot = button.dataset.slot;
  document.querySelectorAll('#slots .slot').forEach((other) => other.setAttribute('aria-checked', String(other === button)));
}

async function bookAppointment(event) {
  event.preventDefault();
  if (!state.slot) { toast('Choose one of the open times.', 'bad'); return; }
  const body = {
    name: $('book-name').value.trim(),
    email: $('book-email').value.trim(),
    date: $('book-date').value,
    time: state.slot,
  };
  $('book-btn').disabled = true;
  const result = await api('/booking/create', { method: 'POST', body, admin: true });
  $('book-btn').disabled = false;

  if (!result.ok) {
    setContent($('book-result'), el('div', { class: 'notice bad', text: errorText(result, 'The appointment could not be booked.') }));
    if (result.status === 409) loadSlots();
    return;
  }
  const reference = result.data.ref_code ? ` Reference ${result.data.ref_code}.` : '';
  setContent($('book-result'), el('div', { class: 'notice good', text: `${result.data.message}${reference}` }));
  $('book-name').value = '';
  $('book-email').value = '';
  loadSlots();
  loadBookingStats();
  loadBookings();
}

async function loadBookingStats() {
  const result = await api('/booking/stats', { admin: true });
  if (!result.ok) { setContent($('booking-stats')); return; }
  const stats = result.data.stats || {};
  setContent($('booking-stats'),
    tile('Upcoming', stats.upcoming || 0),
    tile('Confirmed', stats.confirmed || 0),
    tile('Cancelled', stats.cancelled || 0),
    tile('All time', stats.total || 0),
    tile('Calendar', stats.mode === 'mock' ? 'Mock' : 'Cal.com', stats.mode === 'mock' ? 'warn' : 'good'),
  );
}

async function loadBookings() {
  const rows = $('booking-rows');
  const note = $('booking-empty');
  const query = state.bookingFilter ? `&status=${state.bookingFilter}` : '';
  const result = await api(`/booking/history?limit=50${query}`, { admin: true });
  if (!result.ok) {
    setContent(rows);
    note.textContent = errorText(result, 'Appointments could not be loaded.');
    show(note, true);
    return;
  }
  const bookings = result.data.bookings || [];
  show(note, bookings.length === 0);
  note.textContent = state.bookingFilter ? `No ${state.bookingFilter} appointments.` : 'No appointments yet.';
  setContent(rows, ...bookings.map((booking) => {
    const confirmed = booking.status === 'confirmed';
    const action = confirmed && booking.cal_booking_uid
      ? el('button', {
        class: 'btn btn-sm btn-danger-ghost', type: 'button', text: 'Cancel',
        dataset: { action: 'cancel-booking', uid: booking.cal_booking_uid, when: `${formatDate(booking.date)} at ${formatTime(booking.time_ist)}` },
      })
      : null;
    return el('tr', {},
      el('td', {}, document.createTextNode(formatDate(booking.date)), el('span', { class: 'cell-sub', text: formatTime(booking.time_ist) })),
      el('td', { class: 'cell-wrap' }, document.createTextNode(booking.attendee_name || ''), el('span', { class: 'cell-sub', text: booking.attendee_email || '' })),
      el('td', { class: 'mono', text: booking.ref_code || '–' }),
      el('td', { text: booking.source || '–' }),
      el('td', {}, pill(booking.status || 'unknown', confirmed ? 'good' : '')),
      el('td', { class: 'num' }, action),
    );
  }));
}

function setBookingFilter(button) {
  state.bookingFilter = button.dataset.status;
  document.querySelectorAll('[data-action="booking-filter"]').forEach((other) => {
    other.classList.toggle('active', other === button);
    other.setAttribute('aria-selected', String(other === button));
  });
  loadBookings();
}

async function cancelBooking(button) {
  if (!await confirmAction('Cancel this appointment?', `${button.dataset.when}. The attendee is notified by the calendar.`, 'Cancel appointment')) return;
  const result = await api('/booking/cancel', { method: 'POST', body: { booking_uid: button.dataset.uid }, admin: true });
  if (!result.ok) { toast(errorText(result, 'The appointment could not be cancelled.'), 'bad'); return; }
  toast('Appointment cancelled.');
  loadBookingStats();
  loadBookings();
  loadSlots();
}

// ── Calls ────────────────────────────────────────────────────────────────────

async function loadCalls() {
  const rows = $('call-rows');
  const note = $('calls-empty');
  const result = await api('/calls', { admin: true });
  if (!result.ok) {
    setContent(rows);
    setContent($('call-stats'));
    note.textContent = errorText(result, 'Calls could not be loaded.');
    show(note, true);
    $('calls-config-text').textContent = '';
    return;
  }
  const data = result.data;
  $('call-btn').disabled = !data.outbound_enabled;
  $('calls-config-text').textContent = data.outbound_enabled
    ? 'International format. The telephony worker must be running.'
    : data.telephony_configured
      ? 'Outbound calling needs a SIP trunk: run setup_vobiz_trunk.py and set OUTBOUND_TRUNK_ID.'
      : 'Calling is off until the LIVEKIT and VOBIZ settings are added on the server.';

  const stats = data.stats || {};
  setContent($('call-stats'),
    tile('Calls', stats.total || 0),
    tile('Inbound', stats.inbound || 0),
    tile('Outbound', stats.outbound || 0),
    tile('Completed', stats.completed || 0),
    tile('Not connected', stats.failed || 0, stats.failed ? 'warn' : ''),
    tile('Talk time', formatDuration(stats.total_duration_sec)),
  );

  const calls = data.calls || [];
  show(note, calls.length === 0);
  note.textContent = 'No calls yet.';
  const tones = { completed: 'good', in_progress: 'accent', ringing: 'accent', failed: 'bad' };
  setContent(rows, ...calls.map((call) => {
    const outbound = call.direction === 'outbound';
    return el('tr', {},
      el('td', {}, el('span', { class: `dir ${outbound ? 'out' : ''}`.trim(), title: outbound ? 'Outbound' : 'Inbound' }, icon(outbound ? 'out' : 'in'))),
      el('td', { class: 'mono', text: call.phone || 'Unknown' }),
      el('td', { text: formatTimestamp(call.created_at) }),
      el('td', { class: 'num', text: String(call.turns || 0) }),
      el('td', { class: 'num', text: formatDuration(call.duration_sec) }),
      el('td', {}, pill(String(call.status || '').replace('_', ' '), tones[call.status] || ''),
        call.detail ? el('span', { class: 'cell-sub', text: call.detail }) : null),
    );
  }));
}

async function placeCall(event) {
  event.preventDefault();
  const number = $('call-number').value.replace(/[\s().-]/g, '');
  if (!await confirmAction('Place this call?', `The agent will phone ${number} now and start the conversation when it is answered.`, 'Call')) return;
  $('call-btn').disabled = true;
  const result = await api('/calls/outbound', { method: 'POST', body: { phone_number: number }, admin: true });
  $('call-btn').disabled = false;
  if (!result.ok) { toast(errorText(result, 'The call could not be started.'), 'bad'); return; }
  toast('Call queued. The worker is dialling.');
  $('call-number').value = '';
  loadCalls();
}

// ── Evaluation ───────────────────────────────────────────────────────────────

async function runBenchmark(event) {
  event.preventDefault();
  const count = Math.min(Math.max(parseInt($('bench-n').value, 10) || 5, 1), 20);
  $('bench-btn').disabled = true;
  show($('bench-status'), true);
  $('bench-status-text').textContent = `Running ${plural(count, 'question')}…`;

  const result = await api(`/benchmark?n=${count}`, { admin: true });
  show($('bench-status'), false);
  $('bench-btn').disabled = false;
  if (!result.ok) { toast(errorText(result, 'The evaluation could not be run.'), 'bad'); return; }

  state.benchmark = result.data;
  renderBenchmark(result.data);
  show($('bench-report-btn'), true);
}

function renderBenchmark(data) {
  const latency = data.latency || {};
  const cost = data.cost || {};
  const accuracy = (data.accuracy || {}).mean_score || 0;
  const budget = data.budget_ms || state.budgetMs;
  const set = (id, text, good) => {
    $(id).textContent = text;
    if (good !== undefined) $(id).className = `tile-value ${good ? 'good' : 'warn'}`;
  };

  set('bkpi-p50', formatMs(latency.p50_ms), (latency.p50_ms || 0) <= budget);
  set('bkpi-p95', formatMs(latency.p95_ms), (latency.p95_ms || 0) <= budget);
  set('bkpi-accuracy', `${(accuracy * 100).toFixed(0)}%`, accuracy >= 0.6);
  set('bkpi-cost', `₹${(cost.per_query_avg_inr || 0).toFixed(5)}`);
  $('bkpi-budget-note').textContent = `Budget ${formatMs(budget)}`;
  $('bkpi-within').textContent = `${data.within_budget_pct || 0}% within budget`;
  $('bkpi-queries').textContent = `${data.successful || 0} of ${data.total_queries || 0} answered`;
  $('bench-stack').textContent = (data.api_stack || {}).llm || '';

  const scale = Math.max(budget, ...(data.individual_results || []).map((row) => row.total_ms || 0));
  setContent($('bench-tbody'), ...(data.individual_results || []).map((row) => {
    const fill = el('div', { class: `budget-fill ${row.within_budget ? '' : 'over'}`.trim() });
    fill.style.width = `${Math.max(((row.total_ms || 0) / scale) * 100, 1)}%`;
    return el('tr', {},
      el('td', { class: 'cell-wrap', text: row.query }),
      el('td', {}, el('div', { class: 'budget' }, el('div', { class: 'budget-track' }, fill))),
      el('td', { class: 'num', text: formatMs(row.retrieval_ms) }),
      el('td', { class: 'num', text: formatMs(row.llm_ms) }),
      el('td', { class: 'num', text: formatMs(row.total_ms) }),
      el('td', { class: 'num', text: `${((row.accuracy || 0) * 100).toFixed(0)}%` }),
    );
  }));
  show($('bench-kpis'), true);
  show($('bench-table-card'), true);
}

function downloadReport() {
  if (!state.benchmark) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.benchmark, null, 2)], { type: 'application/json' }));
  el('a', { href: url, download: `evaluation_${localISODate(new Date())}.json` }).click();
  URL.revokeObjectURL(url);
}

// ── Wiring ───────────────────────────────────────────────────────────────────

const ACTIONS = {
  'open-nav': () => openNav(),
  'close-nav': () => closeNav(),
  'refresh-view': () => showView(state.view),
  'open-settings': () => openSettings(),
  'clear-token': () => clearToken(),
  'cycle-theme': () => cycleTheme(),
  'toggle-recording': () => toggleRecording(),
  'clear-conversation': () => clearConversation(),
  'suggest': (target) => sendText(target.textContent.trim()),
  'source-kind': (target) => selectSourceKind(target.dataset.kind),
  'delete-source': (target) => deleteSource(target.dataset.source),
  'clear-kb': () => clearKB(),
  'pick-slot': (target) => pickSlot(target),
  'booking-filter': (target) => setBookingFilter(target),
  'cancel-booking': (target) => cancelBooking(target),
  'download-report': () => downloadReport(),
};

function wireEvents() {
  // One delegated listener covers static controls and rows rendered later.
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-action]');
    if (target && ACTIONS[target.dataset.action]) ACTIONS[target.dataset.action](target);
  });
  window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

  $('admin-token-form').addEventListener('submit', saveToken);
  $('chat-form').addEventListener('submit', submitChat);
  $('text-input').addEventListener('input', (event) => autoGrow(event.target));
  $('text-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('chat-form').requestSubmit();
    }
  });
  $('url-form').addEventListener('submit', ingestURL);
  $('text-form').addEventListener('submit', ingestText);
  $('book-form').addEventListener('submit', bookAppointment);
  $('book-date').addEventListener('change', loadSlots);
  $('call-form').addEventListener('submit', placeCall);
  $('bench-form').addEventListener('submit', runBenchmark);

  const dropZone = $('drop-zone');
  const fileInput = $('pdf-file-input');
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
  });
  dropZone.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
    uploadPDF(event.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => uploadPDF(fileInput.files[0]));
}

document.addEventListener('DOMContentLoaded', () => {
  applyTheme(currentTheme());
  buildMeter();
  wireEvents();
  initBookingDate();
  connectWebSocket();
  showView(location.hash.slice(1) || 'overview');
  setInterval(() => { if (state.view === 'overview') loadOverview(); else checkHealth(); }, 60000);
});
