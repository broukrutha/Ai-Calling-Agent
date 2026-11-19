/**
 * Calling Agent Console — application module.
 *
 * Pages: overview, playground, knowledge base, appointments, calls,
 * monitoring and evaluation, plus a command palette and a trace drawer.
 */

import {
  $, api, confirmAction, countTo, download, el, emptyNote, errorText, formatDate, formatDuration, formatMs,
  formatTime, formatTimestamp, getToken, icon, inr, localISODate, onLockChange, parseTimestamp, pct, pill, plural,
  relativeTime, setContent, setToken, show, skeleton, storage, toast,
} from './dom.js';
import { barChart, lineChart, sparkline } from './charts.js';
import { HandsFree, Microphone, Speaker, blobToBase64 } from './voice.js';

const state = {
  view: 'overview',
  health: null,
  budgetMs: 3000,
  window: '24h',
  traceFilter: '',
  bookingFilter: '',
  slot: '',
  evalSuite: 'all',
  selectedRun: null,
  pollTimer: null,
  ws: null,
  wsRetryMs: 1000,
  wsTimer: null,
  voiceMode: storage.get(localStorage, 'voiceMode') || 'handsfree',
  voiceState: 'idle',
  discardAudio: false,
  streamingEl: null,
  streamingText: '',
  spokenEl: null,
  lastTraceRows: [],
};

const speaker = new Speaker();
const mic = new Microphone();
let handsFree = null;

// ═════════════════════════════════════════════════════════════════════════════
// Navigation, theme, settings
// ═════════════════════════════════════════════════════════════════════════════

const VIEWS = {
  overview: () => loadOverview(),
  playground: () => checkHealth(),
  knowledge: () => loadSources(),
  appointments: () => { loadBookingStats(); loadBookings(); loadSlots(); },
  calls: () => loadCalls(),
  monitoring: () => loadMonitoring(),
  evaluation: () => loadEvaluation(),
};

function showView(name) {
  const view = VIEWS[name] ? name : 'overview';
  const changed = view !== state.view;
  state.view = view;
  document.querySelectorAll('.view').forEach((section) => {
    const active = section.id === `view-${view}`;
    section.hidden = !active;
    if (active && changed) {
      section.classList.remove('entering');
      void section.offsetWidth;          // restart the entrance animation
      section.classList.add('entering');
    }
  });
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
  if (changed) $('main').scrollTo({ top: 0 });
  clearInterval(state.pollTimer);
  VIEWS[view]();
}

function openNav() { $('sidebar').classList.add('open'); show($('scrim'), true); }
function closeNav() { $('sidebar').classList.remove('open'); show($('scrim'), false); }

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
  if (state.view === 'monitoring') loadMonitoring();
}
const cycleTheme = () => applyTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);

onLockChange((locked) => show($('lock-banner'), locked));

function renderTokenState() {
  const note = $('admin-token-state');
  if (getToken()) note.textContent = 'A token is saved for this browser tab.';
  else if (state.health && state.health.auth_required) note.textContent = 'Required to manage documents, bookings, calls and evaluations.';
  else note.textContent = 'This server has no admin token set, so none is needed (development mode).';
}

function openSettings() { renderTokenState(); $('settings-dialog').showModal(); }

function saveToken(event) {
  event.preventDefault();
  const value = $('admin-token-input').value.trim();
  if (value) setToken(value);
  $('admin-token-input').value = '';
  $('settings-dialog').close();
  if (value) toast('Admin token saved for this tab.');
  showView(state.view);
}

// ═════════════════════════════════════════════════════════════════════════════
// Health and overview
// ═════════════════════════════════════════════════════════════════════════════

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
  return health;
}

function statusRow(name, detail, label, tone) {
  return el('li', {},
    el('div', { class: 'row-main' }, el('div', { class: 'row-title', text: name }), el('div', { class: 'row-sub', text: detail, title: detail })),
    pill(label, tone));
}

function renderSystemStatus(health) {
  const model = health.api || {};
  const kb = health.knowledge_base || {};
  const telephony = health.telephony || {};
  const rows = [];
  if (model.status === 'ok') {
    const latency = Number.isFinite(model.latency_ms) ? ` · ${Math.round(model.latency_ms)} ms` : '';
    rows.push(statusRow('Language model', `${model.llm_provider} · ${model.llm_model}${latency}`, 'Operational', 'good'));
  } else if (model.status === 'api_key_missing') {
    rows.push(statusRow('Language model', 'Add provider keys to the server environment', 'Not configured', 'warn'));
  } else {
    rows.push(statusRow('Language model', model.message || 'The provider did not respond', 'Unreachable', 'bad'));
  }
  const benched = (model.providers || []).filter((p) => p.state === 'open');
  if (benched.length) {
    rows.push(statusRow('Fallbacks', benched.map((p) => `${p.name}: ${p.reason}`).join(' · '), `${benched.length} benched`, 'warn'));
  }
  rows.push(model.status === 'api_key_missing'
    ? statusRow('Speech', 'Set SARVAM_API_KEY to enable voice', 'Not configured', 'warn')
    : statusRow('Speech', `${model.stt_model || 'speech-to-text'} · ${model.tts_model || 'text-to-speech'}`, 'Ready', 'good'));
  if (kb.available === false) rows.push(statusRow('Knowledge base', kb.error || 'The embedding model could not be loaded', 'Unavailable', 'bad'));
  else if (!kb.total_chunks) rows.push(statusRow('Knowledge base', 'No documents indexed yet', 'Empty', 'warn'));
  else rows.push(statusRow('Knowledge base', `${plural(kb.total_chunks, 'passage')} · ${kb.retrieval || 'hybrid'} search`, 'Ready', 'good'));
  rows.push((health.bookings || {}).mode === 'mock'
    ? statusRow('Calendar', 'Bookings are saved locally, not to Cal.com', 'Mock mode', 'warn')
    : statusRow('Calendar', 'Cal.com', 'Connected', 'good'));
  if (telephony.configured && telephony.outbound) rows.push(statusRow('Telephony', 'LiveKit and SIP trunk', 'Ready', 'good'));
  else if (telephony.configured) rows.push(statusRow('Telephony', 'Inbound only: no outbound trunk', 'Partial', 'warn'));
  else rows.push(statusRow('Telephony', 'LiveKit and SIP settings are missing', 'Not configured', ''));
  rows.push(health.auth_required
    ? statusRow('Access control', 'Admin token required for changes', 'Protected', 'good')
    : statusRow('Access control', 'No admin token set on the server', 'Open', 'warn'));
  setContent($('status-list'), rows);
  $('overview-env').textContent = health.environment || '';
}

const KIND_TONE = { answer: 'good', booking: 'accent', noise: '', error: 'bad' };

function traceRow(trace) {
  return el('button', { class: 'row row-button', type: 'button', dataset: { action: 'open-trace', id: trace.id } },
    el('span', { class: `dot ${KIND_TONE[trace.kind] || ''}` }),
    el('div', { class: 'row-main' },
      el('div', { class: 'row-title', text: trace.question || '(no text stored)' }),
      el('div', { class: 'row-sub', text: [relativeTime(trace.created_at), trace.channel, trace.provider && `${trace.provider}`].filter(Boolean).join(' · ') })),
    el('div', { class: 'row-aside' },
      trace.rating === 1 ? icon('up', 'good') : trace.rating === -1 ? icon('down', 'bad') : null,
      el('span', { text: formatMs(trace.first_audio_ms || trace.total_ms) })));
}

async function loadOverview() {
  for (const id of ['overview-traces', 'overview-appointments']) if (!$(id).children.length) setContent($(id), skeleton());
  const [health, ops, series, traces, bookings, runs] = await Promise.all([
    checkHealth(), api('/ops/summary?window=24h', { admin: true }), api('/ops/timeseries?window=24h', { admin: true }),
    api('/ops/traces?limit=6', { admin: true }), api('/booking/stats', { admin: true }), api('/evaluation/runs?limit=10', { admin: true }),
  ]);
  if (health) renderSystemStatus(health);

  if (ops.ok) {
    const s = ops.data;
    countTo($('tile-turns'), s.turns);
    $('tile-turns-foot').textContent = s.turns ? `${pct(s.error_rate ?? 0)} errors · ${pct(s.cache_hit_rate ?? 0, 0)} cached` : 'No conversations yet';
    const latency = (s.latency_ms.first_audio_ms.p50 ?? s.latency_ms.total_ms.p50);
    $('tile-latency').textContent = latency === null || latency === undefined ? '–' : formatMs(latency);
    $('tile-latency').className = `tile-value ${latency && latency > state.budgetMs ? 'warn' : ''}`;
    $('tile-latency-foot').textContent = `p95 ${formatMs(s.latency_ms.first_audio_ms.p95 ?? s.latency_ms.total_ms.p95)} · budget ${formatMs(state.budgetMs)}`;
  }
  if (series.ok) {
    const points = series.data.points;
    setContent($('spark-turns'), sparkline(points.map((p) => p.turns)));
    setContent($('spark-latency'), sparkline(points.map((p) => p.first_audio_p50 ?? p.total_p50)));
  }
  if (traces.ok) {
    setContent($('overview-traces'), traces.data.traces.length
      ? traces.data.traces.map(traceRow)
      : [emptyNote('No conversations yet. Try the Playground, or place a call.')]);
  } else {
    setContent($('overview-traces'), emptyNote(errorText(traces, 'Conversations could not be loaded.')));
  }
  if (bookings.ok) {
    const stats = bookings.data.stats || {};
    countTo($('tile-upcoming'), stats.upcoming || 0);
    $('tile-upcoming-foot').textContent = `${stats.total || 0} booked in total · ${stats.mode === 'mock' ? 'mock calendar' : 'Cal.com'}`;
    const next = bookings.data.next_appointments || [];
    setContent($('overview-appointments'), next.length
      ? next.map((b) => el('div', { class: 'row' },
        el('div', { class: 'row-main' },
          el('div', { class: 'row-title', text: `${formatDate(b.date)} · ${formatTime(b.time_ist)}` }),
          el('div', { class: 'row-sub', text: `${b.attendee_name} · ${b.attendee_email}` })),
        el('div', { class: 'row-aside' }, b.ref_code ? el('span', { class: 'mono', text: b.ref_code }) : null, pill(b.source || 'manual', 'plain'))))
      : [emptyNote('Nothing scheduled. Bookings from calls and chat show up here.')]);
  } else {
    setContent($('overview-appointments'), emptyNote(errorText(bookings, 'Appointments could not be loaded.')));
  }
  if (runs.ok) {
    const latest = runs.data.runs.find((run) => run.status === 'completed' && run.suites.some((s) => s.suite === 'retrieval'));
    const retrieval = latest && latest.suites.find((s) => s.suite === 'retrieval');
    if (retrieval) {
      $('tile-quality').textContent = pct(retrieval.headline.context_hit);
      $('tile-quality').className = `tile-value ${retrieval.gates_passed ? 'good' : 'warn'}`;
      $('tile-quality-foot').textContent = `answer found in context · ${relativeTime(latest.created_at)}`;
    } else {
      $('tile-quality').textContent = '–';
      $('tile-quality-foot').textContent = 'Run an evaluation to measure it';
    }
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Playground: socket, messages, voice
// ═════════════════════════════════════════════════════════════════════════════

function connectWebSocket() {
  clearTimeout(state.wsTimer);
  if (state.ws && state.ws.readyState <= WebSocket.OPEN) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/voice`);
  state.ws = socket;
  socket.addEventListener('open', () => { state.wsRetryMs = 1000; socket.send(JSON.stringify({ type: 'ping' })); });
  socket.addEventListener('message', (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch (_) { return; }
    handleSocketMessage(message);
  });
  socket.addEventListener('close', () => {
    finishTurn();
    state.wsTimer = setTimeout(connectWebSocket, state.wsRetryMs);
    state.wsRetryMs = Math.min(state.wsRetryMs * 2, 15000);
  });
}
const socketReady = () => Boolean(state.ws) && state.ws.readyState === WebSocket.OPEN;
const send = (payload) => { if (socketReady()) state.ws.send(JSON.stringify(payload)); };
setInterval(() => send({ type: 'ping' }), 25000);

function handleSocketMessage(message) {
  switch (message.type) {
    case 'processing':
      state.discardAudio = false;
      setVoiceState('thinking');
      break;
    case 'transcript':
      addMessage('user', message.text);
      setVoiceState('thinking');
      break;
    case 'noise':
      finishTurn();
      if (!handsFree?.active) toast(message.message || 'That was hard to make out. Please try again.');
      break;
    case 'text_chunk':
      if (state.discardAudio) break;
      state.streamingText += message.text || '';
      if (!state.streamingEl) state.streamingEl = addMessage('assistant', state.streamingText, { streaming: true });
      else state.streamingEl.querySelector('.msg-text').textContent = state.streamingText;
      followTranscript();
      break;
    case 'audio_chunk':
      if (!state.discardAudio && message.audio) speaker.enqueue(message.audio);
      break;
    case 'done': {
      const node = state.streamingEl || addMessage('assistant', message.answer);
      finalizeAnswer(node, message);
      state.spokenEl = node;
      finishTurn();
      renderTurn(message, 'Voice');
      if (message.is_booking && state.view === 'appointments') VIEWS.appointments();
      break;
    }
    case 'text_result':
      finalizeAnswer(addMessage('assistant', message.answer || 'No answer was returned.'), message);
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

function finishTurn() {
  state.streamingEl?.classList.remove('streaming');
  state.streamingEl = null;
  state.streamingText = '';
  $('send-btn').disabled = false;
  if (!speaker.speaking) setVoiceState(handsFree?.active ? 'listening' : 'idle');
}

// ── Messages ─────────────────────────────────────────────────────────────────

let stickToBottom = true;
function followTranscript() {
  const transcript = $('conversation');
  if (stickToBottom) transcript.scrollTo({ top: transcript.scrollHeight, behavior: 'smooth' });
  else show($('jump-latest'), true);
}

function addMessage(role, text, { streaming = false } = {}) {
  show($('chat-empty'), false);
  const node = el('div', { class: `msg ${role}${streaming ? ' streaming' : ''}` },
    el('div', { class: 'msg-who', text: role === 'user' ? 'You' : 'Agent' }),
    el('div', { class: 'msg-text', text }));
  $('conversation').append(node);
  followTranscript();
  return node;
}

function finalizeAnswer(node, result) {
  node.classList.remove('streaming');
  if (result.answer) node.querySelector('.msg-text').textContent = result.answer;
  const meta = el('div', { class: 'msg-meta' });
  if (result.ref_code) meta.append(el('span', { class: 'tag ref', text: `Ref ${result.ref_code}` }));
  if (result.cached) meta.append(el('span', { class: 'tag', text: 'From cache' }));

  const citations = result.citations || [];
  if (citations.length) {
    const list = el('div', { class: 'citations', hidden: true }, citations.map(citationCard));
    const toggle = el('button', { class: 'tag tag-button', type: 'button', 'aria-expanded': 'false' },
      icon('file'), el('span', { text: plural(citations.length, 'source') }));
    toggle.addEventListener('click', () => {
      list.hidden = !list.hidden;
      toggle.setAttribute('aria-expanded', String(!list.hidden));
      followTranscript();
    });
    meta.append(toggle);
    node.append(meta, list);
  } else {
    node.append(meta);
  }

  if (result.trace_id) {
    const actions = el('div', { class: 'msg-actions' },
      el('button', { class: 'icon-btn sm', type: 'button', title: 'Copy', 'aria-label': 'Copy answer', dataset: { action: 'copy-answer' } }, icon('copy')),
      el('button', { class: 'icon-btn sm', type: 'button', title: 'Helpful', 'aria-label': 'Helpful', 'aria-pressed': 'false', dataset: { action: 'rate', rating: '1', trace: result.trace_id } }, icon('up')),
      el('button', { class: 'icon-btn sm', type: 'button', title: 'Not helpful', 'aria-label': 'Not helpful', 'aria-pressed': 'false', dataset: { action: 'rate', rating: '-1', trace: result.trace_id } }, icon('down')));
    meta.append(actions);
  }
}

function citationCard(citation) {
  const where = [citation.source, citation.section, citation.page ? `page ${citation.page}` : ''].filter(Boolean).join(' › ');
  const bar = el('span', { class: 'score-fill' });
  bar.style.width = `${Math.round(Math.min(Math.max(citation.score || 0, 0), 1) * 100)}%`;
  return el('div', { class: 'citation' },
    el('div', { class: 'citation-head' }, el('span', { class: 'citation-n', text: String(citation.n) }), el('span', { class: 'citation-where', text: where, title: where }),
      el('span', { class: 'score', title: `similarity ${citation.score}` }, bar)),
    el('p', { class: 'citation-snippet', text: citation.snippet }));
}

async function rate(button) {
  const rating = Number(button.dataset.rating);
  const result = await api('/feedback', { method: 'POST', body: { trace_id: button.dataset.trace, rating } });
  if (!result.ok) { toast(errorText(result, 'Your rating could not be saved.'), 'bad'); return; }
  button.parentElement.querySelectorAll('[data-action="rate"]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  toast(rating === 1 ? 'Thanks — marked as helpful.' : 'Thanks — it will show up in Monitoring for review.');
}

async function copyAnswer(button) {
  const text = button.closest('.msg').querySelector('.msg-text').textContent;
  try { await navigator.clipboard.writeText(text); toast('Answer copied.'); } catch (_) { toast('Copying is blocked in this browser.', 'bad'); }
}

function clearConversation() {
  stopVoice();
  const empty = $('chat-empty');
  setContent($('conversation'), empty);
  show(empty, true);
  send({ type: 'cancel' });
  // A new socket gives the agent a fresh conversation memory.
  if (state.ws) state.ws.close();
}

// ── Text turns ───────────────────────────────────────────────────────────────

function sendText(query) {
  if (!query) return;
  speaker.unlock();
  interrupt();
  addMessage('user', query);
  if (!socketReady()) { toast('Reconnecting to the server…', 'bad'); connectWebSocket(); return; }
  $('send-btn').disabled = true;
  setVoiceState('thinking');
  send({ type: 'query', query });
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

// ── Voice ────────────────────────────────────────────────────────────────────

const VOICE_TEXT = {
  idle: ['Tap to talk', null],
  listening: ['Listening…', 'Go ahead, I reply when you pause.'],
  hearing: ['Hearing you…', 'Pause to send.'],
  recording: ['Recording…', 'Tap again, or release Space, to send.'],
  thinking: ['Thinking…', 'Searching your documents.'],
  speaking: ['Speaking…', 'Start talking to interrupt.'],
};

function setVoiceState(next) {
  state.voiceState = next;
  const dock = $('voice-dock');
  dock.dataset.state = next;
  const live = handsFree?.active || mic.recording;
  const [title, hint] = VOICE_TEXT[next] || VOICE_TEXT.idle;
  $('dock-state').textContent = title;
  $('dock-hint').textContent = hint || (state.voiceMode === 'handsfree'
    ? 'Hands-free: just speak, the agent replies when you pause.'
    : 'Tap the orb or hold Space to talk.');
  $('orb-btn').setAttribute('aria-pressed', String(Boolean(live)));
  $('orb-btn').setAttribute('aria-label', live ? 'Stop talking' : 'Start talking');
  $('orb-btn').classList.toggle('live', Boolean(live));
}

speaker.onchange = (speaking) => {
  if (speaking) { setVoiceState('speaking'); startMeter(); }
  else if (!state.streamingEl) setVoiceState(handsFree?.active ? 'listening' : 'idle');
};

/** Stop the agent mid-answer: silence playback and cancel the server turn. */
function interrupt() {
  const speaking = speaker.speaking;
  if (!speaking && !state.streamingEl && state.voiceState !== 'thinking') return;
  const cut = state.streamingEl || (speaking ? state.spokenEl : null);
  speaker.stop();
  state.discardAudio = true;
  send({ type: 'cancel' });
  cut?.classList.add('interrupted');
  finishTurn();
}

async function sendClip(clip) {
  if (!clip || clip.durationMs < 300) { toast('That was too short. Keep talking a little longer.'); setVoiceState('idle'); return; }
  if (!socketReady()) { toast('The connection dropped. Please try again.', 'bad'); return; }
  setVoiceState('thinking');
  send({
    type: 'audio',
    data: await blobToBase64(clip.blob),
    format: clip.mimeType.replace('audio/', ''),
    language: $('language-select').value,
    duration_ms: Math.round(clip.durationMs),
  });
}

async function ensureMic() {
  if (!Microphone.supported()) { toast('This browser cannot record audio. Type your message instead.', 'bad'); return false; }
  try { await mic.start(); return true; } catch (_) {
    toast('Microphone access is blocked. Allow it in your browser settings.', 'bad');
    return false;
  }
}

async function toggleVoice() {
  speaker.unlock();
  if (state.voiceMode === 'handsfree') {
    if (handsFree?.active) { stopVoice(); return; }
    if (!(await ensureMic())) return;
    handsFree = new HandsFree(mic, speaker, {
      onUtterance: (clip) => { interrupt(); sendClip(clip); },
      onState: (next) => { if (next !== 'listening' || !speaker.speaking) setVoiceState(next); },
      onBargeIn: () => interrupt(),
      bargeIn: () => $('barge-in').checked,
    });
    await handsFree.start();
    startMeter();
    return;
  }
  if (mic.recording) { await stopPushToTalk(); return; }
  await startPushToTalk();
}

async function startPushToTalk() {
  if (!(await ensureMic())) return;
  interrupt();
  mic.record();
  setVoiceState('recording');
  startMeter();
}

async function stopPushToTalk() {
  if (!mic.recording) return;
  const clip = await mic.finish();
  setVoiceState('idle');
  await sendClip(clip);
}

function stopVoice() {
  handsFree?.stop();
  handsFree = null;
  if (mic.recording) mic.discard();
  mic.stop();
  setVoiceState(speaker.speaking ? 'speaking' : 'idle');
}

const METER_BARS = 20;
let meterFrame = 0;
function startMeter() {
  cancelAnimationFrame(meterFrame);
  const bars = $('meter').children;
  const glow = $('orb-glow');
  const draw = () => {
    const level = speaker.speaking ? speaker.level() : mic.open ? mic.level() : 0;
    const scaled = Math.min(level * 9, 1);
    glow.style.transform = `scale(${1 + scaled * 0.55})`;
    glow.style.opacity = String(0.25 + scaled * 0.6);
    for (let i = 0; i < bars.length; i += 1) {
      const shape = Math.sin((i / bars.length) * Math.PI);
      bars[i].style.transform = `scaleY(${0.15 + scaled * shape * (0.6 + Math.random() * 0.4)})`;
    }
    if (mic.open || speaker.speaking) meterFrame = requestAnimationFrame(draw);
    else {
      glow.style.transform = '';
      glow.style.opacity = '';
      for (const bar of bars) bar.style.transform = '';
    }
  };
  meterFrame = requestAnimationFrame(draw);
}

function setVoiceMode(mode) {
  stopVoice();
  state.voiceMode = mode;
  storage.set(localStorage, 'voiceMode', mode);
  document.querySelectorAll('[data-action="voice-mode"]').forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-checked', String(active));
  });
  setVoiceState('idle');
}

// ── Turn inspector ───────────────────────────────────────────────────────────

function renderTurn(result, kind) {
  const latency = result.latency;
  if (!latency) return;
  show($('turn-empty'), false);
  show($('turn-detail'), true);
  show($('turn-kind'), true);
  $('turn-kind').textContent = result.is_booking ? `${kind} · booking` : result.cached ? `${kind} · cached` : kind;

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
  ].filter(([, , , d]) => d > 0);
  const rows = stages.map(([name, key, start, duration]) => {
    const bar = el('div', { class: `wf-bar ${key}` });
    bar.style.left = `${Math.min((start / total) * 100, 99)}%`;
    bar.style.width = `${Math.max(Math.min((duration / total) * 100, 100 - (start / total) * 100), 0.6)}%`;
    return el('div', { class: 'wf-row' }, el('span', { class: 'wf-name', text: name }), el('div', { class: 'wf-lane' }, bar), el('span', { class: 'wf-ms', text: formatMs(duration) }));
  });
  if (first > 0) {
    const position = Math.min((first / total) * 100, 100);
    const marker = el('div', { class: `wf-marker ${position > 70 ? 'flip' : ''}`.trim() }, el('span', { text: 'first audio' }));
    marker.style.left = `${position}%`;
    rows.push(el('div', { class: 'wf-axis' }, marker));
  }
  setContent($('waterfall'), rows.length ? rows : [el('div', { class: 'help', text: 'Answered without calling the model.' })]);

  const grade = (ms) => (ms <= state.budgetMs ? 'good' : ms <= state.budgetMs * 1.3 ? 'warn' : 'bad');
  $('val-first').textContent = first ? formatMs(first) : kind === 'Text' ? 'Text only' : '–';
  $('val-first').className = `kv-big ${first ? grade(first) : ''}`.trim();
  $('val-total').textContent = formatMs(total);
  $('val-model').textContent = result.is_booking ? 'Booking dialog (no LLM)' : result.model ? `${result.provider} · ${result.model}` : result.provider || '–';
  $('val-cost').textContent = result.cost_inr !== undefined ? inr(result.cost_inr) : '–';
  $('val-budget').textContent = `${formatMs(state.budgetMs)} to first audio`;

  const citations = result.citations || [];
  show($('passages-panel'), citations.length > 0);
  $('passages-count').textContent = citations.length ? `(${citations.length})` : '';
  setContent($('passages'), citations.map(citationCard));
}

// ═════════════════════════════════════════════════════════════════════════════
// Knowledge base
// ═════════════════════════════════════════════════════════════════════════════

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

async function searchKnowledge(event) {
  event.preventDefault();
  const query = $('kb-search-input').value.trim();
  if (!query) return;
  setContent($('kb-search-results'), skeleton(2));
  const result = await api(`/kb/search?q=${encodeURIComponent(query)}`, { admin: true });
  if (!result.ok) { setContent($('kb-search-results'), emptyNote(errorText(result, 'Search failed.'))); return; }
  const citations = result.data.citations || [];
  setContent($('kb-search-results'), citations.length
    ? [el('p', { class: 'help', text: `${plural(citations.length, 'passage')} in ${formatMs(result.data.latency_ms)} — this is what the agent would read.` }), ...citations.map(citationCard)]
    : [emptyNote('Nothing relevant found. The agent would say it does not have that information.')]);
}

async function loadSources() {
  const list = $('kb-sources');
  if (!list.children.length) setContent(list, skeleton());
  const result = await api('/kb/sources', { admin: true });
  if (!result.ok) { $('sources-count').textContent = ''; setContent(list, emptyNote(errorText(result, 'Sources could not be loaded.'))); return; }
  const sources = result.data.sources || [];
  $('sources-count').textContent = sources.length ? `(${sources.length})` : '';
  if (!sources.length) { setContent(list, emptyNote('No sources yet. Add a PDF, a web page or some text and the agent can answer from it straight away.')); return; }
  setContent(list, sources.map((item) => {
    const kind = /^https?:\/\//i.test(item.source) ? 'Web page' : item.source.toLowerCase().endsWith('.pdf') ? 'PDF' : 'Text';
    return el('div', { class: 'row' },
      el('span', { class: 'row-icon' }, icon('file')),
      el('div', { class: 'row-main' }, el('div', { class: 'row-title', text: item.source, title: item.source }), el('div', { class: 'row-sub', text: `${kind} · ${plural(item.chunks, 'passage')}` })),
      el('button', { class: 'btn btn-sm btn-danger-ghost', type: 'button', text: 'Remove', dataset: { action: 'delete-source', source: item.source } }));
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

// ═════════════════════════════════════════════════════════════════════════════
// Appointments
// ═════════════════════════════════════════════════════════════════════════════

function tile(label, value, tone = '', foot = '') {
  return el('div', { class: 'tile' }, el('div', { class: 'tile-label', text: label }),
    el('div', { class: `tile-value ${tone}`.trim(), text: String(value) }), foot ? el('div', { class: 'tile-foot', text: foot }) : null);
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
  if (day !== $('book-date').value) return;
  if (!result.ok) { setContent(box, el('span', { class: 'help', text: errorText(result, 'Open times could not be loaded.') })); return; }
  const slots = result.data.available_slots || [];
  if (!slots.length) { setContent(box, el('span', { class: 'help', text: 'No open times on this date.' })); return; }
  setContent(box, slots.map((slot) => el('button', { class: 'slot', type: 'button', role: 'radio', 'aria-checked': 'false', text: formatTime(slot), dataset: { action: 'pick-slot', slot } })));
}

function pickSlot(button) {
  state.slot = button.dataset.slot;
  document.querySelectorAll('#slots .slot').forEach((other) => other.setAttribute('aria-checked', String(other === button)));
}

async function bookAppointment(event) {
  event.preventDefault();
  if (!state.slot) { toast('Choose one of the open times.', 'bad'); return; }
  const body = { name: $('book-name').value.trim(), email: $('book-email').value.trim(), date: $('book-date').value, time: state.slot };
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
  const s = result.data.stats || {};
  setContent($('booking-stats'),
    tile('Upcoming', s.upcoming || 0), tile('Confirmed', s.confirmed || 0), tile('Cancelled', s.cancelled || 0),
    tile('All time', s.total || 0), tile('Calendar', s.mode === 'mock' ? 'Mock' : 'Cal.com', s.mode === 'mock' ? 'warn' : 'good'));
}

async function loadBookings() {
  const rows = $('booking-rows');
  const note = $('booking-empty');
  const query = state.bookingFilter ? `&status=${state.bookingFilter}` : '';
  const result = await api(`/booking/history?limit=50${query}`, { admin: true });
  if (!result.ok) { setContent(rows); note.textContent = errorText(result, 'Appointments could not be loaded.'); show(note, true); return; }
  const bookings = result.data.bookings || [];
  show(note, bookings.length === 0);
  note.textContent = state.bookingFilter ? `No ${state.bookingFilter} appointments.` : 'No appointments yet.';
  setContent(rows, bookings.map((b) => {
    const confirmed = b.status === 'confirmed';
    const action = confirmed && b.cal_booking_uid
      ? el('button', { class: 'btn btn-sm btn-danger-ghost', type: 'button', text: 'Cancel', dataset: { action: 'cancel-booking', uid: b.cal_booking_uid, when: `${formatDate(b.date)} at ${formatTime(b.time_ist)}` } })
      : null;
    return el('tr', {},
      el('td', {}, document.createTextNode(formatDate(b.date)), el('span', { class: 'cell-sub', text: formatTime(b.time_ist) })),
      el('td', { class: 'cell-wrap' }, document.createTextNode(b.attendee_name || ''), el('span', { class: 'cell-sub', text: b.attendee_email || '' })),
      el('td', { class: 'mono', text: b.ref_code || '–' }),
      el('td', { text: b.source || '–' }),
      el('td', {}, pill(b.status || 'unknown', confirmed ? 'good' : '')),
      el('td', { class: 'num' }, action));
  }));
}

function setSegment(selector, button) {
  document.querySelectorAll(selector).forEach((other) => {
    const active = other === button;
    other.classList.toggle('active', active);
    other.setAttribute(other.getAttribute('role') === 'tab' ? 'aria-selected' : 'aria-checked', String(active));
  });
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

// ═════════════════════════════════════════════════════════════════════════════
// Calls
// ═════════════════════════════════════════════════════════════════════════════

async function loadCalls() {
  const rows = $('call-rows');
  const note = $('calls-empty');
  const result = await api('/calls', { admin: true });
  if (!result.ok) {
    setContent(rows); setContent($('call-stats'));
    note.textContent = errorText(result, 'Calls could not be loaded.'); show(note, true);
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
  const s = data.stats || {};
  setContent($('call-stats'),
    tile('Calls', s.total || 0), tile('Inbound', s.inbound || 0), tile('Outbound', s.outbound || 0),
    tile('Completed', s.completed || 0), tile('Not connected', s.failed || 0, s.failed ? 'warn' : ''), tile('Talk time', formatDuration(s.total_duration_sec)));
  const calls = data.calls || [];
  show(note, calls.length === 0);
  note.textContent = 'No calls yet.';
  const tones = { completed: 'good', in_progress: 'accent', ringing: 'accent', failed: 'bad' };
  setContent(rows, calls.map((call) => {
    const outbound = call.direction === 'outbound';
    return el('tr', {},
      el('td', {}, el('span', { class: `dir ${outbound ? 'out' : ''}`.trim(), title: outbound ? 'Outbound' : 'Inbound' }, icon(outbound ? 'out' : 'in'))),
      el('td', { class: 'mono', text: call.phone || 'Unknown' }),
      el('td', { text: formatTimestamp(call.created_at) }),
      el('td', { class: 'num', text: String(call.turns || 0) }),
      el('td', { class: 'num', text: formatDuration(call.duration_sec) }),
      el('td', {}, pill(String(call.status || '').replace('_', ' '), tones[call.status] || ''), call.detail ? el('span', { class: 'cell-sub', text: call.detail }) : null));
  }));
}

async function placeCall(event) {
  event.preventDefault();
  const number = $('call-number').value.replace(/[\s().-]/g, '');
  if (!await confirmAction('Place this call?', `The agent will phone ${number} now and start talking when it is answered.`, 'Call', false)) return;
  $('call-btn').disabled = true;
  const result = await api('/calls/outbound', { method: 'POST', body: { phone_number: number }, admin: true });
  $('call-btn').disabled = false;
  if (!result.ok) { toast(errorText(result, 'The call could not be started.'), 'bad'); return; }
  toast('Call queued. The worker is dialling.');
  $('call-number').value = '';
  loadCalls();
}

// ═════════════════════════════════════════════════════════════════════════════
// Monitoring
// ═════════════════════════════════════════════════════════════════════════════

function timeLabel(iso, full = false) {
  const date = parseTimestamp(iso);
  if (!date) return '';
  if (state.window === '7d' || state.window === '30d') {
    return date.toLocaleDateString('en-IN', full ? { weekday: 'short', day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short' });
  }
  return date.toLocaleTimeString('en-IN', { hour: 'numeric', minute: '2-digit' });
}

async function loadMonitoring() {
  const [summary, series] = await Promise.all([
    api(`/ops/summary?window=${state.window}`, { admin: true }), api(`/ops/timeseries?window=${state.window}`, { admin: true }),
  ]);
  if (!summary.ok) {
    setContent($('ops-tiles'), emptyNote(errorText(summary, 'Monitoring data could not be loaded.')));
    return;
  }
  const s = summary.data;
  const firstAudio = s.latency_ms.first_audio_ms;
  setContent($('ops-tiles'),
    tile('Turns', s.turns, '', Object.entries(s.by_channel).map(([k, v]) => `${v} ${k}`).join(' · ') || 'none yet'),
    tile('First audio p50', formatMs(firstAudio.p50), firstAudio.p50 > state.budgetMs ? 'warn' : '', `p95 ${formatMs(firstAudio.p95)}`),
    tile('Error rate', pct(s.error_rate), s.error_rate > 0.05 ? 'warn' : '', `${s.by_kind.error || 0} failed turns`),
    tile('Fallback rate', pct(s.fallback_rate), s.fallback_rate > 0.2 ? 'warn' : '', 'answered by a backup model'),
    tile('Helpful', pct(s.feedback.helpful_rate, 0), '', `${s.feedback.rated} rated · ${s.feedback.not_helpful} down`),
    tile('Cost', inr(s.cost_inr), '', `${(s.tokens.input + s.tokens.output).toLocaleString('en-IN')} tokens`));

  if (series.ok) {
    const points = series.data.points;
    lineChart($('chart-latency'), {
      points, xFormat: timeLabel, format: formatMs, budget: state.budgetMs,
      series: [{ key: 'first_audio_p50', label: 'p50', cls: 'p50' }, { key: 'first_audio_p95', label: 'p95', cls: 'p95' }],
      empty: 'No voice turns in this period yet.',
    });
    barChart($('chart-volume'), { points, key: 'turns', overlayKey: 'errors', format: (v) => String(Math.round(v)), xFormat: timeLabel, label: 'Turns', overlayLabel: 'Errors' });
  }

  const stages = [['Speech in', 'stt_ms', 'stt'], ['Search', 'retrieval_ms', 'search'], ['Model, first token', 'first_token_ms', 'llm'], ['Speech out', 'tts_ms', 'tts']]
    .map(([label, key, cls]) => ({ label, cls, value: s.latency_ms[key].p50 }));
  const maxStage = Math.max(1, ...stages.map((x) => x.value || 0));
  setContent($('stage-bars'), stages.map(({ label, cls, value }) => {
    const fill = el('span', { class: `stage-fill ${cls}` });
    fill.style.width = `${value ? Math.max((value / maxStage) * 100, 1.5) : 0}%`;
    return el('div', { class: 'stage-row' }, el('span', { class: 'stage-name', text: label }), el('span', { class: 'stage-track' }, fill), el('span', { class: 'stage-ms', text: formatMs(value) }));
  }));

  const providers = s.providers_health || [];
  const usage = s.providers || {};
  setContent($('provider-list'), providers.length
    ? providers.map((p, i) => el('div', { class: 'row' },
      el('span', { class: 'rank', text: String(i + 1) }),
      el('div', { class: 'row-main' }, el('div', { class: 'row-title', text: `${p.name} · ${p.model}` }),
        el('div', { class: 'row-sub', text: `${usage[`${p.name}/${p.model}`] || 0} answers in this period${p.state === 'open' ? ` · retry in ${p.retry_in_sec}s` : ''}` })),
      pill(p.state === 'open' ? p.reason : 'healthy', p.state === 'open' ? 'warn' : 'good')))
    : [emptyNote('No language model is configured.')]);

  await loadTraces();
}

async function loadTraces() {
  const filter = state.traceFilter === 'error' ? '&kind=error' : state.traceFilter === 'down' ? '&rating=-1' : '';
  const result = await api(`/ops/traces?limit=50${filter}`, { admin: true });
  const rows = $('trace-rows');
  const note = $('trace-empty');
  if (!result.ok) { setContent(rows); note.textContent = errorText(result, 'Turns could not be loaded.'); show(note, true); return; }
  const traces = result.data.traces;
  show(note, !traces.length);
  note.textContent = state.traceFilter ? 'Nothing here — good news.' : 'No conversation turns yet.';
  setContent(rows, traces.map((t) => el('tr', { dataset: { action: 'open-trace', id: t.id }, tabindex: '0' },
    el('td', { class: 'nowrap', text: relativeTime(t.created_at) }),
    el('td', { text: t.channel }),
    el('td', { class: 'cell-wrap', text: t.question || '(no text stored)' }),
    el('td', {}, pill(t.kind, KIND_TONE[t.kind] || ''), t.followup ? el('span', { class: 'cell-sub', text: 'follow-up' }) : null),
    el('td', { class: 'num', text: formatMs(t.first_audio_ms) }),
    el('td', { class: 'num', text: formatMs(t.total_ms) }),
    el('td', { class: 'num' }, t.rating === 1 ? icon('up', 'good') : t.rating === -1 ? icon('down', 'bad') : null))));
}

async function openTrace(id) {
  const result = await api(`/ops/traces/${encodeURIComponent(id)}`, { admin: true });
  if (!result.ok) { toast(errorText(result, 'That turn could not be loaded.'), 'bad'); return; }
  const t = result.data;
  const timing = [['Speech in', t.stt_ms], ['Search', t.retrieval_ms], ['Model', t.llm_ms], ['First token', t.first_token_ms], ['Speech out', t.tts_ms], ['First audio', t.first_audio_ms], ['Total', t.total_ms]]
    .filter(([, v]) => v);
  setContent($('trace-body'),
    el('div', { class: 'trace-meta' }, pill(t.kind, KIND_TONE[t.kind] || ''), pill(t.channel, 'plain'), el('span', { class: 'help', text: formatTimestamp(t.created_at) })),
    el('div', { class: 'trace-block' }, el('div', { class: 'tile-label', text: 'Question' }), el('p', { text: t.question || '(no text stored)' })),
    t.retrieval_query && t.retrieval_query !== t.question ? el('div', { class: 'trace-block' }, el('div', { class: 'tile-label', text: 'Searched as (follow-up)' }), el('p', { class: 'help', text: t.retrieval_query })) : null,
    el('div', { class: 'trace-block' }, el('div', { class: 'tile-label', text: 'Answer' }), el('p', { text: t.answer || '–' })),
    t.error ? el('div', { class: 'notice bad', text: t.error }) : null,
    el('dl', { class: 'kv' },
      el('dt', { text: 'Model' }), el('dd', { text: t.provider ? `${t.provider} · ${t.model}${t.fallback ? ' (fallback)' : ''}` : '–' }),
      el('dt', { text: 'Tokens' }), el('dd', { text: `${t.input_tokens} in · ${t.output_tokens} out` }),
      el('dt', { text: 'Cost' }), el('dd', { text: inr(t.cost_inr) }),
      el('dt', { text: 'Prompt version' }), el('dd', { text: t.prompt_version || '–' }),
      el('dt', { text: 'Feedback' }), el('dd', { text: t.rating === 1 ? 'Helpful' : t.rating === -1 ? `Not helpful${t.comment ? ` — ${t.comment}` : ''}` : 'Not rated' }),
      ...timing.flatMap(([label, value]) => [el('dt', { text: label }), el('dd', { text: formatMs(value) })])),
    el('div', { class: 'trace-block' }, el('div', { class: 'tile-label', text: `Passages (${t.passages.length})` }),
      t.passages.length
        ? el('ul', { class: 'trace-passages' }, t.passages.map((p) => el('li', {}, el('span', { text: [p.source, p.section].filter(Boolean).join(' › ') }), el('span', { class: 'mono', text: Number(p.score).toFixed(2) }))))
        : el('p', { class: 'help', text: 'No passages were retrieved.' })));
  $('trace-drawer').showModal();
}

async function exportFeedback() {
  const result = await api('/ops/export', { admin: true, raw: true });
  if (!result.ok) { toast('The export failed.', 'bad'); return; }
  const lines = result.data.trim() ? result.data.trim().split('\n').length : 0;
  if (!lines) { toast('No answers have been rated down yet.'); return; }
  download(`low_rated_${localISODate(new Date())}.jsonl`, result.data, 'application/x-ndjson');
  toast(`${plural(lines, 'case')} exported. Add evidence and a reference answer, then add them to your test set.`);
}

// ═════════════════════════════════════════════════════════════════════════════
// Evaluation
// ═════════════════════════════════════════════════════════════════════════════

const SUITE_NAMES = { retrieval: 'Retrieval', answers: 'Answers', dialog: 'Booking dialogs' };
const METRICS = {
  retrieval: [['context_hit', 'Answer in context'], ['hit_at_1', 'Ranked first'], ['mrr', 'MRR', 'ratio'], ['context_precision', 'Context precision']],
  dialog: [['success_rate', 'Task success']],
  answers: [['pass_rate', 'Pass rate'], ['correctness', 'Correctness'], ['faithfulness', 'Faithfulness'], ['refusal_accuracy', 'Refuses when unsure'], ['voice_format_rate', 'Speakable']],
};
const GATE_LABELS = {
  context_hit: 'Answer in context', hit_at_1: 'Ranked first', mrr: 'MRR', context_precision: 'Context precision',
  evidence_recall: 'Evidence recall', success_rate: 'Task success', pass_rate: 'Pass rate', correctness: 'Correctness',
  faithfulness: 'Faithfulness', refusal_accuracy: 'Refuses when unsure', voice_format_rate: 'Speakable',
};
const gateLabel = (metric) => { const key = metric.split('.').pop(); return GATE_LABELS[key] || key.replaceAll('_', ' '); };
const fmtMetric = (value, kind) => (value === null || value === undefined ? '–' : kind === 'ratio' ? Number(value).toFixed(3) : pct(value));

async function loadEvaluation() {
  const [datasets, runs] = await Promise.all([api('/evaluation/datasets', { admin: true }), api('/evaluation/runs', { admin: true })]);
  if (datasets.ok) {
    const { golden, generated } = datasets.data;
    setContent($('eval-datasets'),
      el('div', { class: 'row' }, el('span', { class: 'row-icon' }, icon('check')),
        el('div', { class: 'row-main' }, el('div', { class: 'row-title', text: 'Golden set' }),
          el('div', { class: 'row-sub', text: `${golden.cases} questions on the sample documents · plus 11 booking conversations` }))),
      el('div', { class: 'row' }, el('span', { class: 'row-icon' }, icon('sparkle')),
        el('div', { class: 'row-main' }, el('div', { class: 'row-title', text: 'Your documents' }),
          el('div', { class: 'row-sub', text: generated.available ? `${generated.cases} generated questions` : 'Not generated yet' })),
        el('button', { class: 'btn btn-sm', type: 'button', dataset: { action: 'generate-dataset' } }, generated.available ? 'Regenerate' : 'Generate')));
  } else {
    setContent($('eval-datasets'), emptyNote(errorText(datasets, 'Test sets could not be loaded.')));
  }
  if (!runs.ok) { setContent($('eval-runs'), emptyNote(errorText(runs, 'History could not be loaded.'))); return; }
  renderRuns(runs.data.runs);
  const running = runs.data.runs.find((run) => run.status === 'running');
  setRunBusy(Boolean(running));
  if (running) pollRun(running.id);
  const pick = state.selectedRun && runs.data.runs.some((r) => r.id === state.selectedRun) ? state.selectedRun : runs.data.runs[0]?.id;
  if (pick) openRun(pick);
}

function renderRuns(runs) {
  if (!runs.length) { setContent($('eval-runs'), emptyNote('No runs yet. Start with Retrieval: it is free and takes seconds.')); return; }
  setContent($('eval-runs'), runs.map((run) => {
    const passed = run.status === 'completed' && run.suites.every((s) => s.gates_passed !== false);
    const tone = run.status === 'running' ? 'accent' : run.status === 'failed' ? 'bad' : passed ? 'good' : 'warn';
    const label = run.status === 'running' ? 'running' : run.status === 'failed' ? 'failed' : passed ? 'passed' : 'below threshold';
    return el('button', { class: `row row-button ${run.id === state.selectedRun ? 'selected' : ''}`, type: 'button', dataset: { action: 'open-run', id: run.id } },
      el('div', { class: 'row-main' },
        el('div', { class: 'row-title', text: run.suite === 'all' ? 'All suites' : SUITE_NAMES[run.suite] || run.suite }),
        el('div', { class: 'row-sub', text: `${relativeTime(run.created_at)} · ${run.dataset}` })),
      pill(label, tone));
  }));
}

async function openRun(id) {
  state.selectedRun = id;
  document.querySelectorAll('#eval-runs .row-button').forEach((row) => row.classList.toggle('selected', row.dataset.id === id));
  const result = await api(`/evaluation/runs/${encodeURIComponent(id)}`, { admin: true });
  if (!result.ok) { setContent($('eval-detail'), emptyNote(errorText(result, 'That run could not be loaded.'))); return; }
  const run = result.data;
  $('eval-detail-title').textContent = 'Results';
  $('eval-detail-meta').textContent = `${run.suite === 'all' ? 'All suites' : SUITE_NAMES[run.suite] || run.suite} · ${formatTimestamp(run.created_at)} · ${run.dataset}`;
  const blocks = [];
  if (run.status === 'running') blocks.push(el('div', { class: 'thinking' }, el('span', { class: 'spinner' }), el('span', { text: 'Running… results appear as each suite finishes.' })));
  if (run.status === 'failed') blocks.push(el('div', { class: 'notice bad', text: run.error || 'The run failed.' }));
  for (const summary of Array.isArray(run.results) ? run.results : []) blocks.push(suiteBlock(summary));
  if (!blocks.length) blocks.push(emptyNote('No results yet.'));
  setContent($('eval-detail'), blocks);
}

function suiteBlock(summary) {
  const metrics = METRICS[summary.suite] || [];
  const source = summary.suite === 'retrieval' ? summary.answerable : summary;
  const tiles = el('div', { class: 'tiles tiles-compact' }, metrics.map(([key, label, kind]) => tile(label, fmtMetric(source[key], kind))));
  const gates = (summary.gates || []).map((gate) => el('li', { class: 'gate' }, icon(gate.passed ? 'check' : 'alert', gate.passed ? 'good' : 'bad'),
    el('span', { text: gateLabel(gate.metric) }),
    el('span', { class: 'mono', text: `${fmtMetric(gate.value, gate.metric.endsWith('mrr') ? 'ratio' : '')} ≥ ${fmtMetric(gate.minimum, gate.metric.endsWith('mrr') ? 'ratio' : '')}` })));
  const failing = (summary.rows || []).filter((row) => row.passed === false);
  const details = failing.slice(0, 25).map((row) => {
    const body = [];
    if (row.answer) body.push(el('p', {}, el('strong', { text: 'Answer: ' }), document.createTextNode(row.answer)));
    if (row.reference) body.push(el('p', { class: 'help', text: `Expected: ${row.reference}` }));
    if (row.judge?.reason) body.push(el('p', { class: 'help', text: `Judge: ${row.judge.reason}` }));
    if (row.sources?.length) body.push(el('p', { class: 'help', text: `Retrieved from: ${row.sources.join(', ')}` }));
    if (row.problems?.length) body.push(el('p', { class: 'help', text: row.problems.join('; ') }));
    if (row.error) body.push(el('p', { class: 'help', text: row.error }));
    return el('details', { class: 'failure' }, el('summary', {}, el('span', { text: row.question || row.description || row.id })), ...body);
  });
  const judged = summary.suite === 'answers' && !summary.judged ? el('p', { class: 'help', text: 'Scored by keywords and refusal checks only: the LLM judge was unavailable.' }) : null;
  return el('section', { class: 'suite-block' },
    el('div', { class: 'suite-head' }, el('h3', { text: SUITE_NAMES[summary.suite] || summary.suite }),
      pill(summary.gates?.length ? (summary.gates_passed ? 'all gates passed' : 'below threshold') : 'no gates for this test set', summary.gates?.length ? (summary.gates_passed ? 'good' : 'warn') : 'plain'),
      el('span', { class: 'help', text: `${summary.duration_sec}s` })),
    judged, tiles,
    gates.length ? el('ul', { class: 'gates' }, gates) : null,
    failing.length ? el('div', { class: 'failures' }, el('div', { class: 'tile-label', text: `${plural(failing.length, 'failing case')}` }), details) : el('p', { class: 'help', text: 'Every case passed.' }));
}

function setRunBusy(busy) {
  const button = $('eval-run-btn');
  button.disabled = busy;
  button.lastChild.textContent = busy ? 'Running…' : 'Run evaluation';
}

async function runEvaluation(event) {
  event.preventDefault();
  const kb = $('eval-kb').value;
  setRunBusy(true);
  const result = await api('/evaluation/runs', { method: 'POST', body: { suite: state.evalSuite, kb }, admin: true });
  if (!result.ok) { setRunBusy(false); toast(errorText(result, 'The evaluation could not be started.'), 'bad'); return; }
  state.selectedRun = result.data.id;
  toast('Evaluation started.');
  await loadEvaluation();
}

function pollRun(id) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    if (state.view !== 'evaluation') { clearInterval(state.pollTimer); return; }
    const result = await api(`/evaluation/runs/${encodeURIComponent(id)}`, { admin: true });
    if (!result.ok) return;
    if (state.selectedRun === id) openRun(id);
    if (result.data.status !== 'running') {
      clearInterval(state.pollTimer);
      setRunBusy(false);
      toast(result.data.status === 'completed' ? 'Evaluation finished.' : 'The evaluation failed.', result.data.status === 'completed' ? '' : 'bad');
      const runs = await api('/evaluation/runs', { admin: true });
      if (runs.ok) renderRuns(runs.data.runs);
    }
  }, 1500);
}

async function generateDataset(button) {
  if (!await confirmAction('Generate questions from your documents?', 'The LLM writes about 30 questions from your knowledge base. Review them before trusting the scores.', 'Generate', false)) return;
  button.disabled = true;
  button.textContent = 'Generating…';
  const result = await api('/evaluation/generate', { method: 'POST', body: { limit: 30 }, admin: true });
  button.disabled = false;
  if (!result.ok) { toast(errorText(result, 'Questions could not be generated.'), 'bad'); loadEvaluation(); return; }
  toast(`${result.data.cases} questions written. Choose “Your knowledge base” to evaluate with them.`);
  $('eval-kb').value = 'live';
  loadEvaluation();
}

// ═════════════════════════════════════════════════════════════════════════════
// Command palette
// ═════════════════════════════════════════════════════════════════════════════

const COMMANDS = [
  ...Object.keys(VIEWS).map((view) => ({ label: `Go to ${$(`view-${view}`).dataset.title}`, hint: 'Page', run: () => { location.hash = view; } })),
  { label: 'Start or stop talking', hint: 'Voice', run: () => { location.hash = 'playground'; toggleVoice(); } },
  { label: 'Run retrieval evaluation', hint: 'Evaluation', run: async () => { location.hash = 'evaluation'; state.evalSuite = 'retrieval'; await runEvaluation(new Event('submit')); } },
  { label: 'Switch theme', hint: 'Appearance', run: () => cycleTheme() },
  { label: 'Clear conversation', hint: 'Playground', run: () => clearConversation() },
  { label: 'Open settings', hint: 'Admin', run: () => openSettings() },
];
let paletteIndex = 0;

function openPalette() {
  const dialog = $('palette');
  $('palette-input').value = '';
  renderPalette();
  dialog.showModal();
  $('palette-input').focus();
}

function paletteMatches() {
  const query = $('palette-input').value.trim().toLowerCase();
  return COMMANDS.filter((command) => !query || command.label.toLowerCase().includes(query) || command.hint.toLowerCase().includes(query));
}

function renderPalette() {
  const matches = paletteMatches();
  paletteIndex = Math.min(paletteIndex, Math.max(matches.length - 1, 0));
  setContent($('palette-list'), matches.length
    ? matches.map((command, i) => el('li', { role: 'option', 'aria-selected': String(i === paletteIndex), class: i === paletteIndex ? 'active' : '', dataset: { index: String(i) } },
      el('span', { text: command.label }), el('span', { class: 'help', text: command.hint })))
    : [el('li', { class: 'help', text: 'No matching command.' })]);
}

function runPaletteItem(index) {
  const command = paletteMatches()[index];
  if (!command) return;
  $('palette').close();
  command.run();
}

// ═════════════════════════════════════════════════════════════════════════════
// Wiring
// ═════════════════════════════════════════════════════════════════════════════

const ACTIONS = {
  'open-nav': () => openNav(),
  'close-nav': () => closeNav(),
  'refresh-view': () => showView(state.view),
  'open-settings': () => openSettings(),
  'clear-token': () => { setToken(''); renderTokenState(); toast('Admin token removed.'); },
  'cycle-theme': () => cycleTheme(),
  'open-palette': () => openPalette(),
  'toggle-voice': () => toggleVoice(),
  'voice-mode': (t) => setVoiceMode(t.dataset.mode),
  'clear-conversation': () => clearConversation(),
  'jump-latest': () => { stickToBottom = true; followTranscript(); show($('jump-latest'), false); },
  'suggest': (t) => sendText(t.textContent.trim()),
  'rate': (t) => rate(t),
  'copy-answer': (t) => copyAnswer(t),
  'source-kind': (t) => selectSourceKind(t.dataset.kind),
  'delete-source': (t) => deleteSource(t.dataset.source),
  'clear-kb': () => clearKB(),
  'pick-slot': (t) => pickSlot(t),
  'booking-filter': (t) => { state.bookingFilter = t.dataset.status; setSegment('[data-action="booking-filter"]', t); loadBookings(); },
  'cancel-booking': (t) => cancelBooking(t),
  'window': (t) => { state.window = t.dataset.window; setSegment('[data-action="window"]', t); loadMonitoring(); },
  'trace-filter': (t) => { state.traceFilter = t.dataset.filter; setSegment('[data-action="trace-filter"]', t); loadTraces(); },
  'open-trace': (t) => openTrace(t.dataset.id),
  'export-feedback': () => exportFeedback(),
  'eval-suite': (t) => { state.evalSuite = t.dataset.suite; setSegment('[data-action="eval-suite"]', t); },
  'open-run': (t) => openRun(t.dataset.id),
  'generate-dataset': (t) => generateDataset(t),
};

function isTyping(target) {
  return target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
}

function wireEvents() {
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-action]');
    if (target && ACTIONS[target.dataset.action]) ACTIONS[target.dataset.action](target);
  });
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); openPalette(); return; }
    const row = event.target.closest?.('tr[data-action="open-trace"]');
    if (row && event.key === 'Enter') { openTrace(row.dataset.id); return; }
    // Hold Space to talk (push-to-talk), when not typing.
    if (event.code === 'Space' && !event.repeat && state.view === 'playground' && state.voiceMode === 'ptt' && !isTyping(event.target)) {
      event.preventDefault();
      speaker.unlock();
      startPushToTalk();
    }
  });
  document.addEventListener('keyup', (event) => {
    if (event.code === 'Space' && state.voiceMode === 'ptt' && mic.recording && !isTyping(event.target)) {
      event.preventDefault();
      stopPushToTalk();
    }
  });
  window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

  const transcript = $('conversation');
  transcript.addEventListener('scroll', () => {
    stickToBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 60;
    if (stickToBottom) show($('jump-latest'), false);
  }, { passive: true });

  $('admin-token-form').addEventListener('submit', saveToken);
  $('chat-form').addEventListener('submit', submitChat);
  $('text-input').addEventListener('input', (event) => autoGrow(event.target));
  $('text-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('chat-form').requestSubmit(); }
  });
  $('url-form').addEventListener('submit', ingestURL);
  $('text-form').addEventListener('submit', ingestText);
  $('kb-search-form').addEventListener('submit', searchKnowledge);
  $('book-form').addEventListener('submit', bookAppointment);
  $('book-date').addEventListener('change', loadSlots);
  $('call-form').addEventListener('submit', placeCall);
  $('eval-form').addEventListener('submit', runEvaluation);

  const dropZone = $('drop-zone');
  const fileInput = $('pdf-file-input');
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); } });
  dropZone.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (event) => { event.preventDefault(); dropZone.classList.remove('dragover'); uploadPDF(event.dataTransfer.files[0]); });
  fileInput.addEventListener('change', () => uploadPDF(fileInput.files[0]));

  $('palette-input').addEventListener('input', () => { paletteIndex = 0; renderPalette(); });
  $('palette-input').addEventListener('keydown', (event) => {
    const count = paletteMatches().length;
    if (event.key === 'ArrowDown') { event.preventDefault(); paletteIndex = (paletteIndex + 1) % Math.max(count, 1); renderPalette(); }
    if (event.key === 'ArrowUp') { event.preventDefault(); paletteIndex = (paletteIndex - 1 + Math.max(count, 1)) % Math.max(count, 1); renderPalette(); }
    if (event.key === 'Enter') { event.preventDefault(); runPaletteItem(paletteIndex); }
  });
  $('palette-list').addEventListener('click', (event) => {
    const item = event.target.closest('li[data-index]');
    if (item) runPaletteItem(Number(item.dataset.index));
  });
  for (const dialog of [$('palette'), $('trace-drawer')]) {
    dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  }

  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.view === 'monitoring') loadMonitoring(); }, 200);
  });
}

function boot() {
  applyTheme(currentTheme());
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) $('palette-kbd').textContent = 'Ctrl K';
  else $('palette-kbd').textContent = '⌘K';
  setContent($('meter'), Array.from({ length: METER_BARS }, () => el('i')));
  wireEvents();
  setVoiceMode(state.voiceMode === 'ptt' ? 'ptt' : 'handsfree');
  initBookingDate();
  connectWebSocket();
  showView(location.hash.slice(1) || 'overview');
  setInterval(() => { if (state.view === 'overview') loadOverview(); else if (state.view === 'monitoring') loadMonitoring(); else checkHealth(); }, 60000);
}

boot();
