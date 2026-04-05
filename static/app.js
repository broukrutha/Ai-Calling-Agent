/**
 * app.js – Sarvam AI Voice Agent Frontend
 *
 * Features:
 *  - WebSocket connection to backend
 *  - Push-to-talk microphone recording (WebAudio API)
 *  - Text query support
 *  - Real-time latency visualization
 *  - Document ingestion (PDF, URL, text)
 *  - Benchmark runner + results display
 *  - Cal.com booking: slot check, create, cancel, history
 */

'use strict';

// ──────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────
let BUDGET_MS = 3000;
const WS_PROTOCOL = location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTOCOL}//${location.host}/ws/voice`;
const API_BASE = '';

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let ws = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let waveAnimFrame = null;
let benchmarkData = null;
let ingestHistory = [];

function getAdminHeaders() {
  const token = window.localStorage.getItem('adminApiToken') || '';
  return token ? { 'X-Admin-Token': token } : {};
}

function syncBudgetLabels() {
  const p50Sub = document.querySelector('#bench-kpis .kpi-card:nth-child(1) .kpi-sub');
  const budgetSub = document.querySelector('#bench-kpis .kpi-card:nth-child(3) .kpi-sub');
  if (p50Sub) p50Sub.textContent = `Budget: ${BUDGET_MS}ms`;
  if (budgetSub) budgetSub.textContent = `<=${BUDGET_MS}ms queries`;
}

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildWaveform();
  checkHealth();
  loadKBInfo();
  connectWebSocket();
  setDefaultBookingDates();
  loadBookingStats();
  loadBookingHistory();
});

// ──────────────────────────────────────────────
// Tab switching
// ──────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${tabId}`).classList.add('active');
  document.getElementById(`panel-${tabId}`).classList.add('active');

  // Refresh booking data when switching to booking tab
  if (tabId === 'booking') {
    loadBookingStats();
    loadBookingHistory();
  }
}

// ──────────────────────────────────────────────
// WebSocket
// ──────────────────────────────────────────────
function connectWebSocket() {
  clearTimeout(window._wsRetry);
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'ping' }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleWSMessage(msg);
  };

  ws.onerror = () => {
    setStatus(false, 'WS error');
  };

  ws.onclose = () => {
    // Auto-reconnect
    window._wsRetry = setTimeout(connectWebSocket, 3000);
  };
}

// ── Streaming state ────────────────────────────────────────────────────────
let _streamingAnswer = '';
let _streamingMsgEl  = null;
let audioQueue       = [];
let isPlayingAudio   = false;

function queueAudioChunk(b64) {
  audioQueue.push(b64);
  if (!isPlayingAudio) playNextAudioChunk();
}

function playNextAudioChunk() {
  if (audioQueue.length === 0) { isPlayingAudio = false; return; }
  isPlayingAudio = true;
  const b64 = audioQueue.shift();
  try {
    const bytes = atob(b64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    const blob  = new Blob([arr], { type: 'audio/wav' });
    const url   = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => { URL.revokeObjectURL(url); playNextAudioChunk(); };
    audio.onerror = () => { URL.revokeObjectURL(url); playNextAudioChunk(); };
    audio.play().catch(() => playNextAudioChunk());
  } catch(e) { playNextAudioChunk(); }
}

function handleWSMessage(msg) {
  switch (msg.type) {
    case 'pong':
      setStatus(true, 'Connected');
      break;

    case 'processing':
      showQueryStatus(true, 'Processing audio pipeline…');
      break;

    case 'transcript':
      if (msg.text && msg.text.trim()) {
        addMessage('user', msg.text);
        showQueryStatus(true, 'Thinking…');
        _streamingAnswer = '';
        _streamingMsgEl  = null;
      }
      break;

    case 'noise':
      showQueryStatus(false);
      document.getElementById('mic-hint').textContent = 'Click mic to start recording';
      toast('Background noise — please speak clearly', 'warning');
      break;

    case 'text_chunk':
      if (msg.text) {
        _streamingAnswer += msg.text;
        if (!_streamingMsgEl) {
          _streamingMsgEl = addMessage('assistant', _streamingAnswer);
        } else {
          const t = _streamingMsgEl.querySelector('.msg-text');
          if (t) t.textContent = _streamingAnswer;
        }
        // Auto scroll
        const conv = document.getElementById('conversation');
        if (conv) conv.scrollTop = conv.scrollHeight;
      }
      break;

    case 'audio_chunk':
      if (msg.audio) queueAudioChunk(msg.audio);
      break;

    case 'done':
      showQueryStatus(false);
      document.getElementById('mic-hint').textContent = 'Click mic to start recording';
      _streamingAnswer = '';
      _streamingMsgEl  = null;
      if (msg.latency) updateLatencyDisplay(msg.latency, msg.cost_inr);
      // Refresh booking history if a booking was made
      if (msg.is_booking) {
        loadBookingHistory();
        loadBookingStats();
      }
      break;

    case 'result':
      handleVoiceResult(msg);
      break;

    case 'text_result':
      handleTextResult(msg);
      break;

    case 'error':
      showQueryStatus(false);
      _streamingAnswer = '';
      _streamingMsgEl  = null;
      toast(`Error: ${msg.message}`, 'error');
      break;
  }
}

// ──────────────────────────────────────────────
// Health check
// ──────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (Number.isFinite(data.latency_budget_ms)) {
      BUDGET_MS = data.latency_budget_ms;
      syncBudgetLabels();
    }
    const ok = data.api?.status === 'ok';
    setStatus(ok, ok ? `Connected · ${data.api?.latency_ms?.toFixed(0)}ms` : 'API Error');
    updateKBInfo(data.knowledge_base);
  } catch (e) {
    setStatus(false, 'Server offline');
  }
}

function setStatus(ok, text) {
  const dot = document.getElementById('api-dot');
  const label = document.getElementById('api-status-text');
  dot.className = `status-dot ${ok ? 'live' : 'error'}`;
  label.textContent = text;
}

// Override health rendering with provider-aware status details.
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (Number.isFinite(data.latency_budget_ms)) {
      BUDGET_MS = data.latency_budget_ms;
      syncBudgetLabels();
    }

    const apiStatus = data.api?.status || 'error';
    let statusState = 'error';
    if (apiStatus === 'ok') statusState = 'ok';
    else if (apiStatus === 'degraded') statusState = 'warn';

    let statusText = 'API Error';
    if (apiStatus === 'ok') {
      const latency = Number.isFinite(data.api?.latency_ms) ? ` | ${data.api.latency_ms.toFixed(0)}ms` : '';
      const provider = data.api?.llm_provider ? ` | ${String(data.api.llm_provider).toUpperCase()}` : '';
      statusText = `Connected${provider}${latency}`;
    } else if (apiStatus === 'degraded') {
      statusText = data.api?.summary || 'Degraded';
    } else {
      statusText = data.api?.message || data.api?.summary || 'API Error';
    }

    setStatus(statusState, statusText);
    updateKBInfo(data.knowledge_base);
  } catch (e) {
    setStatus('error', 'Server offline');
  }
}

function setStatus(state, text) {
  const dot = document.getElementById('api-dot');
  const label = document.getElementById('api-status-text');
  const normalized = state === 'ok' ? 'live' : (state === 'warn' ? 'warn' : 'error');
  dot.className = `status-dot ${normalized}`;
  label.textContent = text;
}

// ──────────────────────────────────────────────
// KB Info
// ──────────────────────────────────────────────
async function loadKBInfo() {
  try {
    const res = await fetch(`${API_BASE}/kb/info`);
    const data = await res.json();
    updateKBInfo(data);
  } catch (e) {}
}

function updateKBInfo(info) {
  if (!info) return;
  const el = document.getElementById('kb-info-text');
  if (el) {
    el.textContent = `${info.total_chunks || 0} chunks · Collection: ${info.collection || '—'} · Model: ${info.embedding_model || '—'}`;
  }
}

async function clearKB() {
  if (!confirm('Clear all documents from the knowledge base?')) return;
  try {
    const res = await fetch(`${API_BASE}/kb/clear`, {
      method: 'DELETE',
      headers: getAdminHeaders(),
    });
    if (res.status === 403) {
      toast('Admin token required. Save it in localStorage as adminApiToken.', 'error');
      return;
    }
    toast('Knowledge base cleared.', 'info');
    loadKBInfo();
    ingestHistory = [];
    renderHistory();
  } catch (e) {
    toast('Failed to clear KB.', 'error');
  }
}

// ──────────────────────────────────────────────
// Waveform
// ──────────────────────────────────────────────
function buildWaveform() {
  const container = document.getElementById('waveform');
  container.innerHTML = '';
  for (let i = 0; i < 28; i++) {
    const bar = document.createElement('div');
    bar.className = 'wave-bar';
    container.appendChild(bar);
  }
}

function setWaveformActive(active) {
  document.querySelectorAll('.wave-bar').forEach((bar, i) => {
    if (active) {
      bar.classList.add('active');
      bar.style.animationDelay = `${(i * 0.05) % 0.8}s`;
    } else {
      bar.classList.remove('active');
      bar.style.height = '8px';
    }
  });
}

// ──────────────────────────────────────────────
// Recording
// ──────────────────────────────────────────────
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];

    const mimeType = getSupportedMime();
    mediaRecorder = new MediaRecorder(stream, { mimeType });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      processAudio();
    };

    mediaRecorder.start(100);
    isRecording = true;

    const btn = document.getElementById('mic-btn');
    const wrap = btn.closest('.mic-button-wrap');
    btn.classList.add('recording');
    btn.textContent = '⏹';
    document.getElementById('mic-hint').textContent = 'Recording… click to stop';
    setWaveformActive(true);

    toast('Recording started', 'info');
  } catch (e) {
    toast('Microphone access denied. Please allow mic access.', 'error');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  const btn = document.getElementById('mic-btn');
  btn.classList.remove('recording');
  btn.textContent = '🎤';
  document.getElementById('mic-hint').textContent = 'Processing…';
  setWaveformActive(false);
}

function getSupportedMime() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

async function processAudio() {
  const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
  const arrayBuffer = await blob.arrayBuffer();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    toast('WebSocket not connected. Reconnecting…', 'error');
    connectWebSocket();
    document.getElementById('mic-hint').textContent = 'Click mic to start recording';
    return;
  }

  showQueryStatus(true, 'Sending audio to pipeline…');

  ws.send(JSON.stringify({
    type: 'audio',
    data: b64,
    format: 'webm',
    language: 'en-IN',
  }));
}

// ──────────────────────────────────────────────
// Voice result handler
// ──────────────────────────────────────────────
function handleVoiceResult(msg) {
  showQueryStatus(false);
  document.getElementById('mic-hint').textContent = 'Click mic to start recording';

  // Add messages
  if (msg.transcript) {
    addMessage('user', msg.transcript);
  }
  if (msg.answer) {
    addMessage('assistant', msg.answer, msg.sources || []);
  }

  // Play audio
  if (msg.audio) {
    const audioEl = document.getElementById('response-audio');
    const bytes = atob(msg.audio);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    const blob2 = new Blob([arr], { type: 'audio/wav' });
    audioEl.src = URL.createObjectURL(blob2);
    audioEl.style.display = 'block';
    audioEl.play().catch(() => {});
  }

  // Update latency
  if (msg.latency) {
    updateLatencyDisplay(msg.latency, msg.cost_inr);
  }

  // Refresh booking if relevant
  if (msg.is_booking) {
    loadBookingHistory();
    loadBookingStats();
  }
}

// ──────────────────────────────────────────────
// Text query
// ──────────────────────────────────────────────
async function sendTextQuery() {
  const input = document.getElementById('text-input');
  const query = input.value.trim();
  if (!query) return;

  addMessage('user', query);
  input.value = '';
  showQueryStatus(true, 'Querying knowledge base…');
  document.getElementById('send-btn').disabled = true;

  // Try WebSocket first for real-time feel
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'query', query }));
  } else {
    // Fallback to REST
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, language: 'en-IN' }),
      });
      const data = await res.json();
      handleTextResult(data);
    } catch (e) {
      toast('Query failed: ' + e.message, 'error');
      showQueryStatus(false);
    }
  }

  document.getElementById('send-btn').disabled = false;
}

function handleTextResult(msg) {
  showQueryStatus(false);
  const answer = msg.answer || msg.detail || 'No answer returned.';
  const sources = msg.sources || [];
  addMessage('assistant', answer, sources);

  if (msg.latency) {
    const lat = {
      stt_ms: 0,
      retrieval_ms: msg.latency.retrieval_ms || 0,
      llm_ms: msg.latency.llm_ms || 0,
      tts_ms: 0,
      total_ms: msg.latency.total_ms || 0,
    };
    updateLatencyDisplay(lat, msg.cost_inr);
  }

  // Refresh booking if relevant
  if (msg.is_booking) {
    loadBookingHistory();
    loadBookingStats();
  }
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendTextQuery();
  }
}

// ──────────────────────────────────────────────
// Conversation UI
// ──────────────────────────────────────────────
function addMessage(role, text, sources = []) {
  const conv = document.getElementById('conversation');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'Assistant';
  div.appendChild(label);

  const content = document.createElement('div');
  content.className = 'msg-text';
  content.textContent = text;
  div.appendChild(content);

  if (sources.length > 0) {
    const chips = document.createElement('div');
    chips.className = 'source-chips';
    sources.forEach(src => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.textContent = '📄 ' + (src.length > 30 ? src.slice(0, 30) + '…' : src);
      chips.appendChild(chip);
    });
    div.appendChild(chips);
  }

  conv.appendChild(div);
  conv.scrollTop = conv.scrollHeight;
  return div;
}

function clearConversation() {
  const conv = document.getElementById('conversation');
  conv.innerHTML = '<div class="message system"><div>Conversation cleared. Ready for new questions.</div></div>';
}

// ──────────────────────────────────────────────
// Latency display
// ──────────────────────────────────────────────
function updateLatencyDisplay(lat, costInr) {
  const display = document.getElementById('latency-display');
  display.style.display = 'block';

  const budget = BUDGET_MS;

  function setBar(idBar, idVal, ms) {
    const pct = Math.min((ms / budget) * 100, 100);
    document.getElementById(idBar).style.width = `${pct}%`;
    document.getElementById(idVal).textContent = `${ms.toFixed(0)}ms`;
  }

  setBar('bar-stt', 'val-stt', lat.stt_ms || 0);
  setBar('bar-rag', 'val-rag', lat.retrieval_ms || 0);
  setBar('bar-llm', 'val-llm', lat.llm_ms || 0);
  setBar('bar-tts', 'val-tts', lat.tts_ms || 0);

  const total = lat.total_ms || 0;
  const totalEl = document.getElementById('val-total');
  totalEl.textContent = `${total.toFixed(0)}ms`;
  totalEl.className = `val ${total <= budget ? 'ok' : total <= budget * 1.3 ? 'warn' : 'over'}`;

  if (costInr !== undefined) {
    document.getElementById('cost-display').textContent = `Cost: ₹${Number(costInr).toFixed(5)}`;
  }
}

// ──────────────────────────────────────────────
// Status bar
// ──────────────────────────────────────────────
function showQueryStatus(show, text = '') {
  const bar = document.getElementById('query-status');
  const label = document.getElementById('query-status-text');
  bar.style.display = show ? 'flex' : 'none';
  if (text) label.textContent = text;
}

// ──────────────────────────────────────────────
// Document Ingestion
// ──────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('dragover');
}
function handleDragLeave() {
  document.getElementById('drop-zone').classList.remove('dragover');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.pdf')) uploadPDF(file);
  else toast('Please drop a PDF file.', 'error');
}

async function uploadPDF(file) {
  if (!file) return;
  setPDFStatus(true, `Uploading ${file.name}…`);
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch(`${API_BASE}/ingest/pdf`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'ok') {
      toast(`✅ PDF ingested: ${data.chunks_added} chunks`, 'success');
      addToHistory('PDF', file.name, data.chunks_added);
      loadKBInfo();
    } else {
      toast('Ingestion failed: ' + (data.detail || 'unknown'), 'error');
    }
  } catch (e) {
    toast('Upload failed: ' + e.message, 'error');
  }
  setPDFStatus(false);
}

async function ingestURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) { toast('Please enter a URL.', 'error'); return; }
  setPDFStatus(true, `Scraping ${url}…`);
  try {
    const res = await fetch(`${API_BASE}/ingest/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      toast(`✅ URL ingested: ${data.chunks_added} chunks`, 'success');
      addToHistory('URL', url, data.chunks_added);
      loadKBInfo();
      document.getElementById('url-input').value = '';
    } else {
      toast('Ingestion failed: ' + (data.detail || 'unknown'), 'error');
    }
  } catch (e) {
    toast('URL ingest failed: ' + e.message, 'error');
  }
  setPDFStatus(false);
}

async function ingestText() {
  const text = document.getElementById('text-area-input').value.trim();
  const name = document.getElementById('text-name-input').value.trim() || 'manual_input';
  if (!text) { toast('Please enter some text.', 'error'); return; }

  setTextStatus(true, 'Ingesting text…');
  try {
    const res = await fetch(`${API_BASE}/ingest/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, name }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      toast(`✅ Text ingested: ${data.chunks_added} chunks`, 'success');
      addToHistory('Text', name, data.chunks_added);
      loadKBInfo();
      document.getElementById('text-area-input').value = '';
    } else {
      toast('Ingestion failed: ' + (data.detail || 'unknown'), 'error');
    }
  } catch (e) {
    toast('Text ingest failed: ' + e.message, 'error');
  }
  setTextStatus(false);
}

function setPDFStatus(show, text = '') {
  const el = document.getElementById('pdf-status');
  el.style.display = show ? 'flex' : 'none';
  if (text) document.getElementById('pdf-status-text').textContent = text;
}
function setTextStatus(show, text = '') {
  const el = document.getElementById('text-status');
  el.style.display = show ? 'flex' : 'none';
  if (text) document.getElementById('text-status-text').textContent = text;
}

function addToHistory(type, source, chunks) {
  ingestHistory.unshift({ type, source, chunks, time: new Date().toLocaleTimeString() });
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById('ingest-history');
  if (ingestHistory.length === 0) {
    el.innerHTML = '<span class="text-muted">No documents ingested yet.</span>';
    return;
  }
  el.innerHTML = ingestHistory.map(h =>
    `<div style="padding:8px 0; border-bottom:1px solid var(--border); font-size:0.83rem;">
      <span style="color:var(--accent-light);">[${h.type}]</span>
      <span style="margin:0 8px;">${h.source.length > 50 ? h.source.slice(0,50)+'…' : h.source}</span>
      <span style="color:var(--green);">+${h.chunks} chunks</span>
      <span style="color:var(--text-muted); float:right;">${h.time}</span>
    </div>`
  ).join('');
}

// ──────────────────────────────────────────────
// Benchmark
// ──────────────────────────────────────────────
async function runBenchmark() {
  const n = parseInt(document.getElementById('bench-n').value) || 5;
  const btn = document.getElementById('bench-btn');
  btn.disabled = true;

  document.getElementById('bench-status').style.display = 'flex';
  document.getElementById('bench-status-text').textContent = `Running ${n} queries…`;
  document.getElementById('bench-kpis').style.display = 'none';
  document.getElementById('bench-comp-card').style.display = 'none';
  document.getElementById('bench-table-card').style.display = 'none';

  try {
    const res = await fetch(`${API_BASE}/benchmark?n=${n}`);
    const data = await res.json();
    benchmarkData = data;
    renderBenchmark(data);
    document.getElementById('bench-report-btn').style.display = 'flex';
    toast('Benchmark complete!', 'success');
  } catch (e) {
    toast('Benchmark failed: ' + e.message, 'error');
  }

  document.getElementById('bench-status').style.display = 'none';
  btn.disabled = false;
}

function renderBenchmark(data) {
  const lat = data.latency || {};
  const cost = data.cost || {};
  const comp = lat.component_means || {};
  const within = data.within_budget_pct || 0;

  // KPIs
  const p50El = document.getElementById('bkpi-p50');
  p50El.textContent = `${(lat.p50_ms || 0).toFixed(0)}ms`;
  p50El.className = `kpi-val ${lat.p50_ms <= BUDGET_MS ? 'green' : 'yellow'}`;

  const p95El = document.getElementById('bkpi-p95');
  p95El.textContent = `${(lat.p95_ms || 0).toFixed(0)}ms`;
  p95El.className = `kpi-val ${lat.p95_ms <= BUDGET_MS * 1.3 ? 'green' : 'yellow'}`;

  const budEl = document.getElementById('bkpi-budget');
  budEl.textContent = `${within}%`;
  budEl.className = `kpi-val ${within >= 80 ? 'green' : 'yellow'}`;

  document.getElementById('bkpi-cost').textContent = `₹${(cost.per_query_avg_inr || 0).toFixed(5)}`;
  document.getElementById('bkpi-cost100').textContent = `₹${(cost.per_100_queries_inr || 0).toFixed(3)} / 100 queries`;
  document.getElementById('bkpi-queries').textContent = `${data.successful || 0}/${data.total_queries || 0}`;
  document.getElementById('bkpi-errors').textContent = `${data.errors || 0} errors`;

  // Component bars
  const maxMs = Math.max(comp.stt_ms || 0, comp.retrieval_ms || 0, comp.llm_ms || 0, comp.tts_ms || 0, 1);
  function compBar(id, valId, ms) {
    const pct = Math.min((ms / BUDGET_MS) * 100, 100);
    document.getElementById(id).style.width = `${pct}%`;
    document.getElementById(id).textContent = `${ms.toFixed(0)}ms`;
    document.getElementById(valId).textContent = `${ms.toFixed(0)} ms`;
  }
  compBar('bc-stt', 'bcv-stt', comp.stt_ms || 0);
  compBar('bc-rag', 'bcv-rag', comp.retrieval_ms || 0);
  compBar('bc-llm', 'bcv-llm', comp.llm_ms || 0);
  compBar('bc-tts', 'bcv-tts', comp.tts_ms || 0);

  // Table
  const tbody = document.getElementById('bench-tbody');
  tbody.innerHTML = '';
  (data.individual_results || []).forEach(r => {
    const ok = r.within_budget;
    const tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid var(--border)';
    tr.innerHTML = `
      <td style="padding:9px 12px; color:${ok ? 'var(--text-primary)' : 'var(--yellow)'};">${ok ? '✅' : '⚠️'} ${r.query.slice(0, 55)}</td>
      <td style="padding:9px 12px; text-align:right; color:var(--text-secondary);">${(r.retrieval_ms||0).toFixed(0)}</td>
      <td style="padding:9px 12px; text-align:right; color:var(--text-secondary);">${(r.llm_ms||0).toFixed(0)}</td>
      <td style="padding:9px 12px; text-align:right; font-weight:600; color:${ok ? 'var(--green)' : 'var(--yellow)'};">${(r.total_ms||0).toFixed(0)}</td>
      <td style="padding:9px 12px; text-align:right; color:var(--green);">₹${(r.cost_inr||0).toFixed(5)}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('bench-kpis').style.display = 'grid';
  document.getElementById('bench-comp-card').style.display = 'block';
  document.getElementById('bench-table-card').style.display = 'block';
}

function downloadReport() {
  if (!benchmarkData) return;
  const json = JSON.stringify(benchmarkData, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `benchmark_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ──────────────────────────────────────────────
// Booking (Cal.com) — Enhanced
// ──────────────────────────────────────────────

function setDefaultBookingDates() {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const dateStr = tomorrow.toISOString().split('T')[0];

  const bookingDate = document.getElementById('booking-date');
  const bookDate = document.getElementById('book-date');
  if (bookingDate && !bookingDate.value) bookingDate.value = dateStr;
  if (bookDate && !bookDate.value) bookDate.value = dateStr;

  // Set min date to today
  const today = new Date().toISOString().split('T')[0];
  if (bookingDate) bookingDate.setAttribute('min', today);
  if (bookDate) bookDate.setAttribute('min', today);
}

async function checkSlots() {
  const dateVal = document.getElementById('booking-date').value;
  if (!dateVal) { toast('Please select a date.', 'error'); return; }

  document.getElementById('slots-status').style.display = 'flex';
  document.getElementById('slots-result').innerHTML = '';

  try {
    const res  = await fetch(`${API_BASE}/booking/slots?date=${dateVal}`);
    const data = await res.json();

    if (data.status === 'ok') {
      const slots = data.available_slots || [];
      const dateObj = new Date(dateVal + 'T00:00:00');
      const dateStr = dateObj.toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

      if (slots.length === 0) {
        document.getElementById('slots-result').innerHTML =
          `<div style="color:var(--yellow); padding:12px; background:rgba(245,158,11,0.1); border-radius:8px;">
            ⚠️ No available slots on ${dateStr}.
          </div>`;
      } else {
        const slotBtns = slots.map(s => {
          const h = parseInt(s.split(':')[0]);
          const ampm = h >= 12 ? 'PM' : 'AM';
          const h12  = h > 12 ? h - 12 : (h === 0 ? 12 : h);
          const label = `${h12}:${s.split(':')[1]} ${ampm}`;
          return `<button class="slot-btn" onclick="selectSlot('${s}', '${dateVal}')">${label}</button>`;
        }).join('');

        document.getElementById('slots-result').innerHTML =
          `<div style="margin-bottom:8px; color:var(--text-muted); font-size:0.82rem;">
            ✅ ${slots.length} slot${slots.length > 1 ? 's' : ''} available on ${dateStr}:
          </div>
          <div class="slots-grid">${slotBtns}</div>`;
      }
    } else {
      toast('Failed to fetch slots.', 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
  document.getElementById('slots-status').style.display = 'none';
}

function selectSlot(time, date) {
  // Auto-fill the booking form with selected slot
  document.getElementById('book-date').value = date;
  document.getElementById('book-time').value = time;
  // Switch focus to booking form
  document.getElementById('book-name').focus();
  const h = parseInt(time.split(':')[0]);
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h > 12 ? h - 12 : (h === 0 ? 12 : h);
  toast(`Slot ${h12}:${time.split(':')[1]} ${ampm} selected — fill in your details to confirm.`, 'info');
}

// Auto-load available slots when booking date changes
async function autoLoadSlots() {
  const dateVal = document.getElementById('book-date').value;
  if (!dateVal) return;

  try {
    const res = await fetch(`${API_BASE}/booking/slots?date=${dateVal}`);
    const data = await res.json();

    if (data.status === 'ok') {
      const slots = data.available_slots || [];
      const select = document.getElementById('book-time');
      select.innerHTML = '';

      if (slots.length === 0) {
        select.innerHTML = '<option value="">No slots available</option>';
        toast('No slots available on this date.', 'warning');
        return;
      }

      slots.forEach(s => {
        const h = parseInt(s.split(':')[0]);
        const min = s.split(':')[1];
        const ampm = h >= 12 ? 'PM' : 'AM';
        const h12 = h > 12 ? h - 12 : (h === 0 ? 12 : h);
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = `${h12}:${min} ${ampm}`;
        select.appendChild(opt);
      });

      toast(`${slots.length} available slot${slots.length > 1 ? 's' : ''} loaded.`, 'info');
    }
  } catch (e) {
    // Silently fail — slots dropdown will keep default values
  }
}

async function bookAppointment() {
  const name  = document.getElementById('book-name').value.trim();
  const email = document.getElementById('book-email').value.trim();
  const date  = document.getElementById('book-date').value;
  const time  = document.getElementById('book-time').value;

  if (!name)  { toast('Please enter your name.', 'error'); return; }
  if (!email) { toast('Please enter your email.', 'error'); return; }
  if (!date)  { toast('Please select a date.', 'error'); return; }
  if (!time)  { toast('Please select a time slot.', 'error'); return; }

  document.getElementById('book-status').style.display = 'flex';
  document.getElementById('book-status-text').textContent = 'Booking your appointment…';
  document.getElementById('book-result').innerHTML = '';

  try {
    const res  = await fetch(`${API_BASE}/booking/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAdminHeaders() },
      body: JSON.stringify({ date, time, name, email }),
    });
    const data = await res.json();

    if (res.status === 403) {
      toast('Admin token required. Save it in localStorage as adminApiToken.', 'error');
    } else if (data.status === 'ok') {
      document.getElementById('book-result').innerHTML =
        `<div style="color:var(--green); padding:12px; background:rgba(34,197,94,0.1); border-radius:8px;">
          ✅ ${data.message}
        </div>`;
      toast('Appointment booked!', 'success');
      // Refresh history
      loadBookingHistory();
      loadBookingStats();
    } else {
      document.getElementById('book-result').innerHTML =
        `<div style="color:var(--red, #ef4444); padding:12px; background:rgba(239,68,68,0.1); border-radius:8px;">
          ❌ ${data.detail || 'Booking failed.'}
        </div>`;
    }
  } catch (e) {
    toast('Booking error: ' + e.message, 'error');
  }
  document.getElementById('book-status').style.display = 'none';
}

async function naturalBooking() {
  const query = document.getElementById('nl-booking-input').value.trim();
  const name  = document.getElementById('nl-name').value.trim() || 'User';
  const email = document.getElementById('nl-email').value.trim() || 'user@example.com';

  if (!query) { toast('Please enter your booking request.', 'error'); return; }

  document.getElementById('nl-status').style.display = 'flex';
  document.getElementById('nl-result').innerHTML = '';

  try {
    const res  = await fetch(`${API_BASE}/booking/intent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, name, email }),
    });
    const data = await res.json();

    if (data.status === 'ok') {
      const isSuccess = data.response.toLowerCase().includes('confirmed') ||
                        data.response.toLowerCase().includes('booked') ||
                        data.booking_completed === true;
      document.getElementById('nl-result').innerHTML =
        `<div style="color:${isSuccess ? 'var(--green)' : 'var(--text-primary)'}; padding:12px; background:rgba(124,58,237,0.08); border-radius:8px; line-height:1.6;">
          ${isSuccess ? '✅' : '📅'} ${data.response}
        </div>`;
      // Also show in conversation tab
      addMessage('assistant', data.response);

      if (isSuccess) {
        loadBookingHistory();
        loadBookingStats();
      }
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
  document.getElementById('nl-status').style.display = 'none';
}

// ──────────────────────────────────────────────
// Booking Stats
// ──────────────────────────────────────────────
async function loadBookingStats() {
  try {
    const res = await fetch(`${API_BASE}/booking/stats`, {
      headers: getAdminHeaders(),
    });
    if (res.status === 403) {
      const el = document.getElementById('booking-stats');
      if (el) el.textContent = 'Admin token required to view booking stats.';
      return;
    }
    const data = await res.json();

    if (data.status === 'ok') {
      const s = data.stats || {};
      const el = document.getElementById('booking-stats');
      el.innerHTML = `
        <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.85rem;">
          <div>
            <span style="color:var(--text-muted);">Total:</span>
            <strong style="color:var(--accent-light); margin-left:4px;">${s.total || 0}</strong>
          </div>
          <div>
            <span style="color:var(--text-muted);">Confirmed:</span>
            <strong style="color:var(--green); margin-left:4px;">${s.confirmed || 0}</strong>
          </div>
          <div>
            <span style="color:var(--text-muted);">Cancelled:</span>
            <strong style="color:var(--yellow); margin-left:4px;">${s.cancelled || 0}</strong>
          </div>
          <div>
            <span style="color:var(--text-muted);">Upcoming:</span>
            <strong style="color:var(--accent-light); margin-left:4px;">${s.upcoming || 0}</strong>
          </div>
        </div>
      `;
    }
  } catch (e) {
    document.getElementById('booking-stats').innerHTML =
      '<span class="text-muted">Could not load booking stats.</span>';
  }
}

// ──────────────────────────────────────────────
// Booking History
// ──────────────────────────────────────────────
async function loadBookingHistory() {
  const listEl = document.getElementById('booking-history-list');

  try {
    const res = await fetch(`${API_BASE}/booking/history?limit=20`, {
      headers: getAdminHeaders(),
    });
    if (res.status === 403) {
      listEl.innerHTML = '<span class="text-muted">Admin token required to view booking history.</span>';
      return;
    }
    const data = await res.json();

    if (data.status === 'ok') {
      const bookings = data.bookings || [];

      if (bookings.length === 0) {
        listEl.innerHTML = '<span class="text-muted">No bookings yet. Book your first appointment above!</span>';
        return;
      }

      listEl.innerHTML = bookings.map(b => {
        const statusColor = b.status === 'confirmed' ? 'var(--green)' :
                           b.status === 'cancelled' ? 'var(--yellow)' : 'var(--text-muted)';
        const statusIcon = b.status === 'confirmed' ? '✅' :
                          b.status === 'cancelled' ? '❌' : '⏸';
        const dateObj = new Date(b.date + 'T00:00:00');
        const dateStr = dateObj.toLocaleDateString('en-IN', {
          weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'
        });
        const h = parseInt((b.time_ist || '09:00').split(':')[0]);
        const min = (b.time_ist || '09:00').split(':')[1];
        const ampm = h >= 12 ? 'PM' : 'AM';
        const h12 = h > 12 ? h - 12 : (h === 0 ? 12 : h);
        const timeStr = `${h12}:${min} ${ampm}`;

        const cancelBtn = b.status === 'confirmed' && b.cal_booking_uid
          ? `<button class="btn btn-danger btn-sm" style="padding:2px 8px; font-size:0.75rem; margin-left:8px;"
               onclick="cancelBooking('${b.cal_booking_uid}')">Cancel</button>`
          : '';

        return `
          <div style="padding:10px 0; border-bottom:1px solid var(--border); font-size:0.83rem; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
            <div>
              <span style="color:${statusColor}; font-weight:600;">${statusIcon} ${b.status.toUpperCase()}</span>
              <span style="margin:0 8px; color:var(--text-primary);">${dateStr} at ${timeStr}</span>
            </div>
            <div>
              <span style="color:var(--text-secondary);">${b.attendee_name}</span>
              <span style="color:var(--text-muted); margin:0 6px;">·</span>
              <span style="color:var(--text-muted); font-size:0.78rem;">${b.attendee_email}</span>
              ${cancelBtn}
            </div>
          </div>
        `;
      }).join('');
    } else {
      listEl.innerHTML = '<span class="text-muted">Failed to load booking history.</span>';
    }
  } catch (e) {
    listEl.innerHTML = '<span class="text-muted">Could not connect to server.</span>';
  }
}

// ──────────────────────────────────────────────
// Cancel Booking
// ──────────────────────────────────────────────
async function cancelBooking(bookingUid) {
  if (!confirm('Are you sure you want to cancel this booking?')) return;

  try {
    const res = await fetch(`${API_BASE}/booking/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAdminHeaders() },
      body: JSON.stringify({ booking_uid: bookingUid }),
    });
    const data = await res.json();

    if (res.status === 403) {
      toast('Admin token required. Save it in localStorage as adminApiToken.', 'error');
    } else if (data.status === 'ok') {
      toast('Booking cancelled successfully.', 'success');
      loadBookingHistory();
      loadBookingStats();
    } else {
      toast('Cancel failed: ' + (data.detail || data.message || 'Unknown error'), 'error');
    }
  } catch (e) {
    toast('Cancel error: ' + e.message, 'error');
  }
}

// ──────────────────────────────────────────────
// Toast notifications
// ──────────────────────────────────────────────
function toast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const icons = { success: '✅', error: '❌', info: '💡', warning: '⚠️' };
  const div = document.createElement('div');
  div.className = `toast ${type}`;
  div.innerHTML = `<span>${icons[type] || '💡'}</span><span>${message}</span>`;
  container.appendChild(div);
  setTimeout(() => {
    div.style.transition = '0.3s ease';
    div.style.opacity = '0';
    div.style.transform = 'translateX(20px)';
    setTimeout(() => div.remove(), 300);
  }, 3500);
}
