/**
 * Voice I/O for the playground.
 *
 * Speaker      – gapless playback: every sentence is decoded and scheduled on
 *                one Web Audio timeline, so there is no click or pause between
 *                sentences, and playback can be stopped instantly (barge-in).
 * Microphone   – one open stream, a level meter, and MediaRecorder recordings.
 * HandsFree    – speech detection on top of the microphone: an utterance is
 *                sent after a short silence, and listening resumes on its own.
 *                Speaking while the agent talks interrupts it.
 */

function base64ToArrayBuffer(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function rms(analyser, buffer) {
  analyser.getFloatTimeDomainData(buffer);
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}

export class Speaker {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.samples = null;
    this.nextStart = 0;
    this.sources = new Set();
    this.chain = Promise.resolve();
    this.generation = 0;
    this.speaking = false;
    this.onchange = () => {};
  }

  /** Must first be called from a user gesture (browsers block audio otherwise). */
  unlock() {
    if (!this.ctx) {
      const Context = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Context();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.samples = new Float32Array(this.analyser.fftSize);
      this.analyser.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
  }

  _setSpeaking(value) {
    if (this.speaking !== value) {
      this.speaking = value;
      this.onchange(value);
    }
  }

  /** Queue one WAV clip (base64). Clips play strictly in the order queued. */
  enqueue(base64) {
    this.unlock();
    const generation = this.generation;
    this.chain = this.chain.then(async () => {
      let buffer;
      try {
        buffer = await this.ctx.decodeAudioData(base64ToArrayBuffer(base64));
      } catch (_) {
        return;   // an undecodable clip is skipped, the rest still plays
      }
      if (generation !== this.generation) return;
      const source = this.ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(this.analyser);
      const start = Math.max(this.ctx.currentTime + 0.03, this.nextStart);
      source.start(start);
      this.nextStart = start + buffer.duration;
      this.sources.add(source);
      this._setSpeaking(true);
      source.onended = () => {
        this.sources.delete(source);
        if (!this.sources.size && generation === this.generation) this._setSpeaking(false);
      };
    });
  }

  /** Stop at once and drop anything queued. */
  stop() {
    this.generation += 1;
    for (const source of this.sources) {
      try { source.stop(); } catch (_) { /* already stopped */ }
    }
    this.sources.clear();
    this.nextStart = 0;
    this.chain = Promise.resolve();
    this._setSpeaking(false);
  }

  level() {
    return this.analyser && this.speaking ? rms(this.analyser, this.samples) : 0;
  }
}

const MIME_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];

export class Microphone {
  constructor() {
    this.stream = null;
    this.ctx = null;
    this.analyser = null;
    this.samples = null;
    this.recorder = null;
    this.chunks = [];
    this.startedAt = 0;
  }

  static supported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  }

  get open() { return Boolean(this.stream); }
  get recording() { return Boolean(this.recorder && this.recorder.state === 'recording'); }

  async start() {
    if (this.stream) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const Context = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Context();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.samples = new Float32Array(this.analyser.fftSize);
    this.ctx.createMediaStreamSource(this.stream).connect(this.analyser);
  }

  level() {
    return this.analyser ? rms(this.analyser, this.samples) : 0;
  }

  record() {
    const mimeType = MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) || '';
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.chunks = [];
    this.recorder.addEventListener('dataavailable', (event) => { if (event.data.size) this.chunks.push(event.data); });
    this.recorder.start(250);
    this.startedAt = performance.now();
  }

  /** Stop recording. Resolves to {blob, mimeType, durationMs}, or null if nothing was captured. */
  finish() {
    const recorder = this.recorder;
    this.recorder = null;
    if (!recorder || recorder.state === 'inactive') return Promise.resolve(null);
    const durationMs = performance.now() - this.startedAt;
    return new Promise((resolve) => {
      recorder.addEventListener('stop', () => {
        const mimeType = (recorder.mimeType || 'audio/webm').split(';')[0];
        resolve(this.chunks.length ? { blob: new Blob(this.chunks, { type: mimeType }), mimeType, durationMs } : null);
      }, { once: true });
      recorder.stop();
    });
  }

  discard() {
    const recorder = this.recorder;
    this.recorder = null;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
  }

  stop() {
    this.discard();
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    if (this.ctx) this.ctx.close().catch(() => {});
    this.stream = null;
    this.ctx = null;
    this.analyser = null;
  }
}

/** Silence that ends an utterance. Shorter replies sooner; too short cuts people off mid-pause. */
const END_OF_SPEECH_MS = 600;

/**
 * Hands-free conversation: listens continuously, detects the start and end of
 * each utterance, and hands finished recordings to ``onUtterance``.
 */
export class HandsFree {
  constructor(mic, speaker, { onUtterance, onState, onBargeIn, bargeIn = () => true }) {
    this.mic = mic;
    this.speaker = speaker;
    this.onUtterance = onUtterance;
    this.onState = onState;
    this.onBargeIn = onBargeIn;
    this.bargeIn = bargeIn;
    this.timer = null;
    this.reset();
  }

  reset() {
    this.noiseFloor = 0.01;
    this.above = 0;
    this.below = 0;
    this.inSpeech = false;
    this.speechStarted = 0;
  }

  get active() { return this.timer !== null; }

  async start() {
    await this.mic.start();
    this.reset();
    this.mic.record();
    this.onState('listening');
    this.timer = setInterval(() => this.tick(), 50);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.mic.discard();
    this.onState('idle');
  }

  tick() {
    const level = this.mic.level();
    const agentTalking = this.speaker.speaking;
    if (!this.inSpeech) this.noiseFloor = this.noiseFloor * 0.97 + Math.min(level, 0.05) * 0.03;
    // Speaking over the agent needs a clearly louder voice than its echo.
    const threshold = agentTalking ? Math.max(this.noiseFloor * 5, 0.06) : Math.max(this.noiseFloor * 3, 0.018);
    const now = performance.now();

    if (!this.inSpeech) {
      this.above = level > threshold ? this.above + 50 : 0;
      if (this.above >= (agentTalking ? 250 : 150)) {
        if (agentTalking && !this.bargeIn()) { this.above = 0; return; }
        this.inSpeech = true;
        this.below = 0;
        this.speechStarted = now;
        if (agentTalking) this.onBargeIn();
        this.onState('hearing');
        return;
      }
      // Keep idle recordings short so uploads never carry long silences.
      if (now - this.mic.startedAt > 8000) {
        this.mic.discard();
        this.mic.record();
      }
      return;
    }

    this.below = level < threshold * 0.8 ? this.below + 50 : 0;
    const spoken = now - this.speechStarted;
    if ((this.below >= END_OF_SPEECH_MS && spoken >= 350) || spoken > 30000) this.finishUtterance();
  }

  async finishUtterance() {
    this.inSpeech = false;
    this.above = 0;
    const clip = await this.mic.finish();
    if (this.active) this.mic.record();
    if (clip && clip.durationMs > 400) this.onUtterance(clip);
    else if (this.active) this.onState('listening');
  }
}

export function blobToBase64(blob) {
  // FileReader handles recordings of any length (spreading bytes into
  // String.fromCharCode overflows the call stack around 100 KB).
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] || '');
    reader.readAsDataURL(blob);
  });
}
