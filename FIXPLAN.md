# Habla Fix Plan

Concrete implementation plan for every issue found in the [audit](AUDIT.md), in priority order. Each fix lists the files to change, what to change, and why.

---

## Fix 1: WhisperX Thread Safety (AUDIT #2)

**Problem:** `_run_quick_asr` (partials) and `_run_asr_and_diarize` (finals) both call `self._whisperx_model.transcribe()` from separate threads via `asyncio.to_thread()`. WhisperX is not thread-safe -- concurrent calls can corrupt GPU state or crash.

**Files:** `server/pipeline/orchestrator.py`

**Plan:**
1. Add a `threading.Lock` to `__init__`:
   ```python
   import threading
   self._asr_lock = threading.Lock()
   ```
2. Wrap both `_run_quick_asr` (line 457) and `_run_asr_and_diarize` (line 570) transcribe calls with `with self._asr_lock:`. This serializes all GPU access -- a partial ASR will block until a final ASR finishes and vice versa. Since both run in threads, the threading.Lock is the correct primitive (not asyncio.Lock, which only works in coroutines).
3. Also protect `_last_detected_language` writes inside the same lock scope (already covered since the writes happen inside the locked transcribe blocks).

**Impact:** Partials may be slightly delayed when a final is running, but they already take ~1s so the added latency is acceptable. Prevents crashes and corrupted output.

---

## Fix 2: Single Pipeline / Callback Clobbering (AUDIT #1)

**Problem:** The global `pipeline` has one set of callbacks. A second WebSocket client overwrites the first client's callbacks and session ID.

**Files:** `server/pipeline/orchestrator.py`, `server/routes/websocket.py`

**Plan:** Enforce single-client access. This is a personal-use app (one phone over Tailscale), so multi-client is not a real use case. The simplest fix:

1. In `websocket.py`, add a module-level `_active_session` variable (or use an `asyncio.Lock`):
   ```python
   _active_ws_lock = asyncio.Lock()
   ```
2. In `websocket_endpoint`, try to acquire the lock without blocking. If it's already held, send an error message and close the new connection:
   ```python
   if _active_ws_lock.locked():
       await websocket.accept()
       await websocket.send_text(json.dumps({
           "type": "error",
           "message": "Another client is already connected. Only one client at a time is supported."
       }))
       await websocket.close()
       return
   async with _active_ws_lock:
       # ... existing handler body ...
   ```
3. This guarantees callbacks and session_id are never clobbered. The user sees a clear error if they accidentally open a second tab.

**Future alternative:** If multi-client is ever needed, refactor callbacks into a dict keyed by session ID. But for now, the single-client lock is simpler and matches the use case.

---

## Fix 3: WebM Buffer Re-decode Performance (AUDIT #7a)

**Problem:** Every ~1s decode cycle re-decodes the **entire** `_webm_buffer` (up to 5MB) by spawning a new ffmpeg process, then uses `_last_pcm_len` to extract only new PCM. CPU cost grows linearly with session length.

**Files:** `server/routes/websocket.py` (class `ClientSession`), `server/pipeline/vad_buffer.py` (class `AudioDecoder`)

**Plan:** Replace the "decode everything, diff against last" approach with a persistent ffmpeg subprocess that streams.

1. Add a new method to `AudioDecoder` -- `start_streaming()` -- that spawns a persistent ffmpeg process with `stdin=PIPE, stdout=PIPE`:
   ```python
   async def start_streaming(self):
       self._process = await asyncio.create_subprocess_exec(
           "ffmpeg", "-y",
           "-i", "pipe:0",
           "-ar", str(self.sample_rate),
           "-ac", "1",
           "-f", "s16le",
           "pipe:1",
           stdin=asyncio.subprocess.PIPE,
           stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.DEVNULL,
       )
   ```
2. Add a `feed_chunk(self, webm_bytes)` method that writes to stdin and reads available PCM from stdout (non-blocking read with a short timeout).
3. Add `stop_streaming()` to close stdin and drain remaining stdout.
4. Rewrite `_continuous_decode_loop` in `ClientSession`:
   - On start, call `self.decoder.start_streaming()`
   - On each chunk from `_chunk_queue`, call `self.decoder.feed_chunk(chunk)`
   - Feed the returned PCM directly to `self.vad.feed_pcm()`
   - Remove `_webm_buffer`, `_last_pcm_len`, and the `MAX_BUFFER_BYTES` reset logic entirely
   - On stop, call `self.decoder.stop_streaming()` and feed any remaining PCM
5. Remove `_chunk_queue` -- chunks go directly to the ffmpeg pipe. Use an `asyncio.Event` for the timing signal instead.

**Risk:** WebM is a container format; ffmpeg needs the container header before it can decode. With a persistent pipe, the first few chunks provide the header, then subsequent chunks decode incrementally. This is how ffmpeg pipe mode is designed to work. The key consideration is that `MediaRecorder.start(1000)` produces 1-second WebM segments, each with its own header -- so each chunk is a complete, self-contained WebM blob. We may need to use a concat demuxer or accept that the first chunk per `start()` call needs a header. **Test carefully.**

**Fallback plan:** If persistent ffmpeg piping proves unreliable with WebM segments, use a simpler optimization: keep the current approach but maintain a sliding window instead of the full buffer. After each successful decode, truncate `_webm_buffer` to keep only the last 2 seconds of data (by tracking byte offsets mapped to timestamps). This is less elegant but still O(1) per decode cycle.

---

## Fix 4: saveIdiom Uses Wrong Exchange (AUDIT #6)

**Problem:** `saveIdiom()` always reads `state.exchanges[state.exchanges.length - 1]` instead of the exchange the clicked button belongs to.

**File:** `client/js/ui.js:252-256`

**Plan:** Replace the last-exchange lookup with DOM traversal:
```js
async function saveIdiom(btn) {
  const card = btn.closest('.idiom'), phrase = card.dataset.phrase;
  const exCard = btn.closest('.ex');
  const ex = exCard?._exData;
  const idiom = ex?.idioms?.find(i => i.phrase === phrase);
  // ... rest unchanged
```

This uses the `_exData` property already stored on each `.ex` element (line 124: `el._exData = msg`). The fix is two lines changed.

---

## Fix 5: Double-Tap Race on Listen Button (AUDIT #8)

**Problem:** `startListening()` is async and sets `state.listening = true` only after `getUserMedia` resolves. A second tap during the ~100-500ms `getUserMedia` call starts a second recording, leaking the first mic stream and timer interval.

**File:** `client/js/audio.js:5-60`

**Plan:** Add an immediate guard at the top of `startListening()`:
```js
export async function startListening() {
  if (state.listening || state._startingListen) return;
  if (!state.audioCompat) {
    toast('Audio recording not available in this browser', 'error', 8000);
    return;
  }
  state._startingListen = true;
  try {
    // ... existing getUserMedia + setup code ...
    state.listening = true;
    // ... rest of function ...
  } catch (err) {
    toast('Microphone access required. Please allow in browser settings.', 'error', 8000);
  } finally {
    state._startingListen = false;
  }
}
```

Add `_startingListen: false` to the state object in `core.js`.

---

## Fix 6: Orphaned Partial Card Cleanup (AUDIT #9)

**Problem:** If partials arrive but no final ever comes (empty ASR, pipeline error, short segment), the "Live" card stays on screen forever.

**File:** `client/js/ui.js`

**Plan:** Add a watchdog in `showPartialSource()`:
```js
export function showPartialSource(text) {
  // ... existing code ...

  // Reset stale-partial watchdog
  clearTimeout(state.partialWatchdog);
  state.partialWatchdog = setTimeout(() => {
    if (state.partialEl && Date.now() - state.lastPartialTime > 8000) {
      clearPartial();
    }
  }, 10000);
}
```

Add `partialWatchdog: null` to the state object in `core.js`. Also clear it in `clearPartial()`:
```js
export function clearPartial() {
  clearTimeout(state.partialWatchdog);
  // ... existing cleanup ...
}
```

The watchdog fires 10s after the last partial. If no new partial has arrived in 8s, the card is stale and gets removed.

---

## Fix 7: Wake Lock Not Released (AUDIT #12)

**Problem:** `stopListening()` never calls `state.wakeLock.release()`. Screen stays on after stopping.

**File:** `client/js/audio.js:62-72`

**Plan:** Add release to `stopListening()`:
```js
export function stopListening() {
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') state.mediaRecorder.stop();
  if (state.micStream) state.micStream.getTracks().forEach(t => t.stop());
  state.micStream = null; state.mediaRecorder = null; state.listening = false;

  // Release wake lock
  if (state.wakeLock) {
    state.wakeLock.release().catch(() => {});
    state.wakeLock = null;
  }

  send({ type: 'stop_listening' });
  // ... rest unchanged
```

---

## Fix 8: Unbounded DOM and Exchange Array Growth (AUDIT #7)

**Problem:** `state.exchanges` and the `#transcript` DOM grow without limit. A 4-hour classroom session accumulates hundreds of nodes on a mobile phone.

**File:** `client/js/ui.js:82-132`

**Plan:** Add pruning at the top of `addExchange()`, after the merge check:
```js
const MAX_EXCHANGES = 200;

// ... after the merge check and push ...
state.exchanges.push(msg);

// Prune oldest exchanges if over limit
while (state.exchanges.length > MAX_EXCHANGES) {
  const old = state.exchanges.shift();
  if (old._el && old._el.parentNode) old._el.remove();
}
```

200 exchanges is plenty of visible history (~1.5 hours at 2/min). The DB already persists all exchanges server-side for later retrieval.

---

## Fix 9: Route Continuous Mode Through Orchestrator Queue (AUDIT #4, #5)

**Problem:** `ClientSession._process_segment` calls `pipeline._process_audio_segment_from_wav()` directly, bypassing the orchestrator's queue. The queue worker runs idle. The shutdown drain logic drains nothing.

**Files:** `server/routes/websocket.py:241-272`, `server/pipeline/orchestrator.py`

**Plan:**
1. Add a public method to `PipelineOrchestrator` that accepts a WAV path instead of raw bytes:
   ```python
   async def process_wav(self, wav_path: str) -> Exchange | None:
       if not self._ready:
           return None
       future = asyncio.get_event_loop().create_future()
       await self._queue.put((wav_path, future))
       return await future
   ```
2. Update `_process_queue` worker to handle WAV paths (it currently expects raw audio bytes). Change `_process_audio_segment` to branch on input type, or just always accept WAV paths since `ClientSession` already writes the WAV.
3. Update `ClientSession._process_segment` to call `self.pipeline.process_wav(wav_path)` instead of the private method. Remove the `_processing_lock` since the queue now serializes.
4. Fix the shutdown drain: after the drain timeout, iterate remaining queue items and set their futures to `CancelledError`:
   ```python
   while not self._queue.empty():
       try:
           _, future = self._queue.get_nowait()
           future.cancel()
       except asyncio.QueueEmpty:
           break
   ```

---

## Fix 10: Translation Timeout Tuning (AUDIT #11)

**Problem:** 30s total timeout with 3 retries and exponential backoff means a single failed translation can block processing for 93 seconds.

**File:** `server/pipeline/translator.py`

**Plan:**
1. Use separate connect and read timeouts in `httpx.AsyncClient`:
   ```python
   self.client = httpx.AsyncClient(
       timeout=httpx.Timeout(connect=5.0, read=config.timeout_seconds, write=5.0, pool=5.0)
   )
   ```
2. Reduce retries for local providers from 3 to 1 (Ollama/LM Studio are local -- if they don't respond in 30s, a retry won't help):
   ```python
   MAX_RETRIES_LOCAL = 1
   MAX_RETRIES_CLOUD = 3
   ```
3. Adjust `_call_provider` to use the appropriate retry count based on provider type.

**Impact:** Worst-case local failure drops from 93s to ~32s. Cloud providers keep full retry budget since transient network issues are common.

---

## Fix 11: Direction/Mode Changes Lost During Disconnect (AUDIT #13)

**Problem:** Only `text_input` messages are queued when the WebSocket is down. Direction and mode changes are silently dropped.

**File:** `client/js/core.js:56-64`

**Plan:** Expand the send queue to include state-change messages:
```js
const QUEUEABLE_TYPES = new Set(['text_input', 'toggle_direction', 'set_mode', 'rename_speaker']);

export function send(o) {
  if (state.ws?.readyState === 1) {
    state.ws.send(JSON.stringify(o));
  } else if (QUEUEABLE_TYPES.has(o.type)) {
    state.pendingTextQueue.push(o);
  }
}
```

On reconnect, the queued state changes will replay before any new input. If multiple direction toggles are queued, they all replay in order, which is correct (final state matches what the user expects).

---

## Fix 12: Duplicate Final Detection (AUDIT #14)

**Problem:** If the server sends the same translation twice, two identical cards appear.

**Files:** `server/pipeline/orchestrator.py`, `server/models/schemas.py`, `client/js/ui.js`

**Plan:**
1. Server: Add an `exchange_id` field to `WSTranslation` in `schemas.py`. Set it from the DB-generated exchange ID in `_save_exchange()` (already returns `cursor.lastrowid`). Pass it through `process_text` -> callback.
2. Client: In `finalizeExchange()`, check if an exchange with this ID already exists:
   ```js
   if (msg.exchange_id && state.exchanges.some(e => e.exchange_id === msg.exchange_id)) return;
   ```

---

## Fix 13: CSS Injection via Speaker Color (AUDIT #15)

**Problem:** `msg.speaker.color` is injected into `style` attributes unsanitized.

**File:** `client/js/ui.js:137, 186-187`

**Plan:** Add a color validator:
```js
function safeColor(c) {
  return /^#[0-9a-fA-F]{3,8}$/.test(c) ? c : '#536471';
}
```

Use it everywhere a color is interpolated into style attributes:
```js
const color = safeColor(msg.speaker?.color);
```

Apply in `buildExchangeHTML` (line 137), `addExchange` (line 120), and `updateSpeakers` (line 223).

---

## Fix 14: Toast Removal Fallback (AUDIT #16)

**Problem:** `dismissToast` relies on `animationend` which may never fire if CSS animations are missing.

**File:** `client/js/core.js:84-88`

**Plan:** Add a fallback timeout:
```js
export function dismissToast(el) {
  if (!el || el.classList.contains('out')) return;
  el.classList.add('out');
  el.addEventListener('animationend', () => el.remove());
  setTimeout(() => { if (el.parentNode) el.remove(); }, 500);
}
```

---

## Fix 15: Service Worker Cache Versioning (AUDIT #18)

**Problem:** Most JS files in `SHELL_URLS` lack version query strings. After code changes, stale cached versions may be served.

**File:** `client/sw.js`

**Plan:** Bump `CACHE_NAME` to `habla-v21` and add version parameters to all JS files:
```js
const CACHE_NAME = 'habla-v21';
const SHELL_URLS = [
  '/',
  '/static/styles.css?v=21',
  '/static/js/core.js?v=21',
  '/static/js/ui.js?v=21',
  '/static/js/audio.js?v=21',
  '/static/js/settings.js?v=21',
  '/static/js/websocket.js?v=21',
  '/static/js/app.js?v=21',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
];
```

Going forward, bump the version number on every deploy. Since `index.html` references these with `<script src="/static/js/app.js?v=...">`, the HTML version params need to match.

---

## Fix 16: Service Worker 404 Fallthrough (AUDIT #20)

**Problem:** `/sw.js` endpoint returns `FileResponse` on a non-existent file, causing a 500 error.

**File:** `server/main.py:217-223`

**Plan:**
```python
@app.get("/sw.js")
async def service_worker():
    sw_path = client_dir / "sw.js"
    if sw_path.exists():
        return FileResponse(str(sw_path), media_type="application/javascript")
    from fastapi.responses import Response
    return Response(status_code=404)
```

---

## Fix 17: Remove Dead Code (AUDIT #21)

**File:** `client/js/ui.js:206-214`

**Plan:** Delete the `mergePartialText` function entirely. It is defined but never called.

---

## Fix 18: FTS5 Rebuild on Startup (AUDIT #17)

**Problem:** `INSERT INTO vocab_fts(vocab_fts) VALUES ('rebuild')` runs on every server start, which is slow for large vocab tables.

**File:** `server/db/database.py:148`

**Plan:** Only rebuild if the FTS table is empty (first run or post-migration):
```python
row = await db.execute_fetchall("SELECT COUNT(*) as c FROM vocab_fts")
count = row[0][0] if row else 0
if count == 0:
    vocab_count = await db.execute_fetchall("SELECT COUNT(*) as c FROM vocab")
    if vocab_count and vocab_count[0][0] > 0:
        await db.execute("INSERT INTO vocab_fts(vocab_fts) VALUES ('rebuild')")
```

The FTS triggers (already in place) keep the index in sync during normal operation, so a rebuild is only needed to backfill pre-trigger data or on first run.

---

## Fix 19: VAD Forced-Split Audio Gap (AUDIT #19)

**Problem:** When a long monologue hits `max_segment_frames` and is force-split, `_emit_segment` sets `_is_speaking = False`. The next speech frame starts a fresh utterance, losing the split-boundary frames.

**File:** `server/pipeline/vad_buffer.py:193-196`

**Plan:** After a forced split, keep `_is_speaking = True` and seed the new speech buffer with the last ~200ms of audio from the previous segment (reuse the pre-speech padding amount):
```python
if self._speech_frames >= self._max_segment_frames:
    logger.debug(f"Max segment reached, forcing split")
    # Save tail for crossfade into next segment
    tail_bytes = self.config.pre_speech_padding_ms * self.config.sample_rate * 2 // 1000
    tail = bytes(self._speech_buffer[-tail_bytes:]) if len(self._speech_buffer) > tail_bytes else b''
    await self._emit_segment()
    # Continue speaking -- seed next segment with tail
    self._is_speaking = True
    self._speech_buffer = bytearray(tail)
    self._speech_frames = len(tail) // (self._frame_size * 2)
```

---

## Fix 20: Chunk Queue Memory Waste (AUDIT #22)

**Problem:** Audio chunks are stored in both `_chunk_queue` and `_webm_buffer`. The queue data is never read.

**File:** `server/routes/websocket.py`

**Plan:** This is fully addressed by Fix 3 (streaming ffmpeg). If Fix 3 uses the fallback sliding-window approach instead, replace the queue with a lightweight signal:
```python
self._chunk_event = asyncio.Event()
```

In `handle_audio_chunk`:
```python
self._webm_buffer.extend(chunk)
self._chunk_event.set()
```

In `_continuous_decode_loop`, replace the queue wait with:
```python
await asyncio.wait_for(self._chunk_event.wait(), timeout=remaining)
self._chunk_event.clear()
```

---

## Fix 21: Reconnect Backoff Jitter (AUDIT #23)

**Problem:** Deterministic reconnect delay. Minor issue for single-user, but easy to fix.

**File:** `client/js/websocket.js:56`

**Plan:**
```js
const jitter = 0.5 + Math.random(); // 0.5x to 1.5x
const delay = Math.min(WS_BASE_DELAY * Math.pow(2, state.wsAttempt - 1) * jitter, WS_MAX_DELAY);
```

---

## Implementation Order

The fixes are ordered by priority (crash prevention first, then correctness, then polish):

| Phase | Fixes | Description |
|-------|-------|-------------|
| **Phase 1: Stability** | 1, 2, 5 | Thread safety, single-client lock, double-tap guard |
| **Phase 2: Correctness** | 4, 6, 12, 13 | saveIdiom, orphaned partials, dedup, color sanitization |
| **Phase 3: Performance** | 3, 8, 18 | Streaming decode, DOM pruning, FTS rebuild |
| **Phase 4: UX** | 7, 11, 14, 16 | Wake lock, disconnect queue, toast fallback, sw.js 404 |
| **Phase 5: Architecture** | 9, 10 | Queue routing, timeout tuning |
| **Phase 6: Cleanup** | 15, 17, 19, 20, 21 | SW versioning, dead code, VAD gap, chunk memory, jitter |
