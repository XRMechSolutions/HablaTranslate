# Habla App Audit -- 2026-02-15

Full stability and correctness audit of the Habla transcription/translation app. Focused on transcription overwriting, data loss, race conditions, and stability.

---

## CRITICAL Issues

### 1. Single Pipeline Shared Across All WebSocket Clients

**Files:** `server/routes/websocket.py:310-326`, `server/pipeline/orchestrator.py:200-204`

The `pipeline` is a single global instance. Every new WebSocket connection overwrites the pipeline's callbacks and session ID:

```python
# websocket.py:322-326
pipeline.set_callbacks(
    on_translation=on_translation,
    on_partial=on_partial,
    on_speakers=on_speakers,
)
```

If client B connects while client A is active:
- Client A stops receiving all translations, partials, and speaker updates
- All output routes only to client B
- Client A's session/exchanges get saved under client B's session ID

**Fix:** Either enforce single-client (reject new connections while one is active) or make the pipeline per-connection / route callbacks by session ID.

---

### 2. WhisperX Model Accessed From Multiple Threads Simultaneously

**File:** `server/pipeline/orchestrator.py`

Partial transcription (`process_partial_audio` -> `_run_quick_asr`) and final transcription (`_process_audio_segment` -> `_run_asr_and_diarize`) both call `self._whisperx_model.transcribe()` from separate threads via `asyncio.to_thread()`. WhisperX/faster-whisper models are **not thread-safe**. This can cause:
- Corrupted transcriptions
- GPU memory errors / CUDA crashes
- Silent wrong results

**Fix:** Add a threading lock (`threading.Lock`) around all `_whisperx_model.transcribe()` calls, or serialize partial and final ASR through the same thread.

---

## HIGH Issues

### 3. WebM Buffer Re-decoded From Start Every Cycle (Performance Degrades Over Time)

**File:** `server/routes/websocket.py:180`

The decode loop re-decodes the **entire** `_webm_buffer` every ~1 second by spawning a new ffmpeg process. It then uses `_last_pcm_len` to extract only the new PCM. For a 1-hour session, this means re-parsing up to 5MB of WebM every second. CPU cost grows linearly with session length until the 5MB cap triggers a full reset (losing any in-progress speech).

**Fix:** Use a persistent ffmpeg subprocess with streaming stdin/stdout instead of re-decoding the full buffer each cycle. Or maintain a sliding window that only keeps recent unprocessed WebM data.

---

### 4. Continuous Mode Bypasses the Orchestrator Queue Entirely

**File:** `server/routes/websocket.py:258`

```python
exchange = await self.pipeline._process_audio_segment_from_wav(wav_path)
```

The WebSocket handler calls the private `_process_audio_segment_from_wav` directly, bypassing the orchestrator's queue (which has maxsize=5 backpressure). The queue worker runs but is never fed work during normal operation. The shutdown drain logic drains an empty queue.

**Fix:** Route continuous-mode segments through `pipeline.process_audio()` to use the queue, or remove the queue if it's not needed.

---

### 5. Queued Futures Never Resolved on Shutdown

**File:** `server/pipeline/orchestrator.py:130-157`

During shutdown, the worker task is cancelled. Any items still in the queue have their futures never resolved -- callers awaiting those futures will hang until the event loop shuts down.

**Fix:** On shutdown, drain the queue and set all remaining futures to a `CancelledError` or a sentinel result.

---

### 6. saveIdiom() Uses Last Exchange Instead of Clicked Card's Exchange

**File:** `client/js/ui.js:253-256`

```js
const ex = state.exchanges[state.exchanges.length - 1];
const idiom = ex?.idioms?.find(i => i.phrase === phrase);
```

When saving an idiom, the code always looks at the **last** exchange in the array, not the exchange the clicked idiom card belongs to. If the user clicks "Save" on an idiom from an earlier exchange, the idiom data won't be found and the save silently fails.

**Fix:** Traverse up from the button to the `.ex` card element and use its `_exData` to find the correct exchange.

---

### 7. Unbounded DOM and State Growth (Memory Leak on Long Sessions)

**File:** `client/js/ui.js:116-125`

Every finalized exchange is pushed to `state.exchanges` and a DOM element is created. Neither is ever pruned. A 4-hour classroom session (~480 exchanges) accumulates all objects and DOM nodes. On a mobile phone this will cause increasing jank and memory pressure.

**Fix:** Cap the array and DOM to a reasonable limit (e.g., 200 exchanges). Remove oldest DOM nodes and shift oldest array entries when the cap is reached. Consider persisting to the server/DB if history needs to be preserved.

---

### 8. Double-Tap Race on Listen Button Leaks Mic Streams

**Files:** `client/js/app.js:20-23`, `client/js/audio.js:14-36`

`startListening()` is async and sets `state.listening = true` only after `getUserMedia` resolves (~100-500ms). A second tap during this window calls `startListening()` again, creating:
- Two mic streams (first is leaked, never stopped)
- Two MediaRecorder instances (first is orphaned)
- Two setInterval timers (first handle is lost)

**Fix:** Set a `state.startingListening` guard immediately on entry, before the await. Or disable the button during the async setup.

---

### 9. Orphaned Partial Card if No Final Ever Arrives

**File:** `client/js/ui.js`

If the server sends partial transcriptions but never sends a final (ASR produces empty text, pipeline errors, segment too short), the "Live" partial card remains on screen indefinitely with stale text. There is no watchdog timer to clean up orphaned partials.

**Fix:** Add an independent `setTimeout` (e.g., 10s) that clears the partial card if no update has been received. Reset the timer on each new partial.

---

## MEDIUM Issues

### 10. `_last_detected_language` Written From Multiple Threads

**File:** `server/pipeline/orchestrator.py:464-465, 577-578`

Both `_run_quick_asr` (partial) and `_run_asr_and_diarize` (final) write `self._last_detected_language` from different threads. Partial ASR on garbled audio could detect the wrong language, which the final ASR then uses.

**Fix:** Only update `_last_detected_language` from the final ASR path, or protect with a lock.

---

### 11. Translation Timeout of 30s Can Block Queue Worker for 93s

**File:** `server/pipeline/translator.py:73`

The httpx total timeout is 30s. With 3 retries and exponential backoff (1s, 2s, 4s), a single failed translation blocks the queue worker for up to 93 seconds. All subsequent translations are delayed.

**Fix:** Reduce max retries for slow LLM responses, or use separate connect/read timeouts. Consider a shorter total timeout with fewer retries.

---

### 12. Wake Lock Never Released on Stop

**File:** `client/js/audio.js:62-72, 75-77`

`requestWake()` acquires a screen wake lock but `stopListening()` never calls `state.wakeLock.release()`. The screen stays on after the user stops listening, draining battery.

**Fix:** Call `state.wakeLock?.release()` in `stopListening()`.

---

### 13. Direction/Mode Changes Lost During Disconnect

**File:** `client/js/core.js:60-64`

Only `text_input` messages are queued during WebSocket disconnect. If the user toggles direction or mode while disconnected, those messages are silently dropped. On reconnect, the server sends its current state, overwriting the user's intended change.

**Fix:** Queue `toggle_direction` and `set_mode` messages, or re-apply client-side state after reconnect.

---

### 14. No Duplicate Detection for Final Translations

**File:** `client/js/ui.js`

If the server sends the same `translation` message twice (network retry, bug), two identical cards are created. There is no exchange ID or deduplication.

**Fix:** Add an exchange ID from the server and dedup on the client.

---

### 15. CSS Injection via Unsanitized Speaker Color

**File:** `client/js/ui.js:137, 186-187`

The `color` value from the server is injected directly into `style` attributes without validation. A malicious value could overlay UI elements or exfiltrate data on older browsers.

**Fix:** Validate that `color` matches a hex color pattern (`/^#[0-9a-fA-F]{3,8}$/`) before using it.

---

### 16. Toast Elements May Never Be Removed

**File:** `client/js/core.js:84-88`

`dismissToast` adds a CSS class and waits for `animationend`. If the animation doesn't fire (CSS not loaded, element hidden), the toast stays in the DOM forever.

**Fix:** Add a fallback `setTimeout(() => el.remove(), 500)`.

---

### 17. FTS5 Rebuild on Every Startup

**File:** `server/db/database.py:148`

```python
await db.execute("INSERT INTO vocab_fts(vocab_fts) VALUES ('rebuild')")
```

The FTS index is rebuilt on every server start. For a large vocab table this is slow and unnecessary.

**Fix:** Remove the rebuild or only run it conditionally (e.g., after a migration).

---

## LOW Issues

### 18. Service Worker Cache Versioning Gaps

**File:** `client/sw.js`

Some files have version query strings (`styles.css?v=3`, `app.js?v=20`) while others (`core.js`, `ui.js`, `audio.js`) do not. Unversioned files may serve stale cached versions after updates.

**Fix:** Add version query strings to all cached JS files, or change `CACHE_NAME` on every update.

---

### 19. Small Audio Gap at Forced VAD Split Boundary

**File:** `server/pipeline/vad_buffer.py:193-196`

When a long monologue hits the max segment length and is force-split, `_is_speaking` is set to false. The next speech frame is treated as a new utterance onset, potentially losing a few frames at the split boundary.

**Fix:** Carry forward the trailing audio from the split as the beginning of the next segment.

---

### 20. Service Worker Route Returns 500 on Missing File

**File:** `server/main.py:217-223`

The `/sw.js` endpoint falls through to `FileResponse` on a non-existent path, producing a 500 instead of a clean 404.

**Fix:** Return `Response(status_code=404)` when the file doesn't exist.

---

### 21. Dead Code: `mergePartialText` Never Called

**File:** `client/js/ui.js:206-214`

The function is defined but never referenced. Left over from an earlier implementation.

**Fix:** Remove the dead code.

---

### 22. Chunk Queue Holds Redundant Audio Copies

**File:** `server/routes/websocket.py:146-149`

Audio chunks are added to both `_chunk_queue` and `_webm_buffer`. The queue is only used as a timing signal -- the dequeued chunk data is never read. This doubles memory usage for queued audio.

**Fix:** Use an `asyncio.Event` or a lightweight signal instead of queuing the full chunk data.

---

### 23. No Reconnect Backoff Jitter

**File:** `client/js/websocket.js:56`

Reconnect delay is deterministic (`3000 * 2^n`). Adding random jitter would prevent thundering herd on server restart.

---

## Priority Order for Fixes

1. **WhisperX thread safety** (#2) -- can cause crashes and corrupted transcriptions right now
2. **Single pipeline shared across clients** (#1) -- architectural, but blocks multi-device use
3. **WebM buffer re-decode performance** (#3) -- will degrade any session over ~10 minutes
4. **saveIdiom wrong exchange** (#6) -- core feature broken
5. **Double-tap mic leak** (#8) -- easy to trigger on mobile
6. **Orphaned partial card** (#9) -- confusing stale UI
7. **Wake lock not released** (#12) -- battery drain on mobile
8. **Unbounded DOM growth** (#7) -- long session stability
9. **Bypass of orchestrator queue** (#4) -- architectural cleanup
10. Everything else in severity order
