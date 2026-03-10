# Client

Mobile-first PWA built with vanilla JavaScript, HTML, and CSS. No build step, no framework, no bundler. Dark theme optimized for Chrome on Android over Tailscale.

## Pages

| Page | File | Purpose |
|------|------|---------|
| Translator | `index.html` | Main interface — mic, text input, live translation cards |
| Vocab Review | `vocab.html` | SM-2 flashcard review, browse, search, export |
| Session History | `history.html` | Past sessions, replay, search, audio playback |
| Corrections | `corrections.html` | Whisper fine-tuning — correct ASR transcripts, failed segments |
| Bookmarks | `bookmarks.html` | Saved exchange viewer |

## JavaScript Modules

### app.js (108L) — Entry Point

Initializes all modules in order: audio → audio spool → UI → settings → WebSocket.

Main button handlers: listen toggle, send text, direction cycle (es_to_en → en_to_es → auto), mode toggle, navigation, session save/clear.

Keyboard shortcut: B key bookmarks the most recent exchange.

Keepalive: sends `{type: "ping"}` every 30 seconds, measures round-trip latency.

### websocket.js (177L) — Connection Management

Single WebSocket to `ws[s]://{host}/ws/translate`.

**Reconnection**: Exponential backoff 3s → 60s max, 20 attempts, random jitter. Gives up with a persistent "Connection lost" banner.

**Message dispatch**: Routes 15+ message types to handlers (partial, translation, speakers_updated, direction_changed, etc.).

**Client-authoritative state**: On reconnect, the client sends its localStorage direction/mode to the server rather than accepting server defaults. This prevents settings from resetting on page reload.

**Binary message handling**: TTS audio received as binary frames, decoded and played via Web Audio API.

### audio.js (173L) — Microphone Capture

**Codec negotiation**: Tries `audio/webm;codecs=opus` → `audio/webm` → `audio/mp4` → `audio/ogg;codecs=opus`. Falls back through the list until one works.

**Audio compression** (optional, toggle in settings): Client-side DynamicsCompressor for automatic gain control. Threshold -50dB, ratio 12:1, attack 0, release 0.25s. Boosts quiet speech while preventing clipping. Increases Opus bitrate from 96kbps to 128kbps when active.

**Voice-reactive waveform**: 5 bars driven by Web Audio `AnalyserNode` frequency data (FFT size 32, picks bins 1/2/3/5/7). Updated every animation frame during listening.

**Wake lock**: Requests `screen` wake lock during listening to prevent screen-off on Android.

**MediaRecorder**: 1-second chunks, sent as binary WebSocket frames.

### core.js (348L) — Shared State & Utilities

**Singleton state object** (`state`): WebSocket reference, direction, mode, listening flag, session ID, exchange list, pending text queue, audio spool state, latency tracking.

**DOM helpers**: `$()` (querySelector), `esc()` (HTML escape), `escA()` (attribute escape), `escRx()` (regex escape).

**Messaging**: `send(obj)` queues if WebSocket is closed (for text_input, direction, mode, rename). `sendBin(buffer)` sends audio with backlog awareness.

**Audio spooling** (IndexedDB with memory fallback): Buffers audio chunks during brief disconnections. Configurable limits (default 10 minutes / 50MB). Trimmed by age then size. Flushed on reconnect.

**Notifications**: `toast()` for transient messages (error, warn, info, ok). `showSvcNotice()` / `hideSvcNotice()` for persistent service status.

### settings.js (430L) — Settings Panel

**LLM provider management**: Provider dropdown, model dropdown, status indicators, API key entry/masking/validation.

**Playback controls**: Recording list, segment selector, speed control, ground truth loading.

**ASR settings**: Auto-language detection toggle, recording toggle, audio compression toggle.

**TTS settings**: Enable/disable, voice selection per language.

**Audio spool settings**: Max age and size configuration.

**Metrics display**: Translation request counts, average latency, failures, costs, confidence distribution, correction counts, idiom hits.

### ui.js (844L) — Rendering & Interaction

**Exchange cards**: Built from translation messages. Show speaker name/role (clickable for rename), source text with idiom highlights, corrected line (if different), translation, metadata (timestamp, language badge, confidence), audio play button, bookmark button.

**Merge logic**: Same speaker within 5 minutes → merge into one card. Sources and translations concatenate. Idiom lists combine.

**Partial transcripts**: Live-updating card at reduced opacity. Shows raw ASR and draft translation with a cursor indicator. Never auto-clears.

**Transcript finalization**: `lockTranscript()` creates a pending card (source locked, "Translating..." placeholder). `finalizeExchange()` upgrades it to a final card or merges with the previous card. Dedup by exchange_id. Pruned at 200 cards max.

**Auto-scroll**: Only scrolls to bottom if user is near bottom (80px threshold). Pauses on scroll-up, resumes when scrolled back down. No timer-based scrolling.

**Vocab saving**: Click idiom save button → `POST /api/vocab` + `POST /api/idioms`. Text selection in source → floating Save button → `POST /api/vocab`. Correction save → `POST /api/vocab`.

**Ground truth comparison**: During playback, shows word-level diffs (LCS-based) between pipeline output and reference transcript. Red=deleted, green=added, gray=match.

**Long-press**: 600ms hold on translation text copies to clipboard.

## PWA

- `manifest.json`: App name, icons (192px, 512px), standalone display, dark theme
- `sw.js`: Service worker caches app shell (HTML, CSS, JS) — not API responses
- Install prompt: Chrome "Add to Home Screen" on Android

## Design Rationale

**Why vanilla JS instead of React/Vue?** The client is ~2000 lines of JS across 6 files. A framework would add build tooling, a dependency tree, and bundle size for no real benefit at this scale. Vanilla JS loads instantly, has zero build step, and the developer can read every line.

**Why no build step?** The app is served as static files from FastAPI. No webpack, no Vite, no npm. The developer edits a file, refreshes the browser. This eliminates an entire category of tooling problems.

**Why IndexedDB for audio spooling?** Audio chunks during a brief disconnection can be 5-50MB. localStorage has a 5MB limit. IndexedDB handles arbitrary binary data with no size limit. Memory fallback ensures audio capture continues even if IDB fails.

**Why client-authoritative direction/mode?** The server is stateless between connections. If the user sets a preference and the WebSocket drops, the server has no way to remember it. localStorage is the most reliable persistence for client preferences.

**Why 5-minute merge window?** In natural conversation, the same speaker often talks for several minutes with pauses. Merging prevents a wall of single-sentence cards and produces more readable transcript blocks.

**Why 200 card max?** Memory management. Long classroom sessions can produce hundreds of exchanges. Pruning the oldest keeps DOM size bounded and scroll performance smooth.

## Planned Direction

- **Native Android app** (7.5): Background audio capture (works with screen off), persistent WebSocket through Android lifecycle, offline vocab review from synced local DB.
- **Multi-language UI** (7.9): Replace direction toggle with language pair selector. Show language flags/codes on exchange cards.
- **Content import** (7.12): Add "Import" tab for pasting text or entering URLs for paragraph-by-paragraph translation.
- **Pronunciation practice** (7.13): Add "Practice" button on vocab cards and exchange cards with saved audio.
