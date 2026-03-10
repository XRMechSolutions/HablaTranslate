# Translation

The Translator is a multi-provider LLM client that handles prompt construction, response parsing, cost tracking, rate limiting, retry with backoff, and automatic fallback. It's the core differentiator — the LLM sees conversation context and uses it to fix ASR errors, translate idioms by meaning, and detect grammar corrections.

## Supported Providers

| Provider | Type | Endpoint | Request Format | Response Field |
|----------|------|----------|---------------|----------------|
| **Ollama** | Local | `POST {url}/api/generate` | `prompt` + `system` | `response` |
| **LM Studio** | Local | `POST {url}/v1/chat/completions` | OpenAI `messages` array | `choices[0].message.content` |
| **OpenAI** | Cloud | `POST /v1/responses` | Concatenated `input` string | `output[].content[].text` |
| **Claude** | Cloud | `POST /v1/messages` | `messages` array + `system` | `content[0].text` |

Providers are switchable at runtime via `POST /api/llm/select` — no server restart needed.

## Prompt Architecture

### System Prompt

Dynamically built from direction (es_to_en / en_to_es) and mode (conversation / classroom):

1. **Correct ASR errors** — Use conversation context to fix homophones, dropped words, garbled phrases
2. **Translate idiomatically** — "NEVER leave source language words untranslated"
3. **Flag phrases** — Idioms, slang, false friends, grammar notes worth studying
4. **Speaker role inference** — Suggest role hint (teacher, shopkeeper, student)

**Classroom mode** adds: aggressive vocab flagging, grammar correction detection (one speaker correcting another), false friend warnings.

### User Prompt

Includes conversation context that the LLM uses for error correction and coherent translation:

- **Recent exchanges** (last 5, configurable): `[Speaker A] "hola" → "hello"`
- **Topic summary**: Rolling LLM-maintained summary of what the conversation is about
- **Current utterance**: Speaker label + raw ASR transcript

### Output Format

Single JSON object with fields: `corrected` (ASR-corrected source), `translated`, `confidence` (0-1), `flagged_phrases[]`, `is_correction`, `correction_detail`, `speaker_hint`.

## Response Parsing

Multi-stage graceful degradation for malformed LLM output:

1. Strip `<think>...</think>` blocks and markdown code fences
2. Extract first JSON block using brace-depth tracking
3. Try `json.loads()` on extracted block
4. **Fallback 1**: Regex-extract `corrected` and `translated` fields individually
5. **Fallback 2**: Pattern match for `Translated: "..."` or quoted strings
6. **Fallback 3**: Use raw cleaned text as translation, confidence=0.1

The parser never throws — it always produces a TranslationResult, even if degraded.

## Passthrough Detection

After each translation, the system compares normalized word sets of source and output. If >70% overlap, the LLM likely returned the source language untranslated. This triggers a retry with a forceful prompt: "MUST be translated to [target]. Do NOT repeat [source]."

- Single retry (no infinite loops)
- Success capped at confidence 0.5 (signal of degraded quality)
- Returns None if retry also fails

## Cost Tracking

Cloud providers (OpenAI, Claude) track token usage and calculate cost per request:

```
cost = (input_tokens / 1M * input_price) + (output_tokens / 1M * output_price)
```

**Session costs**: Reset on provider switch. Displayed in the settings panel.
**All-time costs**: Loaded from database on startup, persisted on session close.
**Local providers**: Show "Free (local)" — no cost tracking.

Pricing tables are hardcoded for current models (Mar 2026). OpenAI ranges from $0.05/1M (gpt-5-nano) to $1.75/1M (gpt-5.2). Claude ranges from $1.00/1M (haiku-4-5) to $5.00/1M (opus-4-6).

## Rate Limiting

An async lock serializes requests with a configurable minimum interval (default 0.5s, via `RATE_LIMIT_INTERVAL`). This prevents flooding Ollama during rapid partial updates and avoids cloud rate limits.

## Retry and Backoff

| Provider Type | Max Retries | Timeout |
|---------------|-------------|---------|
| Local (Ollama, LM Studio) | 1 | 15s |
| Cloud (OpenAI, Claude) | 3 | 30s |
| Degraded mode | 0 | 8s |

Exponential backoff: 1s → 2s → 4s → 8s, capped at 15s.

**Retryable**: Timeouts, 5xx errors, connection errors.
**Non-retryable**: 404, invalid model, invalid API key, 4xx client errors.

## Fallback Chain

**Cloud → Cloud (model tiers)**: If the selected model fails, cascade through cheaper models:
- OpenAI: gpt-5.2 → gpt-5 → gpt-4o → gpt-5-mini → gpt-4o-mini → gpt-5-nano
- Claude: opus-4-6 → sonnet-4-6 → haiku-4-5

**Cloud → Local**: If all cloud models fail, try Ollama then LM Studio (single attempt each). Only if `fallback_to_local=True` (default).

**Local → Cloud**: Never. Prevents surprise bills.

## Health and Degradation

Consecutive timeout tracking with thresholds:
- After 3 consecutive timeouts → enter **degraded mode** (shorter 8s timeout, no retries, fail-fast)
- Any successful response → exit degraded mode immediately

## Topic Summary Updates

A separate cheap LLM call (retries=1, max_tokens=200, temperature=0.2) maintains a rolling topic summary. Updated after each exchange. Tracks: main topics, key entities, speaking style, domain context. Kept under 100 words. Falls back to previous summary on error.

## Design Rationale

**Why OpenAI Responses API (not Chat Completions)?** The Responses API supports tighter JSON schema control and reasoning parameters. Input is concatenated (not message-structured) which works fine for translation prompts.

**Why conservative local timeouts (15s)?** Ollama and LM Studio sometimes hang or crash under load. Shorter timeouts prevent long blocks that stall the pipeline.

**Why never fall back from local to cloud?** A user running Ollama locally chose to avoid cloud costs. Silently falling back to OpenAI when Ollama hiccups would surprise them with a bill.

**Why rate limit at 0.5s?** During continuous listening, partial translations fire every ~1s. Without rate limiting, rapid speech can queue 5-10 LLM calls in seconds, overwhelming a local Ollama instance.

**Why `translate()` never raises?** The pipeline must continue even if the LLM is temporarily unavailable. Returning a low-confidence result with an error message lets the client show something rather than nothing.

## Planned Direction

- **Multi-language prompt templates** (7.9): Parameterize language names in prompts instead of hardcoded Spanish/English.
- **Conversation export** (7.8): LLM-generated study summaries of sessions using the same translator infrastructure.
- **Grammar rule engine** (7.6): Port conjugation rules from ConjugationGameKMM to validate LLM corrections against known patterns.
