# Phase 1 Remaining Work — Implementation Plan

Date: 2026-02-25
Status: Ready for implementation
Scope: 2 items remaining to close out Phase 1

---

## 1. Idiom Feedback Loop — Client Wiring

**Problem:** `saveIdiom()` in `ui.js:394-409` saves idioms to `/api/vocab` (the vocabulary list) but never calls `/api/idioms` (the pattern DB). The backend is fully built — `POST /api/idioms` generates a regex, inserts it, and reloads the scanner. The last mile is missing: the client never triggers it.

**Goal:** When a user saves an LLM-detected idiom, it should be added to both the vocab list (for study) AND the pattern DB (so the regex scanner catches it in future utterances without needing the LLM).

### Files to modify

| File | Change |
|------|--------|
| `habla/client/js/ui.js` | Add `POST /api/idioms` call inside `saveIdiom()` |

### Implementation

In `ui.js:saveIdiom()` (line 394), after the successful `/api/vocab` POST on line 406, add a fire-and-forget call to `/api/idioms`:

```javascript
// Inside saveIdiom(), after the vocab save succeeds (line 406):
if (r.ok) {
  btn.textContent = 'Saved';
  // Also add to pattern DB for future regex matching (fire-and-forget)
  fetch('/api/idioms', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      phrase: idiom.phrase,
      meaning: idiom.meaning,
      literal: idiom.literal || '',
      region: idiom.region || 'universal'
    })
  }).catch(() => {}); // Pattern DB insert is best-effort; vocab save is the critical path
}
```

### Design decisions

- **Fire-and-forget:** The idiom pattern insert is secondary to the vocab save. If it fails (409 duplicate, server error), the user still gets their vocab card. No error toast needed.
- **No UI change:** The save button already says "Saved" after success. No need for a second indicator.
- **Dedup is handled server-side:** `api_idioms.py:93-105` checks both DB and JSON patterns before insert, returning 409 on duplicates. Safe to call on every save.
- **Scanner reload is automatic:** `api_idioms.py:115-116` calls `reload_idiom_patterns()` after insert, so the pattern is active immediately.

### Testing

- Manual: Save an idiom from a translation card, then speak the same idiom in a new utterance. It should now be caught by the regex scanner (visible in the response as a pattern-DB match rather than LLM-detected).
- Unit: Existing `test_api.py` idiom endpoint tests cover the backend. No new tests needed for a one-line client change.

### Risk: None

The backend endpoint is tested and in production. The client change is additive (one fetch call after an already-successful save). Failure is silently ignored.

---

## 2. Persist Cumulative OpenAI Cost to Database

**Problem:** `translator.py:168-175` tracks costs in-memory (`_costs` dict) with both session and all-time counters. But all-time counters reset to zero on server restart because nothing writes them to the database.

**Goal:** Persist per-session costs to the `sessions` table on session close. Provide cumulative cost queries across sessions.

### Files to modify

| File | Change | Lines affected |
|------|--------|----------------|
| `habla/server/db/database.py` | Add cost columns to `sessions` table | ~60 (schema) |
| `habla/server/pipeline/orchestrator.py` | Write costs on `close_session()` | ~370-378 |
| `habla/server/pipeline/translator.py` | Load all-time costs from DB on init, add `get_session_costs()` | ~167-175, new method |
| `habla/server/routes/api_llm.py` | Update `/api/llm/costs` to query DB for all-time totals | ~193-210 |

### Step 1: Schema migration — add cost columns to `sessions`

In `database.py:_create_tables()`, the `sessions` table (line 52) needs three new columns:

```sql
ALTER TABLE sessions ADD COLUMN llm_provider TEXT;
ALTER TABLE sessions ADD COLUMN llm_input_tokens INTEGER DEFAULT 0;
ALTER TABLE sessions ADD COLUMN llm_output_tokens INTEGER DEFAULT 0;
ALTER TABLE sessions ADD COLUMN llm_cost_usd REAL DEFAULT 0.0;
```

Since this project uses auto-creating tables (not migrations), handle this with `ALTER TABLE ... ADD COLUMN` wrapped in try/except for the "duplicate column" case, matching the existing pattern. Add this after table creation:

```python
# Add cost columns if they don't exist (migration for existing DBs)
for col, typ in [
    ("llm_provider", "TEXT"),
    ("llm_input_tokens", "INTEGER DEFAULT 0"),
    ("llm_output_tokens", "INTEGER DEFAULT 0"),
    ("llm_cost_usd", "REAL DEFAULT 0.0"),
]:
    try:
        await db.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typ}")
    except Exception:
        pass  # Column already exists
```

### Step 2: Write costs on session close

In `orchestrator.py:close_session()` (line 370), extend the UPDATE to include cost data from the translator:

```python
# Get cost data from translator
costs = self.translator.get_session_costs()

await db.execute(
    """UPDATE sessions SET ended_at = CURRENT_TIMESTAMP,
       topic_summary = ?, speaker_count = ?,
       llm_provider = ?, llm_input_tokens = ?,
       llm_output_tokens = ?, llm_cost_usd = ?
       WHERE id = ?""",
    (
        self.topic_summary or None,
        len(self.speaker_tracker.speakers),
        self.translator.config.provider,
        costs["session_input_tokens"],
        costs["session_output_tokens"],
        costs["session_cost_usd"],
        self.session_id,
    ),
)
```

### Step 3: Add `get_session_costs()` to translator

In `translator.py`, add a method that returns a snapshot of session costs (for the orchestrator to persist):

```python
def get_session_costs(self) -> dict:
    """Return current session cost snapshot for persistence."""
    return {
        "session_input_tokens": self._costs["session_input_tokens"],
        "session_output_tokens": self._costs["session_output_tokens"],
        "session_cost_usd": self._costs["session_cost_usd"],
    }
```

### Step 4: Load all-time costs from DB on startup

In `translator.py:__init__()` or a new `async load_all_time_costs()` method (called from `main.py` after DB init):

```python
async def load_all_time_costs(self):
    """Load cumulative cost totals from all past sessions."""
    db = await get_db()
    row = await db.execute_fetchone(
        """SELECT COALESCE(SUM(llm_input_tokens), 0) as total_input,
                  COALESCE(SUM(llm_output_tokens), 0) as total_output,
                  COALESCE(SUM(llm_cost_usd), 0.0) as total_cost
           FROM sessions
           WHERE llm_provider = 'openai'"""
    )
    if row:
        self._costs["all_time_input_tokens"] = row["total_input"]
        self._costs["all_time_output_tokens"] = row["total_output"]
        self._costs["all_time_cost_usd"] = row["total_cost"]
```

Call this from `main.py` during lifespan startup, after DB init and translator creation.

### Step 5: Update `/api/llm/costs` endpoint

In `api_llm.py:llm_costs()` (line 193), the endpoint already returns `t.costs` which includes all-time fields. No change needed to the endpoint itself — the data will now be populated from DB on startup and updated in-memory during the session.

### Design decisions

- **Per-session granularity:** Costs are stored per session row, not per exchange. This keeps the schema simple and avoids adding cost columns to the high-volume `exchanges` table.
- **All-time is a SUM query:** Rather than maintaining a separate running-total row, all-time cost is computed as `SUM(llm_cost_usd) FROM sessions WHERE llm_provider = 'openai'`. Simple, accurate, no sync issues.
- **In-memory all-time updates during session:** The `_track_openai_cost()` method (line 502) already increments all-time counters. After `load_all_time_costs()` seeds them from DB, they stay accurate for the duration of the server run.
- **No separate cost table:** The 4 columns on `sessions` are sufficient. A dedicated `llm_costs` table would be over-engineering for a single-user app.
- **ALTER TABLE migration:** Safe for SQLite. The try/except pattern handles both fresh DBs (columns created with the table) and existing DBs (ALTER adds them).

### Testing

New tests needed in `tests/db/test_database.py`:
- `test_session_cost_columns_exist` — verify columns after `_create_tables()`
- `test_session_cost_persistence` — create session, update with costs, verify SELECT returns them
- `test_all_time_cost_aggregation` — create 3 sessions with different costs, verify SUM query

New test in `tests/pipeline/test_translator.py`:
- `test_get_session_costs` — verify `get_session_costs()` returns current snapshot

Update in `tests/pipeline/test_orchestrator.py`:
- `test_close_session_persists_costs` — verify costs written on close

### Risk: Low

- SQLite ALTER TABLE is safe (append-only, no data migration).
- Existing session close logic is proven stable.
- Cost tracking is already working in-memory; this just adds persistence.

---

## Implementation order

1. **Idiom wiring first** (5 minutes, 1 file, zero risk)
2. **Cost persistence second** (30 minutes, 4 files, low risk, needs tests)

## Definition of done

- [ ] Saving an idiom from a translation card adds it to both vocab and pattern DB
- [ ] Server restart preserves cumulative OpenAI cost totals
- [ ] `/api/llm/costs` returns accurate all-time totals after restart
- [ ] All existing tests still pass
- [ ] New tests cover cost persistence round-trip
