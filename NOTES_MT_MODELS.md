## Future: Fast EN/ES MT Models (Non-LLM)

If we want lower-latency, lower-overhead translation for English <-> Spanish
without the LLM "reasoning" overhead, consider adding a dedicated MT provider
based on MarianMT (Helsinki-NLP):

- Helsinki-NLP/opus-mt-en-es (English -> Spanish)
- Helsinki-NLP/opus-mt-es-en (Spanish -> English)

Trade-offs vs current LLM pipeline:
- Pros: smaller, faster, consistent literal translation.
- Cons: no JSON output, no idiom reasoning, no corrections/role hints,
  weaker handling of noisy ASR or context-dependent phrasing.

Possible implementation:
- Add a provider option (e.g., `mt_fast`) in translator config.
- Use `transformers` pipeline or direct model/tokenizer calls.
- Keep current LLM path for classroom mode and idiom/correction features.
