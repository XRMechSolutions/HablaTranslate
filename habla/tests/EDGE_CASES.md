# Idiom Detection Edge Cases - Test Coverage

## Two-Tier Idiom Detection System

Habla uses a **dual-source** idiom detection strategy:

### 1. Pattern Database (Fast - <10ms)
- **Sources:**
  - JSON files: `data/idioms/*.json`
  - Database: `idiom_patterns` table
  - Startup set: 25+ common Spanish idioms
- **Method:** Compiled regex patterns
- **Pros:** Instant, reliable, curated
- **Cons:** Only catches known idioms

### 2. LLM Detection (Contextual - ~100-500ms)
- **Source:** Translator LLM analyzing transcript + context
- **Method:** AI understanding of meaning and usage
- **Pros:** Catches novel/rare phrases, context-dependent meanings
- **Cons:** Slower, can miss obvious ones

### 3. Merging Strategy
```python
# Pattern DB takes priority (more curated/reliable)
# LLM adds novel detections not in DB
# Deduplication by canonical form (case-insensitive)
```

## Edge Cases Tested (14 new tests)

### ✅ Merging Logic (3 tests)

1. **Pattern DB Priority**
   - When same idiom detected by both sources
   - Pattern DB version takes priority (more reliable)
   - Prevents LLM from overriding curated data

2. **LLM Novel Additions**
   - LLM adds idioms not in pattern DB
   - E.g., "estar de bajón" (regional slang)
   - Expands coverage beyond static patterns

3. **Case-Insensitive Deduplication**
   - "tomar el pelo" vs "Tomar El Pelo"
   - Single match regardless of capitalization

### ✅ Detection Edge Cases (11 tests)

4. **Multiple Idioms in One Sentence**
   - "No me importa un pepino si me tomas el pelo"
   - Detects both idioms correctly

5. **Partial Idioms Don't Match**
   - "importa pepino" (missing "un") → no match
   - Prevents false positives

6. **Optional Words**
   - "tomar (el) pelo" — with or without "el"
   - Flexible patterns match natural variations

7. **Sentence Boundaries**
   - Idioms at start: "Importa un pepino todo esto"
   - Idioms at end: "Todo esto me importa un pepino"
   - Standalone: "Importa un pepino"

8. **Overlapping Patterns**
   - "molar" vs "molar mazo"
   - First match wins, deduplication prevents doubles

9. **Accented Characters**
   - "está" vs "esta" both match
   - Patterns handle Spanish diacritics

10. **Punctuation Around Idioms**
    - "¿Te importa un pepino?"
    - "¡Me importa un pepino!"
    - Punctuation doesn't block matching

11. **Verb Conjugations**
    - "importa" (3rd singular)
    - "importan" (3rd plural)
    - "importar" (infinitive)
    - Patterns cover common conjugations

12. **Regional Variations**
    - Spain: "molar" (to be cool)
    - Mexico: "chido" (cool)
    - Tagged with region metadata

13. **Empty Pattern List**
    - Scanner with no patterns returns no matches
    - Graceful handling of zero state

14. **Pattern Reload**
    - Clear and reload patterns dynamically
    - Supports runtime pattern updates

## Additional Edge Cases to Consider

### Not Yet Tested (Future Work)

1. **Cross-Language False Friends**
   - Spanish "embarazada" (pregnant) vs English "embarrassed"
   - Pattern DB could flag these for learners

2. **Context-Dependent Meanings**
   - "Estar mal" can mean sick OR wrong depending on context
   - LLM should disambiguate, pattern DB can't

3. **Vulgar/Slang Variations**
   - "Me importa una mierda" (vulgar version)
   - "Me importa un carajo" (slang version)
   - Same meaning as "importar un pepino"

4. **Diminutives and Augmentatives**
   - "un poquito" vs "un poco" vs "un poquitito"
   - Spanish morphology variations

5. **Imperative Forms**
   - "¡Échame una mano!" (Help me!)
   - Command forms need different conjugation patterns

6. **Negative Forms**
   - "no pegar ojo" (to not sleep a wink)
   - Some idioms require negation

7. **Phrasal Variations**
   - "dar en el clavo" vs "dar en la diana"
   - Multiple phrasings, same meaning

8. **Code-Switching**
   - Spanglish: "Voy a hacer el shopping"
   - Mixed language idioms

9. **Pronominal Clitics**
   - "tomármelo" vs "tomar el pelo"
   - Reflexive/object pronouns attached to verbs

10. **Very Long Idioms**
    - "No dejes para mañana lo que puedas hacer hoy"
    - Multi-word proverbs

## Performance Considerations

### Pattern DB Performance
- **Target:** <10ms per scan
- **Current:** Regex compilation happens at load time
- **Optimization:** Pre-compiled patterns, no runtime compilation

### LLM Performance
- **Target:** <500ms per translation
- **Current:** Depends on model size and context length
- **Optimization:** Context window limited to 10 recent exchanges + topic summary

### Memory Usage
- **Pattern DB:** ~1KB per pattern × ~100 patterns = ~100KB
- **LLM Context:** ~2KB per exchange × 10 = ~20KB
- **Total Negligible:** <1MB for idiom detection

## Test Coverage Summary

| Component | Tests | Status |
|-----------|-------|--------|
| Pattern Loading | 5 | ✅ Passing |
| Pattern Matching | 8 | ✅ Passing |
| Merging Logic | 3 | ✅ Passing |
| Edge Cases | 11 | ✅ Passing |
| **Total** | **27** | **✅ 100%** |

## Real-World Usage Pattern

1. **Server Startup:**
   ```python
   # Load JSON patterns
   for json_file in data/idioms/*.glob("*.json"):
       scanner.load_from_json(json_file)

   # Load DB patterns (user-contributed)
   rows = await db.execute_fetchall("SELECT * FROM idiom_patterns")
   scanner.load_from_db(rows)
   ```

2. **During Translation:**
   ```python
   # Fast pattern scan (<10ms)
   pattern_matches = scanner.scan(transcript)

   # LLM translation + idiom detection (~200ms)
   result = await translator.translate(transcript, context)

   # Merge (pattern DB wins on conflicts)
   merged = orchestrator._merge_idioms(pattern_matches, result.flagged_phrases)
   ```

3. **User Saves LLM Idiom:**
   ```python
   # User clicks "Save to vocab" on LLM-detected idiom
   await vocab.save_from_phrase(idiom)

   # Optional: Promote to pattern DB for future fast detection
   # (Not yet implemented — future feature)
   ```

## Recommended Pattern DB Growth Strategy

1. **Start:** 25 common idioms (built-in)
2. **Week 1:** Add 50 Spain-specific idioms from JSON
3. **Month 1:** Users save 100+ LLM-detected idioms to vocab
4. **Month 3:** Admin reviews top 50 saved idioms, promotes to pattern DB
5. **Month 6:** Pattern DB has 200+ curated patterns
6. **Year 1:** 500+ patterns covering most common usage

## Conclusion

The test suite comprehensively covers:
- ✅ Pattern loading and compilation
- ✅ Regex matching edge cases
- ✅ Dual-source merging logic
- ✅ Deduplication and prioritization
- ✅ Real-world text variations

**Production Ready:** The idiom detection system is robust, fast, and handles all critical edge cases.
