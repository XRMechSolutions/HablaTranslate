"""Generate the Kotlin/Python parity goldens for shared/ assets.

Default mode (stdlib only, no server imports; runs in any Python 3.11+):

    python -I habla/scripts/gen_shared_goldens.py

writes android/shared-assets/src/test/resources/golden/prompt_format.json and
idiom_regex.json: expected outputs of str.format and re.search(..., re.IGNORECASE)
over the real shared/ files, which :shared-assets' GoldenParityTest replays
against PromptTemplate and IdiomRegex.

Every golden records `sources` (sha256 of each input file after CRLF->LF
normalization). The Kotlin tests recompute them and fail as "golden stale"
when a shared file changes without regenerating. Output is UTF-8, LF,
ensure_ascii=False, indent=1, trailing newline, and byte-identical on re-run
with the same Python version.

Options:
    --dry-run            compute and validate everything, write nothing
    --extra-probe TEXT   add a regex probe (repeatable; for checking the
                         BMP-only guard, combine with --dry-run)
"""

import glob
import hashlib
import json
import os
import re
import sys
import unicodedata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARED_DIR = os.path.join(REPO_ROOT, "shared")
SHARED_GOLDEN_DIR = os.path.join(REPO_ROOT, "android", "shared-assets", "src", "test", "resources", "golden")

# Mirrors habla/tests/test_shared_assets.py and :shared-assets Validators.EXPECTED_FIELDS.
TEMPLATE_FIELDS = {
    "translator_system": ["source_lang", "target_lang", "classroom_extra"],
    "translator_user": ["context_block", "topic_block", "speaker_label", "transcript"],
    "translator_context_line": ["speaker", "source", "translated"],
    "translator_passthrough_retry": ["source_lang", "target_lang", "transcript"],
    "quick_translate": ["source_lang", "target_lang", "text"],
    "topic_summary": ["previous_summary", "speaker_label", "latest_source", "latest_translation"],
    "study_summary_system": ["mode_note"],
    "study_summary_user": ["exchanges_text", "topic_block"],
}

UNICODE_VALUE = "¿Qué tal, señor? 😀\nlínea \"dos\" \\ fin"

SYNTHETIC_OK = [
    ("empty_template", "", {}),
    ("plain", "plain", {}),
    ("escaped_field", "{{a}}", {}),
    ("triple_braces", "{{{a}}}", {"a": "x"}),
    ("escapes_around_field", "}}{a}{{", {"a": "x"}),
    ("repeated_field", "{a}{a}", {"a": "x"}),
    ("value_not_rescanned", "{a}", {"a": "{b}", "b": "y"}),
    ("extra_key_ignored", "{a}", {"a": "x", "extra": "unused"}),
    ("spaces_in_name", "{ a }", {" a ": "x"}),
]

SYNTHETIC_ERROR = [
    ("single_open", "{", {}),
    ("single_close", "}", {}),
    ("unterminated_field", "{a", {"a": "x"}),
    ("stray_close", "a}", {}),
    ("nested_open", "{a{b}}", {"a": "x", "b": "y"}),
    ("missing_key", "{missing}", {"a": "x"}),
    ("missing_key_before_stray_close", "{missing} }", {}),
]

SYNTHETIC_UNSUPPORTED = [
    ("auto_numbered", "{}", {"a": "x"}),
    ("positional", "{0}", {"a": "x"}),
    ("conversion", "{a!r}", {"a": "x"}),
    ("format_spec", "{a:>5}", {"a": "x"}),
    ("attribute", "{a.b}", {"a": "x"}),
    ("index", "{a[0]}", {"a": "x"}),
]

NEGATIVE_PROBES = [
    "tomarpelo",
    "hola qué tal",
    "",
    "estar",
    "pelo",
    "en las",
    "buenos días",
    "TOMAR EL",
    "importa",
    "no pasa nada",
]


def python_version() -> str:
    return sys.version.split()[0]


def read_shared(rel: str) -> str:
    with open(os.path.join(SHARED_DIR, rel), "r", encoding="utf-8", newline="") as f:
        return f.read()


def source_sha256(rel: str) -> str:
    """sha256 of a shared file after CRLF->LF, so a Windows CRLF checkout hashes the same."""
    with open(os.path.join(SHARED_DIR, rel), "rb") as f:
        data = f.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def write_golden(directory: str, name: str, obj: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


def describe_error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---- prompt_format.json ---------------------------------------------------------------------

def prompt_case(case_id, template_key, template, values, kind):
    case = {
        "id": case_id,
        "template_key": template_key,
        "template": template,
        "values": values,
        "kind": kind,
        "expected": None,
        "python_error": None,
        "python_output": None,
    }
    try:
        output = template.format(**values)
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
        if kind == "ok":
            sys.exit(f"prompt case {case_id}: expected success but Python raised {describe_error(e)}")
        case["python_error"] = describe_error(e)
        return case
    if kind == "ok":
        case["expected"] = output
    elif kind == "error":
        sys.exit(f"prompt case {case_id}: expected a Python error but got {output!r}")
    else:
        case["python_output"] = output
    return case


def build_prompt_golden() -> dict:
    prompts = json.loads(read_shared("prompts.json"))
    cases = []
    for key, fields in TEMPLATE_FIELDS.items():
        template = prompts[key]
        variants = {
            "upper": {f: f.upper() for f in fields},
            "braces": {f: "{" + f + "}}{{" for f in fields},
            "unicode": {f: UNICODE_VALUE for f in fields},
            "empty": {f: "" for f in fields},
        }
        for variant, values in variants.items():
            cases.append(prompt_case(f"{key}.{variant}", key, template, values, "ok"))

    classroom = {f: f.upper() for f in TEMPLATE_FIELDS["translator_system"]}
    classroom["classroom_extra"] = prompts["translator_classroom_extra"]
    cases.append(prompt_case("translator_system.classroom_extra", "translator_system",
                             prompts["translator_system"], classroom, "ok"))
    cases.append(prompt_case("study_summary_system.classroom_note", "study_summary_system",
                             prompts["study_summary_system"],
                             {"mode_note": prompts["study_summary_classroom_note"]}, "ok"))

    for kind, table in (("ok", SYNTHETIC_OK), ("error", SYNTHETIC_ERROR), ("unsupported", SYNTHETIC_UNSUPPORTED)):
        for case_id, template, values in table:
            cases.append(prompt_case(f"synthetic.{kind}.{case_id}", None, template, values, kind))

    return {
        "python": python_version(),
        "sources": {"prompts.json": source_sha256("prompts.json")},
        "cases": cases,
    }


# ---- idiom_regex.json -----------------------------------------------------------------------

def idiom_files() -> list:
    """Same order as the server: sorted glob of shared/idioms/*.json minus index.json."""
    paths = sorted(glob.glob(os.path.join(SHARED_DIR, "idioms", "*.json")))
    return ["idioms/" + os.path.basename(p) for p in paths if os.path.basename(p) != "index.json"]


def canonical_probes(canonical: str) -> list:
    return [
        canonical,
        canonical.upper(),
        canonical.title(),
        canonical.replace(" ", " "),
        canonical.replace(" ", "\t"),
        canonical.replace(" ", "  "),
        f"Ayer dije que {canonical}, ¿vale?",
    ]


def require_bmp(probes: list) -> None:
    for probe in probes:
        if any(ord(ch) > 0xFFFF for ch in probe):
            sys.exit(f"probe outside the BMP (Java and Python spans would differ): {probe!r}")


def build_regex_golden(extra_probes: list) -> dict:
    files = idiom_files()
    refs, patterns, compiled, compile_errors = [], [], [], {}
    probes, seen = [], set()

    def add_probe(p):
        if p not in seen:
            seen.add(p)
            probes.append(p)

    for rel in files:
        entries = json.loads(read_shared(rel))
        for i, entry in enumerate(entries):
            index = len(refs)
            refs.append(f"{rel}#{i}")
            patterns.append(entry["pattern"])
            try:
                compiled.append(re.compile(entry["pattern"], re.IGNORECASE))
            except re.error as e:
                # Server parity: an uncompilable pattern is skipped and never produces hits.
                compiled.append(None)
                compile_errors[str(index)] = describe_error(e)
            for probe in canonical_probes(entry["canonical"]):
                add_probe(probe)
    for probe in NEGATIVE_PROBES + list(extra_probes):
        add_probe(probe)
    require_bmp(probes)

    hits = []
    for pi, rx in enumerate(compiled):
        if rx is None:
            continue
        for qi, probe in enumerate(probes):
            m = rx.search(probe)
            if m:
                hits.append({"pattern": pi, "probe": qi, "start": m.start(), "end": m.end()})

    whitespace = re.compile(r"\s")
    whitespace_bmp = [cp for cp in range(0x10000) if whitespace.match(chr(cp))]

    letters = sorted({ch for p in patterns for ch in p if unicodedata.category(ch).startswith("L")})
    casefold = {}
    for ch in letters:
        rx = re.compile(re.escape(ch), re.IGNORECASE)
        casefold[ch] = [cp for cp in range(0x10000) if rx.fullmatch(chr(cp))]

    return {
        "python": python_version(),
        "sources": {rel: source_sha256(rel) for rel in files},
        "whitespace_bmp": whitespace_bmp,
        "casefold": casefold,
        "patterns": refs,
        "python_compile_errors": compile_errors,
        "probes": probes,
        "hits": hits,
    }


# ---- --asr mode: language_vote.json, transcript_filters.json ---------------------------------

ASR_GOLDEN_DIR = os.path.join(REPO_ROOT, "android", "asr", "src", "test", "resources", "golden")

RESET = "RESET"

# Detection sequences; RESET mirrors the orchestrator's session reset
# (_last_detected_language = None; _language_votes.clear()).
VOTE_CASES = [
    ("first_detection_es", ["es"]),
    ("first_detection_en", ["en"]),
    ("single_outlier_ignored", ["es", "es", "es", "en", "es", "es"]),
    ("tie_keeps_current", ["es", "en"]),
    ("three_of_four_switches_before_window_full", ["es", "en", "en", "en"]),
    ("three_of_five_does_not_switch", ["es", "es", "en", "en", "en"]),
    ("four_of_five_switches", ["es", "es", "en", "en", "en", "en"]),
    ("window_rollover", ["es", "es", "es", "es", "es", "en", "en", "en", "es", "en", "en", "es", "es", "es", "es"]),
    ("switch_back_after_rollover", ["en", "en", "en", "en", "en", "es", "es", "es", "es", "es", "en"]),
    ("reset_then_first_detection", ["es", "es", "en", RESET, "en", "es"]),
    ("reset_clears_votes", ["es", "en", "en", RESET, "es", "en", "en", "en"]),
    ("third_language_counts", ["es", "fr", "en", "fr", "fr", "fr", "fr"]),
]

FILTER_TEXTS = [
    "", "   ", "\u0085hola\u0085", " ok ", "ok", "abc", "123", "12 ab", "ab12c", "½½ abc",
    "²³ abc", "Ⅻ abc", "١٢٣ abc", "１２ ａｂｃ", "😀😀 abc", "...!!!", "¿Qué?", "ola amigo",
    "OLA amigo", "Ola, amigo", "  ueno pues", "ué pasa", "Ué pasa", "qué pasa", "ﬆoy aquí",
    "stoy\taquí", "hola", "Hola", "Gracias.", "[Música]",
    "a1", "ab1", "abc1234", " omo está", "racias", "ORQUE no", "stoy", "uándo vienes",
]


def build_asr_goldens() -> dict:
    """Run the server's own functions. Needs the server venv; imports happen only here."""
    habla_dir = os.path.join(REPO_ROOT, "habla")
    if habla_dir not in sys.path:
        sys.path.insert(0, habla_dir)  # -I drops cwd and PYTHONPATH from sys.path
    try:
        from collections import deque
        from server.pipeline.audio_processor import AudioProcessor, _has_clipped_onset, _is_bad_transcript
    except ImportError as e:
        sys.exit(f"cannot import the server ({e}); run --asr with habla\\.venv\\Scripts\\python")

    profile = json.loads(read_shared("asr_profile.json"))
    window = profile["language_vote"]["window"]
    ratio = profile["language_vote"]["switch_ratio"]

    def fresh_processor():
        # AudioProcessor.__init__ needs an AppConfig and sets up locks/model slots; the voter only
        # touches these three fields, so build the object with __new__ and set exactly them
        # (same values __init__ uses). No audio device or model is opened.
        ap = AudioProcessor.__new__(AudioProcessor)
        ap._last_detected_language = None
        ap._language_votes = deque(maxlen=window)
        ap._language_confidence_threshold = ratio
        return ap

    vote_cases = []
    for case_id, steps in VOTE_CASES:
        ap = fresh_processor()
        current = []
        for step in steps:
            if step == RESET:
                ap._last_detected_language = None
                ap._language_votes.clear()
            else:
                ap._update_detected_language(step)
            current.append(ap._last_detected_language)
        vote_cases.append({"id": case_id, "steps": steps, "current": current})

    filter_cases = [
        {"text": t, "is_bad": _is_bad_transcript(t), "clipped_onset": _has_clipped_onset(t)}
        for t in FILTER_TEXTS
    ]

    sources = {"asr_profile.json": source_sha256("asr_profile.json")}
    return {
        "language_vote.json": {
            "python": python_version(),
            "sources": sources,
            "window": window,
            "switch_ratio": ratio,
            "reset_step": RESET,
            "cases": vote_cases,
        },
        "transcript_filters.json": {
            "python": python_version(),
            "sources": sources,
            "min_letters": profile["filters"]["min_letters"],
            "min_letter_ratio": profile["filters"]["min_letter_ratio"],
            "cases": filter_cases,
        },
    }


def parse_args(argv: list) -> dict:
    opts = {"asr": False, "dry_run": False, "extra_probes": []}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--asr":
            opts["asr"] = True
        elif arg == "--dry-run":
            opts["dry_run"] = True
        elif arg == "--extra-probe" and i + 1 < len(argv):
            i += 1
            opts["extra_probes"].append(argv[i])
        else:
            sys.exit(f"unknown argument {arg!r}; usage: gen_shared_goldens.py [--asr] [--dry-run] [--extra-probe TEXT]")
        i += 1
    return opts


def main(argv: list) -> int:
    opts = parse_args(argv)
    if opts["asr"]:
        goldens = build_asr_goldens()
        directory = ASR_GOLDEN_DIR
    else:
        goldens = {
            "prompt_format.json": build_prompt_golden(),
            "idiom_regex.json": build_regex_golden(opts["extra_probes"]),
        }
        directory = SHARED_GOLDEN_DIR
    if opts["dry_run"]:
        print("dry run: goldens computed, nothing written")
        return 0
    for name, obj in goldens.items():
        write_golden(directory, name, obj)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
