"""A5 engine check: whisper.cpp `small` on the PC vs the server's WhisperX goldens.

Runs the Windows/Linux build of whisper-cli (built from the same submodule the
Android app pins, android/third_party/whisper.cpp) over every post-FinalPrep WAV
that make_goldens.py wrote, with the asr module's FINAL decoding parameters
(design.md section 5, read from shared/asr_profile.json): beam 5 / best_of 5, no
temperature fallback, entropy and no-speech guards off, no context, no
timestamps, language forced to the utterance's golden detected language. A
second whisper-cli call per utterance runs language detection only, to report
language agreement.

WER uses the D6 normalization (compare_wer.normalize_for_wer) with the server
transcript as the reference. Results go to
.dev/.task-state/test-results/a5-engine-check-<date>.json; the default model's
transcript (shared/asr_models.json `default_id`) is written back into each
golden utterance as `pc_whisper_cpp`.

Machine-local locations come from HABLA_CACHE_DIR (default ~/.cache):
models in <cache>/whisper-cpp/, the CMake build in <cache>/whisper-cpp-build/.

Usage (from habla/; run make_goldens.py first, it also drops pc_whisper_cpp):
    .venv/Scripts/python.exe scripts/compare_engines.py --model ggml-small-q8_0 --model ggml-small-q5_1
    .venv/Scripts/python.exe scripts/compare_engines.py --whisper-cli D:/bin/whisper-cli.exe --clip en_aesop_01
"""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_wer import normalize_for_wer, word_errors  # noqa: E402
from server import shared_assets  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_SRC = REPO_ROOT / "android" / "third_party" / "whisper.cpp"
RESULTS_DIR = REPO_ROOT / ".dev" / ".task-state" / "test-results"
EXIT_TOOLCHAIN = 2
EXIT_INPUT = 3

logger = logging.getLogger("habla.compare_engines")


class EngineCheckError(Exception):
    """A precondition failed; carries the process exit code."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def cache_dir() -> Path:
    return Path(os.environ.get("HABLA_CACHE_DIR") or Path.home() / ".cache")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- models --------------------------------------------------------------------------------

def model_entry(model_id: str) -> dict:
    catalog = shared_assets.load_json("asr_models.json")
    for m in catalog["models"]:
        if m["id"] == model_id:
            return m
    raise EngineCheckError(f"unknown model {model_id!r}; shared/asr_models.json lists "
                           f"{[m['id'] for m in catalog['models']]}", EXIT_INPUT)


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    part.replace(dest)


def ensure_model(entry: dict, model_dir: Path, fetch=download) -> Path:
    """Model file verified against the pinned sha256; a mismatch is deleted and fetched once more."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / entry["file"]
    for attempt in range(2):
        if not path.exists():
            fetch(entry["url"], path)
        actual = sha256_file(path)
        if actual == entry["sha256"]:
            return path
        logger.warning("%s sha256 %s != pinned %s (attempt %d)", path.name, actual, entry["sha256"], attempt + 1)
        path.unlink()
    raise EngineCheckError(f"{entry['file']}: sha256 still wrong after re-download", EXIT_INPUT)


# --- whisper-cli build ---------------------------------------------------------------------

def find_cmake() -> str | None:
    found = shutil.which("cmake")
    if found:
        return found
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidate = Path(program_files) / "CMake" / "bin" / "cmake.exe"
        if candidate.exists():
            return str(candidate)
    return None


def built_cli(build_dir: Path) -> Path | None:
    exe = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    for candidate in (build_dir / "bin" / "Release" / exe, build_dir / "bin" / exe):
        if candidate.exists():
            return candidate
    return None


def build_whisper_cli(build_dir: Path, log_dir: Path) -> Path:
    """CMake Release CPU build of whisper-cli from the pinned submodule, outside the repo."""
    if not (WHISPER_SRC / "CMakeLists.txt").exists():
        raise EngineCheckError(f"{WHISPER_SRC} is empty; run `git submodule update --init`", EXIT_TOOLCHAIN)
    cmake = find_cmake()
    if cmake is None:
        raise EngineCheckError("CMake not found on PATH (or %ProgramFiles%\\CMake); install CMake 3.20+ "
                               "or pass --whisper-cli", EXIT_TOOLCHAIN)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "whisper-cpp-build.log"
    steps = [
        [cmake, "-S", str(WHISPER_SRC), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
         "-DBUILD_SHARED_LIBS=OFF", "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_SERVER=OFF",
         "-DWHISPER_SDL2=OFF", "-DGGML_CUDA=OFF"],
        [cmake, "--build", str(build_dir), "--config", "Release", "--target", "whisper-cli", "-j"],
    ]
    with open(log, "w", encoding="utf-8") as out:
        for cmd in steps:
            rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise EngineCheckError(
                    f"`{' '.join(cmd[1:3])}` failed (exit {rc}); a missing C/C++ compiler is the usual "
                    f"cause (install Visual Studio 2022 C++ tools / gcc). Log: {log}", EXIT_TOOLCHAIN)
    cli = built_cli(build_dir)
    if cli is None:
        raise EngineCheckError(f"build finished but no whisper-cli under {build_dir}/bin", EXIT_TOOLCHAIN)
    return cli


def submodule_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(WHISPER_SRC), "describe", "--tags", "--always"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --- decoding ------------------------------------------------------------------------------

def final_args(profile: dict) -> list[str]:
    """whisper-cli flags for the asr module's FINAL pass (design.md section 5)."""
    final = profile["whisper_cpp"]["final"]
    beam = final["beam_size"] or profile["whisper"]["beam_size"]
    return [
        "-bs", str(beam), "-bo", str(beam),
        "-tp", str(final["temperature"]), "-tpi", str(final["temperature_inc"]),
        "-et", str(final["entropy_thold"]), "-lpt", str(final["logprob_thold"]),
        "-nth", str(final["no_speech_thold"]),
        "-mc", "0", "-nt", "-sns", "-t", str(profile["whisper_cpp"]["threads"]),
    ]


def transcribe(cli: Path, model: Path, wav: Path, language: str, args: list[str], tmp: Path) -> str:
    out_base = tmp / wav.stem
    cmd = [str(cli), "-m", str(model), "-f", str(wav), "-l", language, *args, "-np", "-otxt", "-of", str(out_base)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise EngineCheckError(f"whisper-cli failed on {wav.name} (exit {proc.returncode}): "
                               f"{proc.stderr.decode('utf-8', 'replace')[-500:]}", EXIT_INPUT)
    txt = out_base.with_suffix(".txt")
    text = txt.read_text(encoding="utf-8")
    txt.unlink()
    return " ".join(text.split())


def detect_language(cli: Path, model: Path, wav: Path, threads: int) -> str | None:
    proc = subprocess.run([str(cli), "-m", str(model), "-f", str(wav), "-l", "auto", "-dl",
                           "-t", str(threads)], capture_output=True)
    for line in proc.stderr.decode("utf-8", "replace").splitlines():
        if "auto-detected language:" in line:
            return line.split("auto-detected language:")[1].split()[0]
    return None


# --- goldens and scoring -------------------------------------------------------------------

def load_goldens(golden_dir: Path, clip: str | None) -> dict[str, dict]:
    paths = sorted(golden_dir.glob("*.json"))
    if clip:
        paths = [p for p in paths if p.stem == clip]
    goldens = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    if not goldens:
        raise EngineCheckError(f"no goldens in {golden_dir}; run scripts/make_goldens.py", EXIT_INPUT)
    return goldens


def golden_utterances(golden: dict, golden_dir: Path) -> list[tuple[dict, Path]]:
    """ASR-ok utterances with their post-FinalPrep WAV, hash-checked against the golden."""
    out = []
    for utt in golden["utterances"]:
        if utt.get("asr_status") != "ok":
            continue
        rel = utt.get("final_prep_wav")
        if not rel:
            raise EngineCheckError(f"{golden['clip']} utterance {utt['index']} has no final_prep_wav; "
                                   "rerun scripts/make_goldens.py", EXIT_INPUT)
        wav = golden_dir / rel
        if not wav.exists() or sha256_file(wav) != utt["final_prep_sha256"]:
            raise EngineCheckError(f"{wav} missing or not the WAV the golden recorded; "
                                   "rerun scripts/make_goldens.py", EXIT_INPUT)
        out.append((utt, wav))
    return out


def score(pairs: list[tuple[str, str]]) -> dict:
    """Word-weighted WER over (reference, hypothesis) pairs after D6 normalization."""
    errors = words = 0
    for ref, hyp in pairs:
        r, h = normalize_for_wer(ref).split(), normalize_for_wer(hyp).split()
        errors += word_errors(r, h)
        words += len(r)
    return {"errors": errors, "reference_words": words, "wer": round(errors / words, 4) if words else None}


def scaled_audio_ctx(n_samples: int, pad: int, minimum: int) -> int:
    """The asr module's scaled encoder context (WhisperParamsBuilder.scaledAudioCtx)."""
    return min(1500, max(minimum, math.ceil(n_samples / 16000 * 50) + pad))


def wav_samples(wav: Path) -> int:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes()


def run_model(model_id: str, cli: Path, model_dir: Path, goldens: dict[str, dict], golden_dir: Path,
              profile: dict, scaled_ctx: tuple[int, int] | None = None) -> dict:
    """``scaled_ctx`` = (pad, min) runs FINAL with the scaled encoder context instead of the full
    30 s window: the accuracy side of the D5 question whether finals can afford it."""
    model = ensure_model(model_entry(model_id), model_dir)
    args = final_args(profile)
    threads = profile["whisper_cpp"]["threads"]
    clips: dict[str, dict] = {}
    all_pairs: list[tuple[str, str]] = []
    agree = total = 0
    with tempfile.TemporaryDirectory(prefix="habla_a5_") as tmp:
        for stem, golden in goldens.items():
            rows = []
            for utt, wav in golden_utterances(golden, golden_dir):
                started = time.perf_counter()
                utt_args = args
                if scaled_ctx:
                    utt_args = [*args, "-ac", str(scaled_audio_ctx(wav_samples(wav), *scaled_ctx))]
                text = transcribe(cli, model, wav, utt["detected_language"], utt_args, Path(tmp))
                elapsed = time.perf_counter() - started
                detected = detect_language(cli, model, wav, threads)
                total += 1
                agree += detected == utt["detected_language"]
                utt_score = score([(utt["transcript"], text)])
                rows.append({
                    "index": utt["index"], "server_language": utt["detected_language"],
                    "whisper_cpp_detected_language": detected, "server_text": utt["transcript"],
                    "whisper_cpp_text": text, "wer": utt_score["wer"], "seconds": round(elapsed, 2),
                })
                print(f"  {model_id} {stem}#{utt['index']}: WER {utt_score['wer']} lang {detected}", flush=True)
            pairs = [(r["server_text"], r["whisper_cpp_text"]) for r in rows]
            all_pairs += pairs
            clips[stem] = {**score(pairs), "utterances": rows}
    return {
        "model": model_id + (f"+ac(pad={scaled_ctx[0]},min={scaled_ctx[1]})" if scaled_ctx else ""),
        "audio_ctx": {"pad": scaled_ctx[0], "min": scaled_ctx[1]} if scaled_ctx else "full",
        "model_sha256": model_entry(model_id)["sha256"],
        "overall": score(all_pairs),
        "language_agreement": {"agree": agree, "total": total,
                               "rate": round(agree / total, 4) if total else None},
        "clips": clips,
    }


def write_back(goldens: dict[str, dict], golden_dir: Path, result: dict, engine: dict) -> None:
    for stem, golden in goldens.items():
        texts = {r["index"]: r["whisper_cpp_text"] for r in result["clips"][stem]["utterances"]}
        for utt in golden["utterances"]:
            utt["pc_whisper_cpp"] = texts.get(utt["index"])
        golden["pc_whisper_cpp_engine"] = engine
        (golden_dir / f"{stem}.json").write_text(
            json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", dest="models",
                        help="ggml model id from shared/asr_models.json (repeatable; default: its default_id)")
    parser.add_argument("--whisper-cli", type=Path, help="prebuilt whisper-cli (default: build from the submodule)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild whisper-cli even if a build exists")
    parser.add_argument("--clip", help="golden clip stem (default: all)")
    parser.add_argument("--no-write-goldens", action="store_true", help="do not add pc_whisper_cpp to the goldens")
    parser.add_argument("--scaled-ctx", nargs=2, type=int, metavar=("PAD", "MIN"),
                        help="FINAL with the scaled encoder context (pad/min frames) instead of the full "
                             "window; implies --no-write-goldens")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        default_id = shared_assets.load_json("asr_models.json")["default_id"]
        models = args.models or [default_id]
        for m in models:
            model_entry(m)
        if args.whisper_cli:
            if not args.whisper_cli.exists():
                raise EngineCheckError(f"--whisper-cli {args.whisper_cli} does not exist", EXIT_TOOLCHAIN)
            cli = args.whisper_cli
        else:
            build_dir = cache_dir() / "whisper-cpp-build"
            cli = None if args.rebuild else built_cli(build_dir)
            cli = cli or build_whisper_cli(build_dir, RESULTS_DIR.parent / "logs")

        golden_dir = shared_assets.shared_dir() / "testdata" / "goldens"
        goldens = load_goldens(golden_dir, args.clip)
        profile = shared_assets.asr_profile()
        engine = {"whisper_cpp": submodule_commit(), "params": "FINAL (asr design.md section 5)",
                  "flags": final_args(profile)}
        scaled = tuple(args.scaled_ctx) if args.scaled_ctx else None
        results = [run_model(m, cli, cache_dir() / "whisper-cpp", goldens, golden_dir, profile, scaled)
                   for m in models]
    except EngineCheckError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code

    closest = min(results, key=lambda r: r["overall"]["wer"] if r["overall"]["wer"] is not None else 1e9)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": "server WhisperX transcripts in shared/testdata/goldens (asr_status ok)",
        "normalization": "NFC, lowercase, Unicode P* -> space, whitespace collapsed (D6)",
        "engine": {**engine, "whisper_cli": str(cli)},
        "clips": sorted(goldens),
        "models": {r["model"]: r for r in results},
        "closest_to_whisperx": closest["model"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"a5-engine-check-{date.today().isoformat()}.json"
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if previous.get("clips") == report["clips"]:
            report["models"] = {**previous.get("models", {}), **report["models"]}
            report["closest_to_whisperx"] = min(
                report["models"].values(),
                key=lambda r: r["overall"]["wer"] if r["overall"]["wer"] is not None else 1e9)["model"]
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for r in report["models"].values():
        la = r["language_agreement"]
        print(f"{r['model']}: overall WER {r['overall']['wer']} ({r['overall']['reference_words']} words), "
              f"language agreement {la['agree']}/{la['total']}; per clip "
              + ", ".join(f"{c} {v['wer']}" for c, v in r["clips"].items()))
    print(f"closest to WhisperX: {report['closest_to_whisperx']}  -> {out}")

    default = next((r for r in results if r["model"] == default_id), None)
    if default and not args.no_write_goldens and not args.scaled_ctx:
        write_back(goldens, golden_dir, default, {**engine, "model": default_id})
        print(f"wrote pc_whisper_cpp ({default_id}) into {len(goldens)} golden(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
