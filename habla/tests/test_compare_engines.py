"""scripts/compare_engines.py (A5) and the D6 WER helpers in scripts/compare_wer.py."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


compare_wer = _load("compare_wer")
compare_engines = _load("compare_engines")


def test_normalization_matches_d6_rules():
    assert compare_wer.normalize_for_wer("¿Cómo   ESTÁS?  ¡Bien!") == "cómo estás bien"
    # NFD input folds to the same NFC words
    assert compare_wer.normalize_for_wer("Café") == compare_wer.normalize_for_wer("Café")
    # apostrophes and hyphens are punctuation (P*), so they split words
    assert compare_wer.normalize_for_wer("Aesop's well-known") == "aesop s well known"


def test_word_errors_counts_sub_del_ins():
    assert compare_wer.word_errors("a b c".split(), "a b c".split()) == 0
    assert compare_wer.word_errors("a b c".split(), "a x c".split()) == 1
    assert compare_wer.word_errors("a b c".split(), "a c".split()) == 1
    assert compare_wer.word_errors("a b c".split(), "a b c d".split()) == 1
    assert compare_wer.word_errors([], "a".split()) == 1


def test_calculate_wer_unnormalized():
    assert compare_wer.calculate_wer("hola mundo", "hola mundo") == 0.0
    assert compare_wer.calculate_wer("Hola mundo", "hola mundo") == 0.5
    assert compare_wer.calculate_wer("", "") == 0.0


def test_score_is_word_weighted_after_normalization():
    s = compare_engines.score([("Hola, mundo.", "hola mundo"), ("uno dos tres cuatro", "uno dos")])
    assert s == {"errors": 2, "reference_words": 6, "wer": round(2 / 6, 4)}


def test_final_args_follow_profile():
    profile = {"whisper": {"beam_size": 5}, "whisper_cpp": {"threads": 4, "final": {
        "beam_size": None, "temperature": 0.0, "temperature_inc": 0.0, "entropy_thold": 0.0,
        "logprob_thold": -1.0, "no_speech_thold": 1.0}}}
    args = compare_engines.final_args(profile)
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["-bs"] == "5" and pairs["-bo"] == "5"
    assert pairs["-tpi"] == "0.0" and pairs["-et"] == "0.0" and pairs["-nth"] == "1.0"
    assert "-nt" in args and "-sns" in args
    profile["whisper_cpp"]["final"]["beam_size"] = 3
    assert dict(zip(compare_engines.final_args(profile)[::2], compare_engines.final_args(profile)[1::2]))["-bs"] == "3"


def test_hash_mismatch_redownloads_once(tmp_path):
    good = b"real model bytes"
    entry = {"file": "m.bin", "url": "u", "sha256": hashlib.sha256(good).hexdigest()}
    (tmp_path / "m.bin").write_bytes(b"corrupt")
    calls = []

    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes(good)

    assert compare_engines.ensure_model(entry, tmp_path, fetch=fetch).read_bytes() == good
    assert calls == ["u"]


def test_hash_still_wrong_after_redownload_fails(tmp_path):
    entry = {"file": "m.bin", "url": "u", "sha256": "0" * 64}

    def fetch(url, dest):
        dest.write_bytes(b"always wrong")

    with pytest.raises(compare_engines.EngineCheckError) as e:
        compare_engines.ensure_model(entry, tmp_path, fetch=fetch)
    assert e.value.code == compare_engines.EXIT_INPUT
    assert not (tmp_path / "m.bin").exists()


def test_missing_cmake_is_a_clear_toolchain_error(tmp_path, monkeypatch):
    # CI checks out without the whisper.cpp submodule; give the build a source dir to get past that check.
    src = tmp_path / "whisper.cpp"
    src.mkdir()
    (src / "CMakeLists.txt").write_text("")
    monkeypatch.setattr(compare_engines, "WHISPER_SRC", src)
    monkeypatch.setattr(compare_engines, "find_cmake", lambda: None)
    with pytest.raises(compare_engines.EngineCheckError) as e:
        compare_engines.build_whisper_cli(tmp_path / "build", tmp_path / "logs")
    assert e.value.code == compare_engines.EXIT_TOOLCHAIN
    assert "CMake not found" in str(e.value)


def test_golden_wav_hash_mismatch_is_rejected(tmp_path):
    wav = tmp_path / "wav" / "c" / "utt_000.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"prepped")
    golden = {"clip": "c.wav", "utterances": [
        {"index": 0, "asr_status": "ok", "final_prep_wav": "wav/c/utt_000.wav",
         "final_prep_sha256": hashlib.sha256(b"prepped").hexdigest()},
        {"index": 1, "asr_status": "prep_failed", "final_prep_wav": None},
    ]}
    assert [u["index"] for u, _ in compare_engines.golden_utterances(golden, tmp_path)] == [0]
    wav.write_bytes(b"changed")
    with pytest.raises(compare_engines.EngineCheckError):
        compare_engines.golden_utterances(golden, tmp_path)


def test_scaled_audio_ctx_matches_the_android_builder():
    # WhisperParamsBuilderTest values (asr design.md section 11), pad 64 / min 128.
    assert compare_engines.scaled_audio_ctx(0, 64, 128) == 128
    assert compare_engines.scaled_audio_ctx(16_000, 64, 128) == 128
    assert compare_engines.scaled_audio_ctx(32_000, 64, 128) == 164
    assert compare_engines.scaled_audio_ctx(29 * 16_000, 64, 128) == 1500
    assert compare_engines.scaled_audio_ctx(40 * 16_000, 64, 128) == 1500
