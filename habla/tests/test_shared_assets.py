"""shared/ assets: the server reads the same files the Android app packages."""

import json

import pytest

from server import shared_assets
from server.config import AppConfig, LANGUAGES
from server.pipeline.audio_processor import _final_prep_filter
from server.pipeline.vad_buffer import VADConfig


def test_shared_dir_found():
    assert (shared_assets.shared_dir() / "asr_profile.json").is_file()


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HABLA_SHARED_DIR", str(tmp_path))
    assert shared_assets.shared_dir() == tmp_path


def test_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HABLA_SHARED_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(shared_assets, "__file__", str(tmp_path / "a" / "b" / "c.py"))
    with pytest.raises(FileNotFoundError):
        shared_assets.shared_dir()


def test_profile_drives_server_defaults():
    p = shared_assets.asr_profile()
    cfg = AppConfig()
    assert cfg.audio.vad_silence_ms == p["segmenter"]["silence_duration_ms"]
    assert cfg.audio.min_speech_ms == p["segmenter"]["min_speech_ms"]
    assert cfg.audio.vad_frame_ms == p["vad"]["frame_ms"]
    assert cfg.asr.vad_threshold == p["vad"]["speech_threshold"]
    assert cfg.translator.max_context_exchanges == p["context"]["max_context_exchanges"]
    assert VADConfig().frame_ms * 16 == p["vad"]["frame_samples"]


def test_final_prep_filter_string_unchanged():
    assert _final_prep_filter() == "highpass=f=80,lowpass=f=7500,dynaudnorm=f=150:g=15:p=0.95:m=10"


def test_languages_loaded():
    assert LANGUAGES["es"]["name"] == "Spanish"
    assert LANGUAGES["en"]["name"] == "English"


def test_prompts_have_no_stray_placeholders():
    """Every template formats with its documented fields (catches a typo'd brace)."""
    fields = {
        "translator_system": dict(source_lang="S", target_lang="T", classroom_extra=""),
        "translator_user": dict(context_block="c", topic_block="t", speaker_label="s", transcript="x"),
        "translator_context_line": dict(speaker="s", source="a", translated="b"),
        "quick_translate": dict(source_lang="S", target_lang="T", text="x"),
        "topic_summary": dict(previous_summary="p", speaker_label="s", latest_source="a", latest_translation="b"),
        "study_summary_system": dict(mode_note=""),
        "study_summary_user": dict(exchanges_text="e", topic_block="t"),
    }
    prompts = shared_assets.prompts()
    for name, kwargs in fields.items():
        prompts[name].format(**kwargs)


def test_idiom_files_are_valid_json():
    files = shared_assets.idiom_files()
    assert files
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert all({"pattern", "canonical", "meaning"} <= set(e) for e in data)


def test_idiom_index_lists_every_pattern_file():
    """idioms/index.json (Android's file list) must equal the server's sorted glob."""
    idioms_dir = shared_assets.shared_dir() / "idioms"
    globbed = sorted(p.name for p in idioms_dir.glob("*.json") if p.name != shared_assets.IDIOM_INDEX)
    assert [p.name for p in shared_assets.idiom_files()] == globbed
    index = json.loads((idioms_dir / shared_assets.IDIOM_INDEX).read_text(encoding="utf-8"))
    assert index["files"] == globbed, f"index.json files {index['files']} != directory {globbed}"
