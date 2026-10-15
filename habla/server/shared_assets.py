"""Loader for the repo-level shared/ directory.

shared/ holds the assets the server and the Android app both use: prompt
templates, ASR/VAD tuning (asr_profile.json), the language registry,
idiom pattern files and the cloud model table. Both apps read the same
bytes, so a tuning change lands on both sides at once.

Location: HABLA_SHARED_DIR if set, otherwise <repo>/shared next to habla/.
"""

import json
import os
from functools import lru_cache
from pathlib import Path


def shared_dir() -> Path:
    """Return the shared/ directory, raising if it cannot be found."""
    override = os.getenv("HABLA_SHARED_DIR")
    candidates = [Path(override)] if override else []
    candidates.append(Path(__file__).resolve().parent.parent.parent / "shared")
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"shared/ directory not found (looked in: {', '.join(str(p) for p in candidates)}); "
        "set HABLA_SHARED_DIR"
    )


@lru_cache(maxsize=None)
def load_json(name: str) -> dict:
    """Load and cache a JSON file from shared/ (e.g. 'asr_profile.json')."""
    with open(shared_dir() / name, "r", encoding="utf-8") as f:
        return json.load(f)


def asr_profile() -> dict:
    return load_json("asr_profile.json")


def prompts() -> dict:
    return load_json("prompts.json")


def languages() -> dict:
    return load_json("languages.json")["languages"]


def models() -> dict:
    return load_json("models.json")


IDIOM_INDEX = "index.json"


def idiom_files() -> list[Path]:
    """Bundled idiom pattern files, sorted by name.

    idioms/index.json lists these files for the Android app (which cannot
    list asset directories); it is not a pattern file, so it is skipped.
    """
    return sorted(p for p in (shared_dir() / "idioms").glob("*.json") if p.name != IDIOM_INDEX)
