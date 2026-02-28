"""Guard tests to ensure personal/sensitive data stays out of tracked files.

These tests scan the repository for patterns that should never appear in
public-facing code: hardcoded personal paths, Tailscale hostnames, API keys, etc.
"""

import os
import re
import subprocess

import pytest

# Patterns that must never appear in tracked files
FORBIDDEN_PATTERNS = [
    (r"C:\\Users\\clint", "personal Windows path"),
    (r"C:\\Clint file drop", "personal Windows path"),
    (r"tailf87b45", "personal Tailscale network ID"),
    (r"clintlaptop", "personal machine hostname"),
    (r"100\.73\.7\.66", "personal Tailscale IP"),
    (r"hf_[A-Za-z0-9]{20,}", "possible HuggingFace token"),
    (r"sk-[A-Za-z0-9]{20,}", "possible OpenAI API key"),
]

# Files that are expected to contain placeholder examples (not real secrets)
ALLOWLISTED_FILES = {
    ".env.example",  # contains hf_your_token_here, sk-your_key_here
    "test_sanitization.py",  # this file defines the patterns
}

# Only scan text files with these extensions
SCANNABLE_EXTENSIONS = {
    ".py", ".js", ".html", ".css", ".json", ".md", ".yml", ".yaml",
    ".bat", ".sh", ".txt", ".toml", ".ini", ".cfg",
}


def _get_tracked_files():
    """Return list of git-tracked files (relative paths)."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
    )
    if result.returncode != 0:
        pytest.skip("Not in a git repository")
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def _is_scannable(filepath):
    """Check if file should be scanned based on extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in SCANNABLE_EXTENSIONS


def _is_allowlisted(filepath):
    """Check if file is in the allowlist."""
    basename = os.path.basename(filepath)
    return basename in ALLOWLISTED_FILES


class TestNoPersonalData:
    """Ensure no personal paths or secrets leak into tracked files."""

    @pytest.fixture(scope="class")
    def tracked_files(self):
        return _get_tracked_files()

    @pytest.mark.unit
    def test_no_forbidden_patterns_in_tracked_files(self, tracked_files):
        """Scan all tracked text files for forbidden personal data patterns."""
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
        violations = []

        for filepath in tracked_files:
            if not _is_scannable(filepath) or _is_allowlisted(filepath):
                continue

            full_path = os.path.join(repo_root, filepath)
            if not os.path.exists(full_path):
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            for pattern, description in FORBIDDEN_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(
                        f"  {filepath}: {description} ({pattern!r}) "
                        f"found {len(matches)} match(es)"
                    )

        assert not violations, (
            "Personal data found in tracked files:\n" + "\n".join(violations)
        )

    @pytest.mark.unit
    def test_env_example_has_no_real_tokens(self):
        """Verify .env.example uses only placeholder values, not real tokens."""
        env_example = os.path.join(
            os.path.dirname(__file__), "..", ".env.example"
        )
        if not os.path.exists(env_example):
            pytest.skip(".env.example not found")

        with open(env_example, "r") as f:
            content = f.read()

        # These are acceptable placeholder patterns
        assert "hf_your_token_here" in content or "HF_TOKEN" in content
        # Real tokens are 20+ chars of alphanumeric after prefix
        real_hf = re.findall(r"hf_[A-Za-z0-9]{20,}", content)
        real_hf = [t for t in real_hf if t != "hf_your_token_here"]
        assert not real_hf, f"Real HF token in .env.example: {real_hf}"

        real_sk = re.findall(r"sk-[A-Za-z0-9]{20,}", content)
        real_sk = [t for t in real_sk if t != "sk-your_key_here"]
        assert not real_sk, f"Real OpenAI key in .env.example: {real_sk}"

    @pytest.mark.unit
    def test_dockerignore_excludes_secrets(self):
        """Verify .dockerignore excludes sensitive file patterns."""
        dockerignore = os.path.join(
            os.path.dirname(__file__), "..", ".dockerignore"
        )
        if not os.path.exists(dockerignore):
            pytest.skip(".dockerignore not found")

        with open(dockerignore, "r") as f:
            content = f.read()

        required_exclusions = [".env", "*.db", "*.log", ".git/", "tests/"]
        missing = [p for p in required_exclusions if p not in content]
        assert not missing, (
            f".dockerignore missing required exclusions: {missing}"
        )

    @pytest.mark.unit
    def test_start_habla_bat_uses_env_variable(self):
        """Verify start-habla.bat reads TAILSCALE_HOST from .env, not hardcoded."""
        bat_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "start-habla.bat"
        )
        if not os.path.exists(bat_path):
            pytest.skip("start-habla.bat not found")

        with open(bat_path, "r") as f:
            content = f.read()

        # Should read from .env
        assert ".env" in content, "start-habla.bat should read from .env file"
        # Should not have hardcoded hostname
        assert "tailf87b45" not in content, (
            "start-habla.bat contains hardcoded Tailscale hostname"
        )
