"""Generate synthetic audio test samples for pipeline benchmarking.

This script creates test audio samples using TTS (Text-to-Speech) for automated
testing when real recorded samples aren't available.

Uses Piper TTS (CPU-based, fast, good quality) which is already in the roadmap
for the main Habla project.

Requirements:
    pip install piper-tts

Usage:
    python generate_test_audio.py --all
    python generate_test_audio.py --type fast
    python generate_test_audio.py --type idioms
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional
import wave
import struct


AUDIO_DIR = Path(__file__).parent / "audio_samples"
AUDIO_DIR.mkdir(exist_ok=True)


# Test scripts for different categories
TEST_SCRIPTS = {
    "fast_es": {
        "text": "Hola buenos días necesito que me ayudes con algo urgente porque tengo un problema con la traducción y no sé qué hacer ahora mismo estoy muy preocupado y necesito una solución rápida.",
        "lang": "es",
        "rate": 1.3,  # Speed multiplier
    },
    "slow_es": {
        "text": "Hola. Buenos días. Cómo estás hoy. Me gustaría aprender más español.",
        "lang": "es",
        "rate": 0.7,
    },
    "conversation_es": {
        "text": "Qué tal el fin de semana. Yo fui a la playa con mis amigos. Hacía muy buen tiempo y lo pasamos genial. Fue una experiencia muy agradable.",
        "lang": "es",
        "rate": 1.0,
    },
    "idioms_es": {
        "text": "No me importa un pepino lo que pienses de mí. Mi hermano siempre está en las nubes durante las clases. La profesora no tiene pelos en la lengua cuando alguien mete la pata.",
        "lang": "es",
        "rate": 1.0,
    },
    "fast_en": {
        "text": "Hello good morning I need your help with something urgent because I have a problem with the translation and I don't know what to do right now I'm very worried and I need a quick solution.",
        "lang": "en",
        "rate": 1.3,
    },
    "conversation_en": {
        "text": "How was your weekend. I went to the beach with my friends. The weather was great and we had a wonderful time. It was a very pleasant experience.",
        "lang": "en",
        "rate": 1.0,
    },
}


def generate_with_piper(text: str, output_file: Path, voice: str = "es_ES-sharvard-medium", rate: float = 1.0):
    """Generate audio using Piper TTS.

    Args:
        text: Text to synthesize
        output_file: Output WAV file path
        voice: Piper voice model name
        rate: Speech rate multiplier (1.0 = normal)
    """
    try:
        # Piper command: echo "text" | piper --model voice --output_file out.wav
        cmd = [
            "piper",
            "--model", voice,
            "--output_file", str(output_file),
        ]

        if rate != 1.0:
            cmd.extend(["--length_scale", str(1.0 / rate)])  # Piper uses length_scale (inverse of rate)

        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            check=True
        )

        print(f"Generated: {output_file.name}")
        return True

    except FileNotFoundError:
        print("ERROR: Piper TTS not found. Install with: pip install piper-tts")
        print("Or download from: https://github.com/rhasspy/piper")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Piper failed: {e.stderr.decode()}")
        return False


def generate_silence(duration_s: float, output_file: Path, sample_rate: int = 16000):
    """Generate silent audio file.

    Useful for creating quiet speech tests.

    Args:
        duration_s: Duration in seconds
        output_file: Output WAV file path
        sample_rate: Sample rate in Hz
    """
    num_samples = int(duration_s * sample_rate)

    with wave.open(str(output_file), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Write silence (zeros)
        silence = struct.pack(f"{num_samples}h", *([0] * num_samples))
        wav_file.writeframes(silence)

    print(f"Generated silence: {output_file.name} ({duration_s}s)")


def generate_multi_speaker(output_file: Path):
    """Generate multi-speaker sample by combining multiple TTS voices.

    Creates a conversation with 2 speakers using different Piper voices.
    """
    # Speaker A (female voice)
    speaker_a_text = "Hola, cómo estás. Qué tal el día."
    temp_a = AUDIO_DIR / "temp_speaker_a.wav"

    # Speaker B (male voice if available, or different female)
    speaker_b_text = "Muy bien, y tú. Todo va genial, gracias."
    temp_b = AUDIO_DIR / "temp_speaker_b.wav"

    # Generate individual speakers
    success_a = generate_with_piper(speaker_a_text, temp_a, voice="es_ES-sharvard-medium")
    success_b = generate_with_piper(speaker_b_text, temp_b, voice="es_ES-sharvard-medium")

    if not (success_a and success_b):
        print("Failed to generate multi-speaker sample")
        return False

    # Combine with ffmpeg (add silence between speakers)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_a),
            "-i", str(temp_b),
            "-filter_complex",
            "[0][1]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            str(output_file)
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Generated multi-speaker: {output_file.name}")

        # Clean up temp files
        temp_a.unlink()
        temp_b.unlink()

        return True

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: ffmpeg not found or failed. Install ffmpeg for multi-speaker generation.")
        return False


def generate_all_samples():
    """Generate all test audio samples."""
    print("Generating test audio samples...\n")

    # Generate standard samples
    for name, config in TEST_SCRIPTS.items():
        output_file = AUDIO_DIR / f"{name}_01.wav"

        if output_file.exists():
            print(f"Skipping {output_file.name} (already exists)")
            continue

        voice = "es_ES-sharvard-medium" if config["lang"] == "es" else "en_US-lessac-medium"
        generate_with_piper(config["text"], output_file, voice=voice, rate=config["rate"])

    # Generate special cases
    print("\nGenerating special cases...")

    # Quiet sample (very quiet speech simulation - just low amplitude)
    quiet_file = AUDIO_DIR / "quiet_es_01.wav"
    if not quiet_file.exists():
        # Generate normal, then reduce amplitude with ffmpeg
        temp = AUDIO_DIR / "temp_quiet.wav"
        generate_with_piper(TEST_SCRIPTS["conversation_es"]["text"], temp, rate=0.9)

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp),
                "-filter:a", "volume=0.3",  # Reduce to 30% volume
                str(quiet_file)
            ], check=True, capture_output=True)
            print(f"Generated: {quiet_file.name}")
            temp.unlink()
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: ffmpeg required for quiet sample generation")

    # Multi-speaker sample
    multi_file = AUDIO_DIR / "multi_speaker_es_01.wav"
    if not multi_file.exists():
        generate_multi_speaker(multi_file)

    # Noisy sample (add background noise with ffmpeg)
    noisy_file = AUDIO_DIR / "noisy_es_01.wav"
    if not noisy_file.exists():
        # Generate base audio
        temp = AUDIO_DIR / "temp_noisy.wav"
        generate_with_piper(TEST_SCRIPTS["conversation_es"]["text"], temp)

        # Add white noise
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp),
                "-filter_complex",
                "anoisesrc=d=10:c=white:r=16000:a=0.05[noise];[0][noise]amix=inputs=2:duration=shortest",
                str(noisy_file)
            ], check=True, capture_output=True)
            print(f"Generated: {noisy_file.name}")
            temp.unlink()
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: ffmpeg required for noisy sample generation")

    print("\nDone! Audio samples generated in:", AUDIO_DIR)
    print("\nRun benchmarks with:")
    print("  pytest tests/benchmark/test_audio_pipeline.py -v -s -m benchmark")


def main():
    parser = argparse.ArgumentParser(description="Generate test audio samples for Habla benchmarks")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all test samples"
    )
    parser.add_argument(
        "--type",
        choices=["fast", "slow", "conversation", "idioms", "multi_speaker", "noisy", "quiet"],
        help="Generate specific type of sample"
    )
    parser.add_argument(
        "--lang",
        choices=["es", "en"],
        default="es",
        help="Language (default: es)"
    )

    args = parser.parse_args()

    if args.all:
        generate_all_samples()
    elif args.type:
        # Generate specific type
        sample_type = args.type
        lang_suffix = args.lang

        if sample_type in ["multi_speaker", "noisy", "quiet"]:
            # Special cases
            if sample_type == "multi_speaker":
                output = AUDIO_DIR / f"multi_speaker_{lang_suffix}_01.wav"
                generate_multi_speaker(output)
            else:
                print(f"Use --all to generate {sample_type} samples")
        else:
            # Standard samples
            key = f"{sample_type}_{lang_suffix}"
            if key in TEST_SCRIPTS:
                config = TEST_SCRIPTS[key]
                output = AUDIO_DIR / f"{key}_01.wav"
                voice = "es_ES-sharvard-medium" if config["lang"] == "es" else "en_US-lessac-medium"
                generate_with_piper(config["text"], output, voice=voice, rate=config["rate"])
            else:
                print(f"No script for {key}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
