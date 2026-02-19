#!/usr/bin/env python3
"""
Compare Word Error Rate (WER) across different configurations.

Tests saved audio samples with different parameter combinations
to find optimal settings empirically.

Usage:
    python scripts/compare_wer.py
    python scripts/compare_wer.py --samples tests/benchmark/audio_samples
    python scripts/compare_wer.py --config-file wer_test_configs.json
"""

import argparse
import json
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from jiwer import wer as calculate_wer
except ImportError:
    print("Error: jiwer library not installed")
    print("Install with: pip install jiwer")
    sys.exit(1)


@dataclass
class TestConfig:
    """Test configuration parameters."""
    name: str
    speech_threshold: float = 0.35
    pre_speech_padding_ms: int = 300
    silence_duration_ms: int = 600
    whisper_model: str = "small"
    beam_size: int = 3
    vad_onset: float = 0.01


@dataclass
class WERResult:
    """WER test result."""
    config_name: str
    audio_file: str
    reference: str
    hypothesis: str
    wer: float
    duration_seconds: float
    processing_time_seconds: float


async def transcribe_with_config(
    audio_path: Path,
    config: TestConfig
) -> Tuple[str, float]:
    """
    Transcribe audio with given configuration.

    Returns (transcript, processing_time)
    """

    start_time = time.time()

    # Import here to avoid loading models at module load time
    try:
        import whisperx
        import torch
    except ImportError:
        print("Error: whisperx not installed")
        print("This script requires the full Habla environment")
        sys.exit(1)

    # Load model with specified config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if device == "cuda" else "float32"

    try:
        model = whisperx.load_model(
            config.whisper_model,
            device=device,
            compute_type=compute_type,
            language="es",  # Adjust if needed
            vad_options={
                "vad_onset": config.vad_onset,
                "vad_offset": config.vad_onset
            }
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return "", 0.0

    # Transcribe
    try:
        result = model.transcribe(
            str(audio_path),
            beam_size=config.beam_size,
            language="es"
        )
        transcript = result.get("text", "").strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        transcript = ""

    processing_time = time.time() - start_time

    return transcript, processing_time


async def run_wer_test(
    audio_samples: Dict[str, Dict],
    configs: List[TestConfig]
) -> List[WERResult]:
    """Run WER tests for all samples and configs."""

    results = []

    total_tests = len(audio_samples) * len(configs)
    current_test = 0

    for sample_name, sample_data in audio_samples.items():
        audio_path = Path(sample_data["audio"])
        reference = sample_data["transcript"]
        duration = sample_data.get("duration", 0)

        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        print(f"\nTesting: {sample_name}")
        print(f"  Reference: \"{reference}\"")

        for config in configs:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] Config: {config.name}...", end=" ", flush=True)

            # Transcribe with this config
            hypothesis, proc_time = await transcribe_with_config(audio_path, config)

            if not hypothesis:
                print("FAILED")
                continue

            # Calculate WER
            wer_score = calculate_wer(reference, hypothesis)

            result = WERResult(
                config_name=config.name,
                audio_file=sample_name,
                reference=reference,
                hypothesis=hypothesis,
                wer=wer_score,
                duration_seconds=duration,
                processing_time_seconds=proc_time
            )

            results.append(result)

            # Print result
            match_str = "✓" if wer_score < 0.05 else "✗"
            print(f"WER: {wer_score:.1%} {match_str}")
            if wer_score > 0:
                print(f"    Hypothesis: \"{hypothesis}\"")

    return results


def print_summary(results: List[WERResult]):
    """Print summary of WER results."""

    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 80)
    print("  WER COMPARISON SUMMARY")
    print("=" * 80)

    # Group by config
    by_config: Dict[str, List[WERResult]] = {}
    for result in results:
        if result.config_name not in by_config:
            by_config[result.config_name] = []
        by_config[result.config_name].append(result)

    # Calculate averages per config
    print("\nAverage WER by Configuration:")
    print("-" * 80)

    config_averages = []
    for config_name, config_results in by_config.items():
        avg_wer = sum(r.wer for r in config_results) / len(config_results)
        avg_time = sum(r.processing_time_seconds for r in config_results) / len(config_results)

        config_averages.append((config_name, avg_wer, avg_time, len(config_results)))

    # Sort by WER (best first)
    config_averages.sort(key=lambda x: x[1])

    for i, (name, avg_wer, avg_time, count) in enumerate(config_averages, 1):
        rank_str = "🏆" if i == 1 else f"{i}."
        print(f"  {rank_str} {name}")
        print(f"      Avg WER: {avg_wer:.2%} ({count} samples)")
        print(f"      Avg processing time: {avg_time:.1f}s")

    # Best config
    best_config, best_wer, _, _ = config_averages[0]
    print(f"\n✓ Best configuration: {best_config} (WER: {best_wer:.2%})")

    # Show improvement over baseline if exists
    baseline_results = [ca for ca in config_averages if "baseline" in ca[0].lower()]
    if baseline_results and best_config.lower() != "baseline":
        baseline_wer = baseline_results[0][1]
        improvement = (baseline_wer - best_wer) / baseline_wer
        print(f"  Improvement over baseline: {improvement:.1%}")

    # Detailed results
    print("\n" + "=" * 80)
    print("  DETAILED RESULTS")
    print("=" * 80)

    for result in results:
        status = "✓ PASS" if result.wer < 0.05 else f"✗ WER: {result.wer:.1%}"
        print(f"\n{result.audio_file} [{result.config_name}]: {status}")
        print(f"  Reference:  \"{result.reference}\"")
        if result.wer > 0:
            print(f"  Hypothesis: \"{result.hypothesis}\"")


def load_audio_samples(samples_dir: Path) -> Dict[str, Dict]:
    """Load audio samples with ground truth transcripts."""

    samples = {}

    # Look for a ground_truth.json file
    gt_file = samples_dir / "ground_truth.json"

    if gt_file.exists():
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Otherwise, look for WAV files and try to infer from metadata
    print(f"Warning: No ground_truth.json found in {samples_dir}")
    print("Please create ground_truth.json with format:")
    print(json.dumps({
        "sample1": {
            "audio": "tests/benchmark/audio_samples/sample1.wav",
            "transcript": "Hola, ¿cómo estás?",
            "duration": 2.5
        }
    }, indent=2))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Compare WER across configurations")
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("tests/benchmark/audio_samples"),
        help="Directory with audio samples and ground_truth.json"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="JSON file with test configurations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Load audio samples
    if not args.samples.exists():
        print(f"Error: Samples directory not found: {args.samples}")
        print("\nCreate test samples:")
        print("  1. Enable audio recording: export SAVE_AUDIO_RECORDINGS=1")
        print("  2. Use Habla to record audio")
        print("  3. Copy segments to tests/benchmark/audio_samples/")
        print("  4. Create ground_truth.json with correct transcripts")
        sys.exit(1)

    audio_samples = load_audio_samples(args.samples)

    if not audio_samples:
        print("Error: No audio samples loaded")
        sys.exit(1)

    print(f"Loaded {len(audio_samples)} audio samples")

    # Load or create test configurations
    if args.config_file and args.config_file.exists():
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            configs = [TestConfig(**cfg) for cfg in config_data]
    else:
        # Default test configurations
        configs = [
            TestConfig(
                name="baseline",
                speech_threshold=0.35,
                pre_speech_padding_ms=300,
                whisper_model="small",
                beam_size=3
            ),
            TestConfig(
                name="low_threshold",
                speech_threshold=0.25,
                pre_speech_padding_ms=300,
                whisper_model="small",
                beam_size=3
            ),
            TestConfig(
                name="high_padding",
                speech_threshold=0.35,
                pre_speech_padding_ms=500,
                whisper_model="small",
                beam_size=3
            ),
            TestConfig(
                name="high_beam",
                speech_threshold=0.35,
                pre_speech_padding_ms=300,
                whisper_model="small",
                beam_size=5
            ),
            TestConfig(
                name="medium_model",
                speech_threshold=0.35,
                pre_speech_padding_ms=300,
                whisper_model="medium",
                beam_size=3
            ),
            TestConfig(
                name="optimized",
                speech_threshold=0.28,
                pre_speech_padding_ms=450,
                whisper_model="small",
                beam_size=5
            ),
        ]

    print(f"Testing {len(configs)} configurations\n")

    # Run tests
    results = asyncio.run(run_wer_test(audio_samples, configs))

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples_count": len(audio_samples),
            "configs_count": len(configs),
            "results": [asdict(r) for r in results]
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
