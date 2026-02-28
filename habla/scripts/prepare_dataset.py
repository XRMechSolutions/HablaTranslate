"""Assemble corrected ground truth into a HuggingFace DatasetDict for LoRA fine-tuning.

Walks recording session directories, reads corrected_ground_truth.json (preferred)
or ground_truth.json (fallback), pairs each segment with its WAV file, and outputs
an 80/10/10 train/val/test split as a HuggingFace DatasetDict.

Usage (from habla/ directory):
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --input data/audio/recordings --output data/whisper_dataset
    python scripts/prepare_dataset.py --min-confidence 0.5 --max-duration 20.0
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace dataset from corrected ground truth for Whisper fine-tuning"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/audio/recordings"),
        help="Root directory containing recording session subdirectories (default: data/audio/recordings)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/whisper_dataset"),
        help="Output directory for the HuggingFace DatasetDict (default: data/whisper_dataset)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.5,
        help="Skip segments shorter than this (seconds, default: 0.5)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Skip segments longer than this (seconds, default: 30.0)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Skip segments with confidence below this (default: 0.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    return parser.parse_args()


def wav_duration_seconds(wav_path: Path) -> float:
    """Calculate duration from WAV file size (16-bit mono 16kHz = 32000 bytes/sec)."""
    size = wav_path.stat().st_size
    return max(0, (size - 44)) / 32000.0


def collect_samples(input_dir: Path, min_duration: float, max_duration: float, min_confidence: float):
    """Walk recording dirs and collect (wav_path, transcript) pairs."""
    samples = []
    stats = {"sessions_scanned": 0, "sessions_with_gt": 0, "skipped_empty": 0,
             "skipped_inaudible": 0, "skipped_missing_wav": 0,
             "skipped_duration": 0, "skipped_confidence": 0}

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    for session_dir in sorted(input_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        stats["sessions_scanned"] += 1

        # Prefer corrected over original
        gt_path = session_dir / "corrected_ground_truth.json"
        if not gt_path.exists():
            gt_path = session_dir / "ground_truth.json"
        if not gt_path.exists():
            continue

        stats["sessions_with_gt"] += 1

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read {gt_path}: {e}")
            continue

        for seg in data.get("segments", []):
            transcript = seg.get("transcript", "").strip()

            # Filter empty transcripts
            if not transcript:
                stats["skipped_empty"] += 1
                continue

            # Filter [inaudible] and similar markers
            if transcript.lower() in ("[inaudible]", "[unintelligible]", "[silence]", "..."):
                stats["skipped_inaudible"] += 1
                continue

            # Check WAV exists
            wav_path = session_dir / seg.get("filename", "")
            if not wav_path.exists():
                stats["skipped_missing_wav"] += 1
                continue

            # Duration filter
            duration = wav_duration_seconds(wav_path)
            if duration < min_duration or duration > max_duration:
                stats["skipped_duration"] += 1
                continue

            # Confidence filter
            confidence = seg.get("confidence", 1.0)
            if confidence < min_confidence:
                stats["skipped_confidence"] += 1
                continue

            samples.append({
                "audio_path": str(wav_path.resolve()),
                "sentence": transcript,
                "duration": round(duration, 2),
            })

    return samples, stats


def build_dataset(samples: list[dict], output_dir: Path, seed: int):
    """Build and save a HuggingFace DatasetDict with 80/10/10 split."""
    try:
        from datasets import Dataset, DatasetDict, Audio
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install it with: pip install datasets soundfile")
        sys.exit(1)

    # Create Dataset from samples
    ds = Dataset.from_dict({
        "audio": [s["audio_path"] for s in samples],
        "sentence": [s["sentence"] for s in samples],
    })

    # Cast audio column to Audio feature (16kHz)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Split: 80% train, 10% val, 10% test
    # First split off 20% for val+test
    split1 = ds.train_test_split(test_size=0.2, seed=seed)
    # Then split the 20% into 50/50 = 10%/10% of total
    split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)

    dataset_dict = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    return dataset_dict


def main():
    args = parse_args()

    print(f"Scanning recordings in: {args.input}")
    print(f"Filters: duration [{args.min_duration}s, {args.max_duration}s], confidence >= {args.min_confidence}")
    print()

    samples, stats = collect_samples(
        args.input, args.min_duration, args.max_duration, args.min_confidence,
    )

    print(f"Sessions scanned:    {stats['sessions_scanned']}")
    print(f"Sessions with GT:    {stats['sessions_with_gt']}")
    print(f"Valid segments:      {len(samples)}")
    print(f"Skipped (empty):     {stats['skipped_empty']}")
    print(f"Skipped (inaudible): {stats['skipped_inaudible']}")
    print(f"Skipped (no WAV):    {stats['skipped_missing_wav']}")
    print(f"Skipped (duration):  {stats['skipped_duration']}")
    print(f"Skipped (confidence):{stats['skipped_confidence']}")
    print()

    if not samples:
        print("No valid samples found. Nothing to build.")
        sys.exit(0)

    total_duration = sum(s["duration"] for s in samples)
    print(f"Total audio: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print()

    print(f"Building HuggingFace DatasetDict...")
    dataset_dict = build_dataset(samples, args.output, args.seed)

    print(f"Saved to: {args.output}")
    print(f"  train:      {len(dataset_dict['train'])} samples")
    print(f"  validation: {len(dataset_dict['validation'])} samples")
    print(f"  test:       {len(dataset_dict['test'])} samples")
    print()
    print(f"Columns: {list(dataset_dict['train'].column_names)}")
    print()
    print("Load with: datasets.load_from_disk('" + str(args.output) + "')")


if __name__ == "__main__":
    main()
