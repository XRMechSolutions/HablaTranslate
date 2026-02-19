"""Benchmark tests for translation pipeline performance.

These tests measure actual performance with real models/services.
Run with: pytest tests/benchmark/ -v -s

NOTE: Requires running services (Ollama, LM Studio, etc.)
"""

import pytest
import time
import asyncio
import json
from datetime import datetime
from pathlib import Path

from server.config import TranslatorConfig
from server.pipeline.translator import Translator
from server.services.idiom_scanner import IdiomScanner, create_starter_idioms
from server.services.speaker_tracker import SpeakerTracker


# Benchmark configuration
BENCHMARK_RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARK_RESULTS_DIR.mkdir(exist_ok=True)

# Test sentences (various complexity levels)
TEST_SENTENCES = {
    "simple": "Hola, ¿cómo estás?",
    "medium": "No me importa un pepino lo que pienses de mí.",
    "complex": "Está en las nubes y no tiene pelos en la lengua cuando habla.",
    "long": "El profesor nos dijo que deberíamos ponernos las pilas y estudiar más porque el examen final será muy difícil y no quiere que metamos la pata como la última vez.",
}


class BenchmarkResult:
    """Store and format benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.measurements = []

    def add(self, operation: str, duration_ms: float, metadata: dict = None):
        self.measurements.append({
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "metadata": metadata or {}
        })

    def save(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = BENCHMARK_RESULTS_DIR / f"{self.name}_{timestamp}.json"

        summary = {
            "benchmark": self.name,
            "timestamp": timestamp,
            "measurements": self.measurements,
            "summary": {
                "total_duration_ms": sum(m["duration_ms"] for m in self.measurements),
                "avg_duration_ms": sum(m["duration_ms"] for m in self.measurements) / len(self.measurements) if self.measurements else 0,
                "min_duration_ms": min((m["duration_ms"] for m in self.measurements), default=0),
                "max_duration_ms": max((m["duration_ms"] for m in self.measurements), default=0),
            }
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Results saved to: {filename}")
        return summary

    def print_summary(self):
        """Print formatted summary to console."""
        print(f"\n{'='*60}")
        print(f"  Benchmark: {self.name}")
        print(f"{'='*60}")

        for m in self.measurements:
            print(f"  {m['operation']:<40} {m['duration_ms']:>8.2f}ms")
            if m['metadata']:
                for k, v in m['metadata'].items():
                    print(f"    {k}: {v}")

        if self.measurements:
            total = sum(m["duration_ms"] for m in self.measurements)
            avg = total / len(self.measurements)
            print(f"\n  {'Total:':<40} {total:>8.2f}ms")
            print(f"  {'Average:':<40} {avg:>8.2f}ms")
        print(f"{'='*60}\n")


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_idiom_scanner_performance():
    """Benchmark idiom pattern matching speed."""
    benchmark = BenchmarkResult("idiom_scanner")

    # Load patterns
    scanner = IdiomScanner()
    start = time.perf_counter()
    scanner.load_from_db(create_starter_idioms())
    load_time = (time.perf_counter() - start) * 1000
    benchmark.add("Load 25 patterns", load_time, {"pattern_count": scanner.count})

    # Scan different complexity texts
    for name, text in TEST_SENTENCES.items():
        start = time.perf_counter()
        matches = scanner.scan(text)
        scan_time = (time.perf_counter() - start) * 1000
        benchmark.add(f"Scan {name} text", scan_time, {
            "text_length": len(text),
            "matches_found": len(matches),
            "text_preview": text[:50] + "..." if len(text) > 50 else text
        })

    # Multiple scans (simulate real usage)
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        scanner.scan(TEST_SENTENCES["medium"])
    batch_time = (time.perf_counter() - start) * 1000
    avg_scan = batch_time / iterations
    benchmark.add(f"{iterations} scans (medium text)", batch_time, {
        "avg_per_scan_ms": round(avg_scan, 3)
    })

    benchmark.print_summary()
    summary = benchmark.save()

    # Assert performance targets
    assert load_time < 100, f"Pattern loading too slow: {load_time}ms > 100ms"
    assert avg_scan < 1.0, f"Average scan too slow: {avg_scan}ms > 1ms"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_translator_ollama_performance():
    """Benchmark Ollama translation performance (if available)."""
    config = TranslatorConfig(
        provider="ollama",
        ollama_url="http://localhost:11434",
        ollama_model="qwen3:4b",
    )

    translator = Translator(config)
    benchmark = BenchmarkResult("translator_ollama")

    try:
        # Test translations at different complexity levels
        for name, text in TEST_SENTENCES.items():
            start = time.perf_counter()
            result = await translator.translate(
                transcript=text,
                speaker_label="Speaker A",
                direction="es_to_en",
                mode="conversation",
                context_exchanges=[],
            )
            duration = (time.perf_counter() - start) * 1000

            benchmark.add(f"Translate {name} ({len(text)} chars)", duration, {
                "confidence": result.confidence,
                "flagged_phrases": len(result.flagged_phrases),
                "translation_preview": result.translated[:60] + "..." if len(result.translated) > 60 else result.translated
            })

        # Test with context
        context = [
            {"speaker": "Speaker A", "source": "Hola", "translation": "Hello"},
            {"speaker": "Speaker B", "source": "¿Qué tal?", "translation": "How are you?"},
        ]

        start = time.perf_counter()
        result = await translator.translate(
            transcript=TEST_SENTENCES["medium"],
            speaker_label="Speaker A",
            direction="es_to_en",
            mode="conversation",
            context_exchanges=context,
            topic_summary="Casual conversation"
        )
        duration = (time.perf_counter() - start) * 1000
        benchmark.add("Translate with context", duration, {
            "context_items": len(context),
            "has_topic_summary": True
        })

        benchmark.print_summary()
        summary = benchmark.save()

        # Performance targets (adjust based on your hardware)
        avg_time = summary["summary"]["avg_duration_ms"]
        assert avg_time < 5000, f"Average translation too slow: {avg_time}ms > 5000ms"

    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")
    finally:
        await translator.close()


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_translator_lmstudio_model_comparison():
    """Benchmark multiple LM Studio models and compare performance.

    This test:
    1. Discovers available LM Studio models
    2. Runs the same translations through each model
    3. Compares speed, quality (confidence), and idiom detection
    4. Generates a comparison report
    """
    config = TranslatorConfig(
        provider="lmstudio",
        lmstudio_url="http://localhost:1234",
    )

    translator = Translator(config)

    try:
        # Discover available models
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.lmstudio_url}/v1/models")
            response.raise_for_status()
            models_data = response.json()
            available_models = [m["id"] for m in models_data.get("data", [])]

        if not available_models:
            pytest.skip("No LM Studio models available")

        print(f"\n🔍 Found {len(available_models)} LM Studio models:")
        for model in available_models:
            print(f"  - {model}")

        # Benchmark results for each model
        all_results = []

        for model_name in available_models:
            print(f"\n🧪 Testing model: {model_name}")

            # Switch to this model
            translator.switch_provider("lmstudio", model=model_name)

            benchmark = BenchmarkResult(f"lmstudio_{model_name.replace('/', '_')}")
            model_results = {
                "model": model_name,
                "translations": []
            }

            # Test each sentence
            for name, text in TEST_SENTENCES.items():
                start = time.perf_counter()
                try:
                    result = await translator.translate(
                        transcript=text,
                        speaker_label="Speaker A",
                        direction="es_to_en",
                        mode="conversation",
                        context_exchanges=[],
                    )
                    duration = (time.perf_counter() - start) * 1000

                    benchmark.add(f"Translate {name}", duration, {
                        "text_length": len(text),
                        "confidence": result.confidence,
                        "idioms_found": len(result.flagged_phrases),
                    })

                    model_results["translations"].append({
                        "test": name,
                        "duration_ms": round(duration, 2),
                        "confidence": result.confidence,
                        "idioms_found": len(result.flagged_phrases),
                        "translation": result.translated
                    })

                except Exception as e:
                    print(f"  ❌ Failed on {name}: {e}")
                    model_results["translations"].append({
                        "test": name,
                        "error": str(e)
                    })

            summary = benchmark.save()
            benchmark.print_summary()

            model_results["summary"] = summary["summary"]
            all_results.append(model_results)

        # Generate comparison report
        _generate_model_comparison_report(all_results)

    except Exception as e:
        pytest.skip(f"LM Studio not available: {e}")
    finally:
        await translator.close()


def _generate_model_comparison_report(results):
    """Generate a comparison report for multiple models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = BENCHMARK_RESULTS_DIR / f"model_comparison_{timestamp}.json"

    # Calculate rankings
    for result in results:
        if "summary" in result:
            result["rank_speed"] = 0  # Will calculate after
            result["rank_confidence"] = 0

    # Sort by speed
    speed_sorted = sorted(
        [r for r in results if "summary" in r],
        key=lambda x: x["summary"]["avg_duration_ms"]
    )
    for i, result in enumerate(speed_sorted):
        result["rank_speed"] = i + 1

    # Sort by average confidence
    for result in results:
        if "translations" in result:
            confidences = [t.get("confidence", 0) for t in result["translations"] if "confidence" in t]
            result["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0

    confidence_sorted = sorted(
        [r for r in results if "avg_confidence" in r],
        key=lambda x: x["avg_confidence"],
        reverse=True
    )
    for i, result in enumerate(confidence_sorted):
        result["rank_confidence"] = i + 1

    # Save report
    report = {
        "timestamp": timestamp,
        "models_tested": len(results),
        "results": results,
        "recommendations": _generate_recommendations(results)
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print comparison table
    print(f"\n{'='*80}")
    print("  MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"  {'Model':<35} {'Avg Time':<12} {'Confidence':<12} {'Speed Rank':<12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")

    for result in results:
        if "summary" in result:
            model = result["model"][:33]
            avg_time = f"{result['summary']['avg_duration_ms']:.1f}ms"
            confidence = f"{result.get('avg_confidence', 0):.2f}"
            rank = f"#{result.get('rank_speed', '?')}"
            print(f"  {model:<35} {avg_time:<12} {confidence:<12} {rank:<12}")

    print(f"{'='*80}")
    print(f"\n📊 Full report saved to: {report_file}\n")

    # Print recommendations
    recs = report["recommendations"]
    if recs:
        print("💡 RECOMMENDATIONS:")
        for rec in recs:
            print(f"  {rec}")
        print()


def _generate_recommendations(results):
    """Generate recommendations based on benchmark results."""
    recs = []

    if not results:
        return recs

    # Find fastest model
    valid_results = [r for r in results if "summary" in r]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x["summary"]["avg_duration_ms"])
        recs.append(f"🚀 Fastest: {fastest['model']} ({fastest['summary']['avg_duration_ms']:.1f}ms avg)")

        # Find most accurate
        if any("avg_confidence" in r for r in valid_results):
            most_accurate = max(valid_results, key=lambda x: x.get("avg_confidence", 0))
            recs.append(f"🎯 Highest confidence: {most_accurate['model']} ({most_accurate.get('avg_confidence', 0):.2f})")

        # Find best balance
        for result in valid_results:
            result["balance_score"] = (
                (1000 / result["summary"]["avg_duration_ms"]) *
                result.get("avg_confidence", 0.5)
            )

        best_balance = max(valid_results, key=lambda x: x["balance_score"])
        recs.append(f"⚖️  Best balance: {best_balance['model']} (speed × confidence)")

        # Performance warning
        slowest = max(valid_results, key=lambda x: x["summary"]["avg_duration_ms"])
        if slowest["summary"]["avg_duration_ms"] > 2000:
            recs.append(f"⚠️  Warning: {slowest['model']} is slow (>{slowest['summary']['avg_duration_ms']:.0f}ms) - may not be suitable for real-time")

    return recs


@pytest.mark.benchmark
def test_speaker_tracker_performance():
    """Benchmark speaker tracking operations."""
    benchmark = BenchmarkResult("speaker_tracker")

    tracker = SpeakerTracker()

    # Benchmark speaker creation
    iterations = 100
    start = time.perf_counter()
    for i in range(iterations):
        tracker.get_or_create(f"SPEAKER_{i:02d}")
    create_time = (time.perf_counter() - start) * 1000
    benchmark.add(f"Create {iterations} speakers", create_time, {
        "avg_per_speaker_ms": round(create_time / iterations, 4)
    })

    # Benchmark utterance recording
    start = time.perf_counter()
    for i in range(1000):
        tracker.record_utterance(f"SPEAKER_{i % 10:02d}")
    record_time = (time.perf_counter() - start) * 1000
    benchmark.add("Record 1000 utterances", record_time)

    # Benchmark summary generation
    start = time.perf_counter()
    for _ in range(100):
        summary = tracker.get_speaker_list_summary()
    summary_time = (time.perf_counter() - start) * 1000
    benchmark.add("Generate 100 summaries", summary_time, {
        "summary_length": len(summary)
    })

    benchmark.print_summary()
    benchmark.save()

    # Assert performance
    assert create_time / iterations < 1.0, "Speaker creation too slow"


if __name__ == "__main__":
    # Run benchmarks directly
    print("Running benchmark suite...")
    pytest.main([__file__, "-v", "-s", "-m", "benchmark"])
