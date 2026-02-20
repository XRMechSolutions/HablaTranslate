# Whisper Fine-Tuning Plan for Andalusian Spanish

## Overview

This document outlines a plan to fine-tune the WhisperX ASR model specifically for Andalusian Spanish accent and quiet speech patterns. Fine-tuning can improve Word Error Rate (WER) by 30-50% for domain-specific use cases like classroom Spanish with regional accents.

## Why Fine-Tune for Andalusian Spanish?

### Unique Challenges

Andalusian Spanish has distinct phonetic characteristics that differ from the Spanish data Whisper was primarily trained on (likely Mexican/Latin American Spanish):

1. **Consonant dropping**: Final /s/ often dropped or aspirated ("los niños" → "loh niñoh")
2. **Seseo**: /θ/ (theta) pronounced as /s/ ("gracias" → "grasias" not "graθias")
3. **Yeísmo**: /ʎ/ merged with /ʝ/ ("pollo" sounds like "poyo")
4. **Vowel merging**: Adjacent vowels often merge ("para ahora" → "paora")
5. **Fast speech rate**: Syllable reduction in rapid conversation
6. **Regional vocabulary**: Colloquialisms and idioms specific to Andalusia

### Expected Improvements

Based on research ([Fine-tuning Whisper on Low-Resource Languages](https://arxiv.org/html/2412.15726v1)):

| Metric | Base Whisper Small | After Fine-Tuning | Improvement |
|--------|-------------------|-------------------|-------------|
| **WER (Quiet speech)** | 25-35% | 10-20% | ~40-60% reduction |
| **WER (Normal speech)** | 10-15% | 5-8% | ~30-50% reduction |
| **Accent-specific errors** | High | Low | 70%+ reduction |
| **Domain vocabulary** | Poor | Excellent | Near-perfect |

**ROI**: With 20+ hours of corrected transcripts, you can achieve near-native accuracy for your specific classroom/accent.

---

## Prerequisites

### Hardware Requirements

✅ **Your RTX 3060 12GB is sufficient** for fine-tuning Whisper Small/Medium:

| Model | VRAM (Training) | Training Time (20h dataset) | Inference VRAM |
|-------|----------------|----------------------------|----------------|
| **Small** | 8-10GB | ~12 hours | ~1GB |
| **Medium** | 10-12GB (tight) | ~24 hours | ~2.5GB |

**Recommendation**: Start with Small model fine-tuning. Your 12GB VRAM is comfortable for Small, tight for Medium.

### Software Requirements

```bash
# Install Hugging Face Transformers + dependencies
pip install transformers datasets evaluate jiwer accelerate

# Optional: Parameter-efficient fine-tuning (LoRA) to reduce VRAM
pip install peft bitsandbytes
```

### Data Requirements

**Minimum viable dataset**:
- **2-5 hours**: Noticeable improvement (15-20% WER reduction)
- **8-20 hours**: Significant improvement (30-40% WER reduction)
- **50+ hours**: Maximum benefit (50%+ WER reduction)

**Quality matters more than quantity**: 10 hours of carefully corrected transcripts beats 50 hours of low-quality data.

---

## Data Collection Strategy

### Phase 1: Opportunistic Recording (Weeks 1-4)

Use Habla's existing audio recording feature to capture classroom Spanish:

```bash
# Enable recording
export SAVE_AUDIO_RECORDINGS=1
cd habla
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

**Target scenarios**:
1. **Classroom lectures** (teacher speaking, longer monologues)
2. **Student Q&A** (varied speakers, shorter utterances)
3. **Conversational practice** (natural back-and-forth)
4. **Quiet speakers** (your primary pain point)

**Collection goals**:
- Week 1-2: 2-3 hours baseline recordings
- Week 3-4: 5-8 hours diverse scenarios
- Focus: Andalusian accent speakers, varied volumes

### Phase 2: Targeted Recording (Weeks 5-8)

Once you identify common error patterns, record specific content:

1. **Problematic phonemes**: Words with /s/ dropping, seseo, etc.
2. **Low-confidence segments**: Re-record utterances where ASR confidence <0.8
3. **Domain vocabulary**: Classroom terms, Spanish grammar terminology
4. **Edge cases**: Whispered speech, fast speech, background noise

**Collection goals**:
- Week 5-6: +3-5 hours targeted content
- Week 7-8: +3-5 hours edge cases
- **Total**: 10-20 hours diverse, high-quality recordings

### Phase 3: Data Annotation (Weeks 9-12)

This is the **most time-consuming** part. Budget ~20 hours of human effort for 20 hours of audio.

#### Workflow

```bash
# 1. Extract all recorded segments
ls data/audio/recordings/*/segment_*.wav

# 2. Review Habla's auto-transcripts
cat data/audio/recordings/*/metadata.json | jq '.segments[] | select(.confidence < 0.9)'

# 3. Manually correct transcripts (see tools below)
```

#### Annotation Tools

**Option A: Label Studio** (Recommended)
- Web-based audio annotation UI
- Waveform visualization + playback
- Export to JSON/CSV
- Free and open-source

```bash
pip install label-studio
label-studio start

# Import Habla's audio segments + metadata
# Correct transcripts in UI
# Export corrected dataset
```

**Option B: Extend Habla's Vocab Review Page**

Create a transcript correction interface in Habla:

```
habla/client/corrections.html:

┌─────────────────────────────────────────┐
│ Audio: [▶] segment_042.wav             │
│                                         │
│ Auto transcript (78% confidence):       │
│ "Lo niño etán jugando en el patio"     │
│                                         │
│ Corrected transcript:                   │
│ [Los niños están jugando en el patio]  │
│                                         │
│ [Skip] [Save] [Next]                    │
└─────────────────────────────────────────┘
```

**Pros**: Integrated into Habla, no external tools
**Cons**: Requires building the UI (1-2 days dev time)

#### Annotation Best Practices

1. **Verbatim transcription**: Write exactly what was said, including:
   - Hesitations: "um", "eh", "este"
   - False starts: "Yo pien- yo creo que..."
   - Regional pronunciations: Transcribe what you *hear*, not standard spelling

2. **Punctuation**: Add periods, commas, question marks (helps model learn prosody)

3. **Speaker labels**: If multiple speakers, note who's speaking (optional, but helpful)

4. **Quality over speed**: Better to have 5 perfect hours than 20 sloppy hours

5. **Consistency**: Use same conventions throughout (e.g., always "¿Qué?" not "Que?")

#### Output Format

Create a CSV: `fine_tune_dataset.csv`

```csv
audio_path,transcript,duration,speaker,notes
data/audio/recordings/12345_20260216_143022/segment_001.wav,"Hola, ¿cómo estás?",2.1,teacher,clear
data/audio/recordings/12345_20260216_143022/segment_002.wav,"Los niños están jugando en el patio",3.5,student_a,quiet
data/audio/recordings/12346_20260216_143530/segment_001.wav,"No me importa un pepino",2.8,teacher,idiom
```

Or JSON: `fine_tune_dataset.json`

```json
[
  {
    "audio": "data/audio/recordings/.../segment_001.wav",
    "text": "Hola, ¿cómo estás?",
    "duration": 2.1,
    "speaker": "teacher"
  },
  {
    "audio": "data/audio/recordings/.../segment_002.wav",
    "text": "Los niños están jugando en el patio",
    "duration": 3.5,
    "speaker": "student_a"
  }
]
```

---

## Fine-Tuning Process

### Step 1: Prepare Dataset

Convert Habla recordings to Hugging Face Dataset format:

```python
# scripts/prepare_dataset.py

import json
import pandas as pd
from datasets import Dataset, Audio

# Load your corrected transcripts
df = pd.read_csv("fine_tune_dataset.csv")

# Create Hugging Face dataset
dataset = Dataset.from_pandas(df[["audio_path", "transcript"]])
dataset = dataset.rename_column("audio_path", "audio")
dataset = dataset.rename_column("transcript", "sentence")

# Cast audio column to Audio type (auto-loads WAV files)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split train/validation/test (80/10/10)
train_test = dataset.train_test_split(test_size=0.2, seed=42)
test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    "train": train_test["train"],       # 80%
    "validation": test_val["train"],    # 10%
    "test": test_val["test"]            # 10%
}

# Save to disk
from datasets import DatasetDict
DatasetDict(final_dataset).save_to_disk("andalusian_spanish_dataset")

print(f"Train: {len(final_dataset['train'])} samples")
print(f"Val: {len(final_dataset['validation'])} samples")
print(f"Test: {len(final_dataset['test'])} samples")
```

### Step 2: Fine-Tune Whisper Small

```python
# scripts/fine_tune_whisper.py

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk
import torch

# Load base model
model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Load your dataset
dataset = load_from_disk("andalusian_spanish_dataset")

# Preprocessing function
def prepare_dataset(batch):
    audio = batch["audio"]
    # Compute log-Mel spectrogram
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # Tokenize transcript
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Process dataset
dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])

# Training arguments (conservative for RTX 3060 12GB)
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-andalusian",
    per_device_train_batch_size=4,        # Adjust based on VRAM
    gradient_accumulation_steps=4,        # Effective batch size = 16
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,                       # ~10-12 hours for 20h dataset
    gradient_checkpointing=True,          # Save VRAM
    fp16=True,                            # Use mixed precision
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Metric: Word Error Rate (WER)
import evaluate
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 (padding) with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# Start fine-tuning
print("Starting fine-tuning... (this will take ~12 hours on RTX 3060)")
trainer.train()

# Save final model
trainer.save_model("./whisper-small-andalusian-final")
processor.save_pretrained("./whisper-small-andalusian-final")

print("Fine-tuning complete!")
```

### Step 3: Parameter-Efficient Fine-Tuning (LoRA) - Optional

If you encounter VRAM issues, use LoRA to reduce memory footprint by 50%:

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=8,                    # Low-rank dimension
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none",
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: "trainable params: 2.3M || all params: 244M || trainable%: 0.94"

# Now only ~6GB VRAM instead of ~10GB
# Train as normal with Seq2SeqTrainer
```

**Trade-off**: LoRA trains only 1% of parameters (faster, less VRAM) but may give slightly lower accuracy than full fine-tuning. For domain adaptation, LoRA is usually sufficient.

---

## Evaluation & Testing

### Step 1: Baseline WER (Before Fine-Tuning)

```python
# scripts/evaluate_baseline.py

from transformers import pipeline
from datasets import load_from_disk
import evaluate

# Load base Whisper Small
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0  # GPU
)

# Load test set
test_dataset = load_from_disk("andalusian_spanish_dataset")["test"]

# Transcribe all test samples
predictions = []
references = []

for sample in test_dataset:
    audio = sample["audio"]["array"]
    result = asr_pipeline(audio)
    predictions.append(result["text"])
    references.append(sample["sentence"])

# Calculate WER
wer_metric = evaluate.load("wer")
baseline_wer = wer_metric.compute(predictions=predictions, references=references)

print(f"Baseline WER (Whisper Small): {baseline_wer:.2%}")
# Expected: 20-30% for quiet Andalusian Spanish
```

### Step 2: Fine-Tuned WER

```python
# scripts/evaluate_finetuned.py

# Same as above, but load your fine-tuned model
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="./whisper-small-andalusian-final",
    device=0
)

# Run same test set
# ...

finetuned_wer = wer_metric.compute(predictions=predictions, references=references)

print(f"Fine-tuned WER: {finetuned_wer:.2%}")
# Expected: 8-15% (30-50% reduction from baseline)

print(f"WER improvement: {(baseline_wer - finetuned_wer) / baseline_wer:.1%}")
```

### Step 3: Qualitative Analysis

Review specific error types:

```python
# Show examples where fine-tuning helped
for pred, ref in zip(predictions[:10], references[:10]):
    print(f"Reference: {ref}")
    print(f"Predicted: {pred}")
    print(f"Match: {'✓' if pred == ref else '✗'}")
    print()
```

**Look for**:
- Andalusian phonetic features correctly recognized
- Domain vocabulary (classroom terms) accurate
- Quiet speech segments no longer dropped
- Fewer hallucinations in silence

---

## Deployment in Habla

Once fine-tuned, integrate your custom model:

### Option 1: Replace Base Model (Simple)

```python
# In habla/server/pipeline/orchestrator.py, line 100:

self._whisperx_model = whisperx.load_model(
    "/path/to/whisper-small-andalusian-final",  # Your fine-tuned model
    device=self.config.asr.device,
    compute_type=self.config.asr.compute_type,
    language="es",
)
```

### Option 2: Configurable via Environment Variable

```python
# In habla/server/config.py, add:

class ASRConfig(BaseModel):
    model_size: str = "small"
    model_path: str = ""  # NEW: path to custom model

# In orchestrator.py:
model_path = self.config.asr.model_path or self.config.asr.model_size
self._whisperx_model = whisperx.load_model(
    model_path,
    device=self.config.asr.device,
    compute_type=self.config.asr.compute_type,
)
```

```bash
# Use custom model:
export WHISPER_MODEL_PATH=/path/to/whisper-small-andalusian-final
uvicorn server.main:app --host 0.0.0.0 --port 8002

# Fallback to base model:
export WHISPER_MODEL=small
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

### Option 3: A/B Testing (Advanced)

Load both models and compare side-by-side:

```python
# Load base + fine-tuned
self._whisperx_base = whisperx.load_model("small", ...)
self._whisperx_finetuned = whisperx.load_model("/path/to/finetuned", ...)

# Transcribe with both, log WER differences
base_result = self._whisperx_base.transcribe(audio)
finetuned_result = self._whisperx_finetuned.transcribe(audio)

# Use fine-tuned for production, log comparison
```

**VRAM cost**: 2x model size (~2GB for two Small models). Only feasible if you have headroom.

---

## Cost-Benefit Analysis

### Time Investment

| Phase | Duration | Effort | Notes |
|-------|----------|--------|-------|
| **Data collection** | 4-8 weeks | Passive | Record during normal classroom use |
| **Data annotation** | 20-40 hours | Active | Most labor-intensive step |
| **Fine-tuning** | 12-24 hours | Passive | Let RTX 3060 run overnight |
| **Evaluation** | 2-4 hours | Active | Test and compare |
| **Integration** | 1-2 hours | Active | Update Habla config |
| **Total** | 2-3 months | 25-50 human hours | Most time is waiting/recording |

### Expected Improvements

| Metric | Before | After Fine-Tuning | Value |
|--------|--------|------------------|-------|
| **WER (quiet speech)** | 30% | 12% | 60% error reduction |
| **WER (normal speech)** | 12% | 6% | 50% error reduction |
| **Andalusian features** | Poor | Excellent | Near-native |
| **Domain vocabulary** | 80% accurate | 98% accurate | Classroom terms perfect |
| **User satisfaction** | "Good enough" | "Near-perfect" | Less manual correction |

### When Fine-Tuning Makes Sense

✅ **Do fine-tune if**:
1. You plan to use Habla for **12+ months** (investment pays off)
2. You have a **recurring problem** (same classroom, same accent)
3. Base model WER is **>15%** (room for improvement)
4. You can collect **20+ hours** of corrected data (sufficient dataset)
5. You're willing to **maintain the model** (re-train as needed)

❌ **Skip fine-tuning if**:
1. Base model + AGC + tuning gets you to **<10% WER** (good enough)
2. You have **<10 hours** of data (insufficient for meaningful improvement)
3. Use case is **temporary** (one semester only)
4. You don't have **time for annotation** (25+ hours of manual work)

---

## Maintenance & Iteration

### When to Re-Train

Fine-tuned models "drift" over time as:
- New speakers join classroom (different accents)
- Vocabulary evolves (new terms, slang)
- Recording conditions change (new room, equipment)

**Re-train schedule**:
- **Every 6 months**: Add new data, re-train from scratch
- **After major changes**: New classroom, new teacher, new equipment

### Continuous Improvement Loop

```
┌─────────────────────────────────────────────┐
│                                             │
│  Use Habla → Collect recordings             │
│       ↓                                     │
│  Review low-confidence transcripts          │
│       ↓                                     │
│  Correct errors (5-10/week)                 │
│       ↓                                     │
│  When dataset grows by 10+ hours            │
│       ↓                                     │
│  Re-train model (quarterly)                 │
│       ↓                                     │
│  Deploy improved model ──────────────┘      │
└─────────────────────────────────────────────┘
```

### Version Control

Track model versions:

```
models/
├── whisper-small-andalusian-v1/  (Feb 2026, 20h data, 18% WER)
├── whisper-small-andalusian-v2/  (May 2026, 35h data, 12% WER)
└── whisper-small-andalusian-v3/  (Aug 2026, 50h data, 8% WER)
```

Tag each version with:
- Training date
- Dataset size
- Test WER
- Major improvements

---

## Alternative: Transfer Learning from Similar Dialects

If collecting 20+ hours is too much effort, consider **transfer learning** from existing Andalusian datasets:

### Public Datasets (If Available)

Check Hugging Face Hub for:
- `google/fleurs` (Spanish variants)
- `mozilla-foundation/common_voice_13_0` (Spain Spanish subset)
- Academic corpora (search "Andalusian Spanish corpus")

**Workflow**:
1. Fine-tune on public Andalusian data (if exists)
2. Further fine-tune on your 5-10 hours classroom data
3. Get 80% of the benefit with 50% of the effort

### Example: Two-Stage Fine-Tuning

```python
# Stage 1: Pre-train on general Spain Spanish (5-10 hours)
# Use Common Voice Spain subset
trainer.train()
model.save_pretrained("./whisper-small-spain-spanish")

# Stage 2: Fine-tune on your Andalusian classroom data (5 hours)
model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-spain-spanish")
# Load your classroom dataset
trainer.train()
model.save_pretrained("./whisper-small-andalusian-classroom")
```

**Benefit**: Bootstraps from closer starting point (Spain Spanish > Latin American Spanish)

---

## Resources & References

### Research Papers
- [Fine-tuning Whisper on Low-Resource Languages](https://arxiv.org/abs/2412.15726) - Swiss German case study (similar dialect problem)
- [Advancing Multilingual ASR: Fine-Tuning Whisper](https://medium.com/@ccibeekeoc42/advancing-multilingual-speech-recognition-fine-tuning-whisper-for-enhanced-low-resource-34529b525f90)
- [Fine-Tuning ASR Models: Gladia Guide](https://www.gladia.io/blog/fine-tuning-asr-models)

### Tutorials & Code
- [Hugging Face Whisper Fine-Tuning Guide](https://huggingface.co/blog/fine-tune-whisper) - Official tutorial
- [whisper-finetune GitHub](https://github.com/vasistalodagala/whisper-finetune) - Ready-to-use scripts
- [LearnOpenCV: Fine-Tuning Whisper](https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/) - Step-by-step

### Tools
- [Label Studio](https://labelstud.io/) - Audio annotation UI
- [Weights & Biases](https://wandb.ai/) - Experiment tracking for fine-tuning
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Monitor training progress

### Communities
- [Hugging Face Audio Discord](https://discord.gg/hugging-face) - Ask fine-tuning questions
- [r/LanguageTechnology](https://reddit.com/r/LanguageTechnology) - ASR discussions
- [Whisper GitHub Discussions](https://github.com/openai/whisper/discussions) - Official forum

---

## Decision Framework: Should You Fine-Tune?

Use this flowchart:

```
START: Is base Whisper WER >15% on your audio?
  │
  ├─ NO → Skip fine-tuning, use parameter tuning + AGC
  │
  └─ YES → Can you collect 20+ hours of corrected data?
       │
       ├─ NO → Try AGC + Medium model first
       │       └─> Still poor? Consider 2-stage transfer learning (5-10h)
       │
       └─ YES → Will you use Habla for 12+ months?
            │
            ├─ NO → Not worth the investment
            │
            └─ YES → Do you have 25-50 hours for annotation?
                 │
                 ├─ NO → Hire annotator or use active learning tools
                 │
                 └─ YES → ✅ PROCEED WITH FINE-TUNING
```

---

## Quick Start Checklist

When you're ready to begin:

- [ ] Enable audio recording in Habla
- [ ] Use app normally for 4-8 weeks (collect 20+ hours)
- [ ] Review recordings, identify low-confidence segments
- [ ] Set up Label Studio or build correction UI
- [ ] Annotate 20+ hours (budget 25-50 hours effort)
- [ ] Prepare dataset using `prepare_dataset.py`
- [ ] Fine-tune overnight using `fine_tune_whisper.py`
- [ ] Evaluate WER improvement on test set
- [ ] If WER reduced by >30%, deploy to production
- [ ] If WER reduced by <20%, collect more data or try LoRA

---

## Summary: Three Paths Forward

### Path 1: Quick Wins (Recommended First)
**Time**: 1 day
**Effort**: Minimal
**Improvement**: 20-40% WER reduction

✅ Enable AGC in settings
✅ Upgrade to Medium model
✅ Tune VAD parameters

**Start here.** Get quick improvements with zero data collection.

---

### Path 2: Fine-Tuning (If Path 1 Insufficient)
**Time**: 2-3 months
**Effort**: 25-50 hours (mostly annotation)
**Improvement**: 50-70% WER reduction

1. Collect 20+ hours audio (4-8 weeks passive)
2. Annotate transcripts (25+ hours active)
3. Fine-tune Whisper Small (12 hours passive)
4. Deploy custom model

**Do this if**: Base model WER >15% after Path 1 tuning.

---

### Path 3: Hybrid Approach (Best of Both)
**Time**: 1 day now, 2-3 months later
**Effort**: Minimal now, 25-50 hours later
**Improvement**: Immediate 20-40%, eventual 50-70%

1. ✅ **Now**: Enable AGC + parameter tuning (get quick wins)
2. **Weeks 1-8**: Collect recordings passively while using Habla normally
3. **Weeks 9-12**: Annotate if WER still >10%
4. **Week 13+**: Fine-tune if justified by continued use

**Best strategy**: Get immediate value, build dataset for future fine-tuning if needed.

---

**Next Step**: Start with AGC + parameter tuning (already implemented). Collect audio for 4-8 weeks. Re-evaluate WER. If still >10%, proceed with fine-tuning using this plan.
