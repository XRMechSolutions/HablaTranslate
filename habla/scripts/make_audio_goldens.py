"""Generate the :audio parity goldens from the server's own code, Silero JIT and ffmpeg.

The Android `:audio` module (android/audio) ports the server's audio front end. Its JVM tests
compare the port with what this script records:

- Segmenter: `server.pipeline.vad_buffer.StreamingVADBuffer` run over composed sessions, once with
  the Silero JIT model (same weights the server loads, from the local torch.hub cache) and once in
  energy-fallback mode (no model). Per session: the JIT probability of every VAD call, segment
  boundaries / PCM SHA-256 / rms / vad_avg_prob, partial snapshot sizes and the noise floor.
- PartialPrep: `server.pipeline.audio_processor.prepare_partial_pcm` int16 output.
- FinalPrep: the server's exact ffmpeg command (`_final_prep_filter`) s16 output, the ffmpeg
  highpass+lowpass stage as f64, and the R3 clean-room dynaudnorm reference (`ref7`) as f64.

Speech is never copied out of shared/testdata/clips (D033): every fixture is a recipe (clip,
sample range, gain, gaps) that the Kotlin test composes from the clips in place. Only synthetic
noise and derived outputs are committed. `--piper` writes throwaway Piper TTS fixtures to the
gitignored goldens/local/ directory; they are never committed.

Usage (from the repo root; ffmpeg on PATH):
    habla\\.venv\\Scripts\\python.exe habla\\scripts\\make_audio_goldens.py
    habla\\.venv\\Scripts\\python.exe habla\\scripts\\make_audio_goldens.py --print-sos
    habla\\.venv\\Scripts\\python.exe habla\\scripts\\make_audio_goldens.py --piper

Output is deterministic: the same inputs give byte-identical goldens.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import wave
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "habla"))

SR = 16000
FRAME = 512
THRESHOLD_MARGIN = 1e-3
MAX_COMMITTED_BYTES = 15 * 1024 * 1024
MAX_JITTER_TRIES = 64
JITTER_STEP = 37  # samples; coprime with 512 so each try moves the frame grid
CLIPS_DIR = REPO / "shared" / "testdata" / "clips"
OUT_DIR = REPO / "android" / "audio" / "src" / "test" / "resources" / "goldens"
SILERO_HUB = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
PIPER_VOICES = Path(os.environ.get("HABLA_PIPER_VOICES", r"D:\.cache\piper-voices"))
REGENERATE = r"habla\.venv\Scripts\python.exe habla\scripts\make_audio_goldens.py"

# Prep fixtures: cut lengths in seconds (R3 Q4); 1.95/2.0/2.1/2.25 cover the K = 13/14/15 rules.
PREP_CUTS = (0.8, 1.5, 1.95, 2.0, 2.1, 2.25, 2.5, 4.0, 8.0, 15.0, 30.0)
PREP_STAGE_SUBSET = {"0.8", "1.95", "2.0", "2.1", "2.25", "4.0"}  # f64 stage goldens (budget)
PARTIAL_CUTS = (1.0, 3.0, 8.0, 30.0)
PREP_CLIP = "es_marti_02.wav"
PREP_START = SR  # 1 s into the clip
LEVELS = {"x1": 1.0, "x0.1": 0.1}
SEEDS = {"noise_m50": 50, "noise_m30": 30, "hot_synth": 7}
AUDIO_PROFILE_SECTIONS = ("audio", "vad", "calibration", "segmenter", "partial_prep", "final_prep")


class GenError(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_wav(path: Path) -> np.ndarray:
    if not path.is_file():
        raise GenError(f"missing clip: {path}")
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise GenError(f"{path.name}: need 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


def apply_gain(x: np.ndarray, gain: float) -> np.ndarray:
    """Level change the Kotlin side reproduces with Math.rint(s * gain)."""
    if gain == 1.0:
        return x.astype(np.int16)
    return np.clip(np.rint(x.astype(np.float64) * gain), -32768, 32767).astype(np.int16)


class Writer:
    """Writes gzip goldens deterministically and remembers every file's SHA-256."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.files: dict[str, str] = {}

    def put(self, name: str, arr: np.ndarray, dtype: str) -> str:
        raw = np.ascontiguousarray(arr, dtype=dtype).tobytes()
        blob = gzip.compress(raw, compresslevel=9, mtime=0)
        (self.out_dir / name).write_bytes(blob)
        self.files[name] = sha256(blob)
        return name


# ---------------------------------------------------------------------------------------------
# Silero JIT (the server's model) and the instrumented server segmenter
# ---------------------------------------------------------------------------------------------

def load_jit():
    import torch
    if not (SILERO_HUB / "hubconf.py").is_file():
        raise GenError(f"missing torch.hub Silero cache: {SILERO_HUB}")
    torch.set_num_threads(1)
    model = torch.hub.load(str(SILERO_HUB), "silero_vad", source="local", trust_repo=True)
    if isinstance(model, tuple):
        model = model[0]
    return model


class RecordingModel:
    """Wraps the JIT model and records the probability of every call the server makes."""

    def __init__(self, model):
        self.model = model
        self.probs: list[float] = []

    def __call__(self, tensor, sr):
        out = self.model(tensor, sr)
        self.probs.append(float(out.item()))
        return out

    def reset_states(self):
        self.model.reset_states()


class TrackedRing(deque):
    """Pre-speech ring that also records the stream frame index of every frame it holds."""

    def __init__(self, maxlen, clock):
        super().__init__(maxlen=maxlen)
        self.clock = clock
        self.idx: deque[int] = deque(maxlen=maxlen)

    def append(self, item):
        super().append(item)
        self.idx.append(self.clock[0])

    def clear(self):
        super().clear()
        self.idx.clear()


async def run_server_session(pcm: np.ndarray, model) -> dict:
    """Feed `pcm` frame by frame through StreamingVADBuffer and record everything the port must match."""
    from server.pipeline.vad_buffer import StreamingVADBuffer, VADConfig

    buf = StreamingVADBuffer(config=VADConfig())
    recorder = None
    if model is not None:
        model.reset_states()
        recorder = RecordingModel(model)
        buf._vad_model = recorder
        buf._vad_ready = True
    clock = [0]  # index of the frame being processed
    buf._pre_speech_ring = TrackedRing(buf._pre_speech_ring.maxlen, clock)

    segments: list[dict] = []
    partials: list[dict] = []
    floor: list[float] = []
    state = {"start": None, "emitted_this_frame": 0, "splits": 0}

    async def on_segment(pcm_bytes: bytes, duration: float, rms: float, prob: float):
        end = (clock[0] + 1) * FRAME
        segments.append({
            "id": len(segments) + 1,
            "startSample": int(state["start"]),
            "endSample": int(end),
            "samples": len(pcm_bytes) // 2,
            "pcmSha256": sha256(pcm_bytes),
            "rms": float(rms),
            "vadAvgProb": float(prob),
            "durationSec": float(duration),
        })
        state["emitted_this_frame"] += 1

    async def on_partial(pcm_bytes: bytes, duration: float):
        partials.append({"segmentId": len(segments) + 1, "samples": len(pcm_bytes) // 2})

    async def on_calibrated(value: float):
        floor.append(float(value))

    buf.on_segment = on_segment
    buf.on_partial_audio = on_partial
    buf.on_calibration_complete = on_calibrated

    data = pcm.astype("<i2").tobytes()
    n_frames = len(pcm) // FRAME
    for k in range(n_frames):
        clock[0] = k
        was_speaking = buf._is_speaking
        state["emitted_this_frame"] = 0
        await buf.feed_pcm(data[k * FRAME * 2:(k + 1) * FRAME * 2])
        if buf._is_speaking and not was_speaking:
            ring = buf._pre_speech_ring.idx
            state["start"] = (ring[0] if ring else k) * FRAME
        elif buf._is_speaking and state["emitted_this_frame"]:
            # forced split: the next segment starts with the re-seeded tail
            state["splits"] += 1
            state["start"] = (k + 1) * FRAME - len(buf._speech_buffer) // 2
    if len(pcm) > n_frames * FRAME:
        # a partial last frame stays pending and is dropped by flush, as on the server
        await buf.feed_pcm(data[n_frames * FRAME * 2:])
    clock[0] = n_frames - 1
    before_flush = len(segments)
    await buf.flush()
    for i, s in enumerate(segments):
        s["flush"] = i >= before_flush
    return {
        "probs": recorder.probs if recorder else None,
        "segments": segments,
        "partials": partials,
        "noiseFloorRms": floor[0] if floor else None,
        "splits": state["splits"],
    }


# ---------------------------------------------------------------------------------------------
# Fixture composition (recipes)
# ---------------------------------------------------------------------------------------------

def make_noise(seed: int, dbfs: float, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = 32768.0 * 10 ** (dbfs / 20.0)
    return np.clip(np.rint(rng.normal(0.0, rms, int(seconds * SR))), -32768, 32767).astype(np.int16)


def make_hot(seed: int) -> np.ndarray:
    """Speech-like hot signal: 0.5 s of digital silence, then noise bursts whose peaks clip."""
    rng = np.random.default_rng(seed)
    n = 3 * SR
    env = np.repeat(rng.uniform(0.02, 1.6, size=n // 800 + 1), 800)[:n]
    body = np.clip(rng.normal(0.0, 1.0, n) * env * 0.5, -1.0, 1.0)
    x = np.concatenate([np.zeros(SR // 2), body])
    return np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)


def compose(pieces: list[dict], clips: dict[str, np.ndarray], noise: dict[str, np.ndarray]) -> np.ndarray:
    out = []
    for p in pieces:
        if p["kind"] == "clip":
            src = clips[p["clip"]][p["start"]:p["start"] + p["length"]]
            if len(src) != p["length"]:
                raise GenError(f"recipe cut past the end of {p['clip']}: {p}")
            out.append(apply_gain(src, p["gain"]))
        elif p["kind"] == "noise":
            src = noise[p["source"]][p["start"]:p["start"] + p["length"]]
            if len(src) != p["length"]:
                raise GenError(f"recipe cut past the end of {p['source']}: {p}")
            out.append(src.astype(np.int16))
        elif p["kind"] == "zeros":
            out.append(np.zeros(p["length"], dtype=np.int16))
        else:
            raise GenError(f"unknown piece kind {p['kind']}")
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def speech_mask(model, x: np.ndarray) -> np.ndarray:
    import torch
    model.reset_states()
    k = len(x) // FRAME
    p = [model(torch.from_numpy(x[i * FRAME:(i + 1) * FRAME].astype(np.float32) / 32768.0), SR).item()
         for i in range(k)]
    return np.array(p) > 0.35


def monologue_pieces(clip: str, x: np.ndarray, mask: np.ndarray, first: int, last: int, gain: float) -> list[dict]:
    """Continuous speech from frames [first, last): every pause of >= 10 non-speech frames is cut
    down to 4 + 4 frames so the server never sees 18 silent frames and must force a max split."""
    keep = []
    k = first
    while k < last:
        if not mask[k]:
            j = k
            while j < last and not mask[j]:
                j += 1
            if j - k >= 10:
                keep.append((k, k + 4))
                keep.append((j - 4, j))
            else:
                keep.append((k, j))
            k = j
        else:
            j = k
            while j < last and mask[j]:
                j += 1
            keep.append((k, j))
            k = j
    merged = []
    for a, b in keep:
        if merged and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [{"kind": "clip", "clip": clip, "start": a * FRAME, "length": (b - a) * FRAME, "gain": gain}
            for a, b in merged]


def clip_piece(clip: str, f0: int, f1: int, gain: float) -> dict:
    return {"kind": "clip", "clip": clip, "start": f0 * FRAME, "length": (f1 - f0) * FRAME, "gain": gain}


class NoiseCursor:
    """Hands out consecutive slices of a committed noise fixture for gaps."""

    def __init__(self, name: str, total: int):
        self.name, self.total, self.pos = name, total, 0

    def take(self, seconds: float = 0.0, samples: int | None = None) -> dict:
        n = samples if samples is not None else int(round(seconds * SR))
        if self.pos + n > self.total:
            self.pos = 0
        piece = {"kind": "noise", "source": self.name, "start": self.pos, "length": n}
        self.pos += n
        return piece


def session_recipes(clips: dict[str, np.ndarray], masks: dict[str, np.ndarray], noise: dict[str, np.ndarray]):
    es, en = "es_marti_02.wav", "en_aesop_01.wav"
    es_mask = masks[es]

    def recipe_es(jitter: int) -> list[dict]:
        n = NoiseCursor("noise_m50", len(noise["noise_m50"]))
        return [
            n.take(samples=3 * SR + jitter),
            clip_piece(es, 0, 88, 1.0),            # utterance
            n.take(0.3),
            clip_piece(es, 104, 178, 1.0),         # utterance after a 0.3 s gap
            n.take(1.0),
            clip_piece(es, 230, 240, 1.0),         # ~0.3 s word
            n.take(1.0),
            *monologue_pieces(es, clips[es], es_mask, 340, 1880, 1.0),  # ~40 s monologue
            n.take(2.0),
            clip_piece(es, 1906, 1930, 1.0),       # utterance cut mid-speech, ended by flush
        ]

    def recipe_en_quiet(jitter: int) -> list[dict]:
        n = NoiseCursor("noise_m50", len(noise["noise_m50"]))
        return [
            n.take(samples=3 * SR + jitter),
            clip_piece(en, 60, 1410, 0.1),         # the fable read with its natural pauses, x0.1
            n.take(1.0),
            clip_piece(en, 1450, 1454, 0.1),
        ]

    def recipe_noise(jitter: int) -> list[dict]:
        n = NoiseCursor("noise_m30", len(noise["noise_m30"]))
        return [n.take(samples=10 * SR - 64 + jitter)]

    return {"session_es": recipe_es, "session_en_quiet": recipe_en_quiet, "session_noise": recipe_noise}


# ---------------------------------------------------------------------------------------------
# Prep oracles
# ---------------------------------------------------------------------------------------------

def rbj(kind: str, f: float, q: float = 0.707):
    w0 = 2 * np.pi * f / SR
    c = np.cos(w0)
    al = np.sin(w0) / (2 * q)
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2] if kind == "hp" else [(1 - c) / 2, 1 - c, (1 - c) / 2]
    a = [1 + al, -2 * c, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]


def ref7(x: np.ndarray, f=150, g=15, p=0.95, m=10.0, sr=SR) -> np.ndarray:
    """R3 clean-room dynaudnorm model (research/dsp/model7.py `ref7`, fill14='inf'), self-contained."""
    from scipy.special import erf
    F = sr * f // 1000
    r = g // 2
    N = len(x)
    if N == 0:
        return x.copy()
    K = -(-N // F)
    P = 2 * r

    def bnd(v):
        return m * erf(np.sqrt(np.pi) / 2 * v / m)

    e = bnd(1.0)
    L = []
    for k in range(K):
        pk = np.abs(x[k * F:(k + 1) * F]).max()
        L.append(bnd(p / pk) if pk > 0 else m)
    L = np.array(L, dtype=np.float64)
    lens = [min(F, N - k * F) for k in range(K)]
    Kp = K
    if K < P:
        L = np.r_[L, np.full(g - K, 1.0)]
        Kp = g
        lens += [F] * (g - K)
    elif K == P:
        L = np.r_[L, np.inf]
        Kp = K + 1
        lens += [F]
    Lp = np.r_[np.full(P, 1.0), L, np.full(P, e)]
    n = len(Lp)
    M = np.array([Lp[max(0, i - r):i + r + 1].min() for i in range(n)])
    w = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * (g / 6.0) ** 2))
    w /= w.sum()
    S = np.full(n, np.nan)
    for i in range(r, n - r):
        S[i] = np.dot(w, M[i - r:i + r + 1])
    T = np.array([min(S[P + j - 1], Lp[P + j]) for j in range(Kp + 1)])
    T[0] = min(1.0, Lp[P])
    gain = []
    for k in range(Kp):
        nn = lens[k]
        i = np.arange(nn)
        gain.append(T[k] + (T[k + 1] - T[k]) * (i + 1) / nn)
    gain = np.concatenate(gain)[:N]
    return x * gain


def ffmpeg_final(pcm: np.ndarray, filt: str) -> np.ndarray:
    """The server's exact final-prep command (audio_processor._normalize_wav)."""
    with tempfile.TemporaryDirectory(prefix="habla_goldens_") as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.astype("<i2").tobytes())
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", filt,
                        "-ar", "16000", "-ac", "1", "-f", "wav", str(dst)], check=True)
        with wave.open(str(dst), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


def ffmpeg_hplp_f64(pcm: np.ndarray, hp: int, lp: int) -> np.ndarray:
    """ffmpeg highpass+lowpass stage in double. The input is fed as f64le (pcm / 32768.0, exactly
    what the s16 -> dblp conversion produces in the full chain): with s16 input and no dynaudnorm
    ffmpeg would run the biquads in s16p (R3 Q1)."""
    x = pcm.astype(np.float64) / 32768.0
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f64le", "-ar", "16000", "-ac", "1", "-i", "-",
                        "-af", f"highpass=f={hp},lowpass=f={lp}", "-f", "f64le", "-"],
                       input=x.astype("<f8").tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype="<f8").copy()


def print_sos() -> None:
    from scipy.signal import butter
    from server import shared_assets
    prof = shared_assets.asr_profile()
    pp, fp = prof["partial_prep"], prof["final_prep"]
    sos = butter(pp["butterworth_order"], list(pp["bandpass_hz"]), btype="band", fs=SR, output="sos")
    print(f"PartialPrep SOS: butter({pp['butterworth_order']}, {pp['bandpass_hz']}, 'band', fs={SR}, output='sos')")
    print("  [b0, b1, b2, a1, a2] per section")
    for row in sos:
        print("  doubleArrayOf(" + ", ".join(repr(float(v)) for v in (row[0], row[1], row[2], row[4], row[5])) + "),")
    for kind, f in (("hp", fp["highpass_hz"]), ("lp", fp["lowpass_hz"])):
        b, a = rbj(kind, f)
        print(f"FinalPrep RBJ {kind} {f} Hz (Q=0.707): b0, b1, b2, a1, a2 =",
              ", ".join(repr(float(v)) for v in (b[0], b[1], b[2], a[1], a[2])))


# ---------------------------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------------------------

def tool_versions() -> dict:
    import scipy
    import torch
    try:
        ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GenError(f"missing ffmpeg on PATH ({exc})") from exc
    return {"ffmpeg": ff, "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "python": sys.version.split()[0]}


def check_threshold_margin(name: str, probs: list[float]) -> int | None:
    arr = np.asarray(probs)
    near = np.nonzero(np.abs(arr - 0.35) < THRESHOLD_MARGIN)[0]
    return int(near[0]) if len(near) else None


def generate(out_dir: Path) -> None:
    from scipy.signal import lfilter
    from server import shared_assets
    from server.pipeline.audio_processor import _final_prep_filter, prepare_partial_pcm

    tools = tool_versions()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.gz"):
        old.unlink()
    wr = Writer(out_dir)

    clip_names = ["en_aesop_01.wav", "es_marti_02.wav", "mixed_es_en_es.wav"]
    clips = {c: read_wav(CLIPS_DIR / c) for c in clip_names}
    clip_hashes = {c: sha256((CLIPS_DIR / c).read_bytes()) for c in clip_names}
    model = load_jit()
    masks = {c: speech_mask(model, clips[c]) for c in ("es_marti_02.wav", "en_aesop_01.wav")}

    # Committed synthetic noise and the hot clip
    noise = {
        "noise_m50": make_noise(SEEDS["noise_m50"], -50.0, 10.0),
        "noise_m30": make_noise(SEEDS["noise_m30"], -30.0, 10.0),
    }
    noise_meta = {}
    for name, arr in noise.items():
        noise_meta[name] = {"file": wr.put(f"{name}.s16.gz", arr, "<i2"), "samples": len(arr),
                            "seed": SEEDS[name], "dbfs": -50.0 if name == "noise_m50" else -30.0,
                            "recipe": "numpy default_rng(seed).normal(0, 32768*10^(dbfs/20)), rint, clip int16"}
    hot = make_hot(SEEDS["hot_synth"])
    noise_meta["hot_synth"] = {"file": wr.put("hot_synth.s16.gz", hot, "<i2"), "samples": len(hot),
                               "seed": SEEDS["hot_synth"],
                               "recipe": "0.5 s zeros + 3 s normal noise x uniform(0.02,1.6) 800-sample envelope x0.5, clip +-1, rint(x*32767)"}

    # Sessions (Silero JIT + energy mode)
    sessions = {}
    for name, recipe_fn in session_recipes(clips, masks, noise).items():
        chosen = None
        for t in range(MAX_JITTER_TRIES):
            jitter = (t * JITTER_STEP) % FRAME
            pieces = recipe_fn(jitter)
            pcm = compose(pieces, clips, noise)
            silero = asyncio.run(run_server_session(pcm, model))
            near = check_threshold_margin(name, silero["probs"])
            if near is None:
                chosen = (jitter, pieces, pcm, silero)
                break
            print(f"{name}: jitter {jitter} rejected, VAD call {near} p={silero['probs'][near]:.6f} within {THRESHOLD_MARGIN} of 0.35")
        if chosen is None:
            raise GenError(f"{name}: no jitter in {MAX_JITTER_TRIES} tries keeps every probability {THRESHOLD_MARGIN} away from 0.35")
        jitter, pieces, pcm, silero = chosen
        energy = asyncio.run(run_server_session(pcm, None))
        if name == "session_noise" and (silero["segments"] or energy["segments"]):
            raise GenError("session_noise produced segments; the recipe does not exercise 'no speech'")
        if name == "session_es":
            # The cases are defined by the Silero run; the energy run replays the same audio.
            if not silero["splits"]:
                raise GenError("session_es has no max split; lengthen the monologue")
            if not any(s["flush"] for s in silero["segments"]):
                raise GenError("session_es emits nothing at flush; the last utterance must be cut mid-speech")
        probs_file = wr.put(f"{name}.probs.f32.gz", np.asarray(silero["probs"], dtype=np.float32), "<f4")
        sessions[name] = {
            "jitterSamples": jitter,
            "length": len(pcm),
            "pcmSha256": sha256(pcm.astype("<i2").tobytes()),
            "pieces": pieces,
            "silero": {"probs": probs_file, "vadCalls": len(silero["probs"]),
                       "noiseFloorRms": silero["noiseFloorRms"], "splits": silero["splits"], "segments": silero["segments"],
                       "partials": silero["partials"]},
            "energy": {"noiseFloorRms": energy["noiseFloorRms"], "splits": energy["splits"], "segments": energy["segments"],
                       "partials": energy["partials"]},
        }
        print(f"{name}: {len(pcm) / SR:.2f} s, jitter {jitter}, silero {len(silero['segments'])} segments / "
              f"{len(silero['partials'])} partials, energy {len(energy['segments'])} segments / {energy['splits']} splits, silero splits {silero['splits']}")

    # Prep fixtures
    profile = shared_assets.asr_profile()
    fp = profile["final_prep"]
    filt = _final_prep_filter()
    prep = {}
    es = clips[PREP_CLIP]

    def add_prep(key: str, piece: dict | None, pcm: np.ndarray, stages: bool, source: str | None = None):
        entry = {"input": piece, "source": source, "length": len(pcm),
                 "inputSha256": sha256(pcm.astype("<i2").tobytes())}
        final = ffmpeg_final(pcm, filt)
        if len(final) != len(pcm):
            raise GenError(f"{key}: ffmpeg output length {len(final)} != input {len(pcm)}")
        entry["final"] = wr.put(f"final_{key}.s16.gz", final, "<i2")
        if stages:
            hplp = ffmpeg_hplp_f64(pcm, fp["highpass_hz"], fp["lowpass_hz"])
            b1, a1 = rbj("hp", fp["highpass_hz"])
            b2, a2 = rbj("lp", fp["lowpass_hz"])
            ref = lfilter(b2, a2, lfilter(b1, a1, pcm.astype(np.float64) / 32768.0))
            err = float(np.abs(hplp - ref).max()) if len(ref) else 0.0
            if err > 1e-12:
                raise GenError(f"{key}: ffmpeg hp/lp f64 differs from RBJ lfilter by {err}; ffmpeg did not filter in f64")
            dyn = ref7(hplp, f=fp["dynaudnorm"]["frame_ms"], g=fp["dynaudnorm"]["gauss_size"],
                       p=fp["dynaudnorm"]["peak"], m=float(fp["dynaudnorm"]["max_gain"]))
            entry["hplp"] = wr.put(f"hplp_{key}.f64.gz", hplp, "<f8")
            entry["dyn"] = wr.put(f"dyn_{key}.f64.gz", dyn, "<f8")
            entry["hplpVsLfilterMaxAbs"] = err
        prep[key] = entry

    for cut in PREP_CUTS:
        n = int(round(cut * SR))
        for lvl, gain in LEVELS.items():
            piece = {"kind": "clip", "clip": PREP_CLIP, "start": PREP_START, "length": n, "gain": gain}
            pcm = compose([piece], clips, noise)
            stages = repr(cut) in PREP_STAGE_SUBSET and lvl == "x1" or (repr(cut) == "2.25" and lvl == "x0.1")
            add_prep(f"cut{cut}s_{lvl}", piece, pcm, stages)
    odd_piece = {"kind": "clip", "clip": PREP_CLIP, "start": PREP_START, "length": 20 * 2400 + 700, "gain": 1.0}
    add_prep("odd48700_x1", odd_piece, compose([odd_piece], clips, noise), True)
    add_prep("zeros1s", {"kind": "zeros", "length": SR}, np.zeros(SR, dtype=np.int16), False)
    add_prep("hot_synth", None, hot, True, source="hot_synth")

    partial = {}
    for cut in PARTIAL_CUTS:
        n = int(round(cut * SR))
        for lvl, gain in LEVELS.items():
            piece = {"kind": "clip", "clip": PREP_CLIP, "start": PREP_START, "length": n, "gain": gain}
            pcm = compose([piece], clips, noise)
            out = np.frombuffer(prepare_partial_pcm(pcm.astype("<i2").tobytes()), dtype="<i2")
            key = f"cut{cut}s_{lvl}"
            partial[key] = {"input": piece, "length": n, "output": wr.put(f"partial_{key}.s16.gz", out, "<i2")}

    total = sum((out_dir / f).stat().st_size for f in wr.files)
    if total > MAX_COMMITTED_BYTES:
        raise GenError(f"committed goldens are {total} bytes > {MAX_COMMITTED_BYTES}")

    manifest = {
        "_comment": "Generated by habla/scripts/make_audio_goldens.py. Do not edit; regenerate.",
        "regenerate": REGENERATE,
        "tools": tools,
        "sampleRate": SR,
        "frameSamples": FRAME,
        # Only the sections :audio reads; other modules may add keys to the profile freely.
        "profileSections": {k: profile[k] for k in AUDIO_PROFILE_SECTIONS},
        "silero": {"jitSha256": sha256((SILERO_HUB / "src" / "silero_vad" / "data" / "silero_vad.jit").read_bytes()),
                   "onnxSha256": sha256((SILERO_HUB / "src" / "silero_vad" / "data" / "silero_vad.onnx").read_bytes()),
                   "torchThreads": 1},
        "clips": clip_hashes,
        "seeds": SEEDS,
        "thresholdMargin": THRESHOLD_MARGIN,
        "noise": noise_meta,
        "sessions": sessions,
        "finalPrep": {"filter": filt, "command": "ffmpeg -y -i in.wav -af <filter> -ar 16000 -ac 1 -f wav out.wav",
                      "clips": prep},
        "partialPrep": {"clips": partial},
        "totalBytes": total,
        "files": dict(sorted(wr.files.items())),
    }
    text = json.dumps(manifest, indent=1, sort_keys=False, ensure_ascii=True) + "\n"
    (out_dir / "manifest.json").write_bytes(text.encode("utf-8"))
    print(f"wrote {len(wr.files)} goldens, {total} bytes, to {out_dir}")


def generate_piper(out_dir: Path) -> None:
    """Local-only Piper fixtures (never committed, D033): one short session per installed voice."""
    exe = REPO / "habla" / ".venv" / "Scripts" / "piper.exe"
    if not exe.is_file():
        raise GenError(f"missing piper.exe: {exe}")
    local = out_dir / "local"
    local.mkdir(parents=True, exist_ok=True)
    texts = {"es": "Buenos días. ¿Cómo estás hoy? Vamos a probar el detector de voz con una frase corta.",
             "en": "Good morning. How are you today? This is a short sentence to test the voice detector."}
    model = load_jit()
    noise = make_noise(SEEDS["noise_m50"], -50.0, 4.0)
    sessions = {}
    for voice in sorted(PIPER_VOICES.glob("*.onnx")):
        lang = "es" if voice.name.startswith("es") else "en"
        with tempfile.TemporaryDirectory(prefix="habla_piper_") as d:
            raw, wav16 = Path(d) / "raw.wav", Path(d) / "16k.wav"
            subprocess.run([str(exe), "--model", str(voice), "--output_file", str(raw)],
                           input=texts[lang].encode("utf-8"), check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ar", "16000", "-ac", "1",
                            "-sample_fmt", "s16", str(wav16)], check=True)
            speech = read_wav(wav16)
        pcm = np.concatenate([noise[:3 * SR], speech, noise[:SR]])
        res = asyncio.run(run_server_session(pcm, model))
        name = voice.stem
        (local / f"{name}.s16.gz").write_bytes(gzip.compress(pcm.astype("<i2").tobytes(), mtime=0))
        (local / f"{name}.probs.f32.gz").write_bytes(
            gzip.compress(np.asarray(res["probs"], dtype="<f4").tobytes(), mtime=0))
        sessions[name] = {"pcm": f"{name}.s16.gz", "probs": f"{name}.probs.f32.gz", "length": len(pcm),
                          "segments": res["segments"], "noiseFloorRms": res["noiseFloorRms"]}
        print(f"piper {name}: {len(pcm) / SR:.2f} s, {len(res['segments'])} segments")
    (local / "manifest.json").write_text(json.dumps({"sessions": sessions}, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-sos", action="store_true", help="print the PartialPrep SOS and FinalPrep RBJ tables and exit")
    ap.add_argument("--piper", action="store_true", help="also write local-only Piper TTS fixtures to goldens/local/")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory (default: android audio test resources)")
    args = ap.parse_args()
    try:
        if args.print_sos:
            print_sos()
            return 0
        generate(args.out)
        if args.piper:
            generate_piper(args.out)
    except GenError as exc:
        print(f"make_audio_goldens: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
