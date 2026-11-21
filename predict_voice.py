"""Trained Emotion2Vec+ Base + MLP speech emotion recognizer for Beyondwords."""

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from funasr import AutoModel

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(
    os.environ.get("CHECKPOINT_PATH", REPO_ROOT / "model_weights" / "emotion2vec_final_model.pth")
)
MODEL_VERSION = "emotion2vec_plus_base-mlp-v1"


class EmotionClassifier(nn.Module):
    """Must match the trained architecture in emotion2vec_final_model.pth exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


_encoder: Optional[AutoModel] = None
_classifier: Optional[EmotionClassifier] = None
_scaler_mean: Optional[np.ndarray] = None
_scaler_scale: Optional[np.ndarray] = None
_idx_to_emotion: Optional[dict[int, str]] = None
_load_lock = threading.Lock()
_predict_lock = threading.Lock()
_load_count = 0
_load_ms = 0.0
_load_error: Optional[str] = None
_encoder_source: Optional[str] = None


def _is_complete_encoder_dir(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file()
        for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt")
    )


def _cached_encoder_source() -> str:
    configured = os.environ.get("EMOTION2VEC_MODEL_PATH")
    if configured and _is_complete_encoder_dir(Path(configured)):
        return configured

    local_default = REPO_ROOT / "models" / "emotion2vec_plus_base"
    if _is_complete_encoder_dir(local_default):
        return str(local_default)

    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / "models--emotion2vec--emotion2vec_plus_base"
    revision = root / "refs" / "main"
    preferred = root / "snapshots" / revision.read_text().strip() if revision.is_file() else None
    snapshots = (
        sorted((root / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if (root / "snapshots").is_dir()
        else []
    )
    for path in ([preferred] if preferred else []) + snapshots:
        if path and _is_complete_encoder_dir(path):
            return str(path)

    return "iic/emotion2vec_plus_base"


def model_status() -> dict:
    return {
        "loaded": _encoder is not None and _classifier is not None,
        "load_count": _load_count,
        "load_ms": _load_ms,
        "model_version": MODEL_VERSION,
        "checkpoint_exists": CHECKPOINT_PATH.is_file(),
        "encoder_cached": bool(_encoder_source and Path(_encoder_source).is_dir()),
        "error": _load_error,
    }


def warm_model() -> dict:
    _ensure_loaded()
    return model_status()


def _ensure_loaded() -> None:
    if _encoder is not None and _classifier is not None:
        return
    with _load_lock:
        _load_components()


def _load_components() -> None:
    global _encoder, _classifier, _scaler_mean, _scaler_scale, _idx_to_emotion
    global _load_count, _load_ms, _load_error, _encoder_source

    if _encoder is not None and _classifier is not None:
        return

    try:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

        started = perf_counter()
        source = _cached_encoder_source()
        encoder = AutoModel(
            model=source,
            hub="hf",
            device="cpu",
            disable_update=True,
        )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

        model = EmotionClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        if mean.shape != (768,) or scale.shape != (768,):
            raise ValueError(f"Invalid saved scaler dimensions: mean={mean.shape}, scale={scale.shape}")
        if not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError("Invalid scaler values in checkpoint.")

        _scaler_mean = mean
        _scaler_scale = scale
        _idx_to_emotion = {int(k): str(v) for k, v in checkpoint["idx_to_emotion"].items()}
        _encoder = encoder
        _classifier = model
        _encoder_source = source
        _load_count += 1
        _load_ms = round((perf_counter() - started) * 1000, 1)
        _load_error = None
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        raise


def convert_to_wav(input_path: Path, output_path: Path, max_seconds: int = 30) -> float:
    """Convert any audio/video clip to mono 16 kHz PCM16 WAV and return duration in seconds."""
    try:
        with wave.open(str(input_path), "rb") as wav:
            compatible = (
                wav.getnchannels(),
                wav.getframerate(),
                wav.getsampwidth(),
                wav.getcomptype(),
            ) == (1, 16000, 2, "NONE")
            duration = wav.getnframes() / wav.getframerate()
        if compatible and 0 < duration <= max_seconds:
            shutil.copyfile(input_path, output_path)
            return duration
    except (wave.Error, EOFError):
        pass

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-t",
        str(max_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    with wave.open(str(output_path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    if duration <= 0:
        raise ValueError("Audio clip is empty after conversion.")
    return duration


def predict(audio_path: Path) -> dict:
    """Extract 768-D Emotion2Vec+ embedding, apply training scaler, and predict 7 emotion classes."""
    _ensure_loaded()
    started = perf_counter()

    with tempfile.TemporaryDirectory(prefix="bw-ser-") as temp_dir:
        wav_path = Path(temp_dir) / "input.wav"
        duration_sec = convert_to_wav(audio_path, wav_path)

        with _predict_lock:
            result = _encoder.generate(
                input=str(wav_path),
                granularity="utterance",
                extract_embedding=True,
            )

            embedding = np.asarray(result[0]["feats"], dtype=np.float32).reshape(1, -1)
            if embedding.shape != (1, 768):
                raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
            if not np.isfinite(embedding).all():
                raise ValueError("Extracted embedding contains NaN or Inf values.")

            # Apply exact training-time normalization saved in emotion2vec_final_model.pth
            embedding_scaled = (embedding - _scaler_mean) / _scaler_scale
            input_tensor = torch.from_numpy(embedding_scaled).float()

            with torch.no_grad():
                logits = _classifier(input_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_index = int(np.argmax(probabilities))
    predicted_label = _idx_to_emotion[predicted_index].lower()
    confidence = round(probabilities[predicted_index] * 100, 1)

    breakdown = [
        {
            "emotion": _idx_to_emotion[idx].capitalize(),
            "score": round(prob * 100, 1),
        }
        for idx, prob in enumerate(probabilities)
    ]
    breakdown.sort(key=lambda item: item["score"], reverse=True)

    return {
        "label": predicted_label,
        "primary_emotion": predicted_label.capitalize(),
        "confidence": confidence,
        "emotion_breakdown": breakdown,
        "model_version": MODEL_VERSION,
        "duration_sec": round(duration_sec, 2),
        "inference_ms": round((perf_counter() - started) * 1000, 1),
    }
