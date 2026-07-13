#!/usr/bin/env python3
"""Optional audio super-resolution adapters for DramaBox outputs."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import snapshot_download


APP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_DIR / "output"
SR_CACHE_DIR = APP_DIR / ".cache" / "audio_super_resolution"
TARGET_SAMPLE_RATE = 48_000

_LAVASR_MODEL = None
_NOVASR_MODEL = None
_LAVASR_LOCK = threading.Lock()
_NOVASR_LOCK = threading.Lock()


def _add_local_package(package_dir: str) -> None:
    path = str(APP_DIR / package_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def _sr_device() -> str:
    requested = os.environ.get("DRAMABOX_SR_DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _output_path(engine: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return tempfile.mktemp(suffix=".wav", prefix=f"dramabox_{engine}_", dir=str(OUTPUT_DIR))


def _snapshot_to_local(repo_id: str, dirname: str) -> str:
    target = SR_CACHE_DIR / dirname
    target.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id,
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
    )


def _as_audio_tensor(wav: torch.Tensor) -> torch.Tensor:
    wav = wav.detach().float().cpu()
    while wav.dim() > 2:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"Expected audio tensor with 1 or 2 dims, got shape {tuple(wav.shape)}")
    return wav.clamp(-1.0, 1.0)


def _save_wav(wav: torch.Tensor, output: str) -> str:
    torchaudio.save(output, _as_audio_tensor(wav), TARGET_SAMPLE_RATE)
    return output


def _get_lavasr_model():
    global _LAVASR_MODEL
    with _LAVASR_LOCK:
        if _LAVASR_MODEL is not None:
            return _LAVASR_MODEL

        _add_local_package("LavaSR")
        from LavaSR.model import LavaEnhance2

        model_path = os.environ.get("LAVASR_MODEL_PATH") or _snapshot_to_local(
            "YatharthS/LavaSR", "LavaSR"
        )
        device = _sr_device()
        logging.info("Loading LavaSR on %s...", device)
        _LAVASR_MODEL = LavaEnhance2(model_path, device=device)
        logging.info("LavaSR ready.")
        return _LAVASR_MODEL


def _get_novasr_model():
    global _NOVASR_MODEL
    with _NOVASR_LOCK:
        if _NOVASR_MODEL is not None:
            return _NOVASR_MODEL

        _add_local_package("NovaSR")
        from NovaSR import FastSR

        ckpt_path = os.environ.get("NOVASR_CKPT_PATH")
        if not ckpt_path:
            model_path = _snapshot_to_local("YatharthS/NovaSR", "NovaSR")
            ckpt_path = str(Path(model_path) / "pytorch_model_v1.bin")
        half = torch.cuda.is_available() and os.environ.get("NOVASR_HALF", "1") == "1"
        logging.info("Loading NovaSR%s...", " with fp16" if half else "")
        _NOVASR_MODEL = FastSR(ckpt_path=ckpt_path, half=half)
        logging.info("NovaSR ready.")
        return _NOVASR_MODEL


def enhance_audio_file(
    input_path: str,
    engine: str,
    *,
    lavasr_denoise: bool = False,
    lavasr_batch: bool = False,
) -> str:
    """Enhance an audio file with the selected SR engine.

    Args:
        input_path: Source wav/mp3 path.
        engine: "off", "lavasr", or "novasr".
        lavasr_denoise: Enable LavaSR's denoiser before bandwidth extension.
        lavasr_batch: Chunk long LavaSR inputs into batches.

    Returns:
        Path to the original file when disabled, otherwise a new 48 kHz wav.
    """
    engine = (engine or "off").strip().lower()
    if engine in {"off", "none", "disabled"}:
        return input_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file does not exist: {input_path}")

    if engine == "lavasr":
        model = _get_lavasr_model()
        audio, _ = model.load_audio(input_path)
        with torch.inference_mode():
            enhanced = model.enhance(audio, denoise=lavasr_denoise, batch=lavasr_batch)
        return _save_wav(enhanced, _output_path("lavasr"))

    if engine == "novasr":
        model = _get_novasr_model()
        lowres_audio = model.load_audio(input_path)
        with torch.inference_mode():
            enhanced = model.infer(lowres_audio)
        return _save_wav(enhanced, _output_path("novasr"))

    raise ValueError(f"Unknown super-resolution engine: {engine}")
