"""Stable class-based DramaBox API for external applications."""

from __future__ import annotations

import gc
import re
import secrets
import sys
import threading
from pathlib import Path
from typing import Any


class DramaBoxTTSEngine:
    """Thin, CUDA-only adapter around DramaBox's warm ``TTSServer``."""

    def __init__(self) -> None:
        self._load_options: dict[str, Any] = {}
        self._reference_audio_path: Path | None = None
        self._server: Any = None
        self._server_class: Any = None
        self._lock = threading.RLock()

    def close(self) -> None:
        """Release the cached server and its CUDA allocations."""
        with self._lock:
            self._server = None
            self._load_options = {}
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def tts_inference(
        self,
        *,
        text: str,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        reference_audio_path: str | Path | None = None,
        voice_description: str = "A clear, expressive narrator speaks naturally",
        cfg_scale: float = 2.5,
        stg_scale: float = 1.5,
        duration_multiplier: float = 1.1,
        target_duration: float = 0.0,
        reference_duration: float = 10.0,
        seed: int = -1,
        rescale_scale: str | float = "auto",
        watermark: bool = True,
        raw_prompt: bool = False,
        **load_overrides: Any,
    ) -> Path:
        """Generate one WAV file and return its resolved path."""
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("DramaBox inference text must not be empty.")

        requested_model_path = Path(model_path).expanduser().resolve() if model_path else None
        if self._server is None:
            if requested_model_path is None and not load_overrides:
                raise RuntimeError("DramaBox is not loaded. Call tts_load(...) first.")
            self.tts_load(model_path=requested_model_path, **load_overrides)
        elif requested_model_path is not None:
            current_root = self._load_options.get("model_path")
            if current_root != requested_model_path:
                self.tts_load(model_path=requested_model_path, **load_overrides)

        reference = self._resolve_optional_file(
            reference_audio_path,
            "reference audio",
        ) if reference_audio_path else self._reference_audio_path

        destination = self._resolve_output_path(output_path, output_dir)
        prompt = normalized_text if raw_prompt else self._build_prompt(normalized_text, voice_description)

        with self._lock:
            result = self._server.generate_to_file(
                prompt=prompt,
                output=str(destination),
                voice_ref=str(reference) if reference else None,
                cfg_scale=float(cfg_scale),
                stg_scale=float(stg_scale),
                duration_multiplier=float(duration_multiplier),
                seed=self._resolve_seed(seed),
                ref_duration=float(reference_duration),
                rescale_scale=rescale_scale,
                gen_duration=float(target_duration),
                watermark=bool(watermark),
            )

        result_path = Path(result).expanduser().resolve()
        if result_path != destination:
            destination = result_path
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError(f"DramaBox did not produce a non-empty WAV file: {destination}")
        return destination

    def tts_load(
        self,
        *,
        model_path: str | Path | None = None,
        reference_audio_path: str | Path | None = None,
        transformer_path: str | Path | None = None,
        audio_components_path: str | Path | None = None,
        gemma_root: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "bf16",
        compile_model: bool = False,
        bnb_4bit: bool = True,
    ) -> None:
        """Load and cache the DramaBox model stack on CUDA."""
        self._require_cuda(device)
        root = Path(model_path).expanduser().resolve() if model_path else None
        transformer, audio_components, gemma = self._resolve_model_paths(
            root,
            transformer_path,
            audio_components_path,
            gemma_root,
        )
        reference = self._resolve_optional_file(reference_audio_path, "reference audio")

        options = {
            "model_path": root,
            "transformer_path": transformer,
            "audio_components_path": audio_components,
            "gemma_root": gemma,
            "device": str(device),
            "dtype": str(dtype),
            "compile_model": bool(compile_model),
            "bnb_4bit": bool(bnb_4bit),
        }
        with self._lock:
            if self._server is not None and options == self._load_options:
                self._reference_audio_path = reference
                return
            self.close()
            server_class = self._get_server_class()
            try:
                self._server = server_class(
                    checkpoint=str(transformer),
                    full_checkpoint=str(audio_components),
                    gemma_root=str(gemma),
                    device=str(device),
                    dtype=str(dtype),
                    compile_model=bool(compile_model),
                    bnb_4bit=bool(bnb_4bit),
                )
            except Exception as exc:
                self._server = None
                raise RuntimeError(f"Failed to load DramaBox: {exc}") from exc
            self._load_options = options
            self._reference_audio_path = reference

    @staticmethod
    def _build_prompt(text: str, voice_description: str) -> str:
        description = DramaBoxTTSEngine._normalize_voice_description(voice_description)
        spoken_text = text.replace('"', "'")
        return f'{description}, "{spoken_text}"'

    def _get_server_class(self):
        if self._server_class is not None:
            return self._server_class
        repo_root = Path(__file__).resolve().parents[1]
        for source_path in (repo_root / "ltx2", repo_root / "src"):
            source_text = str(source_path)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        try:
            from inference_server import TTSServer
        except Exception as exc:
            raise ImportError(
                "DramaBox backend imports failed. Install the locked CUDA dependencies "
                "and ensure the repository's ltx2/ and src/ directories are present."
            ) from exc
        self._server_class = TTSServer
        return self._server_class

    @staticmethod
    def _normalize_voice_description(voice_description: str) -> str:
        description = " ".join(str(voice_description or "").split()).strip(" ,.;:")
        if not description:
            raise ValueError("DramaBox voice_description must not be empty.")

        participle_replacements = {
            "narrating": "narrates",
            "reading": "reads",
            "saying": "says",
            "shouting": "shouts",
            "speaking": "speaks",
            "whispering": "whispers",
            "yelling": "yells",
        }
        for participle, finite_verb in participle_replacements.items():
            pattern = re.compile(rf"\b{participle}\b", flags=re.IGNORECASE)
            if pattern.search(description):
                description = pattern.sub(finite_verb, description, count=1)
                break

        speech_verb_pattern = re.compile(
            r"\b(?:cries|exclaims|laughs|murmurs|narrates|reads|replies|says|"
            r"shouts|sings|speaks|whispers|yells)\b",
            flags=re.IGNORECASE,
        )
        if speech_verb_pattern.search(description):
            return description

        speaker_noun_pattern = re.compile(
            r"\b(?:boy|character|girl|host|man|narrator|person|speaker|storyteller|voice|woman)\b",
            flags=re.IGNORECASE,
        )
        if speaker_noun_pattern.search(description):
            return f"{description} speaks"
        first_word = description.split(maxsplit=1)[0].lower()
        if first_word.endswith("ly") or first_word in {"at", "in", "with"}:
            return f"A narrator speaks {description}"
        return f"A narrator speaks with {description}"

    @staticmethod
    def _require_cuda(device: str) -> None:
        if not str(device).lower().startswith("cuda"):
            raise ValueError("DramaBox 4.5 integration is CUDA-only; device must be 'cuda'.")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("DramaBox requires a CUDA-enabled PyTorch installation.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("DramaBox requires CUDA, but torch.cuda.is_available() is false.")

    @staticmethod
    def _resolve_model_paths(
        model_root: Path | None,
        transformer_path: str | Path | None,
        audio_components_path: str | Path | None,
        gemma_root: str | Path | None,
    ) -> tuple[Path, Path, Path]:
        if model_root is None and not all((transformer_path, audio_components_path, gemma_root)):
            raise ValueError(
                "Provide model_path containing DramaBox assets, or provide transformer_path, "
                "audio_components_path, and gemma_root explicitly."
            )

        def choose(explicit: str | Path | None, relative_names: tuple[str, ...], label: str) -> Path:
            candidates: list[Path] = []
            if explicit:
                candidates.append(Path(explicit).expanduser())
            if model_root:
                candidates.extend(model_root / relative_name for relative_name in relative_names)
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved.exists():
                    return resolved
            searched = ", ".join(str(path.resolve()) for path in candidates)
            raise FileNotFoundError(f"DramaBox {label} was not found. Checked: {searched}")

        transformer = choose(
            transformer_path,
            ("dramabox-dit-v1.safetensors", "base/dramabox-dit-v1.safetensors"),
            "transformer checkpoint",
        )
        audio_components = choose(
            audio_components_path,
            (
                "dramabox-audio-components.safetensors",
                "base/dramabox-audio-components.safetensors",
            ),
            "audio-components checkpoint",
        )
        gemma = choose(
            gemma_root,
            (
                "gemma-3-12b-it-bnb-4bit",
                "base/gemma-3-12b-it-bnb-4bit",
            ),
            "Gemma model directory",
        )
        if not transformer.is_file() or not audio_components.is_file() or not gemma.is_dir():
            raise ValueError("DramaBox model assets have invalid file/directory types.")
        return transformer, audio_components, gemma

    @staticmethod
    def _resolve_optional_file(path_value: str | Path | None, label: str) -> Path | None:
        if path_value in (None, ""):
            return None
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DramaBox {label} does not exist: {path}")
        return path

    @staticmethod
    def _resolve_output_path(
        output_path: str | Path | None,
        output_dir: str | Path | None,
    ) -> Path:
        if output_path:
            destination = Path(output_path).expanduser()
        elif output_dir:
            destination = Path(output_dir).expanduser() / "dramabox_output.wav"
        else:
            raise ValueError("Provide output_path or output_dir for DramaBox inference.")
        destination = destination.resolve()
        if destination.suffix.lower() != ".wav":
            raise ValueError("DramaBox output_path must end in .wav.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        normalized_seed = int(seed)
        if normalized_seed == -1:
            return secrets.randbelow(10_000_000)
        if normalized_seed < -1:
            raise ValueError("DramaBox seed must be -1 (random) or a non-negative integer.")
        return normalized_seed
