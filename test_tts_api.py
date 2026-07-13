"""Executable CUDA smoke test for the public DramaBox TTS API."""

from __future__ import annotations

import argparse
from pathlib import Path

from dramabox_api import DramaBoxTTSEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Load DramaBox and generate two verified WAV files.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tts_test_outputs"))
    args = parser.parse_args()

    load_options: dict = {"device": "cuda", "compile_model": False}
    if args.model_path:
        load_options["model_path"] = args.model_path
    else:
        from src.model_downloader import get_all_paths

        paths = get_all_paths()
        load_options.update(
            transformer_path=paths["transformer"],
            audio_components_path=paths["audio_components"],
            gemma_root=paths["gemma_root"],
        )
    if args.reference_audio:
        load_options["reference_audio_path"] = args.reference_audio

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = DramaBoxTTSEngine()
    try:
        engine.tts_load(**load_options)
        first = engine.tts_inference(
            text="The lantern still burned beside the window.",
            output_path=output_dir / "dramabox_api_default.wav",
            voice_description="A calm audiobook narrator speaks with measured warmth",
            target_duration=4.0,
            seed=42,
            watermark=False,
        )
        second = engine.tts_inference(
            text="Then the storm arrived, sudden and fierce.",
            output_path=output_dir / "dramabox_api_nondefault.wav",
            reference_audio_path=args.reference_audio,
            voice_description="An urgent storyteller speaks with rising tension",
            cfg_scale=3.0,
            stg_scale=1.0,
            target_duration=4.0,
            seed=43,
            watermark=False,
        )
        for path in (first, second):
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"DramaBox smoke output is missing or empty: {path}")
            print(path)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
