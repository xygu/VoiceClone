#!/usr/bin/env python3
"""
Quick debug script to run only RVC training step (Step 3) with minimal epochs.
This skips download and separation to quickly get to the training error.

Usage:
    python debug_train.py
    
Requirements:
    - input/myshot.m4a (or my_shot_original.mp3) must exist
    - RVC repo must be present at rvc_workspace/Retrieval-based-Voice-Conversion-WebUI/

Note: This is equivalent to running: python pipeline.py --train --quick
"""

import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("debug_train")


def _detect_audio_sr(filepath):
    """Detect audio file sample rate using soundfile or ffmpeg."""
    try:
        import soundfile as sf
        info = sf.info(str(filepath))
        return info.samplerate
    except Exception:
        # Fallback to ffmpeg
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(filepath)],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except Exception:
            return None


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("DEBUG: Running Step 3 (Train) in QUICK mode (2 epochs)")
    log.info("=" * 60)
    
    # Log audio file information
    log.info("=" * 60)
    log.info("Audio Source Check (Debug)")
    log.info("=" * 60)
    log.info(f"User voice file: {config.USER_VOICE_FILE}")
    if config.USER_VOICE_FILE.exists():
        original_sr = _detect_audio_sr(config.USER_VOICE_FILE)
        if original_sr:
            log.info(f"Original audio sample rate: {original_sr} Hz")
        else:
            log.warning("Could not detect original audio sample rate")
        log.info(f"Target RVC sample rate: {config.RVC_SAMPLE_RATE} Hz")
        if original_sr and original_sr != config.RVC_SAMPLE_RATE:
            log.warning(f"Sample rate mismatch! Audio will be resampled from {original_sr} Hz to {config.RVC_SAMPLE_RATE} Hz")
    else:
        log.error(f"User voice file NOT FOUND: {config.USER_VOICE_FILE}")
    log.info("=" * 60)
    
    from pipeline import step_train
    
    try:
        step_train(quick=True)
        log.info("=" * 60)
        log.info("DEBUG: Training completed successfully!")
        log.info("=" * 60)
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
