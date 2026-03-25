#!/usr/bin/env python3
"""
Quick debug script to run only RVC training step (Step 3).
This skips download and separation to quickly get to the training error.

Usage:
    python debug_train.py
    
Requirements:
    - input/myshot.m4a (or my_shot_original.mp3) must exist
    - RVC repo must be present at rvc_workspace/Retrieval-based-Voice-Conversion-WebUI/
"""

import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import step_train
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("debug_train")

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("DEBUG: Running only Step 3 (Train)")
    log.info("=" * 60)
    
    try:
        step_train()
        log.info("=" * 60)
        log.info("DEBUG: Training completed successfully!")
        log.info("=" * 60)
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
