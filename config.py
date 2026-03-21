"""
Configuration for the Hamilton "My Shot" Voice Conversion Pipeline.
All paths, parameters, and settings are centralized here.
"""

import os
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate"
RVC_DIR = PROJECT_ROOT / "rvc_workspace"  # RVC model training workspace

# Ensure directories exist
for d in [INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR, RVC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Step 1: Download Configuration
# =============================================================================
# YouTube search query for Hamilton "My Shot" (Original Broadway Cast Recording)
YOUTUBE_SEARCH_QUERY = "Hamilton My Shot Original Broadway Cast Recording official audio"
# Or set a direct URL if known:
YOUTUBE_URL = None  # e.g., "https://www.youtube.com/watch?v=XXXX"

DOWNLOADED_MP3 = INPUT_DIR / "my_shot_original.mp3"
DOWNLOADED_WAV = INPUT_DIR / "my_shot_original.wav"
MP3_BITRATE = "320k"

# =============================================================================
# Step 2: User Voice Reference
# =============================================================================
USER_VOICE_FILE = INPUT_DIR / "my.m4a"

# Recording guidance:
# - Record 10-15 minutes of speaking voice
# - Varied intonation, emotion, pace (read poetry, news, stories energetically)
# - Quiet room, close to microphone, minimal reverb
# - No need to sing or rap — RVC handles the conversion
# - Format: M4A or WAV, 44.1kHz preferred

# =============================================================================
# Step 3: Demucs Vocal Separation Configuration
# =============================================================================
DEMUCS_MODEL = "htdemucs_ft"  # Best quality model (fine-tuned hybrid transformer)
# Options: "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"
DEMUCS_TWO_STEMS = True  # Only separate into vocals + accompaniment
DEMUCS_DEVICE = "auto"  # "auto", "cuda", "mps", "cpu"
DEMUCS_SHIFTS = 1  # Number of random shifts for prediction (higher = better but slower)
DEMUCS_OVERLAP = 0.25  # Overlap between prediction windows

SEPARATED_VOCALS = INTERMEDIATE_DIR / "vocals.wav"
SEPARATED_ACCOMPANIMENT = INTERMEDIATE_DIR / "accompaniment.wav"

# =============================================================================
# Step 4: RVC Training Configuration
# =============================================================================
RVC_REPO_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
RVC_REPO_DIR = PROJECT_ROOT / "Retrieval-based-Voice-Conversion-WebUI"

# Training parameters
RVC_SAMPLE_RATE = 40000  # 40000 or 48000
RVC_F0_METHOD = "rmvpe"  # "rmvpe" (best quality), "crepe", "pm", "harvest"
RVC_TRAINING_EPOCHS = 200  # 200-300 recommended for 10 min data
RVC_BATCH_SIZE = 8  # Adjust based on GPU VRAM (8 for ~8GB, 16 for ~16GB+)
RVC_MODEL_NAME = "my_voice"
RVC_SPEAKER_ID = 0

# Paths for trained model artifacts
RVC_TRAINED_MODEL = RVC_DIR / "my_voice.pth"
RVC_TRAINED_INDEX = RVC_DIR / "my_voice.index"

# =============================================================================
# Step 5: RVC Inference Configuration
# =============================================================================
RVC_TRANSPOSE = 0  # Pitch shift in semitones (0 = no shift, +12 = one octave up)
RVC_INDEX_RATE = 0.75  # How much to use the retrieval index (0.0-1.0)
RVC_FILTER_RADIUS = 3  # Median filtering radius for f0 (reduces breathiness)
RVC_RESAMPLE_SR = 0  # 0 = no resampling
RVC_RMS_MIX_RATE = 0.25  # Volume envelope mix (0 = output envelope, 1 = input envelope)
RVC_PROTECT = 0.33  # Protect voiceless consonants (0-0.5, higher = more protection)

CONVERTED_VOCALS = INTERMEDIATE_DIR / "vocals_converted.wav"

# =============================================================================
# Step 6: Mixing Configuration
# =============================================================================
VOCAL_VOLUME_ADJUST_DB = -2.0  # Adjust vocal volume relative to accompaniment
ACCOMPANIMENT_VOLUME_ADJUST_DB = 0.0  # Adjust accompaniment volume
NORMALIZE_OUTPUT = True  # Normalize final mix to -1 dBFS
OUTPUT_MP3 = OUTPUT_DIR / "my_shot_my_voice.mp3"
OUTPUT_WAV = OUTPUT_DIR / "my_shot_my_voice.wav"
OUTPUT_MP3_BITRATE = "320k"

# =============================================================================
# Device Detection
# =============================================================================
def get_device():
    """Auto-detect the best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
