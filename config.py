"""
Configuration for the Hamilton "My Shot" Voice Conversion Pipeline.
All paths, parameters, and settings are centralized here.
"""

import os
from pathlib import Path

# =============================================================================
# HuggingFace Mirror Configuration (for China network)
# =============================================================================
# Set global HF mirror to avoid manual configuration in code
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


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
USER_VOICE_FILE = INPUT_DIR / "myshot.m4a"

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
# Voice conversion backend
# =============================================================================
# "rvc" — Retrieval-based Voice Conversion (train + infer via RVC repo or rvc-python).
# "passthrough" — Skip timbre conversion: separated vocals are copied straight to mix.
#   Use this when RVC is blocked by network/GPU, or after converting audio in another tool.
VOICE_CONVERSION_BACKEND = "rvc"  # "rvc" | "passthrough"

# =============================================================================
# Step 4: RVC Training Configuration
# =============================================================================
RVC_REPO_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
RVC_REPO_DIR = PROJECT_ROOT / "rvc_workspace" / "Retrieval-based-Voice-Conversion-WebUI"

# Hugging Face repo id for RVC assets (tools/download_models.py default minimal set)
RVC_HF_REPO_ID = "lj1995/VoiceConversionWebUI"

# Training parameters
RVC_SAMPLE_RATE = 40000  # 40000 for v1 (models already downloaded), or 48000 for v2 (requires f0G48k/f0D48k)
RVC_F0_METHOD = "rmvpe"  # "rmvpe" (best quality), "crepe", "pm", "harvest"
RVC_TRAINING_EPOCHS = 200  # 200-300 recommended for 10 min data
RVC_BATCH_SIZE = 32  # Optimized for 80GB VRAM (A800): 32-64 works well
RVC_MODEL_NAME = "my_voice"
RVC_SPEAKER_ID = 0

# GPU Configuration
# Set to specific GPU index (e.g., "0") or "auto" to use first available
# For multi-GPU servers, you can distribute training by setting different indices
RVC_CUDA_DEVICE = "0"  # GPU index to use for training (A800 has 8 GPUs, indices 0-7)

# Train flow switches
# Set to False to skip pip install during --train when environment is already prepared.
RVC_INSTALL_REQS = False
# Set to False to skip audio slicing and reuse existing intermediate/sliced/*.wav.
RVC_SLICE_AUDIO = False

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
