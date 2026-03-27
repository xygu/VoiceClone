# VoiceClone - AI Voice Conversion Pipeline

A universal voice conversion pipeline that replaces vocals in any song with your own voice using AI (RVC - Retrieval-based Voice Conversion).

## Overview

The pipeline performs the following steps:
1. **Download** - Downloads a song from YouTube
2. **Separate** - Separates vocals and accompaniment using Demucs
3. **Train** - Trains an RVC model on your voice samples
4. **Convert** - Converts the original vocals to your voice timbre
5. **Mix** - Mixes the converted vocals with the original accompaniment

## Example Use Case

For example, you can use this pipeline to sing Hamilton's "My Shot" in your own voice:
- Download "My Shot" from YouTube
- Train on 10-15 minutes of your speaking voice
- Convert and mix to create your own version

## Directory Structure

```
voiceclone/
├── input/                    # Place your input files here
│   ├── song_original.mp3     # Downloaded song (auto-generated)
│   ├── song_original.wav     # Converted WAV (auto-generated)
│   └── my_voice.m4a          # YOUR VOICE RECORDING (place here)
├── intermediate/             # Temporary processing files
│   ├── vocals.wav            # Extracted vocals (auto-generated)
│   ├── accompaniment.wav     # Instrumental track (auto-generated)
│   └── sliced/               # Sliced voice segments for training
├── output/                   # Final output files
│   ├── song_my_voice.mp3     # Final mixed song (MP3)
│   └── song_my_voice.wav     # Final mixed song (WAV)
├── exp/                      # Training experiments (auto-created)
│   └── YYYYMMDD_HHMMSS/      # Timestamped training runs
│       ├── my_voice.pth      # Trained model
│       ├── my_voice.index    # Feature index
│       └── vocals_converted.wav  # Converted vocals
└── rvc_workspace/            # RVC repository and models
    └── Retrieval-based-Voice-Conversion-WebUI/
```

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:
- `torch`, `torchaudio` - Deep learning framework
- `demucs` - Vocal separation
- `rvc-python` - Voice conversion (optional, falls back to RVC repo)
- `pydub`, `soundfile`, `librosa` - Audio processing
- `yt-dlp` - YouTube downloading
- `faiss-cpu` or `faiss-gpu` - Feature indexing
- `scikit-learn` - K-means clustering

### 2. Download RVC Repository (Manual Step)

Due to network restrictions, you need to manually download the RVC repository:

```bash
# On a machine with internet access:
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git

# Copy to the project:
# Place the folder at: ./rvc_workspace/Retrieval-based-Voice-Conversion-WebUI/
```

### 3. Download RVC Models

```bash
cd rvc_workspace/Retrieval-based-Voice-Conversion-WebUI
python tools/download_models.py
```

## Step 1: Prepare Your Voice Recording

### File Placement

Place your voice recording at:
```
input/my_voice.m4a
```

Or modify `config.py` to point to your file:
```python
USER_VOICE_FILE = INPUT_DIR / "your_voice.wav"
```

### Recording Requirements

**Duration:**
- **Minimum:** 10 minutes of clear speech
- **Recommended:** 10-15 minutes for best quality
- **More is better:** Up to 30 minutes can improve results

**Content Guidelines:**
- Speak with varied intonation, emotion, and pace
- Read poetry, news articles, or stories energetically
- Cover different speaking styles: whispering, normal speech, enthusiastic speech
- **You do NOT need to sing or rap** - RVC handles the conversion from speech to singing

**Technical Quality:**
- **Format:** M4A, WAV, or MP3 (44.1 kHz preferred)
- **Environment:** Quiet room with minimal background noise
- **Microphone:** Close to mic (6-12 inches), use pop filter if available
- **Reverb:** Minimal room reverb (record in a closet with clothes for best results)
- **Clipping:** Avoid audio clipping/distortion

**Pro Tips:**
- Record in one continuous session for consistency
- Take breaks but keep the mic position constant
- Read material you're comfortable with to sound natural
- Include some laughter and emotional expressions

## Step 2: Run the Pipeline

### Full Pipeline (All Steps)

```bash
python pipeline.py --all
```

### Individual Steps

```bash
# Step 1: Download the song
python pipeline.py --download

# Step 2: Separate vocals and accompaniment
python pipeline.py --separate

# Step 3: Train voice model
python pipeline.py --train

# Step 4: Convert vocals to your voice
python pipeline.py --convert

# Step 5: Mix final output
python pipeline.py --mix
```

### Quick Debug Mode

For testing the pipeline with minimal training time:

```bash
python pipeline.py --all --quick
```

This uses only 2 training epochs instead of 200, completing much faster but with lower quality.

### Resume Training

If training was interrupted, you can resume from a checkpoint:

```bash
python pipeline.py --train --continue-from ./exp/20260328_143000
```

### Use Specific Experiment

To use a specific trained model for conversion/mixing:

```bash
python pipeline.py --convert --ckpt ./exp/20260328_143000
python pipeline.py --mix --ckpt ./exp/20260328_143000
```

## Configuration

Edit `config.py` to customize the pipeline:

### Voice Conversion Backend

```python
# Use RVC for voice conversion (default)
VOICE_CONVERSION_BACKEND = "rvc"

# Or skip voice conversion (copy vocals as-is)
VOICE_CONVERSION_BACKEND = "passthrough"
```

### RVC Training Parameters

```python
RVC_SAMPLE_RATE = 40000      # 40000 (v1) or 48000 (v2)
RVC_TRAINING_EPOCHS = 200    # 200-300 recommended for 10 min data
RVC_BATCH_SIZE = 32          # Adjust based on your GPU memory
RVC_F0_METHOD = "rmvpe"      # Pitch extraction method
```

### Output Settings

```python
VOCAL_VOLUME_ADJUST_DB = -2.0     # Adjust vocal volume
ACCOMPANIMENT_VOLUME_ADJUST_DB = 0.0  # Adjust instrumental volume
OUTPUT_MP3_BITRATE = "320k"       # Output MP3 quality
```

### GPU Configuration

```python
# Use specific GPU (for multi-GPU systems)
RVC_CUDA_DEVICE = "0"  # Use GPU 0
```

## Output Files

After successful completion, you'll find:

| File | Location | Description |
|------|----------|-------------|
| Final MP3 | `output/song_my_voice.mp3` | Final song in MP3 format (320kbps) |
| Final WAV | `output/song_my_voice.wav` | Final song in uncompressed WAV format |
| Model | `exp/YYYYMMDD_HHMMSS/my_voice.pth` | Trained RVC model (reusable) |
| Index | `exp/YYYYMMDD_HHMMSS/my_voice.index` | Feature index for the model |
| Converted Vocals | `exp/YYYYMMDD_HHMMSS/vocals_converted.wav` | Your voice singing the song |

## Troubleshooting

### YouTube Download Issues

If YouTube download fails due to authentication:

```bash
# Use browser cookies
python pipeline.py --download --cookies-from-browser chrome

# Or use a cookies file
python pipeline.py --download --cookies-file /path/to/cookies.txt
```

### CUDA Out of Memory

Reduce batch size in `config.py`:
```python
RVC_BATCH_SIZE = 16  # Or even 8 for GPUs with less memory
```

### No GPU Available

The pipeline will fall back to CPU, but training will be very slow. Consider:
- Using `--quick` mode for testing
- Using a cloud GPU service
- Using pre-trained models if available

### Audio Quality Issues

- **Robotic voice:** Increase training epochs (300+)
- **Background noise:** Improve recording environment
- **Pitch issues:** Adjust `RVC_TRANSPOSE` in config.py
- **Muffled sound:** Check recording quality, ensure no clipping

## Advanced Usage

### Using a Different Song

To convert a different song, modify in `config.py`:

```python
YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
# Or use search:
YOUTUBE_SEARCH_QUERY = "Your Song Name Artist"
```

You can also change the output filenames:

```python
OUTPUT_MP3 = OUTPUT_DIR / "your_song_your_voice.mp3"
OUTPUT_WAV = OUTPUT_DIR / "your_song_your_voice.wav"
```

### Batch Processing

You can train once and convert multiple songs:

```bash
# Train once
python pipeline.py --train

# For each new song:
python pipeline.py --download --separate
python pipeline.py --convert --ckpt ./exp/YOUR_TIMESTAMP
python pipeline.py --mix --ckpt ./exp/YOUR_TIMESTAMP
```

## License

This project uses RVC which is under MIT License. See the RVC repository for details.
