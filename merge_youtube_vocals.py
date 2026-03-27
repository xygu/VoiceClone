#!/usr/bin/env python3
"""
Download YouTube video and merge with converted vocals.
Usage: python merge_youtube_vocals.py <youtube_url> <vocals_wav_path>
"""

import subprocess
import sys
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path

# Configuration
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"

def download_audio(url: str, output_name: str = "downloaded", skip_existing: bool = False) -> Path:
    """Download audio from YouTube URL.
    
    Args:
        url: YouTube URL
        output_name: Base name for output file
        skip_existing: If True, use existing file without re-downloading
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    mp3_path = INPUT_DIR / f"{output_name}.mp3"
    wav_path = INPUT_DIR / f"{output_name}.wav"
    
    if skip_existing and wav_path.exists():
        print(f"Using existing: {wav_path}")
        return wav_path
    
    print(f"Downloading from: {url}")
    cmd = [
        "yt-dlp", "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "0", "--output", str(mp3_path.with_suffix("")) + ".%(ext)s",
        "--no-playlist",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        url
    ]
    subprocess.run(cmd, check=True)
    
    # Convert to WAV
    if not wav_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", "44100", "-ac", "2", str(wav_path)
        ], check=True, capture_output=True)
    
    print(f"Downloaded: {wav_path}")
    return wav_path


def separate_vocals(input_wav: Path) -> tuple[Path, Path]:
    """Separate vocals and accompaniment using Demucs."""
    intermediate_dir = Path(__file__).parent / "intermediate"
    vocals_path = intermediate_dir / "vocals.wav"
    accomp_path = intermediate_dir / "accompaniment.wav"
    
    if vocals_path.exists() and accomp_path.exists():
        print(f"Using existing separated tracks")
        return vocals_path, accomp_path
    
    print("Separating vocals and accompaniment (Demucs)...")
    demucs_out = intermediate_dir / "demucs_output"
    
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs_ft",
        "--out", str(demucs_out),
        "--shifts", "1",
        "--overlap", "0.25",
        "--two-stems", "vocals",
        "--device", "cpu",
        str(input_wav)
    ]
    subprocess.run(cmd, check=True)
    
    stem = input_wav.stem
    result_dir = demucs_out / "htdemucs_ft" / stem
    
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_dir / "vocals.wav", vocals_path)
    shutil.copy2(result_dir / "no_vocals.wav", accomp_path)
    
    print(f"Vocals: {vocals_path}")
    print(f"Accompaniment: {accomp_path}")
    return vocals_path, accomp_path


def merge_with_converted_vocals(accomp_path: Path, converted_vocals_path: Path, output_name: str = "final_mix", vocals_delay_sec: float = 0.0, vocals_gain_db: float = 0.0) -> Path:
    """Mix accompaniment with converted vocals.
    
    Args:
        accomp_path: Path to accompaniment/instrumental file
        converted_vocals_path: Path to converted vocals file
        output_name: Base name for output files
        vocals_delay_sec: Delay in seconds before vocals start (default: 0)
        vocals_gain_db: Gain in dB to apply to vocals (default: 0, positive = louder)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading accompaniment: {accomp_path}")
    print(f"Loading converted vocals: {converted_vocals_path}")
    
    # Load audio files
    accomp, sr_accomp = sf.read(str(accomp_path))
    vocals, sr_vocals = sf.read(str(converted_vocals_path))
    
    # Ensure same sample rate
    if sr_accomp != sr_vocals:
        print(f"Resampling vocals from {sr_vocals} to {sr_accomp} Hz")
        import librosa
        vocals = librosa.resample(vocals.T, orig_sr=sr_vocals, target_sr=sr_accomp).T
        sr_vocals = sr_accomp
    
    # Convert to mono if stereo
    if accomp.ndim > 1:
        accomp = accomp.mean(axis=1)
    if vocals.ndim > 1:
        vocals = vocals.mean(axis=1)
    
    # Apply vocals gain if specified
    if vocals_gain_db != 0.0:
        gain_linear = 10 ** (vocals_gain_db / 20)
        print(f"Applying {vocals_gain_db}dB gain to vocals ({gain_linear:.2f}x)")
        vocals = vocals * gain_linear
    
    # Apply vocals delay if specified
    if vocals_delay_sec > 0:
        delay_samples = int(vocals_delay_sec * sr_accomp)
        print(f"Applying {vocals_delay_sec}s delay to vocals ({delay_samples} samples)")
        vocals = np.pad(vocals, (delay_samples, 0), mode='constant')
    
    # Match lengths (pad shorter with silence)
    len_accomp = len(accomp)
    len_vocals = len(vocals)
    
    if len_vocals < len_accomp:
        vocals = np.pad(vocals, (0, len_accomp - len_vocals), mode='constant')
    elif len_accomp < len_vocals:
        accomp = np.pad(accomp, (0, len_vocals - len_accomp), mode='constant')
    
    # Mix vocals over accompaniment
    print("Mixing...")
    final = accomp + vocals
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(final))
    if max_val > 1.0:
        final = final / max_val * 0.95
        print(f"Normalized audio (peak was {max_val:.2f})")
    
    # Export WAV
    output_wav = OUTPUT_DIR / f"{output_name}.wav"
    sf.write(str(output_wav), final, sr_accomp)
    
    # Export MP3 using ffmpeg
    output_mp3 = OUTPUT_DIR / f"{output_name}.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(output_wav),
        "-codec:a", "libmp3lame", "-q:a", "0",
        str(output_mp3)
    ], check=True, capture_output=True)
    
    print(f"Final MP3: {output_mp3}")
    print(f"Final WAV: {output_wav}")
    
    return output_wav


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YouTube instrumental and merge with converted vocals")
    parser.add_argument("youtube_url", help="YouTube URL to download instrumental from")
    parser.add_argument("vocals_path", help="Path to converted vocals WAV file")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds before vocals start (default: 0)")
    parser.add_argument("--vocals-gain", type=float, default=0.0, help="Gain in dB to apply to vocals, positive = louder (default: 0)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if file already exists")
    parser.add_argument("--output", type=str, default="final_mix", help="Output filename prefix (default: final_mix)")
    args = parser.parse_args()
    
    converted_vocals_path = Path(args.vocals_path)
    
    if not converted_vocals_path.exists():
        print(f"Error: Converted vocals file not found: {converted_vocals_path}")
        sys.exit(1)
    
    # Step 1: Download YouTube audio (instrumental/accompaniment)
    downloaded_wav = download_audio(args.youtube_url, skip_existing=args.skip_download)
    
    # Step 2: Mix downloaded instrumental with converted vocals
    merge_with_converted_vocals(downloaded_wav, converted_vocals_path, output_name=args.output, vocals_delay_sec=args.delay, vocals_gain_db=args.vocals_gain)
    
    print(f"\nDone! Check the output/ directory for {args.output}.mp3 and {args.output}.wav")


if __name__ == "__main__":
    main()
