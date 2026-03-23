#!/usr/bin/env python3
"""
Hamilton "My Shot" Voice Conversion Pipeline
=============================================
Automated pipeline:
  1. Download Hamilton's "My Shot" from YouTube
  2. Separate vocals / accompaniment via Demucs
  3. Train an RVC model on user's voice (skipped if VOICE_CONVERSION_BACKEND=passthrough)
  4. Convert separated vocals to user's timbre (RVC, or passthrough copy)
  5. Mix converted vocals + accompaniment → final song

RVC assets: run rvc_workspace/.../tools/download_models.py (defaults to minimal set only).
Install huggingface_hub for resumable downloads (requirements.txt). Use VOICE_CONVERSION_BACKEND=passthrough to skip RVC.

Usage:
    python pipeline.py --all
    python pipeline.py --download
    python pipeline.py --separate
    python pipeline.py --train
    python pipeline.py --convert
    python pipeline.py --mix
"""

import subprocess, sys, os, shutil, logging, argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("myshot")

sys.path.insert(0, str(Path(__file__).parent))
import config


# ── STEP 1  Download ────────────────────────────────────────────────────────
def step_download(cookies_from_browser=None, cookies_file=None):
    """Download Hamilton 'My Shot' from YouTube → MP3 + WAV."""
    log.info("=" * 60)
    log.info("STEP 1: Downloading Hamilton 'My Shot'")
    log.info("=" * 60)

    if config.DOWNLOADED_MP3.exists() and config.DOWNLOADED_WAV.exists():
        log.info(f"Already downloaded: {config.DOWNLOADED_MP3}  — skipping.")
        return

    url = config.YOUTUBE_URL or f"ytsearch1:{config.YOUTUBE_SEARCH_QUERY}"
    log.info(f"Source: {url}")

    mp3_tpl = str(config.DOWNLOADED_MP3).replace(".mp3", ".%(ext)s")
    cmd = [
        "yt-dlp", "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "0", "--output", mp3_tpl, "--no-playlist",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
        log.info(f"Using cookies from browser: {cookies_from_browser}")
    elif cookies_file:
        cmd.extend(["--cookies", cookies_file])
        log.info(f"Using cookies file: {cookies_file}")
    cmd.append(url)
    subprocess.run(cmd, check=True)
    log.info(f"MP3 saved: {config.DOWNLOADED_MP3}")

    if not config.DOWNLOADED_WAV.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(config.DOWNLOADED_MP3),
            "-ar", "44100", "-ac", "2", str(config.DOWNLOADED_WAV),
        ], check=True)
    log.info("STEP 1 COMPLETE")


# ── STEP 2  Vocal separation (Demucs) ───────────────────────────────────────
def step_separate():
    """Separate vocals + accompaniment using Demucs."""
    log.info("=" * 60)
    log.info("STEP 2: Vocal separation (Demucs)")
    log.info("=" * 60)

    if config.SEPARATED_VOCALS.exists() and config.SEPARATED_ACCOMPANIMENT.exists():
        log.info("Separated tracks exist — skipping.")
        return

    # Determine input file (WAV preferred, MP3 fallback)
    if config.DOWNLOADED_WAV.exists():
        input_file = config.DOWNLOADED_WAV
    elif config.DOWNLOADED_MP3.exists():
        input_file = config.DOWNLOADED_MP3
        log.info(f"Using MP3 file: {input_file}")
    else:
        raise FileNotFoundError(f"Neither {config.DOWNLOADED_WAV} nor {config.DOWNLOADED_MP3} found. Run --download first.")

    demucs_out = config.INTERMEDIATE_DIR / "demucs_output"
    device = config.DEMUCS_DEVICE
    if device == "auto":
        device = config.get_device()

    cmd = [
        sys.executable, "-m", "demucs",
        "--name", config.DEMUCS_MODEL,
        "--out", str(demucs_out),
        "--shifts", str(config.DEMUCS_SHIFTS),
        "--overlap", str(config.DEMUCS_OVERLAP),
    ]
    if config.DEMUCS_TWO_STEMS:
        cmd += ["--two-stems", "vocals"]
    cmd += ["--device", device, str(input_file)]

    log.info(f"Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0 and device != "cpu":
        log.warning("Retrying with CPU …")
        cmd[-2] = "cpu"
        subprocess.run(cmd, check=True)

    stem = input_file.stem
    result_dir = demucs_out / config.DEMUCS_MODEL / stem
    shutil.copy2(result_dir / "vocals.wav", config.SEPARATED_VOCALS)
    shutil.copy2(result_dir / "no_vocals.wav", config.SEPARATED_ACCOMPANIMENT)
    log.info(f"Vocals → {config.SEPARATED_VOCALS}")
    log.info(f"Accompaniment → {config.SEPARATED_ACCOMPANIMENT}")
    log.info("STEP 2 COMPLETE")


# ── STEP 3  Train RVC model ─────────────────────────────────────────────────
def step_train():
    """Train a voice model (RVC) or skip when using passthrough backend."""
    log.info("=" * 60)
    log.info("STEP 3: Train voice model")
    log.info("=" * 60)

    if getattr(config, "VOICE_CONVERSION_BACKEND", "rvc") == "passthrough":
        log.info("VOICE_CONVERSION_BACKEND=passthrough — skipping training (no timbre model).")
        log.info("STEP 3 COMPLETE")
        return

    if not config.USER_VOICE_FILE.exists():
        msg = (
            f"\nVoice file not found: {config.USER_VOICE_FILE}\n\n"
            "Recording guidance:\n"
            "  • Record 10-15 min of speaking voice (varied intonation & emotion)\n"
            "  • Quiet room, close mic, minimal reverb\n"
            "  • No need to sing — RVC handles conversion from speech\n"
            "  • Save as M4A/WAV 44.1 kHz, place at the path above\n"
        )
        log.error(msg)
        raise FileNotFoundError(msg)

    # Check RVC repo exists (manual download required due to network restrictions)
    if not config.RVC_REPO_DIR.exists():
        raise FileNotFoundError(
            f"\nRVC repository not found: {config.RVC_REPO_DIR}\n\n"
            "Please manually download the RVC repository:\n"
            "  1. On a machine with internet access, run:\n"
            "     git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git\n"
            "  2. Copy the 'Retrieval-based-Voice-Conversion-WebUI' folder to:\n"
            f"     {config.RVC_REPO_DIR}\n"
            "  3. Then re-run this script."
        )
    log.info(f"Using existing RVC repository: {config.RVC_REPO_DIR}")
    req = config.RVC_REPO_DIR / "requirements.txt"
    if req.exists():
        log.info("Installing RVC requirements …")
        # Downgrade pip to <24.1 to avoid metadata issues with omegaconf<2.1 (required by fairseq)
        subprocess.run([sys.executable, "-m", "pip", "install", "pip<24.1"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)

    # Train
    try:
        # Slice into segments for rvc_python method
        sliced_dir = config.INTERMEDIATE_DIR / "sliced"
        sliced_dir.mkdir(exist_ok=True)
        if not any(sliced_dir.glob("*.wav")):
            _slice_audio(config.USER_VOICE_FILE, sliced_dir, seg_len=10.0, sr=config.RVC_SAMPLE_RATE)
        _train_rvc_python(sliced_dir)
    except ImportError:
        # Use original audio file directly for RVC repo method (preserves quality)
        _train_rvc_repo()

    log.info("STEP 3 COMPLETE")


def _slice_audio(src, dst, seg_len=10.0, sr=40000):
    import soundfile as sf, numpy as np, tempfile, os
    
    # Convert to WAV if input is not WAV (e.g., M4A files)
    src_path = Path(src)
    if src_path.suffix.lower() != ".wav":
        log.info(f"Converting {src_path.suffix} to WAV for processing...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(src),
                "-ar", str(sr), "-ac", "1", "-f", "wav", tmp_wav
            ], check=True, capture_output=True)
            data, file_sr = sf.read(tmp_wav)
        finally:
            os.unlink(tmp_wav)
    else:
        data, file_sr = sf.read(str(src))
        if file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = int(seg_len * sr)
    for i in range(len(data) // n):
        sf.write(str(dst / f"seg_{i:04d}.wav"), data[i*n:(i+1)*n], sr)
    tail = data[(len(data)//n)*n:]
    if len(tail) > sr * 2:
        sf.write(str(dst / f"seg_{len(data)//n:04d}.wav"), tail, sr)
    log.info(f"Sliced into {len(list(dst.glob('*.wav')))} segments")


def _train_rvc_python(sliced_dir):
    from rvc_python import RVC
    rvc = RVC()
    rvc.train(
        dataset_path=str(sliced_dir),
        model_name=config.RVC_MODEL_NAME,
        sample_rate=config.RVC_SAMPLE_RATE,
        f0_method=config.RVC_F0_METHOD,
        epochs=config.RVC_TRAINING_EPOCHS,
        batch_size=config.RVC_BATCH_SIZE,
    )


def _train_rvc_repo():
    rd = config.RVC_REPO_DIR
    exp = rd / "logs" / config.RVC_MODEL_NAME
    exp.mkdir(parents=True, exist_ok=True)
    gt = exp / "0_gt_wavs"; gt.mkdir(exist_ok=True)
    # Copy original audio file directly (RVC handles format conversion internally)
    original_audio = config.USER_VOICE_FILE
    shutil.copy2(original_audio, gt / original_audio.name)

    rvc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def _run(script_rel, *args, hint=None):
        s = rd / script_rel
        if s.exists():
            if hint:
                log.info(hint)
            subprocess.run(
                [sys.executable, str(s), *args],
                cwd=str(rd),
                check=True,
                env=rvc_env,
            )
        else:
            log.warning(f"Script not found: {s}")

    _run(
        "infer/modules/train/preprocess.py",
        str(gt),
        str(config.RVC_SAMPLE_RATE),
        "4",
        str(exp),
        "False",
        "3.7",
        hint="RVC: preprocessing audio slices …",
    )
    _run(
        "infer/modules/train/extract/extract_f0_print.py",
        str(exp),
        "4",
        config.RVC_F0_METHOD,
        hint=(
            "RVC: extracting F0 (rmvpe loads a large model per worker on CPU — "
            "often several minutes with little terminal output). "
            f"Tail log: {exp / 'extract_f0_feature.log'}"
        ),
    )
    _run(
        "infer/modules/train/extract_feature_print.py",
        "cuda",
        "1",
        "0",
        str(exp),
        "v2",
        "true",
        hint="RVC: extracting HuBERT features (GPU if available; may be slow on CPU) …",
    )
    _run(
        "infer/modules/train/train.py",
        "-e",
        config.RVC_MODEL_NAME,
        "-sr",
        str(config.RVC_SAMPLE_RATE),
        "-bs",
        str(config.RVC_BATCH_SIZE),
        "-te",
        str(config.RVC_TRAINING_EPOCHS),
        "-se",
        "50",
        "-pg",
        str(rd / "assets/pretrained/f0G40k.pth"),
        "-pd",
        str(rd / "assets/pretrained/f0D40k.pth"),
        "-l",
        "0",
        "-v",
        "v2",
        "-f0",
        "1",
        "-c",
        "0",
        hint=f"RVC: training ({config.RVC_TRAINING_EPOCHS} epochs) — longest step …",
    )

    for pat, dst in [("G_*.pth", config.RVC_TRAINED_MODEL), ("*.index", config.RVC_TRAINED_INDEX)]:
        files = sorted(exp.glob(pat), key=lambda p: p.stat().st_mtime)
        if files:
            shutil.copy2(files[-1], dst)
            log.info(f"Saved: {dst}")


# ── STEP 4  Voice conversion (inference) ────────────────────────────────────
def step_convert():
    """Convert separated vocals to user's timbre (RVC) or copy through (passthrough)."""
    log.info("=" * 60)
    log.info("STEP 4: Voice conversion")
    log.info("=" * 60)

    if config.CONVERTED_VOCALS.exists():
        log.info("Converted vocals exist — skipping.")
        return
    if not config.SEPARATED_VOCALS.exists():
        raise FileNotFoundError("Run --separate first.")

    if getattr(config, "VOICE_CONVERSION_BACKEND", "rvc") == "passthrough":
        log.info("VOICE_CONVERSION_BACKEND=passthrough — copying separated vocals (no RVC).")
        shutil.copy2(config.SEPARATED_VOCALS, config.CONVERTED_VOCALS)
        log.info(f"Vocals (unchanged timbre) → {config.CONVERTED_VOCALS}")
        log.info("STEP 4 COMPLETE")
        return

    if not config.RVC_TRAINED_MODEL.exists():
        raise FileNotFoundError("Run --train first.")

    try:
        _convert_rvc_python()
    except ImportError:
        _convert_rvc_repo()

    log.info(f"Converted vocals → {config.CONVERTED_VOCALS}")
    log.info("STEP 4 COMPLETE")


def _convert_rvc_python():
    from rvc_python import RVC
    rvc = RVC(model_path=str(config.RVC_TRAINED_MODEL))
    rvc.convert(
        input_path=str(config.SEPARATED_VOCALS),
        output_path=str(config.CONVERTED_VOCALS),
        f0_method=config.RVC_F0_METHOD,
        f0_up_key=config.RVC_TRANSPOSE,
        index_path=str(config.RVC_TRAINED_INDEX) if config.RVC_TRAINED_INDEX.exists() else None,
        index_rate=config.RVC_INDEX_RATE,
        filter_radius=config.RVC_FILTER_RADIUS,
        rms_mix_rate=config.RVC_RMS_MIX_RATE,
        protect=config.RVC_PROTECT,
    )


def _convert_rvc_repo():
    rd = config.RVC_REPO_DIR
    infer_cli = rd / "tools" / "infer_cli.py"
    if not infer_cli.exists():
        # try alternative path
        infer_cli = rd / "infer" / "modules" / "vc" / "pipeline.py"

    cmd = [
        sys.executable, str(infer_cli),
        "--model_path", str(config.RVC_TRAINED_MODEL),
        "--input_path", str(config.SEPARATED_VOCALS),
        "--output_path", str(config.CONVERTED_VOCALS),
        "--f0_method", config.RVC_F0_METHOD,
        "--transpose", str(config.RVC_TRANSPOSE),
        "--index_rate", str(config.RVC_INDEX_RATE),
        "--filter_radius", str(config.RVC_FILTER_RADIUS),
        "--rms_mix_rate", str(config.RVC_RMS_MIX_RATE),
        "--protect", str(config.RVC_PROTECT),
    ]
    if config.RVC_TRAINED_INDEX.exists():
        cmd += ["--index_path", str(config.RVC_TRAINED_INDEX)]
    subprocess.run(
        cmd,
        cwd=str(rd),
        check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


# ── STEP 5  Mix & master ────────────────────────────────────────────────────
def step_mix():
    """Mix converted vocals + accompaniment into the final track."""
    log.info("=" * 60)
    log.info("STEP 5: Mixing final output")
    log.info("=" * 60)

    if not config.CONVERTED_VOCALS.exists():
        raise FileNotFoundError("Run --convert first.")
    if not config.SEPARATED_ACCOMPANIMENT.exists():
        raise FileNotFoundError("Run --separate first.")

    from pydub import AudioSegment

    vocals = AudioSegment.from_wav(str(config.CONVERTED_VOCALS))
    accomp = AudioSegment.from_wav(str(config.SEPARATED_ACCOMPANIMENT))

    # Volume adjustments
    vocals = vocals + config.VOCAL_VOLUME_ADJUST_DB
    accomp = accomp + config.ACCOMPANIMENT_VOLUME_ADJUST_DB

    # Match lengths (pad shorter with silence)
    if len(vocals) < len(accomp):
        vocals += AudioSegment.silent(duration=len(accomp) - len(vocals))
    elif len(accomp) < len(vocals):
        accomp += AudioSegment.silent(duration=len(vocals) - len(accomp))

    # Overlay vocals on accompaniment
    final = accomp.overlay(vocals)

    # Normalize
    if config.NORMALIZE_OUTPUT:
        target_dbfs = -1.0
        change = target_dbfs - final.dBFS
        final = final + change

    # Export
    final.export(str(config.OUTPUT_MP3), format="mp3", bitrate=config.OUTPUT_MP3_BITRATE)
    final.export(str(config.OUTPUT_WAV), format="wav")

    log.info(f"Final MP3 → {config.OUTPUT_MP3}")
    log.info(f"Final WAV → {config.OUTPUT_WAV}")
    log.info("STEP 5 COMPLETE — Enjoy your song!")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Hamilton 'My Shot' Voice Conversion Pipeline")
    p.add_argument("--all",       action="store_true", help="Run full pipeline (steps 1-5)")
    p.add_argument("--download",  action="store_true", help="Step 1: Download song")
    p.add_argument("--separate",  action="store_true", help="Step 2: Separate vocals")
    p.add_argument("--train",     action="store_true", help="Step 3: Train voice model (RVC; skipped if passthrough)")
    p.add_argument("--convert",   action="store_true", help="Step 4: Convert vocals (or passthrough copy)")
    p.add_argument("--mix",       action="store_true", help="Step 5: Mix final output")
    p.add_argument("--cookies-from-browser", type=str, metavar="BROWSER",
                   help="Browser to extract cookies from for YouTube authentication (e.g., chrome, safari, firefox)")
    p.add_argument("--cookies-file", type=str, metavar="FILE",
                   help="Path to cookies file for YouTube authentication (exported from browser extension)")
    args = p.parse_args()

    if not any(v for k, v in vars(args).items() if k not in ("cookies_from_browser", "cookies_file")):
        p.print_help()
        return

    steps = []
    if args.all or args.download:  steps.append(("Download",  lambda: step_download(args.cookies_from_browser, args.cookies_file)))
    if args.all or args.separate:  steps.append(("Separate",  step_separate))
    if args.all or args.train:     steps.append(("Train",     step_train))
    if args.all or args.convert:   steps.append(("Convert",   step_convert))
    if args.all or args.mix:       steps.append(("Mix",       step_mix))

    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            log.error(f"Step '{name}' failed: {e}")
            raise


if __name__ == "__main__":
    main()
