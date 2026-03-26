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
    if req.exists() and getattr(config, "RVC_INSTALL_REQS", True):
        log.info("Installing RVC requirements …")
        # Downgrade pip to <24.1 to avoid metadata issues with omegaconf<2.1 (required by fairseq)
        subprocess.run([sys.executable, "-m", "pip", "install", "pip<24.1"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)
    elif req.exists():
        log.info("Skipping RVC requirements install (RVC_INSTALL_REQS=False).")

    # Train
    try:
        # Slice into segments for rvc_python method
        # Organize sliced files by sample rate to avoid clearing when switching rates
        sliced_dir = config.INTERMEDIATE_DIR / "sliced" / str(config.RVC_SAMPLE_RATE)
        sliced_dir.mkdir(parents=True, exist_ok=True)
        if getattr(config, "RVC_SLICE_AUDIO", True):
            if not any(sliced_dir.glob("*.wav")):
                _slice_audio(
                    config.USER_VOICE_FILE,
                    sliced_dir,
                    seg_len=10.0,
                    sr=config.RVC_SAMPLE_RATE,
                )
        else:
            log.info("Skipping audio slicing (RVC_SLICE_AUDIO=False).")
            if not any(sliced_dir.glob("*.wav")):
                raise FileNotFoundError(
                    f"No sliced wav files found under {sliced_dir} while "
                    "RVC_SLICE_AUDIO=False. "
                    "Please enable slicing once or place pre-sliced wav files there."
                )
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


def _generate_filelist(exp_dir, rvc_version, sample_rate):
    """Generate filelist.txt required by train.py.
    
    This combines ground truth wavs, features, and F0 files into a training manifest.
    Based on infer-web.py logic.
    """
    import random
    
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / f"3_feature{'256' if rvc_version == 'v1' else '768'}"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"
    
    # Get all files that exist in both gt_wavs and feature directories
    if not gt_wavs_dir.exists():
        raise FileNotFoundError(f"Ground truth wavs directory not found: {gt_wavs_dir}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    
    gt_names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir) if name.endswith(".wav")])
    feature_names = set([name.split(".")[0] for name in os.listdir(feature_dir) if name.endswith(".npy")])
    names = sorted(list(gt_names & feature_names))
    
    if not names:
        raise FileNotFoundError(
            f"No matching files found between {gt_wavs_dir} and {feature_dir}. "
            "Please check preprocessing and feature extraction steps."
        )
    
    log.info(f"Found {len(names)} valid training samples")
    
    opt = []
    spk_id = "0"  # Single speaker
    
    # Format: gt_wav|feature|f0|f0nsf|spk_id (when f0 is enabled)
    for name in names:
        opt.append(
            f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
        )
    
    # Add mute samples for padding (2 samples as in original RVC)
    rd = config.RVC_REPO_DIR
    sr_name = f"{sample_rate // 1000}k"
    fea_dim = 256 if rvc_version == "v1" else 768
    for _ in range(2):
        opt.append(
            f"{rd}/logs/mute/0_gt_wavs/mute{sr_name}.wav|"
            f"{rd}/logs/mute/3_feature{fea_dim}/mute.npy|"
            f"{rd}/logs/mute/2a_f0/mute.wav.npy|"
            f"{rd}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id}"
        )
    
    random.shuffle(opt)
    
    filelist_path = exp_dir / "filelist.txt"
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
    
    log.info(f"Generated filelist.txt with {len(opt)} entries at {filelist_path}")


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


def _train_rvc_repo():
    rd = config.RVC_REPO_DIR
    exp = rd / "logs" / config.RVC_MODEL_NAME
    exp.mkdir(parents=True, exist_ok=True)
    gt = exp / "0_gt_wavs"
    
    # Determine version based on sample rate (40k only has v1 config)
    sr = config.RVC_SAMPLE_RATE
    if sr == 40000:
        rvc_version = "v1"
        config_template = rd / "configs" / "v1" / "40k.json"
    else:
        rvc_version = "v2"
        config_template = rd / "configs" / "v2" / f"{sr // 1000}k.json"
        if not config_template.exists():
            rvc_version = "v1"
            config_template = rd / "configs" / "v1" / f"{sr // 1000}k.json"

    # Check if we need to reprocess audio (sample rate changed or no existing segments)
    config_json = exp / "config.json"
    need_reprocess = True
    if config_json.exists() and gt.exists() and any(gt.glob("*.wav")):
        import json
        try:
            with open(config_json, "r") as f:
                existing_config = json.load(f)
            existing_sr = existing_config.get("data", {}).get("sampling_rate")
            if existing_sr == sr:
                need_reprocess = False
                log.info(f"Sample rate unchanged ({sr} Hz) and audio segments exist — skipping preprocessing")
            else:
                log.info(f"Sample rate changed from {existing_sr} Hz to {sr} Hz — reprocessing required")
        except Exception:
            pass
    
    if need_reprocess:
        # Clear existing segments to prevent accumulation across runs
        if gt.exists():
            for f in gt.glob("*"):
                if f.is_file():
                    f.unlink()
        gt.mkdir(exist_ok=True)
        
        # Copy original audio file directly (RVC handles format conversion internally)
        original_audio = config.USER_VOICE_FILE
        shutil.copy2(original_audio, gt / original_audio.name)

    # Log original audio information
    log.info("=" * 60)
    log.info("Audio Source Check")
    log.info("=" * 60)
    log.info(f"Original audio file: {config.USER_VOICE_FILE}")
    original_sr = _detect_audio_sr(config.USER_VOICE_FILE)
    if original_sr:
        log.info(f"Original audio sample rate: {original_sr} Hz")
    else:
        log.warning("Could not detect original audio sample rate")
    log.info(f"Target RVC sample rate: {config.RVC_SAMPLE_RATE} Hz")
    if original_sr and original_sr != config.RVC_SAMPLE_RATE:
        log.warning(f"Sample rate mismatch! Audio will be resampled from {original_sr} Hz to {config.RVC_SAMPLE_RATE} Hz")
    log.info("=" * 60)

    # Log CUDA/GPU information
    log.info("=" * 60)
    log.info("GPU Configuration Check")
    log.info("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            log.info(f"GPU count: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / 1024**3
                mem_free = torch.cuda.memory_reserved(i) / 1024**3
                log.info(f"  GPU {i}: {props.name}, {mem_total:.1f} GB total")
            # Use configured GPU device
            cuda_device = getattr(config, "RVC_CUDA_DEVICE", "0")
            if cuda_device == "auto":
                cuda_device = "0"
            log.info(f"Using CUDA device: {cuda_device}")
            # Set CUDA_VISIBLE_DEVICES for training
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        else:
            log.warning("CUDA not available - training will be VERY slow on CPU!")
    except Exception as e:
        log.warning(f"Could not query GPU info: {e}")
    log.info("=" * 60)

    # Determine pretrained model paths based on version and sample rate
    sr_name = f"{sr // 1000}k"
    if rvc_version == "v2":
        pretrained_dir = rd / "assets" / "pretrained_v2"
    else:
        pretrained_dir = rd / "assets" / "pretrained"
    pretrained_g = pretrained_dir / f"f0G{sr_name}.pth"
    pretrained_d = pretrained_dir / f"f0D{sr_name}.pth"

    rvc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def _run(script_rel, *args, hint=None):
        s = rd / script_rel
        if s.exists():
            if hint:
                log.info(hint)
            # Force line-buffered output for real-time logging
            env = {**rvc_env, "PYTHONUNBUFFERED": "1"}
            subprocess.run(
                [sys.executable, "-u", str(s), *args],
                cwd=str(rd),
                check=True,
                env=env,
            )
        else:
            log.warning(f"Script not found: {s}")

    if need_reprocess:
        # Create/overwrite config.json when reprocessing (ensures sample rate consistency)
        if config_template.exists():
            import json
            with open(config_template, "r") as f:
                config_data = json.load(f)
            # Fix sample rate mismatch: ensure config matches target sample rate
            if config_data.get("data", {}).get("sampling_rate") != sr:
                original_sr = config_data.get("data", {}).get("sampling_rate")
                config_data["data"]["sampling_rate"] = sr
                log.info(f"Adjusted config sampling_rate from {original_sr} to {sr}")
            with open(config_json, "w") as f:
                json.dump(config_data, f, indent=4)
            log.info(f"Created config.json from {config_template}")
        else:
            raise FileNotFoundError(f"Config template not found: {config_template}")

        _run(
            "infer/modules/train/preprocess.py",
            str(gt),
            str(config.RVC_SAMPLE_RATE),
            "4",
            str(exp),
            "False",
            "3.7",
            hint=f"RVC: preprocessing audio slices (sample_rate={config.RVC_SAMPLE_RATE}) …",
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
            "cuda",  # GPU device - uses CUDA_VISIBLE_DEVICES set above
            "1",
            "0",
            str(exp),
            rvc_version,
            "true",
            hint="RVC: extracting HuBERT features (GPU if available; may be slow on CPU) …",
        )

    # Generate filelist.txt required by train.py
    _generate_filelist(exp, rvc_version, config.RVC_SAMPLE_RATE)

    # Convert sample rate to RVC format (e.g., 48000 -> "48k")
    sr_for_rvc = f"{config.RVC_SAMPLE_RATE // 1000}k"
    
    _run(
        "infer/modules/train/train.py",
        "-e",
        config.RVC_MODEL_NAME,
        "-sr",
        sr_for_rvc,
        "-bs",
        str(config.RVC_BATCH_SIZE),
        "-te",
        str(config.RVC_TRAINING_EPOCHS),
        "-se",
        "50",
        "-pg",
        str(pretrained_g),
        "-pd",
        str(pretrained_d),
        "-l",
        "0",
        "-v",
        rvc_version,
        "-f0",
        "1",
        "-c",
        "0",
        hint=f"RVC: training (epochs={config.RVC_TRAINING_EPOCHS}, batch_size={config.RVC_BATCH_SIZE}) — longest step …",
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
