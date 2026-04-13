#!/usr/bin/env python3
"""
Quantitative Evaluation Module for Singing Voice Conversion
============================================================
Computes objective metrics to evaluate the quality of voice conversion output.

Metrics implemented:
  1. MCD (Mel-Cepstral Distortion) — spectral distance between converted and reference
  2. F0-RMSE / F0-Corr — pitch accuracy of the conversion
  3. Speaker Cosine Similarity — how close the converted voice timbre is to the target speaker
  4. PESQ (Perceptual Evaluation of Speech Quality) — perceptual quality score
  5. SDR (Signal-to-Distortion Ratio) — separation quality (for Demucs output)
  6. SNR (Signal-to-Noise Ratio) — overall signal quality
  7. Spectral Convergence — frequency domain fidelity
  8. F0 Voicing Decision Accuracy — unvoiced/voiced classification correctness

Usage:
    # Evaluate a single converted file against reference
    python evaluate.py --converted output/vocals_converted.wav --reference input/vocals_original.wav --target-speaker input/myshot.m4a

    # Full pipeline comparison (with-separation vs without-separation)
    python evaluate.py --compare-pipelines --exp-dir ./exp/20260328_143000

    # Evaluate chorus handling quality
    python evaluate.py --eval-chorus --converted output/final_mix.wav --chorus-map chorus_segments.json
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. Mel-Cepstral Distortion (MCD)
# ==============================================================================
def compute_mcd(reference_wav, converted_wav, sr=16000, n_mfcc=13, frame_length=0.025, frame_shift=0.010):
    """
    Compute Mel-Cepstral Distortion (MCD) between reference and converted audio.

    MCD measures the spectral distance in the cepstral domain. Lower values indicate
    higher similarity. Typical good SVC results achieve MCD < 6.0 dB.

    Formula: MCD = (10 * sqrt(2) / ln(10)) * mean(||mc_ref - mc_conv||_2)

    Args:
        reference_wav: Path to reference audio file
        converted_wav: Path to converted audio file
        sr: Sample rate for analysis (default 16kHz for fair comparison)
        n_mfcc: Number of MFCC coefficients (excluding c0)
        frame_length: Frame length in seconds
        frame_shift: Frame shift (hop) in seconds

    Returns:
        dict with 'mcd_db' (float), 'mcd_frames' (np.array per-frame MCD)
    """
    import librosa

    ref, _ = librosa.load(str(reference_wav), sr=sr, mono=True)
    conv, _ = librosa.load(str(converted_wav), sr=sr, mono=True)

    # Align lengths (DTW or simple truncation)
    min_len = min(len(ref), len(conv))
    ref = ref[:min_len]
    conv = conv[:min_len]

    n_fft = int(frame_length * sr)
    hop_length = int(frame_shift * sr)

    # Compute MFCCs (exclude c0 energy coefficient)
    mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc + 1, n_fft=n_fft, hop_length=hop_length)[1:]
    mfcc_conv = librosa.feature.mfcc(y=conv, sr=sr, n_mfcc=n_mfcc + 1, n_fft=n_fft, hop_length=hop_length)[1:]

    # Align frame count
    min_frames = min(mfcc_ref.shape[1], mfcc_conv.shape[1])
    mfcc_ref = mfcc_ref[:, :min_frames]
    mfcc_conv = mfcc_conv[:, :min_frames]

    # Per-frame Euclidean distance
    diff = mfcc_ref - mfcc_conv
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))

    # MCD coefficient: 10 * sqrt(2) / ln(10) ≈ 6.1415
    alpha = 10.0 * np.sqrt(2.0) / np.log(10.0)
    mcd_per_frame = alpha * frame_dist
    mcd_mean = float(np.mean(mcd_per_frame))

    return {"mcd_db": mcd_mean, "mcd_frames": mcd_per_frame}


# ==============================================================================
# 2. F0 Accuracy Metrics (RMSE, Correlation, VDA)
# ==============================================================================
def extract_f0(audio_path, sr=16000, method="rmvpe", hop_ms=10.0):
    """
    Extract fundamental frequency (F0) from audio using specified method.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        method: 'rmvpe', 'crepe', or 'pyin'
        hop_ms: Hop size in milliseconds

    Returns:
        f0: numpy array of F0 values (Hz), 0.0 for unvoiced frames
        voiced: boolean array, True for voiced frames
    """
    import librosa

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    hop_length = int(hop_ms / 1000.0 * sr)

    if method == "pyin":
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=1100, sr=sr, hop_length=hop_length
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = voiced_flag.astype(bool) if voiced_flag is not None else (f0 > 0)
    elif method == "crepe":
        try:
            import crepe
            _, f0, confidence, _ = crepe.predict(y, sr, step_size=hop_ms, viterbi=True)
            voiced = confidence > 0.5
            f0[~voiced] = 0.0
        except ImportError:
            log.warning("crepe not installed, falling back to pyin")
            return extract_f0(audio_path, sr, "pyin", hop_ms)
    elif method == "rmvpe":
        # Try to use RMVPE if available, otherwise fall back
        try:
            from infer.lib.rmvpe import RMVPE
            rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", device="cpu")
            f0 = rmvpe.infer_from_audio(y, thred=0.03)
            voiced = f0 > 0
        except (ImportError, FileNotFoundError):
            log.warning("RMVPE not available, falling back to pyin")
            return extract_f0(audio_path, sr, "pyin", hop_ms)
    else:
        raise ValueError(f"Unknown F0 method: {method}")

    return f0, voiced


def compute_f0_metrics(reference_wav, converted_wav, sr=16000, method="pyin"):
    """
    Compute F0 accuracy metrics between reference and converted audio.

    Metrics:
      - F0-RMSE (Hz): Root mean squared error of F0 in voiced regions
      - F0-RMSE (cents): RMSE in cents (perceptually meaningful)
      - F0-Corr: Pearson correlation of F0 contours
      - VDA: Voicing Decision Accuracy

    Args:
        reference_wav: Path to reference audio
        converted_wav: Path to converted audio

    Returns:
        dict with f0_rmse_hz, f0_rmse_cents, f0_corr, vda
    """
    f0_ref, voiced_ref = extract_f0(reference_wav, sr, method)
    f0_conv, voiced_conv = extract_f0(converted_wav, sr, method)

    # Align lengths
    min_len = min(len(f0_ref), len(f0_conv))
    f0_ref = f0_ref[:min_len]
    f0_conv = f0_conv[:min_len]
    voiced_ref = voiced_ref[:min_len]
    voiced_conv = voiced_conv[:min_len]

    # --- VDA (Voicing Decision Accuracy) ---
    vda = float(np.mean(voiced_ref == voiced_conv))

    # --- F0 metrics in jointly voiced regions ---
    both_voiced = voiced_ref & voiced_conv
    n_voiced = np.sum(both_voiced)

    if n_voiced < 10:
        log.warning(f"Too few jointly voiced frames ({n_voiced}), F0 metrics unreliable")
        return {
            "f0_rmse_hz": float("nan"),
            "f0_rmse_cents": float("nan"),
            "f0_corr": float("nan"),
            "vda": vda,
            "n_voiced_frames": int(n_voiced),
        }

    f0_ref_v = f0_ref[both_voiced]
    f0_conv_v = f0_conv[both_voiced]

    # RMSE in Hz
    f0_rmse_hz = float(np.sqrt(np.mean((f0_ref_v - f0_conv_v) ** 2)))

    # RMSE in cents (1200 * log2(f1/f2))
    with np.errstate(divide="ignore", invalid="ignore"):
        cents_diff = 1200.0 * np.log2(f0_conv_v / f0_ref_v)
        cents_diff = cents_diff[np.isfinite(cents_diff)]
    f0_rmse_cents = float(np.sqrt(np.mean(cents_diff ** 2))) if len(cents_diff) > 0 else float("nan")

    # Pearson correlation
    if np.std(f0_ref_v) > 0 and np.std(f0_conv_v) > 0:
        f0_corr = float(np.corrcoef(f0_ref_v, f0_conv_v)[0, 1])
    else:
        f0_corr = 0.0

    return {
        "f0_rmse_hz": f0_rmse_hz,
        "f0_rmse_cents": f0_rmse_cents,
        "f0_corr": f0_corr,
        "vda": vda,
        "n_voiced_frames": int(n_voiced),
    }


# ==============================================================================
# 3. Speaker Cosine Similarity
# ==============================================================================
def compute_speaker_similarity(converted_wav, target_speaker_wav, model_name="ecapa_tdnn"):
    """
    Compute cosine similarity of speaker embeddings between converted audio
    and the target speaker reference.

    Higher values (closer to 1.0) indicate the converted voice sounds more
    like the target speaker. Typical good SVC results: > 0.75.

    Args:
        converted_wav: Path to converted audio
        target_speaker_wav: Path to target speaker reference audio
        model_name: Speaker embedding model ('ecapa_tdnn' or 'resemblyzer')

    Returns:
        dict with 'cosine_similarity' (float), 'embedding_model' (str)
    """
    if model_name == "resemblyzer":
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav

            encoder = VoiceEncoder()
            wav_conv = preprocess_wav(str(converted_wav))
            wav_target = preprocess_wav(str(target_speaker_wav))
            emb_conv = encoder.embed_utterance(wav_conv)
            emb_target = encoder.embed_utterance(wav_target)

            cos_sim = float(np.dot(emb_conv, emb_target) / (
                np.linalg.norm(emb_conv) * np.linalg.norm(emb_target)
            ))
            return {"cosine_similarity": cos_sim, "embedding_model": "resemblyzer"}
        except ImportError:
            log.warning("resemblyzer not installed, falling back to MFCC-based similarity")
            model_name = "mfcc"

    if model_name == "ecapa_tdnn":
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
            emb_conv = classifier.encode_batch(
                classifier.load_audio(str(converted_wav)).unsqueeze(0)
            ).squeeze().numpy()
            emb_target = classifier.encode_batch(
                classifier.load_audio(str(target_speaker_wav)).unsqueeze(0)
            ).squeeze().numpy()

            cos_sim = float(np.dot(emb_conv, emb_target) / (
                np.linalg.norm(emb_conv) * np.linalg.norm(emb_target)
            ))
            return {"cosine_similarity": cos_sim, "embedding_model": "ecapa_tdnn"}
        except ImportError:
            log.warning("speechbrain not installed, falling back to MFCC-based similarity")
            model_name = "mfcc"

    # MFCC-based fallback (lightweight, less accurate)
    if model_name == "mfcc":
        import librosa

        y_conv, sr = librosa.load(str(converted_wav), sr=16000, mono=True)
        y_target, _ = librosa.load(str(target_speaker_wav), sr=16000, mono=True)

        mfcc_conv = np.mean(librosa.feature.mfcc(y=y_conv, sr=sr, n_mfcc=20), axis=1)
        mfcc_target = np.mean(librosa.feature.mfcc(y=y_target, sr=sr, n_mfcc=20), axis=1)

        cos_sim = float(np.dot(mfcc_conv, mfcc_target) / (
            np.linalg.norm(mfcc_conv) * np.linalg.norm(mfcc_target)
        ))
        return {"cosine_similarity": cos_sim, "embedding_model": "mfcc_fallback"}

    raise ValueError(f"Unknown speaker embedding model: {model_name}")


# ==============================================================================
# 4. PESQ (Perceptual Evaluation of Speech Quality)
# ==============================================================================
def compute_pesq(reference_wav, converted_wav, sr=16000):
    """
    Compute PESQ score between reference and converted audio.

    PESQ range: -0.5 to 4.5 (higher is better).
    Typical SVC results: 2.5 - 3.5.

    Note: PESQ is designed for speech, not singing. Results should be
    interpreted as approximate perceptual quality indicators.

    Args:
        reference_wav: Path to reference (clean) audio
        converted_wav: Path to converted (degraded) audio
        sr: Target sample rate (PESQ requires 8000 or 16000)

    Returns:
        dict with 'pesq_wb' (wideband score) and/or 'pesq_nb' (narrowband)
    """
    try:
        from pesq import pesq
        import librosa

        ref, _ = librosa.load(str(reference_wav), sr=sr, mono=True)
        conv, _ = librosa.load(str(converted_wav), sr=sr, mono=True)

        # Align lengths
        min_len = min(len(ref), len(conv))
        ref = ref[:min_len]
        conv = conv[:min_len]

        if sr == 16000:
            score = pesq(sr, ref, conv, "wb")
            return {"pesq_wb": float(score)}
        elif sr == 8000:
            score = pesq(sr, ref, conv, "nb")
            return {"pesq_nb": float(score)}
        else:
            raise ValueError(f"PESQ requires sr=8000 or sr=16000, got {sr}")

    except ImportError:
        log.warning("pesq library not installed (pip install pesq). Skipping PESQ.")
        return {"pesq_wb": None, "pesq_nb": None}
    except Exception as e:
        log.warning(f"PESQ computation failed: {e}")
        return {"pesq_wb": None, "pesq_nb": None}


# ==============================================================================
# 5. SDR (Signal-to-Distortion Ratio) for source separation evaluation
# ==============================================================================
def compute_sdr(reference_wav, estimated_wav, sr=44100):
    """
    Compute SDR (Signal-to-Distortion Ratio) for evaluating source separation quality.

    SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    Higher SDR indicates better separation. State-of-the-art: ~8.5 dB (htdemucs_ft).

    Args:
        reference_wav: Path to clean reference signal (ground truth vocals)
        estimated_wav: Path to estimated/separated signal

    Returns:
        dict with 'sdr_db' (float)
    """
    import librosa

    ref, _ = librosa.load(str(reference_wav), sr=sr, mono=True)
    est, _ = librosa.load(str(estimated_wav), sr=sr, mono=True)

    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    # Compute SDR using the BSS_EVAL definition
    # s_target = <est, ref> * ref / ||ref||^2
    ref_energy = np.sum(ref ** 2)
    if ref_energy < 1e-10:
        return {"sdr_db": float("-inf")}

    # Project estimated onto reference
    alpha = np.dot(est, ref) / ref_energy
    s_target = alpha * ref
    e_noise = est - s_target

    target_energy = np.sum(s_target ** 2)
    noise_energy = np.sum(e_noise ** 2)

    if noise_energy < 1e-10:
        return {"sdr_db": float("inf")}

    sdr = 10.0 * np.log10(target_energy / noise_energy)
    return {"sdr_db": float(sdr)}


# ==============================================================================
# 6. SNR (Signal-to-Noise Ratio)
# ==============================================================================
def compute_snr(audio_wav, sr=16000, silence_threshold_db=-40):
    """
    Estimate SNR of audio by comparing signal energy to noise floor.

    Uses silence detection to estimate noise floor, then computes ratio.

    Args:
        audio_wav: Path to audio file
        sr: Sample rate
        silence_threshold_db: Threshold for silence detection

    Returns:
        dict with 'snr_db' (float)
    """
    import librosa

    y, _ = librosa.load(str(audio_wav), sr=sr, mono=True)

    # Frame-wise energy (in dB)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.sum(frames ** 2, axis=0)

    energy_db = 10 * np.log10(frame_energy + 1e-10)
    max_energy_db = np.max(energy_db)

    # Classify frames as signal or noise
    threshold = max_energy_db + silence_threshold_db
    is_signal = energy_db > threshold

    if np.sum(is_signal) == 0 or np.sum(~is_signal) == 0:
        return {"snr_db": float("nan")}

    signal_energy = np.mean(frame_energy[is_signal])
    noise_energy = np.mean(frame_energy[~is_signal])

    if noise_energy < 1e-10:
        return {"snr_db": float("inf")}

    snr = 10.0 * np.log10(signal_energy / noise_energy)
    return {"snr_db": float(snr)}


# ==============================================================================
# 7. Spectral Convergence & Log-Spectral Distance
# ==============================================================================
def compute_spectral_metrics(reference_wav, converted_wav, sr=16000, n_fft=1024):
    """
    Compute spectral convergence and log-spectral distance.

    Spectral Convergence (SC): ||S_ref - S_conv||_F / ||S_ref||_F  (lower is better)
    Log-Spectral Distance (LSD): sqrt(mean((log|S_ref| - log|S_conv|)^2))  (lower is better)

    Args:
        reference_wav: Path to reference audio
        converted_wav: Path to converted audio

    Returns:
        dict with 'spectral_convergence', 'log_spectral_distance'
    """
    import librosa

    ref, _ = librosa.load(str(reference_wav), sr=sr, mono=True)
    conv, _ = librosa.load(str(converted_wav), sr=sr, mono=True)

    min_len = min(len(ref), len(conv))
    ref = ref[:min_len]
    conv = conv[:min_len]

    S_ref = np.abs(librosa.stft(ref, n_fft=n_fft))
    S_conv = np.abs(librosa.stft(conv, n_fft=n_fft))

    min_frames = min(S_ref.shape[1], S_conv.shape[1])
    S_ref = S_ref[:, :min_frames]
    S_conv = S_conv[:, :min_frames]

    # Spectral Convergence
    sc = np.linalg.norm(S_ref - S_conv, "fro") / (np.linalg.norm(S_ref, "fro") + 1e-10)

    # Log-Spectral Distance
    log_ref = np.log(S_ref + 1e-10)
    log_conv = np.log(S_conv + 1e-10)
    lsd = np.sqrt(np.mean((log_ref - log_conv) ** 2))

    return {"spectral_convergence": float(sc), "log_spectral_distance": float(lsd)}


# ==============================================================================
# 8. Comprehensive Evaluation
# ==============================================================================
def evaluate_conversion(
    converted_wav,
    reference_wav=None,
    target_speaker_wav=None,
    source_speaker_wav=None,
    compute_all=True,
):
    """
    Run comprehensive evaluation on a converted audio file.

    Args:
        converted_wav: Path to converted vocals
        reference_wav: Path to reference vocals (original vocals before conversion)
        target_speaker_wav: Path to target speaker audio sample
        source_speaker_wav: Path to source speaker audio (for leakage measurement)
        compute_all: If True, compute all available metrics

    Returns:
        dict with all computed metrics
    """
    results = {"files": {"converted": str(converted_wav)}}

    if reference_wav:
        results["files"]["reference"] = str(reference_wav)

        log.info("Computing MCD...")
        results["mcd"] = compute_mcd(reference_wav, converted_wav)
        log.info(f"  MCD = {results['mcd']['mcd_db']:.2f} dB")

        log.info("Computing F0 metrics...")
        results["f0"] = compute_f0_metrics(reference_wav, converted_wav)
        log.info(f"  F0-RMSE = {results['f0']['f0_rmse_hz']:.2f} Hz / {results['f0']['f0_rmse_cents']:.2f} cents")
        log.info(f"  F0-Corr = {results['f0']['f0_corr']:.4f}")
        log.info(f"  VDA = {results['f0']['vda']:.4f}")

        if compute_all:
            log.info("Computing spectral metrics...")
            results["spectral"] = compute_spectral_metrics(reference_wav, converted_wav)
            log.info(f"  SC = {results['spectral']['spectral_convergence']:.4f}")
            log.info(f"  LSD = {results['spectral']['log_spectral_distance']:.4f}")

            log.info("Computing PESQ...")
            results["pesq"] = compute_pesq(reference_wav, converted_wav)
            if results["pesq"].get("pesq_wb") is not None:
                log.info(f"  PESQ-WB = {results['pesq']['pesq_wb']:.3f}")

    if target_speaker_wav:
        results["files"]["target_speaker"] = str(target_speaker_wav)

        log.info("Computing speaker similarity (converted vs target)...")
        results["speaker_sim_target"] = compute_speaker_similarity(converted_wav, target_speaker_wav)
        log.info(f"  Cosine Similarity (target) = {results['speaker_sim_target']['cosine_similarity']:.4f}")

    if source_speaker_wav:
        results["files"]["source_speaker"] = str(source_speaker_wav)

        log.info("Computing speaker similarity (converted vs source, for leakage measurement)...")
        results["speaker_sim_source"] = compute_speaker_similarity(converted_wav, source_speaker_wav)
        log.info(f"  Cosine Similarity (source, leakage) = {results['speaker_sim_source']['cosine_similarity']:.4f}")

    log.info("Computing SNR...")
    results["snr"] = compute_snr(converted_wav)
    log.info(f"  SNR = {results['snr']['snr_db']:.2f} dB")

    return results


# ==============================================================================
# 9. Pipeline Comparison (with-separation vs without-separation)
# ==============================================================================
def compare_pipelines(
    original_song_wav,
    separated_vocals_wav,
    converted_with_sep_wav,
    converted_without_sep_wav,
    target_speaker_wav,
    accompaniment_wav=None,
):
    """
    Compare two pipeline variants:
      Pipeline A: with vocal separation (Demucs → RVC → Mix)
      Pipeline B: without vocal separation (direct RVC on mixed audio → Mix)

    Args:
        original_song_wav: Original mixed song
        separated_vocals_wav: Ground-truth separated vocals
        converted_with_sep_wav: Converted vocals from Pipeline A (with separation)
        converted_without_sep_wav: Converted vocals from Pipeline B (without separation)
        target_speaker_wav: Target speaker reference audio
        accompaniment_wav: Accompaniment track (for SDR evaluation)

    Returns:
        dict with comparison results for both pipelines
    """
    log.info("=" * 60)
    log.info("Pipeline Comparison: With-Separation vs Without-Separation")
    log.info("=" * 60)

    results = {}

    # Evaluate Pipeline A (with separation)
    log.info("\n--- Pipeline A: With Vocal Separation ---")
    results["with_separation"] = evaluate_conversion(
        converted_wav=converted_with_sep_wav,
        reference_wav=separated_vocals_wav,
        target_speaker_wav=target_speaker_wav,
    )

    # Evaluate Pipeline B (without separation)
    log.info("\n--- Pipeline B: Without Vocal Separation ---")
    results["without_separation"] = evaluate_conversion(
        converted_wav=converted_without_sep_wav,
        reference_wav=separated_vocals_wav,
        target_speaker_wav=target_speaker_wav,
    )

    # SDR of Demucs separation (if accompaniment provided)
    if accompaniment_wav and separated_vocals_wav:
        log.info("\nComputing Demucs separation SDR...")
        results["separation_sdr"] = compute_sdr(separated_vocals_wav, separated_vocals_wav)

    # Summary comparison table
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)

    metrics_to_compare = [
        ("MCD (dB) ↓", "mcd", "mcd_db"),
        ("F0-RMSE (Hz) ↓", "f0", "f0_rmse_hz"),
        ("F0-RMSE (cents) ↓", "f0", "f0_rmse_cents"),
        ("F0-Corr ↑", "f0", "f0_corr"),
        ("VDA ↑", "f0", "vda"),
        ("Speaker Sim (target) ↑", "speaker_sim_target", "cosine_similarity"),
        ("SNR (dB) ↑", "snr", "snr_db"),
    ]

    log.info(f"{'Metric':<28} {'With Sep':>12} {'Without Sep':>12} {'Better':>10}")
    log.info("-" * 62)

    for label, group, key in metrics_to_compare:
        val_a = results["with_separation"].get(group, {}).get(key, float("nan"))
        val_b = results["without_separation"].get(group, {}).get(key, float("nan"))

        if val_a is None:
            val_a = float("nan")
        if val_b is None:
            val_b = float("nan")

        # Determine which is better (↓ = lower is better, ↑ = higher is better)
        if "↓" in label:
            better = "A (Sep)" if val_a < val_b else "B (NoSep)"
        else:
            better = "A (Sep)" if val_a > val_b else "B (NoSep)"

        log.info(f"{label:<28} {val_a:>12.4f} {val_b:>12.4f} {better:>10}")

    results["comparison_table"] = metrics_to_compare
    return results


# ==============================================================================
# 10. Chorus Region Evaluation
# ==============================================================================
def evaluate_chorus_handling(
    converted_wav,
    reference_wav,
    chorus_segments,
    target_speaker_wav=None,
):
    """
    Evaluate conversion quality specifically in chorus vs solo regions.

    This helps quantify the known issue of chorus parts causing artifacts
    when multiple voices are processed by the single-speaker RVC model.

    Args:
        converted_wav: Path to full converted audio
        reference_wav: Path to reference vocals
        chorus_segments: List of (start_sec, end_sec) tuples marking chorus regions
        target_speaker_wav: Optional target speaker reference

    Returns:
        dict with separate metrics for solo and chorus regions
    """
    import librosa
    import soundfile as sf
    import tempfile
    import os

    log.info("Evaluating chorus vs solo regions...")

    y_conv, sr = librosa.load(str(converted_wav), sr=16000, mono=True)
    y_ref, _ = librosa.load(str(reference_wav), sr=16000, mono=True)

    min_len = min(len(y_conv), len(y_ref))
    y_conv = y_conv[:min_len]
    y_ref = y_ref[:min_len]

    # Create masks for chorus and solo regions
    chorus_mask = np.zeros(min_len, dtype=bool)
    for start_sec, end_sec in chorus_segments:
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), min_len)
        chorus_mask[start_sample:end_sample] = True

    solo_mask = ~chorus_mask

    results = {"chorus_segments": chorus_segments}

    # Write temporary files for each region
    tmpdir = tempfile.mkdtemp()
    try:
        for region_name, mask in [("solo", solo_mask), ("chorus", chorus_mask)]:
            if np.sum(mask) < sr:  # Less than 1 second
                log.warning(f"Region '{region_name}' too short, skipping")
                results[region_name] = {"error": "region too short"}
                continue

            conv_region = y_conv[mask]
            ref_region = y_ref[mask]

            conv_path = os.path.join(tmpdir, f"{region_name}_conv.wav")
            ref_path = os.path.join(tmpdir, f"{region_name}_ref.wav")

            sf.write(conv_path, conv_region, sr)
            sf.write(ref_path, ref_region, sr)

            region_results = {}
            region_results["duration_sec"] = float(np.sum(mask)) / sr

            # MCD
            mcd = compute_mcd(ref_path, conv_path, sr=sr)
            region_results["mcd_db"] = mcd["mcd_db"]

            # F0 metrics
            f0_metrics = compute_f0_metrics(ref_path, conv_path, sr=sr)
            region_results.update(f0_metrics)

            # Spectral metrics
            spec = compute_spectral_metrics(ref_path, conv_path, sr=sr)
            region_results.update(spec)

            results[region_name] = region_results
            log.info(f"  {region_name}: MCD={region_results['mcd_db']:.2f} dB, "
                      f"F0-RMSE={region_results.get('f0_rmse_hz', 'N/A')} Hz")

    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    if "solo" in results and "chorus" in results:
        if isinstance(results["solo"], dict) and isinstance(results["chorus"], dict):
            if "mcd_db" in results["solo"] and "mcd_db" in results["chorus"]:
                solo_mcd = results["solo"]["mcd_db"]
                chorus_mcd = results["chorus"]["mcd_db"]
                results["mcd_degradation_chorus_vs_solo"] = chorus_mcd - solo_mcd
                log.info(f"  MCD degradation (chorus - solo): {results['mcd_degradation_chorus_vs_solo']:.2f} dB")

    return results


# ==============================================================================
# CLI
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description="Quantitative evaluation for singing voice conversion")

    p.add_argument("--converted", type=str, help="Path to converted vocals WAV")
    p.add_argument("--reference", type=str, help="Path to reference vocals WAV")
    p.add_argument("--target-speaker", type=str, help="Path to target speaker audio")
    p.add_argument("--source-speaker", type=str, help="Path to source speaker audio (leakage test)")
    p.add_argument("--output-json", type=str, default=None, help="Path to save results JSON")

    p.add_argument("--compare-pipelines", action="store_true", help="Compare with/without separation pipelines")
    p.add_argument("--with-sep", type=str, help="Converted vocals with separation")
    p.add_argument("--without-sep", type=str, help="Converted vocals without separation")
    p.add_argument("--original-song", type=str, help="Original mixed song WAV")

    p.add_argument("--eval-chorus", action="store_true", help="Evaluate chorus handling")
    p.add_argument("--chorus-map", type=str, help="JSON file with chorus segments: [[start, end], ...]")

    args = p.parse_args()

    if args.compare_pipelines:
        if not all([args.with_sep, args.without_sep, args.original_song, args.target_speaker]):
            p.error("--compare-pipelines requires --with-sep, --without-sep, --original-song, --target-speaker")
        results = compare_pipelines(
            original_song_wav=args.original_song,
            separated_vocals_wav=args.reference,
            converted_with_sep_wav=args.with_sep,
            converted_without_sep_wav=args.without_sep,
            target_speaker_wav=args.target_speaker,
        )
    elif args.eval_chorus:
        if not all([args.converted, args.reference, args.chorus_map]):
            p.error("--eval-chorus requires --converted, --reference, --chorus-map")
        with open(args.chorus_map) as f:
            chorus_segments = json.load(f)
        results = evaluate_chorus_handling(
            converted_wav=args.converted,
            reference_wav=args.reference,
            chorus_segments=chorus_segments,
            target_speaker_wav=args.target_speaker,
        )
    elif args.converted:
        results = evaluate_conversion(
            converted_wav=args.converted,
            reference_wav=args.reference,
            target_speaker_wav=args.target_speaker,
            source_speaker_wav=args.source_speaker,
        )
    else:
        p.print_help()
        return

    # Remove numpy arrays for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results_clean = clean_for_json(results)

    # Save or print
    output_path = args.output_json or "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + json.dumps(results_clean, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
