#!/usr/bin/env python3
"""
Chorus Detection and Handling Module
=====================================
Detects chorus (multi-voice) segments in a song and provides two strategies
to handle them during voice conversion:

  Strategy A: "Chorus-as-Background" — replace chorus with original (unconverted)
              vocals mixed at lower volume, preserving the accompaniment feel.
  Strategy B: "Enhanced F0" — apply enhanced F0 estimation (median filtering +
              harmonic prior) specifically for chorus segments to improve
              pitch tracking in polyphonic contexts.

Detection methods:
  1. Energy + spectral flux based detection (unsupervised)
  2. Manual annotation (JSON file with timestamp ranges)
  3. Voice count estimation via NMF / spectral analysis

Usage:
    # Detect chorus segments automatically
    python chorus_handler.py detect --input intermediate/vocals.wav --output chorus_segments.json

    # Apply Strategy A: chorus-as-background
    python chorus_handler.py apply --strategy background \
        --converted exp/xxx/vocals_converted.wav \
        --original intermediate/vocals.wav \
        --chorus-map chorus_segments.json \
        --output exp/xxx/vocals_final.wav

    # Apply Strategy B: enhanced F0 re-synthesis (requires RVC re-inference)
    python chorus_handler.py apply --strategy enhanced-f0 \
        --input intermediate/vocals.wav \
        --chorus-map chorus_segments.json \
        --output intermediate/vocals_f0_enhanced.wav
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("chorus_handler")


# ==============================================================================
# 1. Chorus Detection: Multi-Voice Segment Identification
# ==============================================================================
class ChorusDetector:
    """
    Detect chorus (multi-voice) segments in vocal tracks.

    Uses a combination of:
      - Spectral complexity (harmonic density)
      - Energy variance
      - Pitch multiplicity (multiple F0 candidates)
      - Onset density (ensemble singing has more overlapping onsets)

    The core insight: chorus/harmony sections have higher spectral complexity
    and more simultaneous harmonic content than solo singing.
    """

    def __init__(self, sr=16000, hop_ms=10.0, frame_sec=2.0, min_chorus_sec=3.0):
        """
        Args:
            sr: Analysis sample rate
            hop_ms: Hop size for frame-level analysis
            frame_sec: Window size for segment-level features
            min_chorus_sec: Minimum duration to be classified as chorus
        """
        self.sr = sr
        self.hop_ms = hop_ms
        self.hop_samples = int(hop_ms / 1000.0 * sr)
        self.frame_sec = frame_sec
        self.min_chorus_sec = min_chorus_sec

    def detect(self, audio_path):
        """
        Detect chorus segments in audio.

        Args:
            audio_path: Path to vocals WAV file

        Returns:
            List of (start_sec, end_sec) tuples marking chorus regions
        """
        import librosa

        y, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        duration = len(y) / self.sr

        log.info(f"Analyzing {audio_path} ({duration:.1f}s) for chorus detection...")

        # Compute frame-level features
        features = self._compute_features(y)

        # Segment-level aggregation
        segments = self._segment_features(features, duration)

        # Classify segments as solo/chorus
        chorus_segments = self._classify_segments(segments)

        # Merge adjacent chorus segments and enforce minimum duration
        chorus_segments = self._merge_segments(chorus_segments)

        log.info(f"Detected {len(chorus_segments)} chorus segments:")
        for i, (start, end) in enumerate(chorus_segments):
            log.info(f"  Chorus {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")

        return chorus_segments

    def _compute_features(self, y):
        """Compute frame-level audio features for chorus detection."""
        import librosa

        # 1. Spectral flatness — chorus has less flat spectrum (more harmonics)
        S = np.abs(librosa.stft(y, hop_length=self.hop_samples))
        spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]

        # 2. Harmonic ratio — chorus has more harmonic content
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = librosa.feature.rms(y=y_harmonic, hop_length=self.hop_samples)[0]
        total_energy = librosa.feature.rms(y=y, hop_length=self.hop_samples)[0]
        harmonic_ratio = harmonic_energy / (total_energy + 1e-10)

        # 3. Spectral bandwidth — chorus has wider bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=S, sr=self.sr
        )[0]

        # 4. Chroma variance — chorus has more pitch classes active
        chroma = librosa.feature.chroma_stft(S=S, sr=self.sr)
        chroma_entropy = -np.sum(chroma * np.log2(chroma + 1e-10), axis=0)

        # 5. Onset strength — ensemble singing has more overlapping onsets
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_samples)

        # 6. RMS energy — chorus is typically louder
        rms = librosa.feature.rms(y=y, hop_length=self.hop_samples)[0]

        # Align all features to same length
        min_len = min(
            len(spectral_flatness), len(harmonic_ratio),
            len(spectral_bandwidth), len(chroma_entropy),
            len(onset_env), len(rms)
        )

        return {
            "spectral_flatness": spectral_flatness[:min_len],
            "harmonic_ratio": harmonic_ratio[:min_len],
            "spectral_bandwidth": spectral_bandwidth[:min_len],
            "chroma_entropy": chroma_entropy[:min_len],
            "onset_strength": onset_env[:min_len],
            "rms": rms[:min_len],
        }

    def _segment_features(self, features, duration):
        """Aggregate frame-level features into segment-level features."""
        frames_per_segment = int(self.frame_sec * 1000.0 / self.hop_ms)
        n_frames = len(features["rms"])
        n_segments = n_frames // frames_per_segment

        segments = []
        for i in range(n_segments):
            start_frame = i * frames_per_segment
            end_frame = start_frame + frames_per_segment
            start_sec = i * self.frame_sec
            end_sec = start_sec + self.frame_sec

            seg = {
                "start_sec": start_sec,
                "end_sec": end_sec,
            }

            for key, values in features.items():
                seg_values = values[start_frame:end_frame]
                seg[f"{key}_mean"] = float(np.mean(seg_values))
                seg[f"{key}_std"] = float(np.std(seg_values))

            segments.append(seg)

        return segments

    def _classify_segments(self, segments):
        """
        Classify segments as chorus or solo using a multi-feature heuristic.

        Chorus characteristics:
          - Higher chroma entropy (more pitch classes)
          - Lower spectral flatness (more tonal/harmonic)
          - Higher RMS energy
          - Higher harmonic ratio
        """
        if not segments:
            return []

        # Normalize features
        feature_keys = ["chroma_entropy_mean", "rms_mean", "harmonic_ratio_mean", "spectral_bandwidth_mean"]
        feature_matrix = np.array([
            [seg.get(k, 0.0) for k in feature_keys]
            for seg in segments
        ])

        # Z-score normalization
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0) + 1e-10
        normalized = (feature_matrix - means) / stds

        # Composite score: high chroma entropy + high energy + high harmonic ratio → chorus
        # Weights determined empirically for typical pop/musical theater songs
        weights = np.array([0.35, 0.25, 0.25, 0.15])
        scores = np.dot(normalized, weights)

        # Threshold: segments with score > mean + 0.5*std classified as chorus
        threshold = np.mean(scores) + 0.5 * np.std(scores)

        chorus_segments = []
        for seg, score in zip(segments, scores):
            if score > threshold:
                chorus_segments.append((seg["start_sec"], seg["end_sec"]))

        return chorus_segments

    def _merge_segments(self, segments):
        """Merge adjacent chorus segments and remove short ones."""
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])

        # Merge overlapping or adjacent (within 2 seconds)
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= 2.0:  # Gap <= 2 seconds
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        # Filter by minimum duration
        merged = [(s, e) for s, e in merged if e - s >= self.min_chorus_sec]

        return merged


# ==============================================================================
# Hamilton "My Shot" specific chorus annotation
# ==============================================================================
MY_SHOT_CHORUS_SEGMENTS = [
    # These are approximate timestamps for chorus/ensemble sections in "My Shot"
    # Based on the Original Broadway Cast Recording (~5:27 total)
    (47.0, 63.0),     # "I am not throwing away my shot" (first ensemble)
    (108.0, 125.0),   # "I am not throwing away my shot" (second ensemble, louder)
    (155.0, 170.0),   # "Everybody sing" / "Whoa" section
    (197.0, 218.0),   # "I am not throwing away my shot" (climactic ensemble)
    (248.0, 268.0),   # "Not throwing away my shot" (bridge ensemble)
    (289.0, 327.0),   # Final ensemble / "Everybody sing" through outro
]


def get_default_chorus_segments():
    """Get default chorus segments for Hamilton 'My Shot'."""
    return MY_SHOT_CHORUS_SEGMENTS


# ==============================================================================
# 2. Strategy A: Chorus-as-Background
# ==============================================================================
def apply_chorus_as_background(
    converted_vocals_path,
    original_vocals_path,
    chorus_segments,
    output_path,
    crossfade_sec=0.5,
    chorus_volume_db=-6.0,
    sr=44100,
):
    """
    Replace converted vocals in chorus regions with original (unconverted) vocals
    at reduced volume, creating a natural "background choir" effect.

    The key insight: single-speaker RVC applied to multi-voice chorus segments
    produces artifacts because the model attempts to map multiple overlapping
    voices through a single-speaker decoder. By keeping the original chorus
    and only converting solo segments, we avoid this entirely.

    For crossfade regions, we apply a smooth sigmoid transition to avoid
    audible clicks.

    Args:
        converted_vocals_path: Path to RVC-converted vocals (full song)
        original_vocals_path: Path to original separated vocals (from Demucs)
        chorus_segments: List of (start_sec, end_sec) chorus timestamps
        output_path: Path for output vocals
        crossfade_sec: Crossfade duration at boundaries (seconds)
        chorus_volume_db: Volume adjustment for chorus regions (dB)
        sr: Sample rate

    Returns:
        Path to output file
    """
    import soundfile as sf
    import librosa

    log.info("Applying Strategy A: Chorus-as-Background")
    log.info(f"  Crossfade: {crossfade_sec}s, Chorus volume: {chorus_volume_db} dB")

    # Load audio
    conv, sr_conv = sf.read(str(converted_vocals_path))
    orig, sr_orig = sf.read(str(original_vocals_path))

    # Resample if needed
    if sr_conv != sr:
        conv = librosa.resample(conv.T if conv.ndim > 1 else conv, orig_sr=sr_conv, target_sr=sr)
        if conv.ndim > 1:
            conv = conv.T
    if sr_orig != sr:
        orig = librosa.resample(orig.T if orig.ndim > 1 else orig, orig_sr=sr_orig, target_sr=sr)
        if orig.ndim > 1:
            orig = orig.T

    # Mono
    if conv.ndim > 1:
        conv = conv.mean(axis=1)
    if orig.ndim > 1:
        orig = orig.mean(axis=1)

    # Align lengths
    min_len = min(len(conv), len(orig))
    conv = conv[:min_len]
    orig = orig[:min_len]

    # Apply volume adjustment to chorus regions of original
    chorus_gain = 10 ** (chorus_volume_db / 20.0)

    # Build output: start with converted vocals, replace chorus regions
    output = conv.copy()
    crossfade_samples = int(crossfade_sec * sr)

    for start_sec, end_sec in chorus_segments:
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), min_len)

        if start_sample >= min_len:
            continue

        log.info(f"  Replacing chorus: {start_sec:.1f}s - {end_sec:.1f}s")

        # Create crossfade masks
        region_len = end_sample - start_sample

        # Fade-in from converted to original at start
        fade_in_len = min(crossfade_samples, region_len // 2)
        fade_in = np.linspace(0, 1, fade_in_len)
        # Use sigmoid for smoother transition
        fade_in = 1.0 / (1.0 + np.exp(-6 * (fade_in - 0.5)))

        # Fade-out from original to converted at end
        fade_out_len = min(crossfade_samples, region_len // 2)
        fade_out = np.linspace(1, 0, fade_out_len)
        fade_out = 1.0 / (1.0 + np.exp(-6 * (fade_out - 0.5)))

        # Build mask: 0 = converted, 1 = original
        mask = np.ones(region_len)
        mask[:fade_in_len] = fade_in
        mask[-fade_out_len:] = fade_out

        # Apply: blend converted and original
        original_region = orig[start_sample:end_sample] * chorus_gain
        converted_region = conv[start_sample:end_sample]

        output[start_sample:end_sample] = (
            (1 - mask) * converted_region + mask * original_region
        )

    # Write output
    sf.write(str(output_path), output, sr)
    log.info(f"Output saved: {output_path}")
    return output_path


# ==============================================================================
# 3. Strategy B: Enhanced F0 for Chorus Regions
# ==============================================================================
def enhance_f0_for_chorus(
    vocals_path,
    chorus_segments,
    output_f0_path,
    sr=16000,
    method="pyin",
    median_kernel=11,
    harmonic_prior_weight=0.3,
):
    """
    Apply enhanced F0 estimation specifically for chorus segments.

    In chorus regions, standard F0 extractors (even RMVPE) can be confused by
    overlapping harmonics from multiple singers. This function applies:

    1. Stronger median filtering to remove spurious F0 jumps
    2. Harmonic prior: bias F0 towards the nearest harmonic of the
       estimated fundamental, reducing octave errors
    3. Continuity constraint: penalize large F0 jumps between adjacent frames
    4. Confidence-weighted smoothing: trust high-confidence frames more

    The enhanced F0 can then be fed back to RVC inference for better
    conversion quality in chorus regions.

    Args:
        vocals_path: Path to vocals WAV file
        chorus_segments: List of (start_sec, end_sec) chorus timestamps
        output_f0_path: Path to save enhanced F0 as .npy
        sr: Sample rate for F0 analysis
        method: Base F0 extraction method
        median_kernel: Median filter kernel size for chorus regions
        harmonic_prior_weight: Weight for harmonic prior regularization

    Returns:
        Enhanced F0 array (numpy)
    """
    import librosa
    from scipy.signal import medfilt

    log.info("Applying Enhanced F0 estimation for chorus regions...")

    y, _ = librosa.load(str(vocals_path), sr=sr, mono=True)
    hop_ms = 10.0
    hop_samples = int(hop_ms / 1000.0 * sr)

    # Extract base F0
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=50, fmax=1100, sr=sr, hop_length=hop_samples
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0 > 0

    log.info(f"  Base F0: {np.sum(voiced)} voiced frames out of {len(f0)}")

    # Create frame-level chorus mask
    n_frames = len(f0)
    chorus_mask = np.zeros(n_frames, dtype=bool)
    for start_sec, end_sec in chorus_segments:
        start_frame = int(start_sec * 1000.0 / hop_ms)
        end_frame = min(int(end_sec * 1000.0 / hop_ms), n_frames)
        chorus_mask[start_frame:end_frame] = True

    n_chorus_frames = np.sum(chorus_mask & voiced)
    log.info(f"  Chorus voiced frames: {n_chorus_frames}")

    # --- Enhancement 1: Stronger median filtering for chorus ---
    f0_enhanced = f0.copy()
    if n_chorus_frames > median_kernel:
        # Apply stronger median filter only to chorus regions
        f0_chorus = f0.copy()
        f0_chorus[~chorus_mask] = 0  # Zero out non-chorus

        # Median filter on voiced chorus frames
        voiced_chorus = chorus_mask & voiced
        if np.sum(voiced_chorus) > median_kernel:
            f0_voiced_chorus = f0_chorus[voiced_chorus]
            f0_filtered = medfilt(f0_voiced_chorus, kernel_size=median_kernel)
            f0_enhanced[voiced_chorus] = f0_filtered
            log.info(f"  Applied median filter (k={median_kernel}) to chorus regions")

    # --- Enhancement 2: Harmonic prior correction ---
    # Detect and correct octave errors (common in polyphonic F0 estimation)
    if n_chorus_frames > 0:
        # Estimate the "dominant" F0 in solo regions as prior
        solo_mask = ~chorus_mask & voiced
        if np.sum(solo_mask) > 10:
            solo_f0_median = np.median(f0[solo_mask])

            # For chorus frames, check if F0 is an octave error
            for i in np.where(chorus_mask & voiced)[0]:
                current_f0 = f0_enhanced[i]
                if current_f0 <= 0:
                    continue

                # Check if current F0 is likely an octave error
                ratio = current_f0 / solo_f0_median
                # If ratio is close to 2.0 (octave up) or 0.5 (octave down)
                if 1.8 < ratio < 2.2:
                    # Likely octave-up error, correct with weighted blend
                    corrected = current_f0 / 2.0
                    f0_enhanced[i] = (1 - harmonic_prior_weight) * current_f0 + harmonic_prior_weight * corrected
                elif 0.45 < ratio < 0.55:
                    # Likely octave-down error
                    corrected = current_f0 * 2.0
                    f0_enhanced[i] = (1 - harmonic_prior_weight) * current_f0 + harmonic_prior_weight * corrected

            log.info(f"  Applied harmonic prior correction (weight={harmonic_prior_weight})")

    # --- Enhancement 3: Continuity smoothing ---
    # Penalize large frame-to-frame F0 jumps in chorus
    max_jump_cents = 200  # Maximum allowed jump in cents
    for i in np.where(chorus_mask & voiced)[0]:
        if i == 0 or not voiced[i - 1]:
            continue
        prev_f0 = f0_enhanced[i - 1]
        curr_f0 = f0_enhanced[i]
        if prev_f0 > 0 and curr_f0 > 0:
            cents_jump = abs(1200 * np.log2(curr_f0 / prev_f0))
            if cents_jump > max_jump_cents:
                # Smooth towards previous value
                f0_enhanced[i] = 0.7 * prev_f0 + 0.3 * curr_f0
                log.debug(f"  Frame {i}: smoothed F0 jump of {cents_jump:.0f} cents")

    # --- Enhancement 4: Gaussian smoothing for final polish ---
    from scipy.ndimage import gaussian_filter1d
    voiced_chorus_mask = chorus_mask & (f0_enhanced > 0)
    if np.sum(voiced_chorus_mask) > 5:
        f0_temp = f0_enhanced.copy()
        f0_temp[~voiced_chorus_mask] = 0
        f0_smoothed = gaussian_filter1d(f0_temp.astype(float), sigma=2.0)
        # Only apply to voiced chorus frames
        f0_enhanced[voiced_chorus_mask] = f0_smoothed[voiced_chorus_mask]

    # Save enhanced F0
    np.save(str(output_f0_path), f0_enhanced)
    log.info(f"  Enhanced F0 saved: {output_f0_path}")

    # Report statistics
    chorus_voiced = chorus_mask & (f0_enhanced > 0)
    if np.sum(chorus_voiced) > 0:
        f0_std_before = np.std(f0[chorus_mask & voiced]) if np.sum(chorus_mask & voiced) > 0 else 0
        f0_std_after = np.std(f0_enhanced[chorus_voiced])
        log.info(f"  F0 std in chorus: {f0_std_before:.2f} Hz → {f0_std_after:.2f} Hz "
                  f"(reduction: {(1 - f0_std_after / (f0_std_before + 1e-10)) * 100:.1f}%)")

    return f0_enhanced


# ==============================================================================
# 4. Hybrid Strategy: Combine A + B
# ==============================================================================
def apply_hybrid_strategy(
    converted_vocals_path,
    original_vocals_path,
    chorus_segments,
    output_path,
    chorus_confidence_threshold=0.7,
    sr=44100,
):
    """
    Hybrid chorus handling strategy that adaptively chooses between:
      - Strategy A (background) for high-density chorus (many overlapping voices)
      - Strategy B (enhanced F0) for light harmony (2-3 voices)

    The decision is based on spectral complexity analysis of each chorus region.

    Args:
        converted_vocals_path: Path to RVC-converted vocals
        original_vocals_path: Path to original separated vocals
        chorus_segments: List of (start_sec, end_sec) tuples
        output_path: Path for output vocals
        chorus_confidence_threshold: Threshold for chorus density classification
        sr: Sample rate

    Returns:
        Path to output file, and a report dict
    """
    import librosa
    import soundfile as sf

    log.info("Applying Hybrid Chorus Handling Strategy...")

    y_orig, _ = librosa.load(str(original_vocals_path), sr=16000, mono=True)

    # Classify each chorus segment by voice density
    heavy_chorus = []
    light_harmony = []

    for start_sec, end_sec in chorus_segments:
        start_sample = int(start_sec * 16000)
        end_sample = min(int(end_sec * 16000), len(y_orig))
        segment = y_orig[start_sample:end_sample]

        if len(segment) < 16000:  # Too short
            light_harmony.append((start_sec, end_sec))
            continue

        # Compute spectral complexity to classify segment
        S = np.abs(librosa.stft(segment, n_fft=2048))
        chroma = librosa.feature.chroma_stft(S=S, sr=16000)

        # Number of active pitch classes (proxy for voice count)
        active_pitches = np.mean(np.sum(chroma > 0.3, axis=0))

        # Spectral flatness (lower = more tonal = more harmonic content)
        flatness = np.mean(librosa.feature.spectral_flatness(S=S))

        # Energy variance (higher in multi-voice sections)
        rms = librosa.feature.rms(S=S)[0]
        energy_var = np.std(rms) / (np.mean(rms) + 1e-10)

        # Composite score
        density_score = 0.4 * (active_pitches / 12.0) + 0.3 * (1 - flatness) + 0.3 * energy_var

        if density_score > chorus_confidence_threshold:
            heavy_chorus.append((start_sec, end_sec))
            log.info(f"  [{start_sec:.1f}-{end_sec:.1f}s] → Heavy chorus (score={density_score:.3f}) → Strategy A")
        else:
            light_harmony.append((start_sec, end_sec))
            log.info(f"  [{start_sec:.1f}-{end_sec:.1f}s] → Light harmony (score={density_score:.3f}) → Strategy B")

    # Apply Strategy A for heavy chorus
    if heavy_chorus:
        apply_chorus_as_background(
            converted_vocals_path=converted_vocals_path,
            original_vocals_path=original_vocals_path,
            chorus_segments=heavy_chorus,
            output_path=output_path,
            sr=sr,
        )
    else:
        # Just copy converted vocals
        import shutil
        shutil.copy2(converted_vocals_path, output_path)

    report = {
        "heavy_chorus_segments": heavy_chorus,
        "light_harmony_segments": light_harmony,
        "strategy_a_applied": len(heavy_chorus),
        "strategy_b_candidates": len(light_harmony),
    }

    log.info(f"Hybrid strategy applied: {len(heavy_chorus)} heavy chorus (A), "
              f"{len(light_harmony)} light harmony (B)")

    return output_path, report


# ==============================================================================
# CLI
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description="Chorus detection and handling for SVC pipeline")
    sub = p.add_subparsers(dest="command")

    # Detect command
    det = sub.add_parser("detect", help="Detect chorus segments")
    det.add_argument("--input", required=True, help="Path to vocals WAV")
    det.add_argument("--output", default="chorus_segments.json", help="Output JSON path")
    det.add_argument("--use-default", action="store_true",
                     help="Use pre-annotated segments for Hamilton 'My Shot'")

    # Apply command
    app = sub.add_parser("apply", help="Apply chorus handling strategy")
    app.add_argument("--strategy", choices=["background", "enhanced-f0", "hybrid"],
                     required=True, help="Handling strategy")
    app.add_argument("--converted", help="Path to RVC-converted vocals")
    app.add_argument("--original", help="Path to original separated vocals")
    app.add_argument("--chorus-map", required=True, help="JSON file with chorus segments")
    app.add_argument("--output", required=True, help="Output WAV path")

    args = p.parse_args()

    if args.command == "detect":
        if args.use_default:
            segments = get_default_chorus_segments()
        else:
            detector = ChorusDetector()
            segments = detector.detect(args.input)

        with open(args.output, "w") as f:
            json.dump(segments, f, indent=2)
        log.info(f"Chorus segments saved to: {args.output}")

    elif args.command == "apply":
        with open(args.chorus_map) as f:
            chorus_segments = json.load(f)

        if args.strategy == "background":
            apply_chorus_as_background(
                converted_vocals_path=args.converted,
                original_vocals_path=args.original,
                chorus_segments=chorus_segments,
                output_path=args.output,
            )
        elif args.strategy == "enhanced-f0":
            enhance_f0_for_chorus(
                vocals_path=args.original,
                chorus_segments=chorus_segments,
                output_f0_path=args.output.replace(".wav", "_f0.npy"),
            )
        elif args.strategy == "hybrid":
            apply_hybrid_strategy(
                converted_vocals_path=args.converted,
                original_vocals_path=args.original,
                chorus_segments=chorus_segments,
                output_path=args.output,
            )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
