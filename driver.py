#!/usr/bin/env python3
"""
driver file for raga detection pipeline - do not run directly, run using run_pipeline.sh

modes:
    preprocess - ingest YouTube/recorded audio to local MP3 and exit
    detect  - run phase 1 of analysis (histogram-based raga filtering/ranking)
    analyze - run phase 2 of analysis (note sequence analysis given tonic/raga)
"""

import os
import sys
from pathlib import Path
from time import perf_counter

# add package to path if running directly
if __name__ == "__main__":
    package_dir = Path(__file__).parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

import librosa

from raga_pipeline.config import PipelineConfig, load_config_from_cli, create_config
from raga_pipeline.audio import (
    separate_stems,
    extract_pitch,
    extract_pitch_from_config,
    load_audio_direct,
    download_youtube_audio,
    ingest_recorded_audio_file,
    record_microphone_audio_interactive,
    get_tonic_from_tanpura_key,
    _pitch_csv_path,
)
from raga_pipeline.analysis import (
    compute_cent_histograms, 
    detect_peaks, 
    fit_gmm_to_peaks,
    compute_gmm_bias_cents,
    compute_cent_histograms_from_config,
    detect_peaks_from_config,
)
from raga_pipeline.raga import (
    RagaDatabase, 
    generate_candidates, 
    RagaScorer,
    _parse_tonic,
    _parse_tonic_list,
    apply_raga_correction_to_notes,
    get_raga_notes,
    get_tonic_candidates,
    build_aaroh_avroh_subset,
    load_aaroh_avroh_patterns,
    get_aaroh_avroh_pattern_for_raga,
)
from raga_pipeline.sequence import (
    detect_notes, 
    detect_phrases, 
    cluster_phrases,
    compute_transition_matrix,
    transcribe_notes,
    build_transition_matrix_corrected,
    extract_melodic_sequences,
    find_common_patterns,
    analyze_raga_patterns,
    merge_consecutive_notes,
    split_phrases_by_silence,
)
from raga_pipeline.output import (
    AnalysisResults,
    AnalysisStats,
    ExtractorTranscription,
    plot_histograms,
    plot_absolute_note_histogram,
    plot_gmm_overlay,
    plot_note_segments,
    plot_note_duration_histogram,
    generate_html_report,
    generate_analysis_report,
    plot_pitch_with_sargam_lines,
    plot_pitch_with_sargam_lines,
    plot_transition_heatmap_v2,
    save_notes_to_csv,
    write_detection_report_metadata,
)
from raga_pipeline import transcription
from raga_pipeline.runtime_fingerprint import get_runtime_fingerprint


def _format_seconds(seconds: float) -> str:
    """Format seconds for compact logging."""
    return f"{max(0.0, float(seconds)):.2f}s"


def _safe_get_audio_duration_seconds(audio_path: str | None) -> float | None:
    """Best-effort audio duration lookup for timing speedup metrics."""
    if not audio_path:
        return None
    try:
        duration = float(librosa.get_duration(path=audio_path))
    except Exception as exc:
        print(f"[TIMER] Could not read audio duration for speed metrics: {exc}")
        return None
    if duration <= 0 or not duration == duration:
        return None
    return duration


def _print_timing(label: str, elapsed_s: float, audio_duration_s: float | None) -> None:
    """Print elapsed time and song-relative realtime factor."""
    if audio_duration_s is None or audio_duration_s <= 0:
        print(f"[TIMER] {label}: {_format_seconds(elapsed_s)}")
        return

    ratio = elapsed_s / audio_duration_s
    if elapsed_s <= 1e-9:
        speedup_txt = "infx"
    else:
        speedup_txt = f"{(audio_duration_s / elapsed_s):.2f}x"
    print(
        f"[TIMER] {label}: {_format_seconds(elapsed_s)} | "
        f"{ratio:.3f}x song length | {speedup_txt} realtime"
    )


def _transcribe_for_extractor(
    pitch_data, config, results, raga_db, aaroh_avroh_lookup,
    aaroh_avroh_subset_path, extractor_name, confidence_threshold,
):
    """Run full transcription pipeline for a single extractor's pitch data.
    Returns an ExtractorTranscription."""
    import copy

    derivative_profile = {}
    raw_notes = transcription.transcribe_to_notes(
        pitch_hz=pitch_data.pitch_hz,
        timestamps=pitch_data.timestamps,
        voicing_mask=pitch_data.voiced_mask,
        tonic=results.detected_tonic,
        energy=pitch_data.energy,
        energy_threshold=config.energy_threshold,
        smoothing_sigma_ms=config.transcription_smoothing_ms,
        min_event_duration=config.transcription_min_duration,
        derivative_threshold=config.transcription_derivative_threshold,
        snap_mode='chromatic',
        transcription_min_duration=config.transcription_min_duration,
        bias_cents=results.gmm_bias_cents or 0.0,
        derivative_profile_out=derivative_profile,
    )

    # Raga correction
    correction_summary = {}
    if config.skip_raga_correction:
        notes = raw_notes
    elif raga_db and results.detected_raga:
        strict_raga_max_cents = max(float(getattr(config, "strict_raga_max_cents", 35.0)), 0.0)
        raga_max_distance = (strict_raga_max_cents / 100.0) if config.strict_raga_35c_filter else 1.0
        keep_impure = config.keep_impure_notes
        if config.strict_raga_35c_filter and keep_impure:
            keep_impure = False
        notes, correction_stats, _ = apply_raga_correction_to_notes(
            raw_notes, raga_db, results.detected_raga, results.detected_tonic,
            max_distance=raga_max_distance, keep_impure=keep_impure,
        )
        correction_summary = correction_stats
    else:
        notes = raw_notes

    # Merge
    notes = merge_consecutive_notes(notes, max_gap=0.1, pitch_tolerance=0.7,
                                     max_dropout_gap=0.18, dropout_fragment_duration=0.12)

    # Phrase detection + silence split + collapse
    phrases = detect_phrases(notes, max_gap=config.phrase_max_gap,
                              min_length=config.phrase_min_length,
                              min_phrase_duration=config.phrase_min_duration)
    silence_thresh = config.silence_threshold
    if silence_thresh <= 0 and config.energy_threshold > 0:
        silence_thresh = config.energy_threshold
    if silence_thresh > 0:
        phrases = split_phrases_by_silence(
            phrases=phrases, energy=pitch_data.energy,
            timestamps=pitch_data.timestamps,
            silence_threshold=silence_thresh,
            silence_min_duration=config.silence_min_duration,
        )

    # Phrase-level collapse
    if phrases:
        collapsed = []
        for phrase in phrases:
            collapsed_notes = merge_consecutive_notes(
                phrase.notes, max_gap=config.phrase_max_gap, pitch_tolerance=0.7,
                max_dropout_gap=config.phrase_max_gap,
                dropout_fragment_duration=config.phrase_max_gap,
            )
            collapsed.append(phrase.__class__(notes=collapsed_notes))
        phrases = collapsed
        notes = [n for p in phrases for n in p.notes]

    # Filter
    phrases = [p for p in phrases
               if p.duration >= config.phrase_min_duration
               and len(p.notes) >= config.phrase_min_length]

    phrase_clusters = cluster_phrases(phrases)

    # Pattern analysis
    expected_pattern = None
    if aaroh_avroh_lookup and results.detected_raga:
        expected_pattern = get_aaroh_avroh_pattern_for_raga(
            results.detected_raga, aaroh_avroh_lookup)
    if expected_pattern:
        pattern_results = analyze_raga_patterns(
            phrases, results.detected_tonic,
            expected_aaroh=expected_pattern.aaroh_pattern,
            expected_avroh=expected_pattern.avroh_pattern,
        )
    else:
        pattern_results = analyze_raga_patterns(phrases, results.detected_tonic)

    return ExtractorTranscription(
        extractor=extractor_name,
        pitch_data=pitch_data,
        notes=notes,
        phrases=phrases,
        phrase_clusters=phrase_clusters,
        pattern_analysis=pattern_results,
        correction_summary=correction_summary,
        confidence_threshold=confidence_threshold,
        derivative_timestamps=derivative_profile.get("timestamps"),
        derivative_values=derivative_profile.get("values"),
        derivative_voiced_mask=derivative_profile.get("voiced_mask"),
    )


def _run_compare_extractors(config, results, stem_dir, melody_path,
                             fmin, fmax, raga_db, aaroh_avroh_lookup,
                             aaroh_avroh_subset_path):
    """Run both SwiftF0 and pYIN, calibrate note counts, populate results."""
    from raga_pipeline.config import EXTRACTOR_CONFIDENCE_DEFAULTS

    print("\n[COMPARE] Running dual-extractor comparison (SwiftF0 + pYIN)...")

    # 1. Extract pitch with both backends at confidence=0
    extractors = {}
    for ext_name in ["swiftf0", "pyin"]:
        print(f"  [{ext_name.upper()}] Extracting pitch at confidence=0...")
        pd_raw = extract_pitch(
            audio_path=melody_path,
            output_dir=stem_dir,
            prefix="melody" if config.source_type != "vocal" else "vocals",
            fmin=fmin, fmax=fmax,
            confidence_threshold=0.0,
            force_recompute=config.force_recompute,
            energy_metric=config.energy_metric,
            extractor=ext_name,
            hop_ms=config.pitch_hop_ms if ext_name == "pyin" else 0.0,
        )
        extractors[ext_name] = pd_raw

    # 2. Count raw notes from each at confidence=0
    counts = {}
    for ext_name, pd_raw in extractors.items():
        raw_notes = transcription.transcribe_to_notes(
            pitch_hz=pd_raw.pitch_hz,
            timestamps=pd_raw.timestamps,
            voicing_mask=pd_raw.voiced_mask,
            tonic=results.detected_tonic,
            energy=pd_raw.energy,
            energy_threshold=config.energy_threshold,
            smoothing_sigma_ms=config.transcription_smoothing_ms,
            min_event_duration=config.transcription_min_duration,
            derivative_threshold=config.transcription_derivative_threshold,
            snap_mode='chromatic',
            transcription_min_duration=config.transcription_min_duration,
            bias_cents=results.gmm_bias_cents or 0.0,
        )
        counts[ext_name] = len(raw_notes)
        print(f"  [{ext_name.upper()}] Raw notes at conf=0: {len(raw_notes)}")

    # 3. Calibrate both extractors to the same proportion of their max notes.
    #    Use the extractor with fewer notes to determine the target percentage
    #    (that extractor keeps 100% at conf=0; the other is reduced to match).
    min_ext = min(counts, key=counts.get)
    max_ext = max(counts, key=counts.get)
    if counts[min_ext] == counts[max_ext]:
        target_pct = 1.0
    else:
        target_pct = counts[min_ext] / counts[max_ext]
    print(f"  [COMPARE] Target proportion: {target_pct:.3f} "
          f"({counts[min_ext]}/{counts[max_ext]} notes)")

    calibrated_thresholds = {}

    def _binary_search_threshold(ext_name, target_count):
        """Find confidence threshold that produces target_count raw notes."""
        pd_raw = extractors[ext_name]
        lo, hi = 0.0, 1.0
        best_t, best_diff = 0.0, abs(counts[ext_name] - target_count)
        for _ in range(20):
            mid = (lo + hi) / 2.0
            pd_threshed = pd_raw.apply_confidence_threshold(mid)
            notes = transcription.transcribe_to_notes(
                pitch_hz=pd_threshed.pitch_hz,
                timestamps=pd_threshed.timestamps,
                voicing_mask=pd_threshed.voiced_mask,
                tonic=results.detected_tonic,
                energy=pd_threshed.energy,
                energy_threshold=config.energy_threshold,
                smoothing_sigma_ms=config.transcription_smoothing_ms,
                min_event_duration=config.transcription_min_duration,
                derivative_threshold=config.transcription_derivative_threshold,
                snap_mode='chromatic',
                transcription_min_duration=config.transcription_min_duration,
                bias_cents=results.gmm_bias_cents or 0.0,
            )
            n = len(notes)
            diff = abs(n - target_count)
            if diff < best_diff:
                best_diff = diff
                best_t = mid
            if n > target_count:
                lo = mid
            elif n < target_count:
                hi = mid
            else:
                best_t = mid
                break
        return best_t

    for ext_name in extractors:
        target_count = int(round(counts[ext_name] * target_pct))
        if target_count >= counts[ext_name]:
            calibrated_thresholds[ext_name] = 0.0
        else:
            calibrated_thresholds[ext_name] = _binary_search_threshold(
                ext_name, target_count)
        print(f"  [{ext_name.upper()}] Calibrated confidence={calibrated_thresholds[ext_name]:.4f} "
              f"(target={target_count}/{counts[ext_name]} notes, {target_pct:.1%})")

    # 4. Run full transcription at calibrated thresholds
    for ext_name in ["swiftf0", "pyin"]:
        t = calibrated_thresholds[ext_name]
        pd_calibrated = extractors[ext_name]
        if t > 0:
            pd_calibrated = pd_calibrated.apply_confidence_threshold(t)
        print(f"  [{ext_name.upper()}] Running full transcription pipeline (conf={t:.4f})...")
        ext_result = _transcribe_for_extractor(
            pd_calibrated, config, results, raga_db,
            aaroh_avroh_lookup, aaroh_avroh_subset_path,
            ext_name, t,
        )
        results.extractor_transcriptions[ext_name] = ext_result
        print(f"  [{ext_name.upper()}] {len(ext_result.notes)} notes, "
              f"{len(ext_result.phrases)} phrases")

    # 5. Populate primary fields from config.pitch_extractor for backward compat
    primary = config.pitch_extractor
    if primary in results.extractor_transcriptions:
        prim = results.extractor_transcriptions[primary]
        results.notes = prim.notes
        results.phrases = prim.phrases
        results.phrase_clusters = prim.phrase_clusters
        results.pitch_data_vocals = prim.pitch_data
        results.transcription_derivative_timestamps = prim.derivative_timestamps
        results.transcription_derivative_values = prim.derivative_values
        results.transcription_derivative_voiced_mask = prim.derivative_voiced_mask


def run_pipeline(
    config: PipelineConfig,
) -> AnalysisResults:
    """
    run the raga detection pipeline in the configured mode

    - preprocess: Ingest YouTube/recorded audio to local MP3 and exit.
    - detect: Run up to raga candidate scoring, generate summary report.
    - analyze: Load cached pitch, run sequence/phrase analysis using provided tonic/raga.
    
    Source Types:
    - mixed: uses stem separation (default)
    - instrumental: uses stem separation + instrument-specific tonic bias
    - vocal: uses stem separation + gender-specific tonic bias

    Detect skip mode:
    - --skip-separation: bypass stem separation and use original audio as the melody source
      (requires --tonic, and melody_source is forced to composite).
    
    args:
        config: pipeline configuration
        
    outputs:
        AnalysisResults with computed data
    """
    results = AnalysisResults(config=config)

    if config.mode == "preprocess":
        print("=" * 60)
        print("RAGA DETECTION PIPELINE")
        print("MODE: PREPROCESS")
        print("=" * 60)
        print(f"Ingest: {config.preprocess_ingest}")
        print(f"Audio Dir: {config.audio_dir}")
        print(f"Filename: {config.filename_override}.mp3")
        if config.preprocess_ingest == "yt":
            print(f"YouTube URL: {config.yt_url}")
            print(f"Start Time: {config.preprocess_start_time or '0:00'}")
            print(f"End Time: {config.preprocess_end_time or 'track end'}")
        else:
            if config.preprocess_ingest == "tanpura_recording":
                print(f"Tanpura Key: {config.preprocess_tanpura_key}")
            print(
                "Recorded Source: "
                f"{config.preprocess_recorded_audio or 'live microphone capture'}"
            )
        print()

        preprocess_tonic: str | None = None
        try:
            if config.preprocess_ingest == "yt":
                downloaded_audio_path = download_youtube_audio(
                    yt_url=config.yt_url or "",
                    audio_dir=config.audio_dir or "",
                    filename_base=config.filename_override or "",
                    start_time=config.preprocess_start_time,
                    end_time=config.preprocess_end_time,
                )
            else:
                tanpura_key = (
                    config.preprocess_tanpura_key
                    if config.preprocess_ingest == "tanpura_recording"
                    else None
                )
                if config.preprocess_recorded_audio:
                    downloaded_audio_path = ingest_recorded_audio_file(
                        recorded_audio_path=config.preprocess_recorded_audio,
                        audio_dir=config.audio_dir or "",
                        filename_base=config.filename_override or "",
                    )
                else:
                    downloaded_audio_path = record_microphone_audio_interactive(
                        audio_dir=config.audio_dir or "",
                        filename_base=config.filename_override or "",
                        tanpura_key=tanpura_key,
                    )
                if tanpura_key:
                    preprocess_tonic = get_tonic_from_tanpura_key(tanpura_key)
        except Exception as exc:
            print(f"[PREPROCESS] ERROR: {exc}")
            if config.preprocess_ingest == "yt":
                print("[PREPROCESS] Tip: try updating yt-dlp and retrying with a different/public video URL.")
            else:
                print("[PREPROCESS] Tip: check microphone permissions, ffmpeg/ffplay availability, and tanpura key selection.")
            raise SystemExit(1)

        config.audio_path = downloaded_audio_path

        print("[PREPROCESS] Ingest complete")
        print(f"  Saved: {downloaded_audio_path}")

        print("\n" + "=" * 60)
        print("Next: Run detect mode with ingested audio:")
        print("=" * 60)
        print(f"./run_pipeline.sh detect \\")
        print(f'  --audio "{downloaded_audio_path}" \\')
        if preprocess_tonic:
            print(f'  --output "{config.output_dir}" \\')
            print(f'  --tonic "{preprocess_tonic}" \\')
            print(f"  --skip-separation")
        else:
            print(f'  --output "{config.output_dir}"')
        print("=" * 60)

        return results
    
    print("=" * 60)
    print("RAGA DETECTION PIPELINE")
    print(f"MODE: {config.mode.upper()}")
    print("=" * 60)
    print(f"Audio: {config.filename}")
    print(f"Output: {config.output_dir}")
    pipeline_start = perf_counter()
    audio_duration_s = _safe_get_audio_duration_seconds(config.audio_path)
    runtime_fingerprint = None
    try:
        runtime_fingerprint = get_runtime_fingerprint(cache_ttl_seconds=0.0)
    except Exception as exc:
        print(f"[VERSION] WARN: Failed to compute runtime fingerprint: {exc}")
    if audio_duration_s is not None:
        print(f"[TIMER] Track duration: {_format_seconds(audio_duration_s)}")
    print()
    
    # check ragaDB path early
    raga_db = None
    aaroh_avroh_lookup = {}
    aaroh_avroh_subset_path = os.path.join(
        Path(__file__).parent,
        "raga_pipeline",
        "data",
        "aarohavroha_subset.csv",
    )

    if config.raga_db_path and os.path.isfile(config.raga_db_path):
        raga_db = RagaDatabase(config.raga_db_path)
        aaroh_avroh_source_path = os.path.join(Path(__file__).parent, "aarohavroha.csv")
        if os.path.isfile(aaroh_avroh_subset_path):
            try:
                aaroh_avroh_lookup = load_aaroh_avroh_patterns(aaroh_avroh_subset_path)
                print(f"[AAROH/AVROH] Loaded {len(aaroh_avroh_lookup)} patterns from subset DB")
            except Exception as e:
                print(f"[AAROH/AVROH] Failed to load subset DB: {e}")
        elif os.path.isfile(aaroh_avroh_source_path):
            try:
                subset_df = build_aaroh_avroh_subset(
                    aaroh_avroh_csv_path=aaroh_avroh_source_path,
                    raga_db_csv_path=config.raga_db_path,
                    output_csv_path=aaroh_avroh_subset_path,
                )
                aaroh_avroh_lookup = load_aaroh_avroh_patterns(aaroh_avroh_subset_path)
                print(
                    f"[AAROH/AVROH] Loaded {len(aaroh_avroh_lookup)} aligned patterns "
                    f"(subset rows: {len(subset_df)})"
                )
            except Exception as e:
                print(f"[AAROH/AVROH] Failed to build/load subset DB: {e}")
        else:
            print(f"[AAROH/AVROH] Source CSV not found: {aaroh_avroh_source_path}")
    
    # =========================================================================
    # PHASE 1: DETECTION (or load cache)
    # =========================================================================
    
    step1_start = perf_counter()
    if config.mode == "analyze":
        # ANALYZE MODE: skip detection, load cache
        print("[PHASE 1] Skipping detection (Analyze Mode)")
        
        # load pitch data from cache
        print("[CACHE] Loading pitch data...")
        
        # Determine correct prefix based on source type (matches detect mode logic)
        preferred_prefix = "vocals" if config.source_type == "vocal" else "melody"
        fallback_prefix = "melody" if preferred_prefix == "vocals" else "vocals"
        prefix_candidates = [preferred_prefix, fallback_prefix]

        melody_prefix = None
        melody_pitch_csv = None
        for prefix in prefix_candidates:
            # Try extractor-specific cache first, then legacy name
            candidate_csv = _pitch_csv_path(config.stem_dir, prefix, config.pitch_extractor)
            if os.path.exists(candidate_csv):
                melody_prefix = prefix
                melody_pitch_csv = candidate_csv
                break
            legacy_csv = os.path.join(config.stem_dir, f"{prefix}_pitch_data.csv")
            if os.path.exists(legacy_csv):
                melody_prefix = prefix
                melody_pitch_csv = legacy_csv
                break

        accomp_pitch_csv = _pitch_csv_path(config.stem_dir, "accompaniment", config.pitch_extractor)
        if not os.path.exists(accomp_pitch_csv):
            accomp_pitch_csv = os.path.join(config.stem_dir, "accompaniment_pitch_data.csv")
        composite_pitch_csv = _pitch_csv_path(config.stem_dir, "composite", config.pitch_extractor)
        if not os.path.exists(composite_pitch_csv):
            composite_pitch_csv = os.path.join(config.stem_dir, "composite_pitch_data.csv")
        if melody_prefix is None and os.path.exists(composite_pitch_csv):
            melody_prefix = "composite"
            melody_pitch_csv = composite_pitch_csv
            print("  [CACHE] Missing melody/vocals pitch cache; falling back to composite_pitch_data.csv")
            if config.melody_source != "composite":
                config.melody_source = "composite"
                print("  [CACHE] Forcing melody_source=composite due to cache fallback")
        elif melody_prefix is None:
            melody_prefix = preferred_prefix
            melody_pitch_csv = os.path.join(config.stem_dir, f"{melody_prefix}_pitch_data.csv")
        assert melody_pitch_csv is not None

        if not os.path.exists(melody_pitch_csv):
            raise FileNotFoundError(
                f"Cached melody pitch data not found at {melody_pitch_csv}. Run 'detect' phase first."
            )
            
        # dummy wrapper for loading parameters
        fmin = float(librosa.note_to_hz(config.fmin_note))
        fmax = float(librosa.note_to_hz(config.fmax_note))
        
        # Load Primary Melody Stem
        melody_audio_path = config.audio_path if melody_prefix == "composite" else config.vocals_path
        pitch_data_stems = extract_pitch(
            audio_path=melody_audio_path, # Path reference for metadata
            output_dir=config.stem_dir,
            prefix=melody_prefix,
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.vocal_confidence,
            force_recompute=config.force_recompute,
            energy_metric=config.energy_metric,
            extractor=config.pitch_extractor,
            hop_ms=config.pitch_hop_ms,

        )
        results.pitch_data_stem = pitch_data_stems
        if melody_prefix == "composite":
            results.pitch_data_composite = pitch_data_stems
        
        # Load Composite if exists
        if os.path.exists(composite_pitch_csv) and results.pitch_data_composite is None:
            results.pitch_data_composite = extract_pitch(
                audio_path=config.audio_path,
                output_dir=config.stem_dir,
                prefix="composite",
                fmin=fmin,
                fmax=fmax,
                confidence_threshold=config.vocal_confidence,
                force_recompute=config.force_recompute,
                energy_metric=config.energy_metric,
                extractor=config.pitch_extractor,
                hop_ms=config.pitch_hop_ms,
    
            )
            
        # Assign primary melody based on config
        if config.melody_source == "composite" and results.pitch_data_composite:
            print(f"  [CACHE] Using COMPOSITE pitch for melody analysis")
            results.pitch_data_vocals = results.pitch_data_composite
        else:
            print(f"  [CACHE] Using SEPARATED STEM pitch for melody analysis")
            results.pitch_data_vocals = pitch_data_stems

        # Load Accompaniment
        if os.path.exists(accomp_pitch_csv):
            results.pitch_data_accomp = extract_pitch(
                audio_path=config.accompaniment_path,
                output_dir=config.stem_dir,
                prefix="accompaniment",
                fmin=fmin,
                fmax=fmax,
                confidence_threshold=config.accomp_confidence,
                force_recompute=config.force_recompute,
                energy_metric=config.energy_metric,
                extractor=config.pitch_extractor,
                hop_ms=config.pitch_hop_ms,
    
            )
        else:
            results.pitch_data_accomp = None
        
        # validate required inputs for analysis
        if not config.tonic_override or not config.raga_override:
            raise ValueError("Analyze mode requires --tonic and --raga arguments")
            
        results.detected_tonic = _parse_tonic(config.tonic_override)
        results.detected_raga = config.raga_override
        
        print(f"  Using Force Tonic: {config.tonic_override} -> {results.detected_tonic}")
        print(f"  Using Force Raga: {config.raga_override}")
        _print_timing("Step 1/7 cache load", perf_counter() - step1_start, audio_duration_s)

    else:
        # DETECT or FULL MODE: run detection pipeline
        
        stem_dir = config.stem_dir
        os.makedirs(stem_dir, exist_ok=True)

        # STEP 1: Stems
        step1_start = perf_counter()
        if config.skip_separation:
            print("[STEP 1/7] Skipping stem separation (--skip-separation)")
            if getattr(config, "skip_separation_forced_composite", False):
                print("[CONFIG] Forcing melody_source=composite because --skip-separation is enabled")
            melody_path = config.audio_path
            accompaniment_path = None
        else:
            print("[STEP 1/7] Audio preprocessing...")
            print(f"  Source type: {config.source_type} - Running stem separation")
            melody_path, accompaniment_path = separate_stems(
                audio_path=config.audio_path,
                output_dir=config.output_dir,
                engine=config.separator_engine,
                model=config.demucs_model,
                force_recompute=config.force_stem_recompute,
            )
        _print_timing("Step 1/7 stem preparation", perf_counter() - step1_start, audio_duration_s)
        
        # STEP 2: Pitch
        step2_start = perf_counter()
        print("\n[STEP 2/7] Extracting pitch...")
        fmin = float(librosa.note_to_hz(config.fmin_note))
        fmax = float(librosa.note_to_hz(config.fmax_note))
        
        # 2a. Composite Pitch (Always computed)
        print("  - Composite (Original Mix)...")
        results.pitch_data_composite = extract_pitch(
            audio_path=config.audio_path,
            output_dir=stem_dir,
            prefix="composite",
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.vocal_confidence, # Use vocal confidence for now
            force_recompute=config.force_recompute,
            energy_metric=config.energy_metric,
            extractor=config.pitch_extractor,
            hop_ms=config.pitch_hop_ms,

        )
        
        # 2b. Stem Pitch (Vocals/Melody)
        if config.skip_separation:
            print("  - Melody from Composite (--skip-separation)...")
            pitch_data_stems = results.pitch_data_composite
        else:
            print("  - Separated Melody Stem...")
            pitch_data_stems = extract_pitch(
                audio_path=melody_path,
                output_dir=stem_dir,
                prefix="vocals" if config.source_type == "vocal" else "melody",
                fmin=fmin,
                fmax=fmax,
                confidence_threshold=config.vocal_confidence,
                force_recompute=config.force_recompute,
                energy_metric=config.energy_metric,
                extractor=config.pitch_extractor,
                hop_ms=config.pitch_hop_ms,
    
            )
        assert pitch_data_stems is not None
        results.pitch_data_stem = pitch_data_stems
        
        # 2c. Accompaniment Pitch
        if accompaniment_path is not None:
            print("  - Accompaniment...")
            results.pitch_data_accomp = extract_pitch(
                audio_path=accompaniment_path,
                output_dir=stem_dir,
                prefix="accompaniment",
                fmin=fmin,
                fmax=fmax,
                confidence_threshold=config.accomp_confidence,
                force_recompute=config.force_recompute,
                energy_metric=config.energy_metric,
                extractor=config.pitch_extractor,
                hop_ms=config.pitch_hop_ms,
    
            )
        else:
            results.pitch_data_accomp = None
            
        # 2d. Assign Primary Melody Data based on Configuration
        if config.melody_source == "composite":
            print(f"  [CONFIG] Using COMPOSITE pitch for melody analysis")
            results.pitch_data_vocals = results.pitch_data_composite
        else:
            print(f"  [CONFIG] Using SEPARATED STEM pitch for melody analysis")
            results.pitch_data_vocals = pitch_data_stems
        _print_timing("Step 2/7 pitch extraction", perf_counter() - step2_start, audio_duration_s)

        # --- PITCH-ONLY EARLY EXIT ---
        if config.pitch_only and config.mode == "detect":
            print("\n[--pitch-only] Stems + pitch CSVs cached. Skipping steps 3-7.")
            _print_timing("Total detect pipeline (pitch-only)", perf_counter() - pipeline_start, audio_duration_s)
            return results

        step3_start = perf_counter()
        print("\n[STEP 3/7] Computing histograms and detecting peaks...")
        results.histogram_vocals = compute_cent_histograms_from_config(results.pitch_data_vocals, config)
        if results.pitch_data_accomp is not None:
            results.histogram_accomp = compute_cent_histograms_from_config(results.pitch_data_accomp, config)
        results.peaks_vocals = detect_peaks_from_config(results.histogram_vocals, config)

        bias_cents = None
        if config.bias_rotation:
            bias_gmm_results = fit_gmm_to_peaks(
                results.histogram_vocals,
                results.peaks_vocals.validated_indices,
                window_cents=config.gmm_window_cents,
                n_components=config.gmm_components,
                tonic=None,
            )
            bias_cents = compute_gmm_bias_cents(bias_gmm_results)
            results.gmm_bias_cents = bias_cents
            if bias_cents is not None:
                print(f"[GMM] Bias rotation: {bias_cents:.1f} cents")
        
        print(f"  Detected {len(results.peaks_vocals.validated_indices)} validated peaks")
        print(f"  [DEBUG] Detected pitch classes: {sorted(results.peaks_vocals.pitch_classes)}")
        
        # Plot histograms (vocals/melody)
        hist_path = os.path.join(stem_dir, "histogram_melody.png")
        plot_histograms(
            results.histogram_vocals,
            results.peaks_vocals,
            hist_path,
            title="Melody Pitch Distribution",
            bias_cents=results.gmm_bias_cents,
        )
        results.plot_paths["histogram_vocals"] = hist_path
        print(f"  Saved: {hist_path}")
        
        # Plot histograms (accompaniment) - only if available
        if results.histogram_accomp is not None:
            hist_accomp_path = os.path.join(stem_dir, "histogram_accompaniment.png")
            peaks_accomp = detect_peaks_from_config(results.histogram_accomp, config)
            plot_histograms(
                results.histogram_accomp,
                peaks_accomp,
                hist_accomp_path,
                title="Accompaniment Pitch Distribution",
            )
            results.plot_paths["histogram_accomp"] = hist_accomp_path
            print(f"  Saved: {hist_accomp_path}")
        _print_timing("Step 3/7 histogram + peaks", perf_counter() - step3_start, audio_duration_s)

        step35_start = perf_counter()
        print("\n[STEP 3.5/7] Building duration-weighted octave-wrapped stationary-note histogram from vocal SwiftF0...")
        stationary_frame_period = 0.01
        if len(results.pitch_data_stem.timestamps) > 1:
            stationary_frame_period = float(
                results.pitch_data_stem.timestamps[1] - results.pitch_data_stem.timestamps[0]
            )

        stationary_events = transcription.detect_stationary_events(
            pitch_hz=results.pitch_data_stem.pitch_hz,
            timestamps=results.pitch_data_stem.timestamps,
            voicing_mask=results.pitch_data_stem.voiced_mask,
            tonic=60.0,
            energy=results.pitch_data_stem.energy,
            energy_threshold=config.energy_threshold,
            smoothing_sigma_ms=config.transcription_smoothing_ms,
            frame_period_s=max(stationary_frame_period, 1e-3),
            derivative_threshold=config.transcription_derivative_threshold,
            min_event_duration=config.transcription_min_duration,
            snap_mode='chromatic',
        )
        stationary_midis = [evt.snapped_midi for evt in stationary_events]
        stationary_durations = [max(0.0, evt.duration) for evt in stationary_events]
        stationary_weighted_hist_path = os.path.join(stem_dir, "stationary_note_histogram_duration_weighted.png")
        stationary_weighted_counts = plot_absolute_note_histogram(
            stationary_midis,
            stationary_weighted_hist_path,
            title="Stationary Notes (Duration-Weighted, Octave-Wrapped)",
            weights=stationary_durations,
            ylabel="Total stationary duration (s)",
        )
        stationary_weighted_csv_path = os.path.join(stem_dir, "stationary_note_histogram_duration_weighted.csv")
        stationary_weighted_counts.to_csv(stationary_weighted_csv_path, index=False)

        results.plot_paths["stationary_note_histogram"] = stationary_weighted_hist_path
        print(f"  Stationary events: {len(stationary_events)}")
        print(f"  Saved: {stationary_weighted_hist_path}")
        print(f"  Saved: {stationary_weighted_csv_path}")
        _print_timing("Step 3.5/7 stationary-note histogram", perf_counter() - step35_start, audio_duration_s)
        
        # STEP 4 & 5: Raga Matching
        step45_start = perf_counter()
        print("\n[STEP 4/7] Loading raga database...")
        # Already loaded raga_db at start if available
        if raga_db:
            print(f"  Loaded {len(raga_db.all_ragas)} ragas")
            
            print("\n[STEP 5/7] Scoring candidates...")
            
            # Get tonic candidates based on source type
            tonic_candidates = get_tonic_candidates(
                source_type=config.source_type,
                vocalist_gender=config.vocalist_gender,
                instrument_type=config.instrument_type,
            )
            tonic_constraints = _parse_tonic_list(config.tonic_override)
            if tonic_constraints:
                tonic_candidates = tonic_constraints
                tonic_names = ", ".join(_tonic_name(t) for t in tonic_candidates)
                print(f"  Tonic constraint: {tonic_names}")
            else:
                print(f"  Tonic candidates: {tonic_candidates}")

            if config.raga_override:
                print(f"  Raga constraint: {config.raga_override}")
            
            scorer = RagaScorer(raga_db=raga_db, model_path=config.model_path, use_ml=config.use_ml_model)
            results.candidates = scorer.score(
                pitch_data_vocals=results.pitch_data_vocals,
                pitch_data_accomp=results.pitch_data_accomp,
                detected_peak_count=len(results.peaks_vocals.validated_indices),
                instrument_mode=config.source_type,
                tonic_candidates=tonic_candidates,  # Pass the bias list!
                bias_cents=results.gmm_bias_cents,
                raga_filter=config.raga_override,
            )
            
            # Save candidates
            candidates_path = os.path.join(stem_dir, "candidates.csv")
            results.candidates.to_csv(candidates_path, index=False)
            print(f"  Saved: {candidates_path}")
            
            # Determine Top Candidate / Override
            if tonic_constraints and len(tonic_constraints) == 1:
                results.detected_tonic = tonic_constraints[0]
                print(f"  Force Tonic: {_tonic_name(tonic_constraints[0])}")
            elif len(results.candidates) > 0:
                top = results.candidates.iloc[0]
                results.detected_tonic = int(top["tonic"])
                print(f"  Top Tonic: {top.get('tonic_name', top['tonic'])}")
            
            if config.raga_override:
                results.detected_raga = config.raga_override
                print(f"  Force Raga: {config.raga_override}")
            elif len(results.candidates) > 0:
                top = results.candidates.iloc[0]
                results.detected_raga = top["raga"]
                print(f"  Top Raga: {results.detected_raga}")

        else:
            print("  [WARN] Raga database not found, skipping matching")
        _print_timing("Step 4-5/7 raga matching", perf_counter() - step45_start, audio_duration_s)

        # --- STEP 5.5: LM RE-RANKING (optional) ---
        if config.use_lm_scoring and config.mode == "detect" and len(results.candidates) > 0:
            step55_start = perf_counter()
            print("\n[STEP 5.5/7] LM re-ranking...")

            import json as _json
            from raga_pipeline.language_model import NgramModel
            from raga_pipeline.sequence import tokenize_notes_for_lm

            # Load LM
            lm_path = config.lm_model_path
            if not lm_path or not os.path.exists(lm_path):
                print(f"  [ERROR] --lm-model not found: {lm_path}")
            else:
                with open(lm_path, "r", encoding="utf-8") as _fh:
                    lm_model = NgramModel.from_dict(_json.load(_fh))
                print(f"  Loaded LM: {len(lm_model.ragas())} ragas, order {lm_model.order}")

                # Run chromatic transcription on melody pitch data
                pitch_data = results.pitch_data_vocals
                raw_notes = transcription.transcribe_to_notes(
                    pitch_hz=pitch_data.pitch_hz,
                    timestamps=pitch_data.timestamps,
                    voicing_mask=pitch_data.voiced_mask,
                    tonic=results.detected_tonic or 0,
                    energy=pitch_data.energy,
                    energy_threshold=config.energy_threshold,
                    smoothing_sigma_ms=config.transcription_smoothing_ms,
                    min_event_duration=config.transcription_min_duration,
                    derivative_threshold=config.transcription_derivative_threshold,
                    snap_mode='chromatic',
                    transcription_min_duration=config.transcription_min_duration,
                    bias_cents=results.gmm_bias_cents or 0.0,
                )
                print(f"  Chromatic transcription: {len(raw_notes)} notes")

                # Get unique tonics from candidates
                unique_tonics = sorted(results.candidates["tonic"].unique())

                # For each (tonic, raga): correct, tokenize, score.
                # Candidates may have comma-grouped raga names (e.g.
                # "Basant, Puriya Dhanashree, Shri"); split and score each
                # raga individually against the LM.
                lm_rows = []
                for _, cand_row in results.candidates.iterrows():
                    cand_tonic = int(cand_row["tonic"])
                    raga_group = str(cand_row["raga"])
                    hist_score = float(cand_row.get("fit_score", cand_row.get("score", 0.0)))

                    # Split comma-grouped raga names
                    individual_ragas = [r.strip() for r in raga_group.split(",") if r.strip()]

                    for cand_raga in individual_ragas:
                        # Apply this raga's correction
                        try:
                            corrected_notes, corr_stats, _ = apply_raga_correction_to_notes(
                                raw_notes, raga_db, cand_raga, cand_tonic,
                                max_distance=1.0, keep_impure=False,
                            )
                        except Exception:
                            continue  # Raga not in DB
                        total = corr_stats.get("total", len(raw_notes))
                        discarded = corr_stats.get("discarded", 0)
                        deletion_rate = discarded / total if total > 0 else 1.0

                        # Scale-normalized deletion residual
                        scale_size = len(get_raga_notes(raga_db, cand_raga, cand_tonic))
                        expected_del = config.lm_deletion_slope * scale_size + config.lm_deletion_intercept
                        del_residual = deletion_rate - expected_del

                        # Match training pipeline: merge consecutive notes
                        # (same post-processing analyze applies before saving CSV)
                        corrected_notes = merge_consecutive_notes(
                            corrected_notes, max_gap=0.1, pitch_tolerance=0.7,
                            max_dropout_gap=0.18, dropout_fragment_duration=0.12,
                        )

                        # Tokenize corrected notes with this tonic
                        tonic_midi = 60.0 + cand_tonic
                        phrases = tokenize_notes_for_lm(corrected_notes, tonic_midi)

                        # Score against this raga's LM
                        lm_score = lm_model.score_sequence(cand_raga, phrases) if phrases else -999.0

                        lm_rows.append({
                            "tonic": cand_tonic,
                            "tonic_name": cand_row.get("tonic_name", _tonic_name(cand_tonic)),
                            "raga": cand_raga,
                            "histogram_score": round(hist_score, 4),
                            "lm_score": round(lm_score, 4),
                            "deletion_rate": round(deletion_rate, 4),
                            "scale_size": scale_size,
                            "expected_deletion": round(expected_del, 4),
                            "del_residual": round(del_residual, 4),
                            "notes_before_correction": total,
                            "notes_after_correction": total - discarded,
                        })

                # Histogram gate: keep candidates with positive histogram score.
                # If none pass, keep top 20 by histogram score as fallback.
                gated = [r for r in lm_rows if r["histogram_score"] > 0]
                if not gated:
                    gated = sorted(lm_rows, key=lambda r: r["histogram_score"], reverse=True)[:20]
                gated_ragas = {(r["tonic"], r["raga"]) for r in gated}

                # Normalize histogram scores within gated candidates to [0, 1].
                gated_hist = [r["histogram_score"] for r in gated]
                hist_min = min(gated_hist) if gated_hist else 0.0
                hist_max = max(gated_hist) if gated_hist else 1.0
                hist_range = hist_max - hist_min if hist_max > hist_min else 1.0

                # Normalize LM scores within gated candidates to [0, 1].
                gated_lm = [r["lm_score"] for r in lm_rows
                            if (r["tonic"], r["raga"]) in gated_ragas and r["lm_score"] > -900]
                lm_min = min(gated_lm) if gated_lm else 0.0
                lm_max = max(gated_lm) if gated_lm else 1.0
                lm_range = lm_max - lm_min if lm_max > lm_min else 1.0

                # Combined score:
                #   alpha * norm(histogram) + beta * norm(lm) - gamma * del_residual
                # Defaults: alpha=1.0, beta=1.0, gamma=2.0 (lambda)
                # norm(histogram) and norm(lm) are in [0,1]; del_residual is ~[-0.2, +0.2]
                lam = config.lm_deletion_lambda
                alpha = 1.0  # histogram weight
                beta = 1.0   # LM weight
                for row in lm_rows:
                    is_gated = (row["tonic"], row["raga"]) in gated_ragas
                    row["gated"] = is_gated
                    if is_gated:
                        norm_hist = (row["histogram_score"] - hist_min) / hist_range
                        norm_lm = (row["lm_score"] - lm_min) / lm_range
                        row["norm_histogram"] = round(norm_hist, 4)
                        row["norm_lm"] = round(norm_lm, 4)
                        row["combined_score"] = round(
                            alpha * norm_hist + beta * norm_lm - lam * row["del_residual"], 4
                        )
                    else:
                        row["norm_histogram"] = 0.0
                        row["norm_lm"] = 0.0
                        row["combined_score"] = -999.0

                # Sort by combined_score descending, assign ranks
                lm_rows.sort(key=lambda r: r["combined_score"], reverse=True)
                for i, row in enumerate(lm_rows):
                    row["lm_rank"] = i + 1

                import pandas as pd
                lm_df = pd.DataFrame(lm_rows)
                lm_csv_path = os.path.join(stem_dir, "lm_candidates.csv")
                lm_df.to_csv(lm_csv_path, index=False)
                print(f"  Saved: {lm_csv_path}")

                if lm_rows:
                    top = lm_rows[0]
                    print(f"  LM Top: {top['raga']} (tonic={top['tonic_name']}, "
                          f"combined={top['combined_score']}, lm={top['lm_score']}, "
                          f"del_resid={top['del_residual']})")

            _print_timing("Step 5.5/7 LM re-ranking", perf_counter() - step55_start, audio_duration_s)

        # --- DETECT MODE EXIT POINT ---
        if config.mode == "detect":
            step7_start = perf_counter()
            if config.skip_report:
                print("\n[STEP 7/7] Skipping report generation (--skip-report)")
            else:
                print("\n[STEP 7/7] Generating detection summary...")
                from raga_pipeline.output import generate_detection_report
                report_path = os.path.join(stem_dir, "detection_report.html")
                generate_detection_report(results, report_path)
                print(f"  Saved: {report_path}")
                try:
                    detection_meta_path = write_detection_report_metadata(
                        results,
                        report_path,
                        runtime_fingerprint=runtime_fingerprint,
                    )
                    print(f"  Saved: {detection_meta_path}")
                except Exception as exc:
                    print(f"  [WARN] Failed to write detection report metadata: {exc}")
            print("\n DETECTION COMPLETE")
            
            # Print suggested analyze command
            if results.detected_tonic is not None and results.detected_raga:
                tonic_name = _tonic_name(results.detected_tonic)
                # Use original paths as provided by user
                audio_path_arg = config.audio_path
                output_dir_arg = config.output_dir
                
                print("\n" + "=" * 60)
                print("Next: Run analyze mode with detected parameters:")
                print("=" * 60)
                print(f"./run_pipeline.sh analyze \\")
                print(f'  --audio "{audio_path_arg}" \\')
                print(f'  --output "{output_dir_arg}" \\')
                print(f'  --tonic "{tonic_name}" \\')
                print(f'  --raga "{results.detected_raga}"')
                print("=" * 60)
            else:
                print(f"Run 'analyze' mode with --tonic and --raga to continue.")
            _print_timing("Step 7/7 detection report", perf_counter() - step7_start, audio_duration_s)
            _print_timing("Total detect pipeline", perf_counter() - pipeline_start, audio_duration_s)
            
            return results

    # =========================================================================
    # PHASE 2: SEQUENCE ANALYSIS (Analyze & Full Mode)
    # =========================================================================
    
    step6_start = perf_counter()
    print("\n[STEP 6/7] Detecting notes and phrases...")
    
    stats_obj = None
    
    # Needs a tonic to proceed with sargam/transcription
    if results.detected_tonic is None:
        print("  [WARN] No tonic detected/provided. Skipping sargam analysis.")
    elif config.compare_extractors:
        # Dual-extractor compare mode
        _run_compare_extractors(
            config, results, config.stem_dir, melody_audio_path, fmin, fmax,
            raga_db, aaroh_avroh_lookup, aaroh_avroh_subset_path,
        )
        # Use primary extractor's data for plots/stats
        primary_key = config.pitch_extractor
        prim_ext = results.extractor_transcriptions.get(primary_key)
        correction_summary = prim_ext.correction_summary if prim_ext else {}
        pattern_results = prim_ext.pattern_analysis if prim_ext else {}

        # CSV + histogram (from primary)
        csv_path = os.path.join(config.stem_dir, "transcribed_notes.csv")
        save_notes_to_csv(results.notes, csv_path)
        print(f"  Saved notes: {csv_path}")

        note_duration_hist_path = os.path.join(config.stem_dir, "note_duration_histogram.png")
        plot_note_duration_histogram(results.notes, note_duration_hist_path)
        results.plot_paths["note_duration_histogram"] = note_duration_hist_path

        # Transition matrix (from primary)
        tm_matrix, tm_labels, tm_stats = build_transition_matrix_corrected(
            results.phrases, results.detected_tonic)
        tm_path = os.path.join(config.stem_dir, "transition_matrix.png")
        plot_transition_heatmap_v2(tm_matrix, tm_labels, tm_path)

        # Pitch sargam plot (from primary)
        pp_path = os.path.join(config.stem_dir, "pitch_sargam.png")
        raga_name = results.detected_raga or "Unknown"
        plot_pitch_with_sargam_lines(
            results.pitch_data_vocals.voiced_times,
            results.pitch_data_vocals.midi_vals,
            results.detected_tonic, raga_name, pp_path,
            phrase_ranges=[(p.start, p.end) for p in results.phrases if p.end > p.start],
            bias_cents=results.gmm_bias_cents,
        )

        # Note segments
        notes_path = os.path.join(config.stem_dir, "note_segments.png")
        plot_note_segments(results.pitch_data_vocals, results.notes, notes_path,
                          tonic=results.detected_tonic)

        # Stats
        stats_obj = AnalysisStats(
            correction_summary=correction_summary,
            pattern_analysis=pattern_results,
            raga_name=raga_name,
            tonic=_tonic_name(results.detected_tonic),
            transition_matrix_path=tm_path,
            pitch_plot_path=pp_path,
        )
    else:
        # Note detection - Unified Transcription
        print(f"  Using Unified Transcription (Stationary + Inflection)")
        derivative_profile = {}
        raw_notes = transcription.transcribe_to_notes(
            pitch_hz=results.pitch_data_vocals.pitch_hz,
            timestamps=results.pitch_data_vocals.timestamps,
            voicing_mask=results.pitch_data_vocals.voiced_mask,
            tonic=results.detected_tonic,
            energy=results.pitch_data_vocals.energy,
            energy_threshold=config.energy_threshold,
            smoothing_sigma_ms=config.transcription_smoothing_ms,
            min_event_duration=config.transcription_min_duration,
            derivative_threshold=config.transcription_derivative_threshold,
            snap_mode='chromatic', # Or 'raga' if we wanted strict snapping here
            transcription_min_duration=config.transcription_min_duration,
            bias_cents=results.gmm_bias_cents or 0.0,
            derivative_profile_out=derivative_profile,
        )
        results.transcription_derivative_timestamps = derivative_profile.get("timestamps")
        results.transcription_derivative_values = derivative_profile.get("values")
        results.transcription_derivative_voiced_mask = derivative_profile.get("voiced_mask")
        print(f"  Detected {len(raw_notes)} raw notes")

        stationary_count = sum(1 for n in raw_notes if n.confidence >= 0.99)
        inflection_count = len(raw_notes) - stationary_count
        print(f"  Note breakdown: Stationary={stationary_count}, Inflection={inflection_count}")
        if config.energy_threshold > 0:
            print(
                f"  Energy gating handled in transcription "
                f"(threshold={config.energy_threshold})"
            )
            
        # Apply Raga Correction
        correction_summary = {}
        if config.skip_raga_correction:
            results.notes = raw_notes
            print("  Raga correction skipped (--skip-raga-correction)")
        elif raga_db and results.detected_raga:
            print(f"  Applying raga correction for {results.detected_raga}...")
            strict_raga_max_cents = max(float(getattr(config, "strict_raga_max_cents", 35.0)), 0.0)
            raga_max_distance = (strict_raga_max_cents / 100.0) if config.strict_raga_35c_filter else 1.0
            keep_impure_notes = config.keep_impure_notes
            if config.strict_raga_35c_filter and keep_impure_notes:
                keep_impure_notes = False
                print(
                    "  [INFO] --strict-raga-35c-filter enabled; "
                    "--keep-impure-notes is ignored for correction."
                )
            if config.strict_raga_35c_filter:
                print(f"  [INFO] Strict raga filter window: +/-{strict_raga_max_cents:.1f} cents")
            corrected_notes, correction_stats, _ = apply_raga_correction_to_notes(
                raw_notes,
                raga_db,
                results.detected_raga,
                results.detected_tonic,
                max_distance=raga_max_distance,
                keep_impure=keep_impure_notes,
            )
            results.notes = corrected_notes
            correction_summary = correction_stats
            print(f"  Corrected notes: {len(results.notes)} (Discarded: {correction_stats['discarded']})")
        else:
            results.notes = raw_notes
            print("  [WARN] Raga DB or name missing, skipping correction.")
        
        # First-pass merge for small-gap fragmentation.
        results.notes = merge_consecutive_notes(
            results.notes,
            max_gap=0.1,
            pitch_tolerance=0.7,
            max_dropout_gap=0.18,
            dropout_fragment_duration=0.12,
        )
        print(f"  Initial note merge: {len(results.notes)} notes")

        # Phrase detection (using corrected notes)
        results.phrases = detect_phrases(
            results.notes,
            max_gap=config.phrase_max_gap,
            min_length=config.phrase_min_length,
            min_phrase_duration=config.phrase_min_duration,
        )
        print(f"  Detected {len(results.phrases)} phrases (gap-based)")

        # Silence-based phrase splitting (optional, uses vocal RMS energy)
        silence_thresh = config.silence_threshold
        if silence_thresh <= 0 and config.energy_threshold > 0:
            # Default: reuse the transcription energy threshold so the user
            # gets silence-aware splitting automatically when energy gating is on.
            silence_thresh = config.energy_threshold

        if silence_thresh > 0 and results.pitch_data_vocals is not None:
            pre_count = len(results.phrases)
            results.phrases = split_phrases_by_silence(
                phrases=results.phrases,
                energy=results.pitch_data_vocals.energy,
                timestamps=results.pitch_data_vocals.timestamps,
                silence_threshold=silence_thresh,
                silence_min_duration=config.silence_min_duration,
            )
            print(f"  Silence split ({silence_thresh:.2f} thresh, "
                  f"{config.silence_min_duration:.2f}s min): "
                  f"{pre_count} -> {len(results.phrases)} phrases")

        # Collapse repeated consecutive note segments within each phrase so
        # sustained notes are represented once per phrase.
        if results.phrases:
            pre_collapse_notes = sum(len(phrase.notes) for phrase in results.phrases)
            collapsed_phrases = []
            for phrase in results.phrases:
                collapsed_notes = merge_consecutive_notes(
                    phrase.notes,
                    max_gap=config.phrase_max_gap,
                    pitch_tolerance=0.7,
                    max_dropout_gap=config.phrase_max_gap,
                    dropout_fragment_duration=config.phrase_max_gap,
                )
                collapsed_phrases.append(phrase.__class__(notes=collapsed_notes))
            results.phrases = collapsed_phrases
            results.notes = [note for phrase in results.phrases for note in phrase.notes]
            post_collapse_notes = len(results.notes)
            print(
                f"  Phrase-level collapse: {pre_collapse_notes} -> "
                f"{post_collapse_notes} note segments"
            )

        # Note duration histogram + CSV use the phrase-collapsed note list.
        note_duration_hist_path = os.path.join(config.stem_dir, "note_duration_histogram.png")
        plot_note_duration_histogram(
            results.notes,
            note_duration_hist_path,
            title="Transcribed Note Duration Distribution",
        )
        results.plot_paths["note_duration_histogram"] = note_duration_hist_path
        print(f"  Saved: {note_duration_hist_path}")
        
        csv_path = os.path.join(config.stem_dir, "transcribed_notes.csv")
        save_notes_to_csv(results.notes, csv_path)
        print(f"  Saved notes: {csv_path}")

        # --- TRANSCRIPTION-ONLY EARLY EXIT ---
        if config.transcription_only:
            print("\n[--transcription-only] transcribed_notes.csv saved. Skipping plots, GMM, and report.")
            _print_timing("Total analyze pipeline (transcription-only)", perf_counter() - pipeline_start, audio_duration_s)
            return results

        # Final phrase exclusion pass after optional silence re-splitting.
        pre_filter_count = len(results.phrases)
        results.phrases = [
            phrase
            for phrase in results.phrases
            if phrase.duration >= config.phrase_min_duration
            and len(phrase.notes) >= config.phrase_min_length
        ]
        print(f"  Phrase filter (>= {config.phrase_min_duration:.2f}s, "
              f">= {config.phrase_min_length} notes): "
              f"{pre_filter_count} -> {len(results.phrases)} phrases")
        
        # Phrase clustering
        results.phrase_clusters = cluster_phrases(results.phrases)
        print(f"  Found {len(results.phrase_clusters)} phrase clusters")
        
        # Transition matrix (Corrected)
        tm_matrix, tm_labels, tm_stats = build_transition_matrix_corrected(
            results.phrases, results.detected_tonic
        )
        tm_path = os.path.join(config.stem_dir, "transition_matrix.png")
        plot_transition_heatmap_v2(tm_matrix, tm_labels, tm_path)
        print(f"  Saved: {tm_path}")
        
        # Advanced Pattern Analysis
        # Includes sequences, motifs, and Aaroh/Avroh runs
        expected_pattern = None
        if aaroh_avroh_lookup and results.detected_raga:
            expected_pattern = get_aaroh_avroh_pattern_for_raga(
                results.detected_raga,
                aaroh_avroh_lookup,
            )

        if expected_pattern:
            pattern_results = analyze_raga_patterns(
                results.phrases,
                results.detected_tonic,
                expected_aaroh=expected_pattern.aaroh_pattern,
                expected_avroh=expected_pattern.avroh_pattern,
            )
            pattern_results["aaroh_avroh_reference"] = {
                "matched_name": expected_pattern.raga_name,
                "source_name": expected_pattern.source_raga_name,
                "aaroh_raw": expected_pattern.aaroh_raw,
                "avroh_raw": expected_pattern.avroh_raw,
                "subset_csv": aaroh_avroh_subset_path,
            }
        else:
            pattern_results = analyze_raga_patterns(results.phrases, results.detected_tonic)

        print(f"  Pattern Analysis:")
        print(f"    - Common Motifs: {len(pattern_results['common_motifs'])}")
        print(f"    - Aaroh Runs: {pattern_results['total_aaroh_runs']}")
        print(f"    - Avroh Runs: {pattern_results['total_avroh_runs']}")
        checker = pattern_results.get("aaroh_avroh_checker")
        if checker:
            print(
                f"    - Aaroh/Avroh Check: score={checker['score']:.3f} "
                f"({checker['matched_checks']}/{checker['total_checks']})"
            )
            mismatches = (
                len(checker["missing_aaroh"]) +
                len(checker["missing_avroh"]) +
                len(checker["unexpected_aaroh"]) +
                len(checker["unexpected_avroh"])
            )
            print(f"    - Directional Mismatches: {mismatches}")
        elif results.detected_raga:
            print(f"    - Aaroh/Avroh Check: No pattern entry found for '{results.detected_raga}'")
        
        # Display top motifs
        if pattern_results['common_motifs']:
            print("    Top Motif: " + pattern_results['common_motifs'][0]['pattern'])
        
        # Detailed Pitch Plot with Sargam Lines
        pp_path = os.path.join(config.stem_dir, "pitch_sargam.png")
        raga_name = results.detected_raga or "Unknown"
        plot_pitch_with_sargam_lines(
            results.pitch_data_vocals.voiced_times,
            results.pitch_data_vocals.midi_vals,
            results.detected_tonic,
            raga_name,
            pp_path,
            phrase_ranges=[(phrase.start, phrase.end) for phrase in results.phrases if phrase.end > phrase.start],
            bias_cents=results.gmm_bias_cents,
        )
        print(f"  Saved: {pp_path}")

        # Note Segments (Legacy plot, still useful)
        notes_path = os.path.join(config.stem_dir, "note_segments.png")
        plot_note_segments(
            results.pitch_data_vocals,
            results.notes,
            notes_path,
            tonic=results.detected_tonic,
        )
        
        # Prepare AnalysisStats
        stats_obj = AnalysisStats(
            correction_summary=correction_summary,
            pattern_analysis=pattern_results,
            raga_name=raga_name,
            tonic=_tonic_name(results.detected_tonic),
            transition_matrix_path=tm_path,
            pitch_plot_path=pp_path
        )
    _print_timing("Step 6/7 note + phrase analysis", perf_counter() - step6_start, audio_duration_s)

    # =========================================================================
    # STEP 6.5: GMM Analysis
    # =========================================================================
    
    if results.histogram_vocals is None and results.pitch_data_vocals is not None:
         results.histogram_vocals = compute_cent_histograms_from_config(results.pitch_data_vocals, config)
         results.peaks_vocals = detect_peaks_from_config(results.histogram_vocals, config)

    step65_start = perf_counter()
    if results.peaks_vocals is not None and results.histogram_vocals is not None:
        print("\n[STEP 6.5/7] GMM microtonal analysis...")
        results.gmm_results = fit_gmm_to_peaks(
            results.histogram_vocals,
            results.peaks_vocals.validated_indices,
            window_cents=config.gmm_window_cents,
            n_components=config.gmm_components,
            tonic=results.detected_tonic, # Pass tonic for rotation
        )

        if config.bias_rotation and results.gmm_bias_cents is None:
            results.gmm_bias_cents = compute_gmm_bias_cents(results.gmm_results)
        
        if results.gmm_results:
            gmm_path = os.path.join(config.stem_dir, "gmm_overlay.png")
            plot_gmm_overlay(
                results.histogram_vocals,
                results.gmm_results,
                gmm_path,
                tonic=results.detected_tonic,
                bias_cents=results.gmm_bias_cents,
            )
            results.plot_paths["gmm_overlay"] = gmm_path
            print(f"  Saved: {gmm_path}")
        _print_timing("Step 6.5/7 GMM analysis", perf_counter() - step65_start, audio_duration_s)
    else:
        print("[TIMER] Step 6.5/7 GMM analysis: skipped")

    # =========================================================================
    # STEP 7: HTML Report (Analysis/Full)
    # =========================================================================
    step7_start = perf_counter()
    if config.skip_report:
        print("\n[STEP 7/7] Skipping report generation (--skip-report)")
        report_path = os.path.join(config.stem_dir, "analysis_report.html")
    else:
        print("\n[STEP 7/7] Generating full analysis report...")

        if config.mode == "analyze" and stats_obj:
            report_path = generate_analysis_report(
                results,
                stats_obj,
                config.stem_dir,
                runtime_fingerprint=runtime_fingerprint,
            )
        else:
            if stats_obj:
                report_path = generate_analysis_report(
                    results,
                    stats_obj,
                    config.stem_dir,
                    runtime_fingerprint=runtime_fingerprint,
                )
            else:
                report_path = os.path.join(config.stem_dir, "report.html")
                generate_html_report(results, report_path)

        print(f"  Saved: {report_path}")
    results.plot_paths["report"] = report_path
    _print_timing("Step 7/7 analysis report", perf_counter() - step7_start, audio_duration_s)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Detected Raga: {results.detected_raga}")
    print(f"Detected Tonic: {_tonic_name(results.detected_tonic)}")
    _print_timing("Total analysis pipeline", perf_counter() - pipeline_start, audio_duration_s)
    
    return results


def _tonic_name(tonic):
    """Convert tonic to note name."""
    if tonic is None:
        return "Unknown"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[tonic % 12]


def main():
    """Main entry point for CLI."""
    # Use centralized config parsing
    config = load_config_from_cli()
    
    # Run pipeline
    run_pipeline(config)
    
    return


if __name__ == "__main__":
    main()
