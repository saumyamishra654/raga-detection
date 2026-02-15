#!/usr/bin/env python3
"""
driver file for raga detection pipeline - do not run directly, run using run_pipeline.sh

modes:
    preprocess - download YouTube audio to local MP3 and exit
    detect  - run phase 1 of analysis (histogram-based raga filtering/ranking)
    analyze - run phase 2 of analysis (note sequence analysis given tonic/raga)
"""

import os
import sys
from pathlib import Path

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
    plot_histograms,
    plot_absolute_note_histogram,
    plot_gmm_overlay,
    plot_note_segments,
    generate_html_report,
    generate_analysis_report,
    plot_pitch_with_sargam_lines,
    plot_pitch_with_sargam_lines,
    plot_transition_heatmap_v2,
    save_notes_to_csv,
)
from raga_pipeline import transcription


def run_pipeline(
    config: PipelineConfig,
    run_histogram_analysis: bool = True,  # Legacy flag, kept for compatibility but effectively controlled by mode
    run_sequence_analysis: bool = True,   # Legacy flag
) -> AnalysisResults:
    """
    run the raga detection pipeline in the configured mode

    - preprocess: Download YouTube audio to local MP3 and exit.
    - detect: Run up to raga candidate scoring, generate summary report.
    - analyze: Load cached pitch, run sequence/phrase analysis using provided tonic/raga.
    
    Source Types:
    - mixed: uses stem separation (default)
    - instrumental: skips stem separation if --skip-separation is set, otherwise uses stem separation
    - vocal: uses stem separation (vocal isolation) + gender-specific tonic bias
    
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
        print(f"YouTube URL: {config.yt_url}")
        print(f"Audio Dir: {config.audio_dir}")
        print(f"Filename: {config.filename_override}.mp3")
        print(f"Start Time: {config.preprocess_start_time or '0:00'}")
        print(f"End Time: {config.preprocess_end_time or 'track end'}")
        print()

        try:
            downloaded_audio_path = download_youtube_audio(
                yt_url=config.yt_url or "",
                audio_dir=config.audio_dir or "",
                filename_base=config.filename_override or "",
                start_time=config.preprocess_start_time,
                end_time=config.preprocess_end_time,
            )
        except Exception as exc:
            print(f"[PREPROCESS] ERROR: {exc}")
            print("[PREPROCESS] Tip: try updating yt-dlp and retrying with a different/public video URL.")
            raise SystemExit(1)

        config.audio_path = downloaded_audio_path

        print("[PREPROCESS] Download complete")
        print(f"  Saved: {downloaded_audio_path}")

        print("\n" + "=" * 60)
        print("Next: Run detect mode with downloaded audio:")
        print("=" * 60)
        print(f"./run_pipeline.sh detect \\")
        print(f'  --audio "{downloaded_audio_path}" \\')
        print(f'  --output "{config.output_dir}"')
        print("=" * 60)

        return results
    
    print("=" * 60)
    print("RAGA DETECTION PIPELINE")
    print(f"MODE: {config.mode.upper()}")
    print("=" * 60)
    print(f"Audio: {config.filename}")
    print(f"Output: {config.output_dir}")
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
            candidate_csv = os.path.join(config.stem_dir, f"{prefix}_pitch_data.csv")
            if os.path.exists(candidate_csv):
                melody_prefix = prefix
                melody_pitch_csv = candidate_csv
                break

        if melody_prefix is None:
            melody_prefix = preferred_prefix
            melody_pitch_csv = os.path.join(config.stem_dir, f"{melody_prefix}_pitch_data.csv")
        accomp_pitch_csv = os.path.join(config.stem_dir, "accompaniment_pitch_data.csv")
        composite_pitch_csv = os.path.join(config.stem_dir, "composite_pitch_data.csv")
        assert melody_pitch_csv is not None
        
        if not os.path.exists(melody_pitch_csv):
            raise FileNotFoundError(
                f"Cached melody pitch data not found at {melody_pitch_csv}. Run 'detect' phase first."
            )
            
        # dummy wrapper for loading parameters
        fmin = float(librosa.note_to_hz(config.fmin_note))
        fmax = float(librosa.note_to_hz(config.fmax_note))
        
        # Load Primary Melody Stem
        pitch_data_stems = extract_pitch(
            audio_path=config.vocals_path, # Path reference for metadata
            output_dir=config.stem_dir,
            prefix=melody_prefix,
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.vocal_confidence,
            force_recompute=config.force_recompute,
            energy_metric=config.energy_metric,
        )
        results.pitch_data_stem = pitch_data_stems
        
        # Load Composite if exists
        if os.path.exists(composite_pitch_csv):
            results.pitch_data_composite = extract_pitch(
                audio_path=config.audio_path,
                output_dir=config.stem_dir,
                prefix="composite",
                fmin=fmin,
                fmax=fmax,
                confidence_threshold=config.vocal_confidence,
                force_recompute=config.force_recompute,
                energy_metric=config.energy_metric,
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

    else:
        # DETECT or FULL MODE: run detection pipeline
        
        # STEP 1: Stems
        print("[STEP 1/7] Audio preprocessing...")
        
        # Always run stem separation for all modes (mixed, vocal, instrumental)
        # unless specifically skipped for debugging (but we remove the skip-separation flag logic for instrumental)
        print(f"  Source type: {config.source_type} - Running stem separation")
        melody_path, accompaniment_path = separate_stems(
            audio_path=config.audio_path,
            output_dir=config.output_dir,
            engine=config.separator_engine,
            model=config.demucs_model,
        )
        stem_dir = config.stem_dir
        
        # STEP 2: Pitch
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
        )
        
        # 2b. Stem Pitch (Vocals/Melody) - Always computed for reference/fallback
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
        )
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
        
        # STEP 4 & 5: Raga Matching
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

        # --- DETECT MODE EXIT POINT ---
        if config.mode == "detect":
            print("\n[STEP 7/7] Generating detection summary...")
            from raga_pipeline.output import generate_detection_report
            report_path = os.path.join(stem_dir, "detection_report.html")
            generate_detection_report(results, report_path)
            print(f"  Saved: {report_path}")
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
            
            return results

    # =========================================================================
    # PHASE 2: SEQUENCE ANALYSIS (Analyze & Full Mode)
    # =========================================================================
    
    print("\n[STEP 6/7] Detecting notes and phrases...")
    
    stats_obj = None
    
    # Needs a tonic to proceed with sargam/transcription
    if results.detected_tonic is None:
        print("  [WARN] No tonic detected/provided. Skipping sargam analysis.")
    else:
        # Note detection
        # Note detection - Unified Transcription
        print(f"  Using Unified Transcription (Stationary + Inflection)")
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
            transcription_min_duration=config.transcription_min_duration
        )
        print(f"  Detected {len(raw_notes)} raw notes")
        
        # Energy Filtering
        if config.energy_threshold > 0:
            original_count = len(raw_notes)
            raw_notes = [n for n in raw_notes if n.energy >= config.energy_threshold]
            filtered_count = len(raw_notes)
            print(f"  Energy Filter ({config.energy_threshold}): Kept {filtered_count}/{original_count} notes")
            
        # Apply Raga Correction
        correction_summary = {}
        if raga_db and results.detected_raga:
            print(f"  Applying raga correction for {results.detected_raga}...")
            corrected_notes, correction_stats, _ = apply_raga_correction_to_notes(
                raw_notes, 
                raga_db, 
                results.detected_raga, 
                results.detected_tonic,
                keep_impure=config.keep_impure_notes
            )
            results.notes = corrected_notes
            correction_summary = correction_stats
            print(f"  Corrected notes: {len(results.notes)} (Discarded: {correction_stats['discarded']})")
        else:
            results.notes = raw_notes
            print("  [WARN] Raga DB or name missing, skipping correction.")
        
        # Merge consecutive notes to fix fragmentation
        results.notes = merge_consecutive_notes(results.notes, max_gap=0.1, pitch_tolerance=0.7)
        print(f"  Merged {len(results.notes)} notes (joined consecutive identical notes)")
        
        # Save transcription to CSV
        csv_path = os.path.join(config.stem_dir, "transcribed_notes.csv")
        save_notes_to_csv(results.notes, csv_path)
        print(f"  Saved notes: {csv_path}")

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

    # =========================================================================
    # STEP 6.5: GMM Analysis
    # =========================================================================
    
    if results.histogram_vocals is None and results.pitch_data_vocals is not None:
         results.histogram_vocals = compute_cent_histograms_from_config(results.pitch_data_vocals, config)
         results.peaks_vocals = detect_peaks_from_config(results.histogram_vocals, config)

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

    # =========================================================================
    # STEP 7: HTML Report (Analysis/Full)
    # =========================================================================
    print("\n[STEP 7/7] Generating full analysis report...")
    
    if config.mode == "analyze" and stats_obj:
        report_path = generate_analysis_report(results, stats_obj, config.stem_dir)
    else:
        # Fallback for full mode or if somethings missing (though Full usually implies analysis)
        # If in FULL mode, we also have stats_obj from above.
        if stats_obj:
            report_path = generate_analysis_report(results, stats_obj, config.stem_dir)
        else:
            report_path = os.path.join(config.stem_dir, "report.html")
            generate_html_report(results, report_path)
            
    results.plot_paths["report"] = report_path
    
    print(f"  Saved: {report_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Detected Raga: {results.detected_raga}")
    print(f"Detected Tonic: {_tonic_name(results.detected_tonic)}")
    
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
