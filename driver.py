#!/usr/bin/env python3
"""
Raga Detection Pipeline - Main Driver

Run the complete raga detection pipeline on an audio file:
    python driver.py --audio path/to/audio.mp3 --output path/to/output

Modes:
    full      - Run both histogram and sequence analysis (default)
    histogram - Run only histogram-based raga detection
    sequence  - Run only note sequence analysis
"""

import os
import sys
from pathlib import Path

# Add package to path if running directly
if __name__ == "__main__":
    package_dir = Path(__file__).parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

import librosa

from raga_pipeline.config import PipelineConfig, load_config_from_cli, create_config
from raga_pipeline.audio import separate_stems, extract_pitch, extract_pitch_from_config
from raga_pipeline.analysis import (
    compute_cent_histograms, 
    detect_peaks, 
    fit_gmm_to_peaks,
    compute_cent_histograms_from_config,
    detect_peaks_from_config,
)
from raga_pipeline.raga import (
    RagaDatabase, 
    generate_candidates, 
    RagaScorer,
    apply_raga_correction_to_notes,
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
)
from raga_pipeline.output import (
    AnalysisResults,
    AnalysisStats,
    plot_histograms,
    plot_gmm_overlay,
    plot_note_segments,
    generate_html_report,
    generate_analysis_report,
    plot_pitch_with_sargam_lines,
    plot_transition_heatmap_v2,
)


def run_pipeline(
    config: PipelineConfig,
    run_histogram_analysis: bool = True,  # Legacy flag, kept for compatibility but effectively controlled by mode
    run_sequence_analysis: bool = True,   # Legacy flag
) -> AnalysisResults:
    """
    Run the raga detection pipeline in the configured mode.
    
    Modes:
    - detect: Run up to raga candidate scoring, generate summary report.
    - analyze: Load cached pitch, run sequence/phrase analysis using provided tonic/raga.
    - full: Run both phases (auto-detect then analyze top candidate).
    
    Args:
        config: Pipeline configuration
        
    Returns:
        AnalysisResults with computed data
    """
    results = AnalysisResults(config=config)
    
    print("=" * 60)
    print("🎵 RAGA DETECTION PIPELINE")
    print(f"MODE: {config.mode.upper()}")
    print("=" * 60)
    print(f"Audio: {config.filename}")
    print(f"Output: {config.output_dir}")
    print()
    
    # Check Raga DB path early
    raga_db = None
    if config.raga_db_path and os.path.isfile(config.raga_db_path):
        raga_db = RagaDatabase(config.raga_db_path)
    
    # =========================================================================
    # PHASE 1: DETECTION (or load cache)
    # =========================================================================
    
    if config.mode == "analyze":
        # ANALYZE MODE: Skip detection, load cache
        print("[PHASE 1] Skipping detection (Analyze Mode)")
        
        # Load pitch data from cache
        print("[CACHE] Loading pitch data...")
        # Re-initialize pitch data objects from cache files
        # We need to construct paths manually since they are usually returned by extract_pitch
        vocals_pitch_csv = os.path.join(config.stem_dir, "vocals_pitch_data.csv")
        accomp_pitch_csv = os.path.join(config.stem_dir, "accompaniment_pitch_data.csv")
        
        if not os.path.exists(vocals_pitch_csv):
            raise FileNotFoundError(f"Cached pitch data not found at {vocals_pitch_csv}. Run 'detect' phase first.")
            
        # We need to re-load pitch data. existing extract_pitch function handles cache loading
        # if we call it with same params and files exist.
        # But we need the stem paths.
        vocals_path = config.vocals_path
        accomp_path = config.accompaniment_path
        
        # Create dummy wrapper for loading
        fmin = librosa.note_to_hz(config.fmin_note)
        fmax = librosa.note_to_hz(config.fmax_note)
        
        results.pitch_data_vocals = extract_pitch(
            audio_path=vocals_path,
            output_dir=config.stem_dir,
            prefix="vocals",
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.vocal_confidence,
            force_recompute=False # Always load cache
        )
        
        results.pitch_data_accomp = extract_pitch(
            audio_path=accomp_path,
            output_dir=config.stem_dir,
            prefix="accompaniment",
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.accomp_confidence,
            force_recompute=False
        )
        
        # Validate required inputs for analysis
        if not config.tonic_override or not config.raga_override:
             raise ValueError("Analyze mode requires --tonic and --raga arguments")
            
        # Parse tonic
        from raga_pipeline.raga import _parse_tonic
        results.detected_tonic = _parse_tonic(config.tonic_override)
        results.detected_raga = config.raga_override
        
        print(f"  Using Force Tonic: {config.tonic_override} -> {results.detected_tonic}")
        print(f"  Using Force Raga: {config.raga_override}")

    else:
        # DETECT or FULL MODE: Run detection pipeline
        
        # STEP 1: Stems
        print("[STEP 1/7] Separating stems...")
        vocals_path, accompaniment_path = separate_stems(
            audio_path=config.audio_path,
            output_dir=config.output_dir,
            engine=config.separator_engine,
            model=config.demucs_model,
        )
        
        # STEP 2: Pitch
        print("\n[STEP 2/7] Extracting pitch...")
        fmin = librosa.note_to_hz(config.fmin_note)
        fmax = librosa.note_to_hz(config.fmax_note)
        
        print("  - Vocals...")
        results.pitch_data_vocals = extract_pitch(
            audio_path=vocals_path,
            output_dir=config.stem_dir,
            prefix="vocals",
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.vocal_confidence,
            force_recompute=config.force_recompute,
        )
        
        print("  - Accompaniment...")
        results.pitch_data_accomp = extract_pitch(
            audio_path=accompaniment_path,
            output_dir=config.stem_dir,
            prefix="accompaniment",
            fmin=fmin,
            fmax=fmax,
            confidence_threshold=config.accomp_confidence,
            force_recompute=config.force_recompute,
        )
        
        # STEP 3: Histograms & Peaks
        print("\n[STEP 3/7] Computing histograms and detecting peaks...")
        results.histogram_vocals = compute_cent_histograms_from_config(results.pitch_data_vocals, config)
        results.histogram_accomp = compute_cent_histograms_from_config(results.pitch_data_accomp, config)
        results.peaks_vocals = detect_peaks_from_config(results.histogram_vocals, config)
        
        print(f"  Detected {len(results.peaks_vocals.validated_indices)} validated peaks")
        
        # Plot histograms
        hist_path = os.path.join(config.stem_dir, "histogram_vocals.png")
        plot_histograms(results.histogram_vocals, results.peaks_vocals, hist_path, title="Vocal Pitch Distribution")
        results.plot_paths["histogram_vocals"] = hist_path
        print(f"  Saved: {hist_path}")
        
        # STEP 4 & 5: Raga Matching
        print("\n[STEP 4/7] Loading raga database...")
        # Already loaded raga_db at start if available
        if raga_db:
            print(f"  Loaded {len(raga_db.all_ragas)} ragas")
            
            print("\n[STEP 5/7] Scoring candidates...")
            scorer = RagaScorer(raga_db=raga_db, model_path=config.model_path, use_ml=config.use_ml_model)
            results.candidates = scorer.score(
                pitch_data_vocals=results.pitch_data_vocals,
                pitch_data_accomp=results.pitch_data_accomp,
                detected_peak_count=len(results.peaks_vocals.validated_indices),
                instrument_mode=config.instrument_mode,
            )
            
            # Save candidates
            candidates_path = os.path.join(config.stem_dir, "candidates.csv")
            results.candidates.to_csv(candidates_path, index=False)
            print(f"  Saved: {candidates_path}")
            
            # Helper to parse tonic
            from raga_pipeline.raga import _parse_tonic
            
            # Determine Top Candidate / Override
            if config.tonic_override:
                results.detected_tonic = _parse_tonic(config.tonic_override)
                print(f"  Force Tonic: {config.tonic_override}")
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
            report_path = os.path.join(config.stem_dir, "detection_report.html")
            generate_detection_report(results, report_path)
            print(f"  Saved: {report_path}")
            print("\n✅ DETECTION COMPLETE")
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
        raw_notes = detect_notes(results.pitch_data_vocals, config)
        print(f"  Detected {len(raw_notes)} raw notes")
        
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
        
        # Phrase detection (using corrected notes)
        results.phrases = detect_phrases(
            results.notes,
            max_gap=config.phrase_max_gap,
            min_length=config.phrase_min_length,
        )
        print(f"  Detected {len(results.phrases)} phrases")
        
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
        
        # Motif Analysis
        sequences = extract_melodic_sequences(results.phrases, results.detected_tonic)
        common_motifs = find_common_patterns(sequences)
        print(f"  Found {len(common_motifs)} common motifs")
        
        # Detailed Pitch Plot with Sargam Lines
        pp_path = os.path.join(config.stem_dir, "pitch_sargam.png")
        plot_pitch_with_sargam_lines(
            results.pitch_data_vocals.voiced_times,
            results.pitch_data_vocals.midi_vals,
            results.detected_tonic,
            results.detected_raga,
            pp_path
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
            motif_analysis=common_motifs,
            raga_name=results.detected_raga,
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

    if results.peaks_vocals is not None:
        print("\n[STEP 6.5/7] GMM microtonal analysis...")
        results.gmm_results = fit_gmm_to_peaks(
            results.histogram_vocals,
            results.peaks_vocals.validated_indices,
            window_cents=config.gmm_window_cents,
            n_components=config.gmm_components,
            tonic=results.detected_tonic, # Pass tonic for rotation
        )
        
        if results.gmm_results:
            gmm_path = os.path.join(config.stem_dir, "gmm_overlay.png")
            plot_gmm_overlay(results.histogram_vocals, results.gmm_results, gmm_path)
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
    print("✅ ANALYSIS COMPLETE")
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
