# Raga Detection Pipeline TODO

## Short Term
- [x] Fix and experiment with instrumental pitch tracking, both with and without stem separation
- [x] Align pipeline peak detection with reference notebook (solost.mp3 fix)
- [x] Get Unified Transcription (Stationary + Inflection) working
- [x] Implement Multi-Track Audio Sync (Original/Vocals/Accompaniment) in reports
- [x] Add energy over time plot in analyze report
- [x] Add optional GMM bias rotation and mass-gated scoring
- [x] Default batch `--ground-truth` to `<input_dir>_gt.csv` stored alongside the input directory
- [x] Save separated stems as MP3 (with legacy WAV fallback)
- [x] Implement scrollable karaoke-style text transcription overlay
- [x] Add optional RMS overlay on pitch plots in reports
- [x] Split phrases using silence thresholds from RMS energy
- [x] Update README CLI flags for energy overlay and silence splitting
- [ ] Add CLI flags to expose `cluster_notes_into_phrases` parameters (method, threshold, etc.)
- [ ] Tune energy-based gating for different instrument types (sitar vs vocal)
- [ ] Refactor and clean up redundancies in `raga_pipeline/sequence.py` (legacy transition matrices, overlapping note detection methods, duplicate sargam logic)

## Future Ideas
- [ ] Integration with more pitch tracking engines (e.g., CREPE), and stem separation engines
- [ ] Implement multi-raga scoring segments (for Ragamala detection)
- [ ] Experiment with Taal detection, think about possible approaches there

## Python Backend Migration
- [ ] Refactor `driver.py` into a stateless worker task for Celery.
- [ ] Implement a FastAPI wrapper for job management and progress tracking.
- [ ] Migrate local outputs (PNG/CSV) to cloud storage (S3/GCS).
- [ ] Expose Plotly JSON generators via API endpoints.
