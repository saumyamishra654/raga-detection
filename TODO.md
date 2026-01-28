# Raga Detection Pipeline TODO

## Short Term
- [x] Fix and experiment with instrumental pitch tracking, both with and without stem separation
- [x] Get Unified Transcription (Stationary + Inflection) working
- [x] Implement Multi-Track Audio Sync (Original/Vocals/Accompaniment) in reports
- [ ] Implement scrollable karaoke-style text transcription overlay
- [ ] Add CLI flags to expose `cluster_notes_into_phrases` parameters (method, threshold, etc.)
- [ ] Tune energy-based gating for different instrument types (sitar vs vocal)

## Future Ideas
- [ ] Integration with more pitch tracking engines (e.g., CREPE), and stem separation engines
- [ ] Implement multi-raga scoring segments (for Ragamala detection)
- [ ] Experiment with Taal detection, think about possible approaches there

## Python Backend Migration
- [ ] Refactor `driver.py` into a stateless worker task for Celery.
- [ ] Implement a FastAPI wrapper for job management and progress tracking.
- [ ] Migrate local outputs (PNG/CSV) to cloud storage (S3/GCS).
- [ ] Expose Plotly JSON generators via API endpoints.
