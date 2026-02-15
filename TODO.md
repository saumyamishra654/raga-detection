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
- [x] Add portable HTML audio source fallback (local file + relative path)
- [x] Update README CLI flags for energy overlay and silence splitting
- [x] Set analyze-mode defaults for energy and silence thresholds
- [x] Fix analyze-mode cache validation indentation error
- [x] Update transcription snapping to raga fallback or skip
- [x] Add detect-mode tonic/raga constraints with comma-separated tonics
- [x] Implement aaroh/avroh reconstruction from directional note context (see `aaroh-avroh-reconstruction.md`)
- [x] Skip AppleDouble/hidden files in batch processing
- [x] Add detect-mode duration-weighted stationary-note histogram (octave-wrapped, 12-bin)
- [x] Remove detect-mode unweighted stationary-note histogram (keep duration-weighted only)
- [x] Add standalone preprocess mode (`--yt --audio-dir --filename`) with copyable next-step detect command
- [x] Set preprocess default `--audio-dir` to `../audio_test_files`
- [x] Add preprocess `--start-time`/`--end-time` trimming with duration validation
- [x] Resolve `output.py` report-generation type errors after preprocess mode migration
- [x] Trim accidental CLI whitespace in path-like args (`--audio`, `--audio-dir`, `--filename`) before validation
- [ ] Tune gating for different instrument types (sitar vs vocal)
- [ ] Refactor and clean up redundancies in `raga_pipeline/sequence.py` (legacy transition matrices, overlapping note detection methods, duplicate sargam logic)
- [ ] Implement sequence mining for raga identification (see `sequence-mining-plan.md`)

## Future Ideas

- [ ] Integration with more pitch tracking engines (e.g., CREPE), and stem separation engines
- [ ] Implement multi-raga scoring segments (for Ragamala detection)
- [ ] Experiment with Taal detection, think about possible approaches there

## Web Application & Mobile App

- [ ] Review and refine implementation plan (see `WEB_APP_IMPLEMENTATION_PLAN.md`)
- [ ] Set up infrastructure (Supabase, Cloudflare R2, RunPod)
- [ ] Build FastAPI backend with job queue
- [ ] Develop React web frontend
- [ ] Build React Native mobile app (Expo)
- [ ] Beta testing and launch

## Thesis Demo (FastAPI + React)

- [ ] Review implementation plan (see `THESIS_DEMO_PLAN.md`)
- [ ] Week 1-2: FastAPI backend setup with background job processing
- [ ] Week 3-5: React frontend development with TypeScript + Tailwind
- [ ] Week 6-7: Sample library curation and pre-computation
- [ ] Week 8: Deploy to Railway + Vercel, documentation, class presentation prep

## Pitch Matching Game (Future Project)

- [ ] Review comprehensive implementation plan (see `PITCH_MATCHING_GAME_PLAN.md`)
- [ ] Week 1: Proof of concept - mic access, real-time pitch detection, latency validation
- [ ] Week 2-3: Reference track processing (Demucs + SwiftF0integration)
- [ ] Week 4-5: Real-time visualization (scrolling notation, color-coded feedback)
- [ ] Week 6-7: Post-analysis (DTW alignment, multi-dimensional scoring)
- [ ] Week 8-9: Full stack integration (user accounts, progress tracking)
- [ ] Week 10-12: Polish, gamification, beta testing, launch
