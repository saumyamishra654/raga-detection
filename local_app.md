# Local Desktop App Plan

## Goal
Turn the current local web app and pipeline into a robust, installable desktop application that runs fully on a user laptop, with no cloud dependency.

Phone-assisted recording is a secondary extension, not part of the first release gate.

## Product Scope

### Primary (Desktop First)
- Native desktop app shell that launches and manages the existing local app UI and Python backend.
- Full preprocess, detect, analyze workflows.
- Local audio recording, tanpura-assisted flow, artifact browsing, and report opening.
- Works offline after install.

### Secondary (Later)
- Phone companion for recording only.
- Transfer recorded audio to desktop app for full processing.

## Architecture Decision

### Recommended
- Use a desktop shell around the existing app (Tauri preferred).
- Keep current Python backend (`driver.py`, `raga_pipeline`, `local_app/server.py`) as the processing engine.
- Keep current frontend (`local_app/static/app.js`) as the operator UI.

### Why
- Minimum rewrite risk.
- Fastest path to production-quality desktop distribution.
- Preserves existing tested pipeline behavior.

## System Design

### App Components
1. Desktop shell process:
- Starts/stops backend.
- Hosts embedded webview.
- Handles app lifecycle, logs, and updates.

2. Backend service (local-only):
- FastAPI server.
- Job queue/runner.
- Pipeline execution and artifact generation.

3. Frontend UI:
- Existing schema-driven forms and job controls.
- Progress, logs, reports, and artifacts.

4. Runtime assets:
- Models, tanpura tracks, static assets, ffmpeg tools, and config defaults.

### Runtime Flow
1. User opens desktop app.
2. Shell boots backend on localhost and waits for health check.
3. Webview loads UI.
4. User runs preprocess/detect/analyze as today.
5. Artifacts and reports are stored locally in app-managed directories.

## Packaging Strategy

### Backend Packaging
- Bundle Python runtime + dependencies with deterministic lock.
- Include required binaries (ffmpeg/ffplay) and verify availability at startup.
- Ship as sidecar executable or managed embedded environment.

### Frontend Packaging
- Bundle static UI with desktop shell.
- No external network dependency for normal operation.

### Platform Targets
1. macOS first (Apple Silicon + Intel where feasible).
2. Windows second.
3. Linux optional later.

## Data and File Layout

### Standardized App Paths
- App settings.
- Job metadata and logs.
- Audio cache and uploads.
- Output artifacts/reports.
- Temporary processing workspace.

### Requirements
- Clear separation between temporary and user-exported outputs.
- Stable naming and cleanup policy.
- Explicit “Open Output Folder” behavior.

## UX Plan

### Release 1 UX
- One-click start experience.
- Existing mode workflow preserved.
- Clear diagnostics panel:
  - microphone access,
  - ffmpeg availability,
  - model presence,
  - writable directories.
- Job view with logs and final artifacts.

### Interaction Changes
- No constant artifact refresh while jobs are running.
- Progress updates should be useful but non-jumpy.
- Report links activate only when artifacts are finalized.

## Figma Design Approach

### Design Objectives
- Keep workflow clarity for preprocess -> detect -> analyze.
- Prioritize fast, low-error operation over decorative UI.
- Maintain parity with current pipeline behavior and terminology.

### Workflow
1. Map core user journeys in Figma first:
- First-time setup + diagnostics.
- Record/upload in preprocess.
- Detect run and result review.
- Analyze run and report review.
- Error and recovery flows.

2. Create a lightweight design system:
- Color, typography, spacing, and interaction states.
- Reusable components for forms, logs, progress, and artifact lists.
- Standard status patterns for running/completed/failed jobs.

3. Design key screens before edge cases:
- App shell/navigation.
- Mode workspace (preprocess/detect/analyze).
- Job monitor panel.
- Reports/artifacts panel.
- Settings/diagnostics panel.

4. Build an interactive prototype:
- Simulate complete end-to-end run flow.
- Include long logs, failures, and cancellation states.
- Validate that no screen requires ambiguous decisions by user.

5. Run quick usability checks:
- 5-8 users performing scripted tasks.
- Track time-to-complete, misclicks, and confusion points.
- Iterate once before implementation starts.

6. Handoff to implementation:
- Export component specs and interaction notes.
- Define acceptance criteria per screen/state.
- Keep a change log in Figma tied to app milestones.

### Practical Guardrails
- Avoid redesigning pipeline semantics in Figma; UI should reflect backend truth.
- Design for laptop constraints first (13-14" screens).
- Keep mobile/phone companion mocks in a separate Figma page and treat them as Phase 2+.

## Reliability and Safety

### Error Handling
- Human-readable failures for:
  - missing mic permission,
  - missing binaries,
  - invalid args,
  - model/runtime failures,
  - insufficient disk space.

### Recovery
- Retry failed jobs.
- Preserve logs and partial outputs for debugging.
- Safe cancellation semantics.

## Testing Plan

### Automated
1. Backend smoke tests for preprocess/detect/analyze.
2. Job lifecycle tests (queue, run, cancel, fail, complete).
3. Packaging integrity tests (assets present, binaries available).
4. Cross-platform startup and health-check tests.

### Manual QA
1. Fresh install startup.
2. Preprocess from recording and from file.
3. Detect and analyze end-to-end.
4. Artifact/report availability after completion.
5. Cancel/interrupt handling.

## Release Phases

### Phase 0: Stabilization (Current Codebase)
- Freeze current pipeline behavior.
- Remove experimental toggles not intended for release.
- Baseline timing and quality metrics.
- Finalize Figma journey maps and base component system.

### Phase 1: Desktop Wrapper Prototype
- Implement shell + backend lifecycle.
- Load existing UI in desktop context.
- Validate local-only operation.
- Translate approved Figma screens for core workflows.

### Phase 2: Packaging and Installers
- Produce signed installers for target platform(s).
- Bundle dependencies and runtime assets.
- Add startup diagnostics and support bundle export.
- Polish UI using Figma-reviewed states for diagnostics/errors.

### Phase 3: Beta
- Limited user testing on real machines.
- Fix installation/runtime edge cases.
- Finalize upgrade/migration behavior.

### Phase 4: General Availability
- Public release.
- Versioned changelog and update channel.
- Ongoing bugfix and performance iteration.

## Phone Companion (Secondary Track)

### Scope
- Recording-only app.
- Transfer file to desktop over LAN, QR-assisted local transfer, or manual import.

### Non-goals
- No on-device detect/analyze initially.
- No duplicated heavy model stack on phone.

## Open Decisions
1. Final desktop shell choice (Tauri vs Electron).
2. Python packaging approach (sidecar executable vs embedded env).
3. FFmpeg bundling strategy per OS and licensing notes.
4. Auto-update mechanism and release hosting.
5. Minimum OS version support matrix.

## Success Criteria
1. User can install and run without terminal commands.
2. End-to-end preprocess/detect/analyze works offline.
3. Output quality matches current local pipeline.
4. Crash-free job completion rate is high on target hardware.
5. Support burden is low due to diagnostics and clear errors.
