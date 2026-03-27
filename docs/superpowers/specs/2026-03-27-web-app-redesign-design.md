# Raga Detection Web App Redesign - Design Spec

## Overview

Redesign the local-only FastAPI app into a production-ready web application with user accounts, a community archive, and an educational focus for intermediate Hindustani classical music listeners developing their ear.

**Primary audience:** Intermediate listeners who know the basics (what a raga is, what sargam means) but are building their ability to hear and distinguish ragas. Not beginners, not scholarly performers.

**Scale:** Pilot of a few hundred users, architecture that supports growth.

## Tech Stack

- **Frontend:** Next.js (React), dark theme with warm brown palette (#bf6e13 accent)
- **Auth:** Firebase Authentication (standard email/password + OAuth)
- **Backend:** FastAPI (existing pipeline, adapted for multi-user)
- **Storage:** Firebase/cloud storage for audio files, user data, analysis artifacts
- **Processing:** Server-side pipeline execution with job queue

## Screens & Routes

| Route | Screen | Access |
|---|---|---|
| `/` | Sign in / Sign up | Public |
| `/library` | Personal song library | Authenticated |
| `/upload` | Upload file or YouTube URL | Authenticated |
| `/song/[id]` | Results page (full analysis) | Public for community songs, private for personal |
| `/song/[id]/edit` | Transcript editor | Authenticated (own forks only) |
| `/explore` | Community archive feed | Public (browsable), authenticated to interact |

## Data Model

### Songs & Ownership

Three visibility levels:
- **Public YouTube analysis** - default for all YouTube uploads. Canonical version visible to everyone.
- **Public file upload** - user opts in to make their uploaded file publicly visible.
- **Private file upload** - default for file uploads. Only the uploader sees it.

### Canonical Versions & Forks

Each unique YouTube video has one canonical analysis. When a user uploads a YouTube URL that already exists:
1. They get the existing results instantly (no re-processing).
2. They can **fork** to re-analyze with different parameters (tonic, raga, etc.).

**Shared base artifacts (analysis data only):** Pitch extraction (SwiftF0) data and analysis results are deterministic for the same audio. Computed once, shared across all forks. Only the analysis phase (fast, 24x real-time) re-runs on a fork.

**Copyright-safe storage model:**
- **YouTube sources:** Audio is downloaded ephemerally during processing, then deleted from the server. Only non-audio artifacts are persisted (pitch CSVs, transcription data, histogram data, candidate scores). Stems are never stored server-side for YouTube content.
- **User-uploaded files:** Original audio and derived stems are stored on the server (user bears responsibility for having rights to the file).
- **On-demand stem regeneration:** When a user requests stem playback for a YouTube source, the server re-downloads the audio and re-runs Demucs (~4x real-time). Stems are streamed to the client and cached in IndexedDB for repeat visits, then deleted from the server.
- **YouTube embed is independent:** Sits separately on the results page for watching the original performance. Not synced with the pitch analysis player.

### Transcription Hierarchy

Three-tier default cascade:
1. **System output** (always exists) - pipeline-generated transcription
2. **Moderator version** (optional) - if a moderator has corrected the transcription, this becomes the default everyone sees
3. **Personal fork** (per user) - user's own edits, visible only to them

No wiki/PR model. Users cannot edit the canonical version. Moderators can override the system output. Everyone else edits their own fork.

## Screen Designs

### 1. Sign In (`/`)

Standard Firebase Auth UI. Email/password + Google OAuth. Redirects to `/library` on success.

### 2. Library (`/library`)

The home screen after sign-in. Shows all songs the user has uploaded or forked.

**Layout:**
- **Header:** App name (left), user avatar + sign out (right)
- **Action bar:** "Your Library" title + song count, search bar, Grid/List toggle, "+ Upload" button
- **Filter pills:** All, Processing, Complete, YouTube, Uploaded
- **Content area:** Card grid (default) or list view, persisted to user preferences

**Card view (default for new accounts):**
- 3 columns desktop, 2 tablet, 1 mobile
- Each card shows:
  - Thumbnail area: YouTube thumbnail (pulled from `img.youtube.com/vi/{ID}/hqdefault.jpg`) or music note icon for file uploads
  - Source badge: "YT" (red) or "File" (accent color) in top-right of thumbnail
  - Song title
  - Raga + tonic pills (once analysis complete)
  - Date
  - Status: green dot "Complete", amber dot "Processing" with progress bar, red dot "Error"
- Ghost card at end of grid: dashed border, "+ Upload or drop file"

**List view:**
- Compact table: Title, Source, Raga, Tonic, Date, Status
- Sortable columns

**Drag-and-drop:** Dragging an audio file onto the library page shows a full-page drop zone overlay (dashed border, accent color), then navigates to `/upload` with the file pre-attached.

**View toggle persisted** to user preferences (Firebase). Last used view is the default.

### 3. Upload (`/upload`)

Reached via "+ Upload" button or drag-and-drop from library.

**Layout:** Centered narrow form (max-width 640px), back link to library.

**Two tabs** sharing the same form fields below:

**Upload File tab:**
- Drag-and-drop zone with "Browse files" button
- Supports MP3, WAV, FLAC, M4A
- If reached via drag-and-drop, file is pre-attached
- Title auto-populated from filename (editable)
- Public/private toggle (defaults to private)

**YouTube URL tab:**
- URL input field
- Live preview on valid URL: thumbnail, video title, channel name, duration
- Title auto-populated from YouTube metadata (editable)
- If the video already exists in the community archive: show a banner "This video has already been analyzed. View results?" with a link. User can still proceed to fork.
- Public by default (noted in UI)

**Shared fields (below tabs):**
- Tonic (optional, dropdown, default "Auto-detect")
- Raga (optional, searchable dropdown from raga DB, default "Auto-detect")
- Instrument (optional, default "Vocal")
- Vocalist Gender (optional, default "Auto-detect")
- Collapsed "Advanced options" panel (separator model, confidence thresholds, etc.)

**Submit:** "Start Analysis" button. Redirects to `/library` where the new card appears in "Processing" state.

### 4. Results Page (`/song/[id]`)

The main analysis view. Long scrolling page, eight sections top to bottom.

**Section 1: Correction Bar**
- Subtle banner at page top: "Think the results are wrong?" with "Adjust & Re-analyze" link
- Expands (inline, not a separate page) to reveal: tonic dropdown, raga dropdown, instrument selector, "Re-analyze" button
- Re-analyze reuses cached stems and pitch data, runs only the analysis phase (~12s for a 5-min recording), results update in place
- Collapses back after submission

**Section 2: Hero**
- "Detected Raga" label (small, uppercase)
- Raga name (large, ~42px, prominent)
- Tonic pill + confidence badge (high/medium/low with color coding)
- Song title, source, duration, analysis date

**Section 3: Raga Context (Educational)**
- Card/panel with structured information:
  - **Aroha (ascending)** and **Avroh (descending)** in sargam notation, side by side
  - **Distinguishing notes** - plain English explanation of what makes this raga unique (e.g., "Uses tivra Ma -- this separates Yaman from Bilawal")
  - **Characteristic phrases** - displayed as pills, highlighted in accent color if found in the recording, dimmed/struck-through if not found
  - **Time of day** - traditional performance time
- Data sources: raga database (aroha/avroh, notes), scoring results (which phrases matched). Designed as a flexible container to accommodate richer motif-based explanations as that system matures.

**Section 4: YouTube Embed (conditional)**
- Only shown for YouTube-sourced songs
- Standard YouTube iframe embed, 16:9 aspect ratio
- Independent from the pitch analysis player (not synced)
- Section header: "Performance"

**Section 5: Pitch Analysis (synced player)**
- **Stem toggle bar:** Three buttons - Original, Vocals, Accompaniment. Active stem highlighted in accent color. Buttons toggle which audio track plays. Multiple can potentially be mixed.
- **Transport controls:** Rewind, Play/Pause, Fast-forward, timestamp display (current / total)
- **Pitch contour plot:**
  - Y-axis: sargam note labels (S, R, G, M', P, D, N, S'). Distinguishing notes highlighted.
  - X-axis: time
  - Pitch curve rendered as a continuous line (from cached pitch data)
  - Horizontal grid lines at each sargam note
  - Playhead: vertical line synced to audio playback, moves in real-time
  - Transcribed note overlays: small labels on the curve showing detected sargam notes
  - Click anywhere on the plot to seek audio to that timestamp
  - Timestamped comment markers shown on the time axis (see Section 7)
- Technology: Plotly.js (existing) or canvas-based for performance with long recordings

**Section 6: Karaoke Transcription**
- Phrase-by-phrase display of transcribed sargam notation
- Synced to playback: current phrase highlighted (accent background + left border)
- Each phrase shows: timestamp (left), sargam sequence with dot separators between sub-phrases
- Distinguishing notes rendered in accent color throughout
- Collapsible: shows first ~8 phrases, expandable to show all
- Click a phrase to seek playback to that timestamp

**Section 7: Deep Analysis (collapsible)**
- Collapsed by default, expandable
- **Vocal pitch histogram** - bar chart, sargam labels on X-axis, distinguishing notes highlighted
- **Note transition matrix** - heatmap grid (sargam x sargam), warm color scale
- **Raga candidates table** - columns: Raga, Score, Notes Match, Phrases Match. Top result highlighted. Top 3-5 shown, expandable.

**Section 8: Timestamped Comments**
- Comment input: text field + "Add comment" button. Timestamp auto-filled from current playback position (editable).
- Comment feed: chronological by timestamp. Each comment shows: user avatar, username, timestamp (clickable - seeks playback), comment text.
- Comments also appear as small markers on the pitch plot timeline (Section 5), so users can see where discussion clusters.
- Threaded replies within a timestamped comment (one level deep).

### 5. Transcript Editor (`/song/[id]/edit`)

Separate page for detailed transcription editing. Lifted from the current local app's editor with visual refinements.

**Layout:** Two-panel view.
- **Top/Left:** Pitch contour plot (same as results page Section 5, with stem toggles and transport)
- **Bottom/Right:** Editable transcription grid

**Transcription grid:**
- Notes table: columns [Sargam, Octave, Start Time, End Time, Duration]
- Phrases table: columns [Phrase Index, Start Time, End Time, Notes]
- Inline cell editing (click to edit)
- Add/delete rows
- Changes highlighted (diff from the default version)

**Version management:**
- Shows which base you're editing from (system output or moderator version)
- Save creates a new version in your personal fork
- Version history dropdown to load previous saves
- "Reset to default" to discard personal edits

**Sync:** Clicking a note/phrase in the grid seeks the pitch plot to that timestamp. Clicking on the pitch plot highlights the corresponding note in the grid.

### 6. Explore / Community Archive (`/explore`)

Public-facing feed of all community-analyzed songs.

**Layout:**
- **Header:** Same global nav as authenticated pages
- **Search bar:** Prominent, searches by artist, song title, raga name
- **Sort tabs:** Most Viewed, Most Recent, Most Discussed
- **Filter sidebar/pills:**
  - Raga (searchable dropdown or pill cloud of popular ragas)
  - Song type: Classical, Semi-classical, Filmy
- **Feed:** Card grid (reuses the same card component as the library). Each card shows:
  - YouTube thumbnail
  - Song title
  - Raga + tonic pills
  - View count, comment count
  - "Analyzed by [username]" or "Community" attribution

**Interaction:**
- Click a card to go to `/song/[id]` (public results page)
- "Fork & Analyze" button on each card to create a personal fork with different params
- If not signed in, can browse and view results but cannot fork, comment, or upload

## Navigation & Global Elements

**Global header (all authenticated pages):**
- Left: App name/logo (links to `/library`)
- Center/right: Nav links - "Library", "Explore"
- Right: User avatar + dropdown (Sign out, Settings)

**Processing notifications:**
- Short recordings (<10 min audio): progress stepper on the library card (user can wait or navigate away)
- Long recordings (>10 min audio): banner suggesting "We'll notify you when it's ready" + browser notification or email when complete
- Library cards show real-time progress (percentage bar) while processing

## Responsive Design

**Desktop-first, responsive.** Key breakpoints:

- **Desktop (>1024px):** Full layout, 3-column card grid, side-by-side histograms, full pitch plot width
- **Tablet (768-1024px):** 2-column card grid, histograms stack vertically, pitch plot full width
- **Mobile (<768px):** 1-column card grid, all sections stack, pitch plot scrolls horizontally, simplified transport controls

The pitch contour is inherently wide - on mobile it gets horizontal scroll rather than being squished.

## Color Palette

Warm brown theme (not purple, not "AI-generated" feeling, no glowy areas, and no gradients):
- **Background:** #1a1208 (deep brown-black)
- **Card/panel background:** #241a0c
- **Elevated surface:** #2e1f0e
- **Border:** #3a2a14
- **Primary accent:** #bf6e13 (warm amber-brown)
- **Secondary accent:** #d4942a (gold, for raga/tonic pills, distinguishing notes)
- **Text primary:** #f5e6d0 (warm cream)
- **Text secondary:** #c4a882
- **Text muted:** #7a6040
- **Text faint:** #5a4a30
- **Success:** #4ade80 (green, for "complete" status, high confidence)
- **Warning:** #f59e0b (amber, for processing state)
- **Error:** #ef4444 (red, for failed state)

Minimal gradients and glow effects. Flat, warm, understated.

## Key Technical Considerations

### Pipeline Integration
- The existing FastAPI backend runs `driver.py detect` then `driver.py analyze` as subprocesses. No changes to the pipeline itself.
- For the web app, this becomes a server-side job queue (existing `jobs.py` pattern scales)
- Detect phase: stem separation + pitch extraction (~4x real-time). Pitch CSVs cached per unique audio. Stems cached only for user-uploaded files.
- Analyze phase: histogram scoring + transcription (~24x real-time). Re-run on forks/corrections.
- YouTube audio and stems are ephemeral (deleted after processing). Only pitch data and analysis artifacts are persisted.

### Community Deduplication
- YouTube videos keyed by video ID. Second upload of same video skips detect entirely.
- File uploads: no deduplication (different files could be different recordings of the same piece)

### YouTube Thumbnails
- Pulled from `https://img.youtube.com/vi/{VIDEO_ID}/hqdefault.jpg`
- Store video ID on upload, render thumbnail client-side or cache server-side

### Transcription Editor
- Lifted from current `local_app/static/transcription_editor.js` (414 lines) with UI updates
- Version storage in Firebase per user per song
- Base version (system or moderator) is read-only; user edits create personal fork versions

### Authentication & Authorization
- Firebase Auth for sign-in/sign-up
- Firestore rules or backend middleware for access control:
  - Public songs: anyone can read results, only owner/moderators can edit canonical
  - Private songs: only owner can read/write
  - Comments: any authenticated user on public songs
  - Forks: any authenticated user on public songs
