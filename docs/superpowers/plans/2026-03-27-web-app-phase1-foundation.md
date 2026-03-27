# Web App Redesign - Phase 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the monorepo structure, infrastructure architecture, Firebase auth, Firestore schema, and adapted FastAPI backend so that subsequent phases (UI screens, rich features, community) have a working foundation to build on.

**Architecture:** Separate project (`raga-web-app/`) alongside the existing `raga-detection/` repo under the shared `RagaDetection/` parent directory. Next.js frontend + new FastAPI backend. Pipeline code accessed via symlink. Firebase for auth and metadata. File storage abstracted behind a storage layer that uses local disk in dev and Cloud Storage in production. nginx reverse proxy in production.

**Tech Stack:** Next.js 14 (App Router), TypeScript, Tailwind CSS, Firebase Auth, Cloud Firestore, FastAPI, Python 3.10, Docker (production)

---

## Infrastructure Architecture

### Development Environment

```
RagaDetection/                           # shared parent directory
├── raga-detection/                      # existing repo (untouched)
│   ├── raga_pipeline/                   # pipeline source of truth
│   └── driver.py
└── raga-web-app/                        # NEW project
    ├── web/                             # Next.js on :3000
    ├── api/                             # FastAPI on :8765
    ├── raga_pipeline -> ../raga-detection/raga_pipeline   # symlink
    └── driver.py -> ../raga-detection/driver.py           # symlink

localhost:3000  (Next.js dev server, in raga-web-app/web/)
    |
    |--- /api/* proxied via next.config.ts rewrites --->  localhost:8765 (FastAPI)
    |
    |--- Firebase Auth (real Firebase project, not emulators)
    |--- Firestore (real Firebase project)
    |
FastAPI (localhost:8765, in raga-web-app/)
    |--- Reads/writes files to local disk: raga-web-app/.local_app_data/
    |--- Pipeline execution via symlinked driver.py + raga_pipeline/
    |--- Verifies Firebase ID tokens on protected endpoints
```

**How it works:**
- Run `npm run dev` in `raga-web-app/web/` for the Next.js dev server on :3000
- Run `uvicorn` in `raga-web-app/` for FastAPI on :8765
- Pipeline code accessed via symlinks to `../raga-detection/raga_pipeline/` and `../raga-detection/driver.py`. Always in sync with pipeline development.
- Next.js `rewrites` in `next.config.ts` proxy `/api/*` requests to FastAPI, so the browser only talks to :3000
- Firebase Auth runs client-side in the browser. The frontend sends Firebase ID tokens in `Authorization: Bearer <token>` headers. FastAPI verifies these tokens via the Firebase Admin SDK.
- **Uploaded files** (MP3s from users) stored on local disk under `.local_app_data/uploads/`, along with their derived stems
- **YouTube audio** downloaded ephemerally during pipeline processing, then deleted. Only non-audio analysis artifacts (pitch CSVs, transcription data, histogram data) are persisted.
- When a user requests stem playback for a YouTube source, audio is re-downloaded and Demucs re-runs on demand. Stems are served to the client for browser-side caching (IndexedDB), not persisted on server.

### Production Environment

```
                  Internet
                     |
               [nginx :443]
              /             \
     /*  (static/SSR)     /api/*
           |                 |
    [Node.js :3000]    [FastAPI :8765]
    (Next.js app)      (Pipeline API)
           |                 |
      Firebase Auth     Firebase Admin SDK
      Firestore         (token verification)
           |                 |
    [Cloud Firestore]   [Cloud Storage / Disk]
    (metadata, comments,  (audio, stems, pitch
     user data, forks)     CSVs, artifacts)
```

**How it works:**
- Single VM (or two containers) runs both Next.js and FastAPI
- nginx terminates TLS, routes `/*` to Next.js (:3000), `/api/*` to FastAPI (:8765)
- Cloud Firestore stores all metadata: songs, users, forks, comments, transcription edits
- **User-uploaded audio** and their derived stems stored on VM disk (pilot scale). Clear migration path to GCS/S3 via the storage abstraction layer.
- **YouTube audio is never persisted.** Downloaded ephemerally for processing, deleted after. Only analysis artifacts (pitch data, transcription, scores) are stored.
- **On-demand stem regeneration** for YouTube sources: when a user requests stem playback, the server re-downloads and re-runs Demucs (~4x real-time), streams stems to the client, then deletes the audio. Client caches stems in IndexedDB for repeat visits.
- Pipeline execution happens on the FastAPI server (requires ffmpeg, torch, demucs, swift-f0)

### File Storage Architecture

```
.local_app_data/                      # Dev: local disk. Prod: VM disk (or Cloud Storage)
├── uploads/                          # User-uploaded audio files (we have rights to store)
│   └── {user_id}/
│       └── {song_id}/
│           ├── original.mp3          # The uploaded file
│           ├── vocals.mp3            # Stem separation output (persisted for uploads)
│           └── accompaniment.mp3
├── artifacts/                        # Analysis artifacts ONLY (no audio for YT sources)
│   └── {audio_hash}/                 # Hash of source audio (dedup key)
│       ├── vocals_pitch_data.csv     # SwiftF0 pitch extraction
│       ├── accompaniment_pitch_data.csv
│       └── composite_pitch_data.csv
├── analyses/                         # Per-fork analysis results
│   └── {analysis_id}/
│       ├── detection_report.json     # Structured detection results (not HTML)
│       ├── analysis_report.json      # Structured analysis results (not HTML)
│       ├── candidates.csv
│       ├── transcribed_notes.csv
│       ├── histogram_data.json       # Raw data for client-side rendering
│       └── transition_matrix.json
├── tmp/                              # Ephemeral processing (auto-cleaned)
│   └── {job_id}/                     # Temp dir per job, deleted after completion
│       ├── audio.mp3                 # YouTube download (deleted after processing)
│       ├── vocals.mp3                # Stems (deleted after processing for YT sources)
│       └── accompaniment.mp3
└── jobs/                             # Job queue persistence
    └── {job_id}.json
```

**Key differences from current app:**
- Reports are no longer generated as HTML on the server. The pipeline outputs structured JSON data, and the Next.js frontend renders it.
- **YouTube audio and stems are never persisted.** They exist only in `tmp/{job_id}/` during processing and are deleted after. Only pitch CSVs and analysis data are kept.
- **User-uploaded files and their stems ARE persisted** (the user uploaded the file, so storing it and derivatives is defensible).
- When a user requests stem playback for a YouTube source, the server re-downloads and re-runs Demucs on demand. The client caches the resulting stems in IndexedDB.

### Firestore Schema

```
users/{uid}
  - email: string
  - displayName: string
  - createdAt: timestamp
  - preferences: { viewMode: "grid" | "list" }

songs/{songId}
  - title: string
  - source: "youtube" | "file"
  - youtubeVideoId: string | null
  - audioHash: string                    # Dedup key for shared artifacts
  - uploadedBy: uid
  - visibility: "public" | "private"
  - status: "processing" | "complete" | "failed"
  - songType: "classical" | "semi-classical" | "filmy" | null
  - createdAt: timestamp
  - viewCount: number
  - commentCount: number

songs/{songId}/analyses/{analysisId}
  - type: "canonical" | "moderator" | "fork"
  - ownerId: uid
  - parentAnalysisId: string | null      # null for canonical, points to parent for forks
  - params: { tonic, raga, instrument, vocalistGender, ... }
  - results: {
      detectedRaga: string,
      detectedTonic: string,
      confidence: number,
      candidateRagas: [{ name, score, notesMatch, phrasesMatch }],
      aroha: string,
      avroh: string,
      distinguishingNotes: string,
      characteristicPhrases: [{ pattern, found }],
      timeOfDay: string
    }
  - artifactPaths: {
      detectionReport: string,
      analysisReport: string,
      candidates: string,
      transcribedNotes: string,
      histogramData: string,
      transitionMatrix: string
    }
  - status: "processing" | "complete" | "failed"
  - createdAt: timestamp

songs/{songId}/comments/{commentId}
  - authorId: uid
  - authorName: string
  - text: string
  - timestampSeconds: number             # Position in the recording
  - parentCommentId: string | null       # For threaded replies
  - createdAt: timestamp

songs/{songId}/analyses/{analysisId}/transcriptionEdits/{editId}
  - ownerId: uid
  - baseVersion: "system" | "moderator"
  - notes: [{ sargam, octave, startTime, endTime, duration }]
  - phrases: [{ index, startTime, endTime, notes }]
  - isDefault: boolean
  - createdAt: timestamp
```

### API Communication

**Frontend -> Backend flow:**

1. User action in browser (e.g., click "Start Analysis")
2. Next.js client component calls a Next.js Server Action or API route
3. Server Action verifies the Firebase session, then calls FastAPI with the user's context
4. FastAPI processes the request (e.g., queues a job, returns song data)
5. Response flows back through Next.js to the client

**Auth flow:**

1. User signs in via Firebase Auth (client-side SDK)
2. Firebase returns an ID token (JWT)
3. Frontend stores token and sends it on every API request: `Authorization: Bearer <idToken>`
4. FastAPI middleware verifies the token via `firebase_admin.auth.verify_id_token()`
5. Extracts `uid` and attaches to the request context

**File serving:**

- Audio files and artifacts served by FastAPI via authenticated endpoints
- URLs are not guessable (use signed paths or token-gated access)
- For the pitch plot, the frontend fetches pitch CSV data as JSON from the API and renders client-side

---

## Project Structure

```
RagaDetection/                             # Shared parent directory
├── raga-detection/                        # EXISTING repo (untouched)
│   ├── raga_pipeline/                     # Pipeline source of truth
│   ├── driver.py
│   ├── local_app/
│   └── ...
│
└── raga-web-app/                          # NEW: separate project (new git repo)
    ├── raga_pipeline -> ../raga-detection/raga_pipeline    # symlink
    ├── driver.py -> ../raga-detection/driver.py            # symlink
    ├── web/                               # Next.js frontend
    │   ├── package.json
    │   ├── tsconfig.json
    │   ├── tailwind.config.ts
    │   ├── next.config.ts
    │   ├── .env.local                     # Firebase config, API URL
    │   ├── src/
    │   │   ├── app/                       # Next.js App Router
    │   │   │   ├── layout.tsx             # Root layout (global nav, providers)
    │   │   │   ├── page.tsx               # / - sign in
    │   │   │   ├── library/
    │   │   │   │   └── page.tsx           # /library
    │   │   │   ├── upload/
    │   │   │   │   └── page.tsx           # /upload
    │   │   │   ├── song/
    │   │   │   │   └── [id]/
    │   │   │   │       ├── page.tsx       # /song/[id] - results
    │   │   │   │       └── edit/
    │   │   │   │           └── page.tsx   # /song/[id]/edit - editor
    │   │   │   └── explore/
    │   │   │       └── page.tsx           # /explore
    │   │   ├── components/
    │   │   │   ├── ui/                    # Reusable primitives (Button, Card, Input, etc.)
    │   │   │   ├── layout/               # Header, Nav, Footer
    │   │   │   ├── library/              # SongCard, SongList, LibraryGrid, etc.
    │   │   │   ├── upload/               # UploadForm, FileDropZone, YouTubePreview
    │   │   │   ├── results/              # HeroSection, RagaContext, PitchPlayer, Karaoke, etc.
    │   │   │   ├── editor/               # TranscriptionGrid, VersionManager
    │   │   │   ├── explore/              # ExploreFeed, FilterBar, SortTabs
    │   │   │   └── comments/             # TimestampedComment, CommentFeed
    │   │   ├── lib/
    │   │   │   ├── firebase.ts           # Firebase app init, auth helpers
    │   │   │   ├── firestore.ts          # Firestore read/write helpers
    │   │   │   ├── api.ts                # FastAPI client (fetch wrapper with auth)
    │   │   │   └── types.ts              # Shared TypeScript types
    │   │   ├── hooks/
    │   │   │   ├── useAuth.ts            # Auth state hook
    │   │   │   ├── useJob.ts             # Job polling hook
    │   │   │   └── usePitchPlayer.ts     # Audio + pitch plot sync hook
    │   │   └── styles/
    │   │       └── globals.css            # Tailwind imports + brown theme tokens
    │   └── public/
    │       └── (static assets)
    ├── api/                               # FastAPI backend (multi-user)
    │   ├── __init__.py
    │   ├── main.py                        # FastAPI app with CORS, Firebase middleware
    │   ├── auth.py                        # Firebase token verification middleware
    │   ├── routes/
    │   │   ├── songs.py                   # CRUD for songs, upload, dedup check
    │   │   ├── analysis.py                # Job submission, results, re-analyze
    │   │   ├── artifacts.py               # File serving (audio, stems, pitch data)
    │   │   ├── comments.py                # Timestamped comments CRUD
    │   │   ├── transcription.py           # Transcription edit CRUD
    │   │   └── explore.py                 # Community feed, search, filters
    │   ├── storage.py                     # Storage abstraction (local disk / cloud)
    │   ├── firestore_client.py            # Firestore read/write from Python
    │   └── schemas.py                     # Pydantic models for request/response
    ├── requirements.txt                   # Python deps (firebase-admin, etc.)
    ├── docker-compose.yml                 # Production deployment
    ├── nginx.conf                         # Reverse proxy config
    ├── Dockerfile.api                     # FastAPI container
    ├── .gitignore
    └── .local_app_data/                   # Runtime file storage (gitignored)
```

---

## Phase Plan Overview

This is **Phase 1 of 4**. Each phase produces working, testable software.

| Phase | Scope | Dependency |
|---|---|---|
| **Phase 1 (this plan)** | Monorepo setup, Firebase auth, Firestore schema, adapted FastAPI backend, storage layer, basic Next.js shell with auth flow | None |
| Phase 2 | Library page (cards + list), upload flow, job submission + polling, processing status | Phase 1 |
| Phase 3 | Results page (all 8 sections), pitch player, karaoke, transcript editor | Phase 2 |
| Phase 4 | Explore feed, forks, timestamped comments, public/private visibility, community features | Phase 3 |

---

## Tasks

### Task 1: Initialize Project and Next.js Frontend

**Files:**
- Create: `raga-web-app/` project directory
- Create: `raga-web-app/web/` (Next.js scaffold)
- Create: symlinks to raga_pipeline and driver.py

- [ ] **Step 1: Create the project directory and git repo**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection
mkdir raga-web-app
cd raga-web-app
git init
```

- [ ] **Step 2: Create symlinks to the pipeline**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
ln -s ../raga-detection/raga_pipeline raga_pipeline
ln -s ../raga-detection/driver.py driver.py
```

Verify: `ls -la raga_pipeline` should show the symlink, and `python -c "import raga_pipeline; print('ok')"` should work.

- [ ] **Step 3: Create .gitignore**

Create `raga-web-app/.gitignore`:

```
.local_app_data/
__pycache__/
*.pyc
.env
node_modules/
```

- [ ] **Step 4: Scaffold Next.js with TypeScript and Tailwind**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
npx create-next-app@latest web --typescript --tailwind --eslint --app --src-dir --import-alias "@/*" --no-turbopack
```

Accept defaults. This creates the full scaffold.

- [ ] **Step 2: Verify the scaffold runs**

```bash
cd web && npm run dev
```

Expected: Next.js dev server on http://localhost:3000, default page renders.

Stop the server (Ctrl+C).

- [ ] **Step 3: Configure the brown theme in Tailwind**

Replace `web/tailwind.config.ts`:

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#1a1208",
          card: "#241a0c",
          elevated: "#2e1f0e",
        },
        border: {
          DEFAULT: "#3a2a14",
        },
        accent: {
          DEFAULT: "#bf6e13",
          gold: "#d4942a",
        },
        text: {
          primary: "#f5e6d0",
          secondary: "#c4a882",
          muted: "#7a6040",
          faint: "#5a4a30",
        },
        status: {
          success: "#4ade80",
          warning: "#f59e0b",
          error: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["system-ui", "sans-serif"],
        mono: ["ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
```

- [ ] **Step 4: Set up global styles**

Replace `web/src/styles/globals.css` (or `web/src/app/globals.css` depending on scaffold):

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  background-color: #1a1208;
  color: #f5e6d0;
}
```

- [ ] **Step 5: Create a minimal root layout**

Replace `web/src/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Raga Detection",
  description: "Identify and study Hindustani ragas in audio recordings",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text-primary font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
```

- [ ] **Step 6: Create placeholder landing page**

Replace `web/src/app/page.tsx`:

```tsx
export default function Home() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-2">Raga Detection</h1>
        <p className="text-text-muted">Sign in page coming soon</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Configure API proxy**

Replace `web/next.config.ts`:

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8765/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 8: Set up .env.local placeholder**

Create `web/.env.local`:

```
# Firebase config (fill in after Task 2)
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=

# API URL (for server-side calls)
API_URL=http://localhost:8765
```

- [ ] **Step 9: Add web/.gitignore entries**

Ensure `web/.gitignore` includes:

```
node_modules/
.next/
.env.local
```

- [ ] **Step 10: Verify everything works**

```bash
cd web && npm run dev
```

Expected: Page renders at localhost:3000 with "Raga Detection" title on brown background.

- [ ] **Step 11: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add .gitignore web/
git commit -m "feat: scaffold Next.js frontend with brown theme and API proxy"
```

---

### Task 2: Set Up Firebase Project and Auth

**Files:**
- Create: `web/src/lib/firebase.ts`
- Create: `web/src/hooks/useAuth.ts`
- Create: `web/src/components/layout/AuthProvider.tsx`
- Modify: `web/src/app/layout.tsx`
- Modify: `web/src/app/page.tsx`
- Modify: `web/.env.local`

**Prerequisites:** Create a Firebase project at https://console.firebase.google.com. Enable Authentication (Email/Password + Google sign-in). Create a Firestore database. Copy the web app config values.

- [ ] **Step 1: Install Firebase SDK**

```bash
cd web && npm install firebase
```

- [ ] **Step 2: Create Firebase initialization**

Create `web/src/lib/firebase.ts`:

```typescript
import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];

export const auth = getAuth(app);
export const db = getFirestore(app);
```

- [ ] **Step 3: Create auth state hook**

Create `web/src/hooks/useAuth.ts`:

```typescript
"use client";

import { useState, useEffect } from "react";
import { onAuthStateChanged, User } from "firebase/auth";
import { auth } from "@/lib/firebase";

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  return { user, loading };
}
```

- [ ] **Step 4: Create AuthProvider context**

Create `web/src/components/layout/AuthProvider.tsx`:

```tsx
"use client";

import { createContext, useContext } from "react";
import { User } from "firebase/auth";
import { useAuth } from "@/hooks/useAuth";

type AuthContextType = {
  user: User | null;
  loading: boolean;
};

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();

  return (
    <AuthContext.Provider value={{ user, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuthContext() {
  return useContext(AuthContext);
}
```

- [ ] **Step 5: Wire AuthProvider into root layout**

Update `web/src/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/components/layout/AuthProvider";

export const metadata: Metadata = {
  title: "Raga Detection",
  description: "Identify and study Hindustani ragas in audio recordings",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text-primary font-sans antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 6: Build sign-in page**

Replace `web/src/app/page.tsx`:

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";
import { auth } from "@/lib/firebase";
import { useAuthContext } from "@/components/layout/AuthProvider";

export default function SignIn() {
  const { user, loading } = useAuthContext();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState("");

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-text-muted">Loading...</p>
      </div>
    );
  }

  if (user) {
    router.push("/library");
    return null;
  }

  async function handleEmailAuth(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      router.push("/library");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    }
  }

  async function handleGoogleAuth() {
    setError("");
    try {
      await signInWithPopup(auth, new GoogleAuthProvider());
      router.push("/library");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Google sign-in failed");
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="w-full max-w-sm">
        <h1 className="text-3xl font-bold text-center mb-2">Raga Detection</h1>
        <p className="text-text-muted text-center mb-8">
          Identify and study Hindustani ragas
        </p>

        <form onSubmit={handleEmailAuth} className="space-y-4">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-bg-elevated border border-border rounded-lg px-4 py-3 text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-bg-elevated border border-border rounded-lg px-4 py-3 text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
          />

          {error && (
            <p className="text-status-error text-sm">{error}</p>
          )}

          <button
            type="submit"
            className="w-full bg-accent text-white font-semibold rounded-lg px-4 py-3 hover:opacity-90 transition-opacity"
          >
            {isSignUp ? "Create Account" : "Sign In"}
          </button>
        </form>

        <div className="my-6 flex items-center gap-4">
          <div className="flex-1 h-px bg-border" />
          <span className="text-text-faint text-sm">or</span>
          <div className="flex-1 h-px bg-border" />
        </div>

        <button
          onClick={handleGoogleAuth}
          className="w-full bg-bg-elevated border border-border text-text-secondary rounded-lg px-4 py-3 hover:border-accent transition-colors"
        >
          Continue with Google
        </button>

        <p className="text-center text-text-muted text-sm mt-6">
          {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
          <button
            onClick={() => setIsSignUp(!isSignUp)}
            className="text-accent hover:underline"
          >
            {isSignUp ? "Sign in" : "Sign up"}
          </button>
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Fill in .env.local with Firebase config**

Update `web/.env.local` with the actual values from the Firebase console. (This step is manual -- copy from Firebase Console > Project Settings > Web App.)

- [ ] **Step 8: Test sign-in flow**

```bash
cd web && npm run dev
```

Expected: Sign-in page renders. Create an account with email/password. After sign-in, redirects to /library (which will 404 for now -- that's correct).

- [ ] **Step 9: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add web/src/lib/firebase.ts web/src/hooks/useAuth.ts web/src/components/layout/AuthProvider.tsx web/src/app/layout.tsx web/src/app/page.tsx web/package.json web/package-lock.json
git commit -m "feat: add Firebase auth with sign-in/sign-up page"
```

---

### Task 3: Create Shared TypeScript Types

**Files:**
- Create: `web/src/lib/types.ts`

- [ ] **Step 1: Define all shared types**

Create `web/src/lib/types.ts`:

```typescript
// Song visibility and source
export type SongSource = "youtube" | "file";
export type SongVisibility = "public" | "private";
export type SongStatus = "processing" | "complete" | "failed";
export type SongType = "classical" | "semi-classical" | "filmy";

// Analysis types
export type AnalysisType = "canonical" | "moderator" | "fork";

export interface Song {
  id: string;
  title: string;
  source: SongSource;
  youtubeVideoId: string | null;
  audioHash: string;
  uploadedBy: string;
  visibility: SongVisibility;
  status: SongStatus;
  songType: SongType | null;
  createdAt: Date;
  viewCount: number;
  commentCount: number;
}

export interface CandidateRaga {
  name: string;
  score: number;
  notesMatch: string;
  phrasesMatch: string;
}

export interface CharacteristicPhrase {
  pattern: string;
  found: boolean;
}

export interface AnalysisResults {
  detectedRaga: string;
  detectedTonic: string;
  confidence: number;
  candidateRagas: CandidateRaga[];
  aroha: string;
  avroh: string;
  distinguishingNotes: string;
  characteristicPhrases: CharacteristicPhrase[];
  timeOfDay: string;
}

export interface AnalysisParams {
  tonic: string | null;
  raga: string | null;
  instrument: string;
  vocalistGender: string | null;
  [key: string]: string | null;
}

export interface Analysis {
  id: string;
  songId: string;
  type: AnalysisType;
  ownerId: string;
  parentAnalysisId: string | null;
  params: AnalysisParams;
  results: AnalysisResults | null;
  artifactPaths: Record<string, string>;
  status: SongStatus;
  createdAt: Date;
}

export interface Comment {
  id: string;
  songId: string;
  authorId: string;
  authorName: string;
  text: string;
  timestampSeconds: number;
  parentCommentId: string | null;
  createdAt: Date;
}

export interface TranscriptionNote {
  sargam: string;
  octave: number;
  startTime: number;
  endTime: number;
  duration: number;
}

export interface TranscriptionPhrase {
  index: number;
  startTime: number;
  endTime: number;
  notes: string;
}

export interface TranscriptionEdit {
  id: string;
  ownerId: string;
  baseVersion: "system" | "moderator";
  notes: TranscriptionNote[];
  phrases: TranscriptionPhrase[];
  isDefault: boolean;
  createdAt: Date;
}

export interface JobStatus {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  songId: string;
  analysisId: string;
  error: string | null;
  createdAt: string;
}

// Pitch data for the synced player
export interface PitchPoint {
  time: number;
  frequency: number;
  midi: number;
  confidence: number;
}

export interface UserPreferences {
  viewMode: "grid" | "list";
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add web/src/lib/types.ts
git commit -m "feat: add shared TypeScript types for songs, analyses, comments"
```

---

### Task 4: Create FastAPI Auth Middleware

**Files:**
- Create: `api/__init__.py`
- Create: `api/auth.py`
- Create: `api/main.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Install firebase-admin Python package**

```bash
pip install firebase-admin google-cloud-firestore
```

- [ ] **Step 2: Update requirements.txt**

Add to `requirements.txt`:

```
firebase-admin>=6.0.0
google-cloud-firestore>=2.11.0
```

- [ ] **Step 3: Create api/__init__.py**

Create `api/__init__.py`:

```python
```

(Empty file, makes it a package.)

- [ ] **Step 4: Create Firebase auth middleware**

Create `api/auth.py`:

```python
"""Firebase ID token verification middleware for FastAPI."""

import os
from typing import Optional

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from fastapi import Depends, HTTPException, Request

# Initialize Firebase Admin SDK
# In production, uses GOOGLE_APPLICATION_CREDENTIALS env var.
# In dev, can use a service account JSON file path.
_cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
if _cred_path:
    _cred = credentials.Certificate(_cred_path)
else:
    _cred = credentials.ApplicationDefault()

if not firebase_admin._apps:
    firebase_admin.initialize_app(_cred)


def _extract_token(request: Request) -> Optional[str]:
    """Extract Bearer token from Authorization header."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:]
    return None


async def get_current_user(request: Request) -> dict:
    """Dependency that verifies Firebase ID token and returns user info.

    Returns dict with 'uid', 'email', and other Firebase token claims.
    Raises 401 if token is missing or invalid.
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


async def get_optional_user(request: Request) -> Optional[dict]:
    """Dependency that returns user info if authenticated, None otherwise.

    Used for endpoints that work for both authenticated and anonymous users
    (e.g., viewing public songs).
    """
    token = _extract_token(request)
    if not token:
        return None
    try:
        return firebase_auth.verify_id_token(token)
    except Exception:
        return None
```

- [ ] **Step 5: Create the new FastAPI app entrypoint**

Create `api/main.py`:

```python
"""Multi-user FastAPI backend for the Raga Detection web app.

This is the new API server that sits alongside the existing local_app.
It adds authentication, multi-user support, and Firestore integration
while reusing the existing pipeline (driver.py, raga_pipeline/).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Raga Detection API",
    version="1.0.0",
    description="Multi-user raga detection and analysis API",
)

# CORS: allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

- [ ] **Step 6: Test the new API server starts**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
python -m uvicorn api.main:app --host 127.0.0.1 --port 8765 --reload
```

Expected: Server starts. `curl http://localhost:8765/api/health` returns `{"status":"ok","version":"1.0.0"}`.

- [ ] **Step 7: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add api/ requirements.txt
git commit -m "feat: add FastAPI backend with Firebase auth middleware"
```

---

### Task 5: Create Storage Abstraction Layer

**Files:**
- Create: `api/storage.py`

- [ ] **Step 1: Write the storage abstraction**

Create `api/storage.py`:

```python
"""Storage abstraction layer.

Uses local disk in development, designed to swap to Cloud Storage in production.
All file paths are relative to the storage root. Callers never deal with absolute paths.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

# Storage root: defaults to .local_app_data/ in the project root.
# In production, set STORAGE_ROOT to a persistent volume mount.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = Path(os.environ.get("STORAGE_ROOT", str(_PROJECT_ROOT / ".local_app_data")))


def _full_path(relative: str) -> Path:
    """Resolve a relative path against the storage root. Prevents traversal."""
    full = (STORAGE_ROOT / relative).resolve()
    if not str(full).startswith(str(STORAGE_ROOT.resolve())):
        raise ValueError(f"Path traversal attempt: {relative}")
    return full


def ensure_dir(relative: str) -> Path:
    """Create directory if it doesn't exist. Returns the full path."""
    path = _full_path(relative)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_file(relative: str, data: bytes) -> str:
    """Write bytes to a file. Creates parent dirs. Returns the relative path."""
    path = _full_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return relative


def read_file(relative: str) -> bytes:
    """Read bytes from a file. Raises FileNotFoundError if missing."""
    path = _full_path(relative)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {relative}")
    return path.read_bytes()


def file_exists(relative: str) -> bool:
    """Check if a file exists."""
    return _full_path(relative).exists()


def get_absolute_path(relative: str) -> Path:
    """Get the absolute path for a relative storage path. For serving files."""
    path = _full_path(relative)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {relative}")
    return path


def delete_file(relative: str) -> bool:
    """Delete a file. Returns True if deleted, False if it didn't exist."""
    path = _full_path(relative)
    if path.exists():
        path.unlink()
        return True
    return False


def list_files(relative_dir: str, pattern: str = "*") -> list[str]:
    """List files in a directory matching a glob pattern. Returns relative paths."""
    dir_path = _full_path(relative_dir)
    if not dir_path.exists():
        return []
    return [
        str(p.relative_to(STORAGE_ROOT))
        for p in dir_path.glob(pattern)
        if p.is_file()
    ]


def upload_dir(user_id: str, song_id: str) -> str:
    """Generate the relative storage path for a user-uploaded song and its stems."""
    return f"uploads/{user_id}/{song_id}"


def upload_path(user_id: str, song_id: str, filename: str) -> str:
    """Generate the relative storage path for an uploaded audio file."""
    return f"uploads/{user_id}/{song_id}/{filename}"


def artifact_dir(audio_hash: str) -> str:
    """Generate the relative storage path for shared analysis artifacts (pitch data only, no audio)."""
    return f"artifacts/{audio_hash}"


def analysis_dir(analysis_id: str) -> str:
    """Generate the relative storage path for analysis-specific outputs."""
    return f"analyses/{analysis_id}"


def tmp_dir(job_id: str) -> str:
    """Generate ephemeral temp dir for a processing job. Cleaned up after completion."""
    return f"tmp/{job_id}"


def cleanup_tmp(job_id: str) -> None:
    """Delete the temporary processing directory for a job."""
    import shutil
    path = _full_path(f"tmp/{job_id}")
    if path.exists():
        shutil.rmtree(path)


def is_youtube_source(song: dict) -> bool:
    """Check if a song is from YouTube (audio not persisted on server)."""
    return song.get("source") == "youtube"
```

- [ ] **Step 2: Write a quick test for the storage layer**

Create `tests/test_storage.py`:

```python
"""Tests for the storage abstraction layer."""

import os
import tempfile
import unittest

# Override STORAGE_ROOT before importing storage
_tmpdir = tempfile.mkdtemp()
os.environ["STORAGE_ROOT"] = _tmpdir

from api.storage import (
    write_file,
    read_file,
    file_exists,
    delete_file,
    list_files,
    ensure_dir,
    upload_path,
    upload_dir,
    artifact_dir,
    analysis_dir,
    tmp_dir,
    cleanup_tmp,
    is_youtube_source,
)


class TestStorage(unittest.TestCase):
    def test_write_and_read(self):
        path = write_file("test/hello.txt", b"hello world")
        self.assertEqual(path, "test/hello.txt")
        self.assertEqual(read_file("test/hello.txt"), b"hello world")

    def test_file_exists(self):
        write_file("test/exists.txt", b"yes")
        self.assertTrue(file_exists("test/exists.txt"))
        self.assertFalse(file_exists("test/nope.txt"))

    def test_delete_file(self):
        write_file("test/delete_me.txt", b"bye")
        self.assertTrue(delete_file("test/delete_me.txt"))
        self.assertFalse(file_exists("test/delete_me.txt"))
        self.assertFalse(delete_file("test/delete_me.txt"))

    def test_list_files(self):
        write_file("listdir/a.txt", b"a")
        write_file("listdir/b.txt", b"b")
        write_file("listdir/c.csv", b"c")
        txt_files = list_files("listdir", "*.txt")
        self.assertEqual(len(txt_files), 2)

    def test_path_traversal_blocked(self):
        with self.assertRaises(ValueError):
            write_file("../../etc/passwd", b"nope")

    def test_path_helpers(self):
        self.assertEqual(upload_path("u1", "s1", "song.mp3"), "uploads/u1/s1/song.mp3")
        self.assertEqual(upload_dir("u1", "s1"), "uploads/u1/s1")
        self.assertEqual(artifact_dir("abc123"), "artifacts/abc123")
        self.assertEqual(analysis_dir("a1"), "analyses/a1")
        self.assertEqual(tmp_dir("j1"), "tmp/j1")

    def test_cleanup_tmp(self):
        write_file("tmp/testjob/audio.mp3", b"fake audio")
        write_file("tmp/testjob/vocals.mp3", b"fake vocals")
        self.assertTrue(file_exists("tmp/testjob/audio.mp3"))
        cleanup_tmp("testjob")
        self.assertFalse(file_exists("tmp/testjob/audio.mp3"))

    def test_is_youtube_source(self):
        self.assertTrue(is_youtube_source({"source": "youtube"}))
        self.assertFalse(is_youtube_source({"source": "file"}))

    def test_ensure_dir(self):
        path = ensure_dir("new/nested/dir")
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the storage tests**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
python -m pytest tests/test_storage.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add api/storage.py tests/test_storage.py
git commit -m "feat: add storage abstraction layer with local disk backend"
```

---

### Task 6: Create Firestore Client Helpers

**Files:**
- Create: `api/firestore_client.py`

- [ ] **Step 1: Write Firestore CRUD helpers**

Create `api/firestore_client.py`:

```python
"""Firestore client helpers for reading and writing song/analysis/comment data.

Wraps the google-cloud-firestore SDK with typed helpers matching the schema
defined in the design spec.
"""

from datetime import datetime
from typing import Optional
import uuid

from google.cloud import firestore

_db: Optional[firestore.Client] = None


def get_db() -> firestore.Client:
    """Get or create the Firestore client. Lazy-initialized."""
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


# --- Songs ---

def create_song(
    title: str,
    source: str,
    uploaded_by: str,
    audio_hash: str,
    youtube_video_id: Optional[str] = None,
    visibility: str = "private",
    song_type: Optional[str] = None,
) -> str:
    """Create a new song document. Returns the song ID."""
    db = get_db()
    song_id = str(uuid.uuid4())
    db.collection("songs").document(song_id).set({
        "title": title,
        "source": source,
        "youtubeVideoId": youtube_video_id,
        "audioHash": audio_hash,
        "uploadedBy": uploaded_by,
        "visibility": visibility,
        "status": "processing",
        "songType": song_type,
        "createdAt": datetime.utcnow(),
        "viewCount": 0,
        "commentCount": 0,
    })
    return song_id


def get_song(song_id: str) -> Optional[dict]:
    """Get a song document by ID. Returns None if not found."""
    db = get_db()
    doc = db.collection("songs").document(song_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = doc.id
    return data


def update_song(song_id: str, **fields) -> None:
    """Update fields on a song document."""
    db = get_db()
    db.collection("songs").document(song_id).update(fields)


def find_song_by_youtube_id(video_id: str) -> Optional[dict]:
    """Find a canonical song by YouTube video ID. Returns None if not found."""
    db = get_db()
    docs = (
        db.collection("songs")
        .where("youtubeVideoId", "==", video_id)
        .where("visibility", "==", "public")
        .limit(1)
        .get()
    )
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        return data
    return None


def list_user_songs(user_id: str) -> list[dict]:
    """List all songs uploaded by a user, ordered by creation date descending."""
    db = get_db()
    docs = (
        db.collection("songs")
        .where("uploadedBy", "==", user_id)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .get()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        results.append(data)
    return results


def list_public_songs(
    order_by: str = "createdAt",
    song_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """List public songs for the explore feed."""
    db = get_db()
    query = db.collection("songs").where("visibility", "==", "public")
    if song_type:
        query = query.where("songType", "==", song_type)
    query = query.order_by(order_by, direction=firestore.Query.DESCENDING).limit(limit)
    docs = query.get()
    results = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        results.append(data)
    return results


# --- Analyses ---

def create_analysis(
    song_id: str,
    analysis_type: str,
    owner_id: str,
    params: dict,
    parent_analysis_id: Optional[str] = None,
) -> str:
    """Create a new analysis document. Returns the analysis ID."""
    db = get_db()
    analysis_id = str(uuid.uuid4())
    db.collection("songs").document(song_id).collection("analyses").document(
        analysis_id
    ).set({
        "type": analysis_type,
        "ownerId": owner_id,
        "parentAnalysisId": parent_analysis_id,
        "params": params,
        "results": None,
        "artifactPaths": {},
        "status": "processing",
        "createdAt": datetime.utcnow(),
    })
    return analysis_id


def get_analysis(song_id: str, analysis_id: str) -> Optional[dict]:
    """Get an analysis document."""
    db = get_db()
    doc = (
        db.collection("songs")
        .document(song_id)
        .collection("analyses")
        .document(analysis_id)
        .get()
    )
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = doc.id
    data["songId"] = song_id
    return data


def update_analysis(song_id: str, analysis_id: str, **fields) -> None:
    """Update fields on an analysis document."""
    db = get_db()
    (
        db.collection("songs")
        .document(song_id)
        .collection("analyses")
        .document(analysis_id)
        .update(fields)
    )


def get_canonical_analysis(song_id: str) -> Optional[dict]:
    """Get the canonical (or moderator, if exists) analysis for a song."""
    db = get_db()
    # Try moderator first
    docs = (
        db.collection("songs")
        .document(song_id)
        .collection("analyses")
        .where("type", "==", "moderator")
        .limit(1)
        .get()
    )
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        data["songId"] = song_id
        return data

    # Fall back to canonical
    docs = (
        db.collection("songs")
        .document(song_id)
        .collection("analyses")
        .where("type", "==", "canonical")
        .limit(1)
        .get()
    )
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        data["songId"] = song_id
        return data

    return None


# --- Comments ---

def create_comment(
    song_id: str,
    author_id: str,
    author_name: str,
    text: str,
    timestamp_seconds: float,
    parent_comment_id: Optional[str] = None,
) -> str:
    """Create a timestamped comment. Returns the comment ID."""
    db = get_db()
    comment_id = str(uuid.uuid4())
    db.collection("songs").document(song_id).collection("comments").document(
        comment_id
    ).set({
        "authorId": author_id,
        "authorName": author_name,
        "text": text,
        "timestampSeconds": timestamp_seconds,
        "parentCommentId": parent_comment_id,
        "createdAt": datetime.utcnow(),
    })
    # Increment comment count on song
    db.collection("songs").document(song_id).update({
        "commentCount": firestore.Increment(1)
    })
    return comment_id


def list_comments(song_id: str) -> list[dict]:
    """List all comments for a song, ordered by timestamp in the recording."""
    db = get_db()
    docs = (
        db.collection("songs")
        .document(song_id)
        .collection("comments")
        .order_by("timestampSeconds")
        .get()
    )
    results = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        data["songId"] = song_id
        results.append(data)
    return results
```

- [ ] **Step 2: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add api/firestore_client.py
git commit -m "feat: add Firestore client helpers for songs, analyses, comments"
```

---

### Task 7: Create API Route Stubs with Auth

**Files:**
- Create: `api/schemas.py`
- Create: `api/routes/__init__.py`
- Create: `api/routes/songs.py`
- Create: `api/routes/analysis.py`
- Create: `api/routes/artifacts.py`
- Modify: `api/main.py`

- [ ] **Step 1: Create Pydantic request/response schemas**

Create `api/schemas.py`:

```python
"""Pydantic models for API request and response bodies."""

from typing import Optional
from pydantic import BaseModel


class UploadRequest(BaseModel):
    title: str
    source: str  # "youtube" | "file"
    youtube_url: Optional[str] = None
    visibility: str = "private"
    song_type: Optional[str] = None
    tonic: Optional[str] = None
    raga: Optional[str] = None
    instrument: str = "vocal"
    vocalist_gender: Optional[str] = None


class ReanalyzeRequest(BaseModel):
    tonic: Optional[str] = None
    raga: Optional[str] = None
    instrument: str = "vocal"
    vocalist_gender: Optional[str] = None


class CommentRequest(BaseModel):
    text: str
    timestamp_seconds: float
    parent_comment_id: Optional[str] = None


class SongResponse(BaseModel):
    id: str
    title: str
    source: str
    youtube_video_id: Optional[str]
    visibility: str
    status: str
    song_type: Optional[str]
    view_count: int
    comment_count: int


class JobResponse(BaseModel):
    job_id: str
    song_id: str
    analysis_id: str
    status: str
```

- [ ] **Step 2: Create route stubs for songs**

Create `api/routes/__init__.py`:

```python
```

Create `api/routes/songs.py`:

```python
"""Song CRUD routes: list, get, upload, YouTube dedup check."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional

from api.auth import get_current_user, get_optional_user
from api.schemas import SongResponse

router = APIRouter(prefix="/api/songs", tags=["songs"])


@router.get("")
async def list_songs(user: dict = Depends(get_current_user)):
    """List songs for the authenticated user's library."""
    from api.firestore_client import list_user_songs
    songs = list_user_songs(user["uid"])
    return {"songs": songs}


@router.get("/{song_id}")
async def get_song(song_id: str, user: Optional[dict] = Depends(get_optional_user)):
    """Get a song by ID. Public songs visible to anyone, private songs require owner."""
    from api.firestore_client import get_song as fs_get_song
    song = fs_get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song["visibility"] == "private":
        if not user or user["uid"] != song["uploadedBy"]:
            raise HTTPException(status_code=403, detail="Access denied")
    return song


@router.get("/youtube/check/{video_id}")
async def check_youtube_exists(video_id: str):
    """Check if a YouTube video has already been analyzed in the community archive."""
    from api.firestore_client import find_song_by_youtube_id
    existing = find_song_by_youtube_id(video_id)
    if existing:
        return {"exists": True, "songId": existing["id"], "title": existing["title"]}
    return {"exists": False}


@router.post("/upload-file")
async def upload_file(
    title: str = Form(...),
    visibility: str = Form("private"),
    song_type: Optional[str] = Form(None),
    tonic: Optional[str] = Form(None),
    raga: Optional[str] = Form(None),
    instrument: str = Form("vocal"),
    vocalist_gender: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload an audio file and start analysis."""
    import hashlib
    from api import storage, firestore_client

    # Read file and compute hash
    content = await file.read()
    audio_hash = hashlib.sha256(content).hexdigest()[:16]

    # Create song record
    song_id = firestore_client.create_song(
        title=title,
        source="file",
        uploaded_by=user["uid"],
        audio_hash=audio_hash,
        visibility=visibility,
        song_type=song_type,
    )

    # Save the file
    filename = file.filename or f"{song_id}.mp3"
    rel_path = storage.upload_path(user["uid"], song_id, filename)
    storage.write_file(rel_path, content)

    # Create canonical analysis record
    analysis_id = firestore_client.create_analysis(
        song_id=song_id,
        analysis_type="canonical",
        owner_id=user["uid"],
        params={
            "tonic": tonic,
            "raga": raga,
            "instrument": instrument,
            "vocalistGender": vocalist_gender,
        },
    )

    # TODO (Phase 2): Queue pipeline job here

    return {"songId": song_id, "analysisId": analysis_id, "status": "processing"}


@router.post("/upload-youtube")
async def upload_youtube(
    title: str = Form(...),
    youtube_url: str = Form(...),
    song_type: Optional[str] = Form(None),
    tonic: Optional[str] = Form(None),
    raga: Optional[str] = Form(None),
    instrument: str = Form("vocal"),
    vocalist_gender: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    """Submit a YouTube URL for analysis."""
    import re
    from api import firestore_client

    # Extract video ID from URL
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", youtube_url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    video_id = match.group(1)

    # Check if already analyzed
    existing = firestore_client.find_song_by_youtube_id(video_id)

    if existing:
        # Create a fork analysis on the existing song
        analysis_id = firestore_client.create_analysis(
            song_id=existing["id"],
            analysis_type="fork",
            owner_id=user["uid"],
            params={
                "tonic": tonic,
                "raga": raga,
                "instrument": instrument,
                "vocalistGender": vocalist_gender,
            },
        )
        return {
            "songId": existing["id"],
            "analysisId": analysis_id,
            "status": "processing",
            "reusedExisting": True,
        }

    # Create new song record (YouTube = public by default)
    audio_hash = f"yt_{video_id}"
    song_id = firestore_client.create_song(
        title=title,
        source="youtube",
        uploaded_by=user["uid"],
        audio_hash=audio_hash,
        youtube_video_id=video_id,
        visibility="public",
        song_type=song_type,
    )

    analysis_id = firestore_client.create_analysis(
        song_id=song_id,
        analysis_type="canonical",
        owner_id=user["uid"],
        params={
            "tonic": tonic,
            "raga": raga,
            "instrument": instrument,
            "vocalistGender": vocalist_gender,
        },
    )

    # TODO (Phase 2): Queue pipeline job (download + detect + analyze)

    return {"songId": song_id, "analysisId": analysis_id, "status": "processing"}
```

- [ ] **Step 3: Create route stubs for analysis**

Create `api/routes/analysis.py`:

```python
"""Analysis routes: get results, re-analyze, fork."""

from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_current_user, get_optional_user
from api.schemas import ReanalyzeRequest
from typing import Optional

router = APIRouter(prefix="/api/songs/{song_id}/analysis", tags=["analysis"])


@router.get("")
async def get_analysis(
    song_id: str,
    user: Optional[dict] = Depends(get_optional_user),
):
    """Get the best available analysis for a song.

    Returns moderator version if exists, otherwise canonical.
    If user is authenticated and has a fork, returns that instead.
    """
    from api.firestore_client import get_song, get_canonical_analysis

    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    if song["visibility"] == "private":
        if not user or user["uid"] != song["uploadedBy"]:
            raise HTTPException(status_code=403, detail="Access denied")

    analysis = get_canonical_analysis(song_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis found")

    return analysis


@router.post("/reanalyze")
async def reanalyze(
    song_id: str,
    request: ReanalyzeRequest,
    user: dict = Depends(get_current_user),
):
    """Re-run analysis with different parameters. Creates a fork."""
    from api.firestore_client import get_song, get_canonical_analysis, create_analysis

    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    canonical = get_canonical_analysis(song_id)
    if not canonical:
        raise HTTPException(status_code=400, detail="No canonical analysis to fork from")

    analysis_id = create_analysis(
        song_id=song_id,
        analysis_type="fork",
        owner_id=user["uid"],
        params={
            "tonic": request.tonic,
            "raga": request.raga,
            "instrument": request.instrument,
            "vocalistGender": request.vocalist_gender,
        },
        parent_analysis_id=canonical["id"],
    )

    # TODO (Phase 2): Queue analysis-only job (reuse cached stems + pitch)

    return {"analysisId": analysis_id, "status": "processing"}
```

- [ ] **Step 4: Create route stubs for artifact serving**

Create `api/routes/artifacts.py`:

```python
"""Artifact serving routes: audio files, pitch data, analysis data."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

from api.auth import get_optional_user
from api import storage

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("/audio/{song_id}/{filename}")
async def get_audio(
    song_id: str,
    filename: str,
    user: Optional[dict] = Depends(get_optional_user),
):
    """Serve an audio file (original, vocals, accompaniment)."""
    # TODO: access control check against song visibility
    from api.firestore_client import get_song

    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    audio_hash = song.get("audioHash", "")

    # Try artifact dir first (stems)
    rel_path = f"{storage.artifact_dir(audio_hash)}/{filename}"
    if storage.file_exists(rel_path):
        return FileResponse(
            str(storage.get_absolute_path(rel_path)),
            media_type="audio/mpeg",
        )

    raise HTTPException(status_code=404, detail="Audio file not found")


@router.get("/pitch-data/{song_id}")
async def get_pitch_data(
    song_id: str,
    stem: str = "vocals",
    user: Optional[dict] = Depends(get_optional_user),
):
    """Serve pitch data as JSON for the synced pitch plot."""
    from api.firestore_client import get_song

    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    audio_hash = song.get("audioHash", "")
    csv_path = f"{storage.artifact_dir(audio_hash)}/{stem}_pitch_data.csv"

    if not storage.file_exists(csv_path):
        raise HTTPException(status_code=404, detail="Pitch data not found")

    # Read CSV and convert to JSON
    import csv
    import io

    raw = storage.read_file(csv_path)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    points = []
    for row in reader:
        points.append({
            "time": float(row.get("time", 0)),
            "frequency": float(row.get("frequency", 0)),
            "confidence": float(row.get("confidence", 0)),
        })

    return {"stem": stem, "points": points}


@router.get("/analysis-data/{song_id}/{analysis_id}/{filename}")
async def get_analysis_data(
    song_id: str,
    analysis_id: str,
    filename: str,
    user: Optional[dict] = Depends(get_optional_user),
):
    """Serve analysis artifact files (histogram data, transition matrix, etc.)."""
    rel_path = f"{storage.analysis_dir(analysis_id)}/{filename}"
    if not storage.file_exists(rel_path):
        raise HTTPException(status_code=404, detail="Analysis data not found")

    raw = storage.read_file(rel_path)

    if filename.endswith(".json"):
        import json
        return JSONResponse(content=json.loads(raw))
    elif filename.endswith(".csv"):
        return FileResponse(
            str(storage.get_absolute_path(rel_path)),
            media_type="text/csv",
        )

    raise HTTPException(status_code=400, detail="Unsupported file type")
```

- [ ] **Step 5: Wire routes into main app**

Update `api/main.py`:

```python
"""Multi-user FastAPI backend for the Raga Detection web app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import songs, analysis, artifacts

app = FastAPI(
    title="Raga Detection API",
    version="1.0.0",
    description="Multi-user raga detection and analysis API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(songs.router)
app.include_router(analysis.router)
app.include_router(artifacts.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/ragas")
async def list_ragas():
    """List available raga names from the database."""
    import csv
    from pathlib import Path

    csv_path = Path(__file__).parent.parent / "raga_pipeline" / "data" / "raga_list_final.csv"
    if not csv_path.exists():
        return {"ragas": []}

    ragas = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("raga_name") or row.get("name") or ""
            if name:
                ragas.append(name)

    return {"ragas": sorted(set(ragas))}
```

- [ ] **Step 6: Verify the API starts with all routes**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
python -m uvicorn api.main:app --host 127.0.0.1 --port 8765 --reload
```

Expected: Server starts without import errors. `curl http://localhost:8765/api/health` returns ok. `curl http://localhost:8765/api/ragas` returns a list of raga names. `curl http://localhost:8765/api/songs` returns 401 (no auth token).

- [ ] **Step 7: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add api/schemas.py api/routes/ api/main.py
git commit -m "feat: add API route stubs for songs, analysis, artifacts with auth"
```

---

### Task 8: Create Frontend API Client

**Files:**
- Create: `web/src/lib/api.ts`

- [ ] **Step 1: Write the API client**

Create `web/src/lib/api.ts`:

```typescript
/**
 * FastAPI client with Firebase auth token injection.
 * All API calls go through this module.
 */

import { auth } from "./firebase";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function getAuthHeaders(): Promise<HeadersInit> {
  const user = auth.currentUser;
  if (!user) return {};
  const token = await user.getIdToken();
  return { Authorization: `Bearer ${token}` };
}

async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const authHeaders = await getAuthHeaders();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...authHeaders,
      ...options.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `API error: ${res.status}`);
  }

  return res;
}

// --- Songs ---

export async function listSongs() {
  const res = await apiFetch("/api/songs");
  return res.json();
}

export async function getSong(songId: string) {
  const res = await apiFetch(`/api/songs/${songId}`);
  return res.json();
}

export async function checkYoutubeExists(videoId: string) {
  const res = await apiFetch(`/api/songs/youtube/check/${videoId}`);
  return res.json();
}

export async function uploadFile(formData: FormData) {
  const res = await apiFetch("/api/songs/upload-file", {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function uploadYoutube(formData: FormData) {
  const res = await apiFetch("/api/songs/upload-youtube", {
    method: "POST",
    body: formData,
  });
  return res.json();
}

// --- Analysis ---

export async function getAnalysis(songId: string) {
  const res = await apiFetch(`/api/songs/${songId}/analysis`);
  return res.json();
}

export async function reanalyze(
  songId: string,
  params: { tonic?: string; raga?: string; instrument?: string; vocalistGender?: string }
) {
  const res = await apiFetch(`/api/songs/${songId}/analysis/reanalyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return res.json();
}

// --- Artifacts ---

export function audioUrl(songId: string, filename: string): string {
  return `${API_BASE}/api/artifacts/audio/${songId}/${filename}`;
}

export async function getPitchData(songId: string, stem: string = "vocals") {
  const res = await apiFetch(`/api/artifacts/pitch-data/${songId}?stem=${stem}`);
  return res.json();
}

// --- Ragas ---

export async function listRagas() {
  const res = await apiFetch("/api/ragas");
  return res.json();
}

// --- Explore ---

export async function listPublicSongs(params?: {
  orderBy?: string;
  songType?: string;
  limit?: number;
}) {
  const query = new URLSearchParams();
  if (params?.orderBy) query.set("order_by", params.orderBy);
  if (params?.songType) query.set("song_type", params.songType);
  if (params?.limit) query.set("limit", String(params.limit));
  const res = await apiFetch(`/api/explore?${query.toString()}`);
  return res.json();
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add web/src/lib/api.ts
git commit -m "feat: add frontend API client with Firebase auth token injection"
```

---

### Task 9: Create Protected Route Wrapper and Library Placeholder

**Files:**
- Create: `web/src/components/layout/ProtectedRoute.tsx`
- Create: `web/src/components/layout/Header.tsx`
- Create: `web/src/app/library/page.tsx`

- [ ] **Step 1: Create route protection component**

Create `web/src/components/layout/ProtectedRoute.tsx`:

```tsx
"use client";

import { useAuthContext } from "./AuthProvider";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuthContext();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push("/");
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-text-muted">Loading...</p>
      </div>
    );
  }

  if (!user) return null;

  return <>{children}</>;
}
```

- [ ] **Step 2: Create global header**

Create `web/src/components/layout/Header.tsx`:

```tsx
"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { useAuthContext } from "./AuthProvider";

export function Header() {
  const { user } = useAuthContext();
  const router = useRouter();

  async function handleSignOut() {
    await signOut(auth);
    router.push("/");
  }

  if (!user) return null;

  const initials = (user.displayName || user.email || "U")
    .split(" ")
    .map((s) => s[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-border">
      <Link href="/library" className="text-xl font-semibold text-text-primary">
        Raga Detection
      </Link>

      <nav className="flex items-center gap-6">
        <Link
          href="/library"
          className="text-text-secondary hover:text-text-primary transition-colors text-sm"
        >
          Library
        </Link>
        <Link
          href="/explore"
          className="text-text-secondary hover:text-text-primary transition-colors text-sm"
        >
          Explore
        </Link>

        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-bg-elevated flex items-center justify-center text-xs text-text-secondary">
            {initials}
          </div>
          <button
            onClick={handleSignOut}
            className="text-text-muted hover:text-text-primary text-sm transition-colors"
          >
            Sign out
          </button>
        </div>
      </nav>
    </header>
  );
}
```

- [ ] **Step 3: Create library page placeholder**

Create `web/src/app/library/page.tsx`:

```tsx
"use client";

import { ProtectedRoute } from "@/components/layout/ProtectedRoute";
import { Header } from "@/components/layout/Header";

export default function LibraryPage() {
  return (
    <ProtectedRoute>
      <Header />
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Your Library</h1>
          <button className="bg-accent text-white px-5 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity">
            + Upload
          </button>
        </div>
        <p className="text-text-muted">
          Your song library will appear here. Upload a recording to get started.
        </p>
      </main>
    </ProtectedRoute>
  );
}
```

- [ ] **Step 4: Verify the full auth flow**

```bash
cd web && npm run dev
```

Expected:
1. Visit localhost:3000 -- see sign-in page
2. Sign in -- redirected to /library
3. Library page shows header with "Library" and "Explore" nav, user initials, sign out button
4. Click "Sign out" -- redirected to sign-in page
5. Visit /library directly when signed out -- redirected to sign-in

- [ ] **Step 5: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add web/src/components/layout/ProtectedRoute.tsx web/src/components/layout/Header.tsx web/src/app/library/page.tsx
git commit -m "feat: add protected routes, global header, and library placeholder"
```

---

### Task 10: Production Deployment Config

**Files:**
- Create: `Dockerfile.api`
- Create: `docker-compose.yml`
- Create: `nginx.conf`
- Create: `.env.example`

- [ ] **Step 1: Create FastAPI Dockerfile**

Create `Dockerfile.api`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (resolve symlinks -- Docker COPY follows them)
# In production, copy the actual files rather than relying on symlinks
COPY raga_pipeline/ raga_pipeline/
COPY api/ api/
COPY driver.py .

# Create storage directory
RUN mkdir -p /data

ENV STORAGE_ROOT=/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8765

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8765"]
```

- [ ] **Step 2: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8765:8765"
    volumes:
      - app-data:/data
    env_file:
      - .env
    restart: unless-stopped

  web:
    build:
      context: ./web
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    env_file:
      - ./web/.env.production
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - api
      - web
    restart: unless-stopped

volumes:
  app-data:
```

- [ ] **Step 3: Create nginx config**

Create `nginx.conf`:

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 500M;

    # API routes -> FastAPI
    location /api/ {
        proxy_pass http://api:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Long timeout for pipeline jobs
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }

    # Everything else -> Next.js
    location / {
        proxy_pass http://web:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

- [ ] **Step 4: Create Next.js Dockerfile**

Create `web/Dockerfile`:

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner

WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server.js"]
```

- [ ] **Step 5: Update Next.js config for standalone output**

Add `output: "standalone"` to `web/next.config.ts`. The rewrite only applies in dev (where there's no nginx). In production, nginx handles `/api/*` routing:

```typescript
import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === "development";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    // Only proxy in dev. In production, nginx routes /api/* to FastAPI.
    if (!isDev) return [];
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8765/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 6: Create .env.example**

Create `.env.example`:

```bash
# Firebase Admin SDK (for FastAPI backend)
# Download service account key from Firebase Console > Project Settings > Service Accounts
FIREBASE_SERVICE_ACCOUNT_KEY=/path/to/service-account-key.json

# Storage root (local disk path for audio/artifacts)
STORAGE_ROOT=/data

# Google Cloud project (for Firestore)
GOOGLE_CLOUD_PROJECT=your-firebase-project-id
```

- [ ] **Step 7: Commit**

```bash
cd /Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-web-app
git add Dockerfile.api docker-compose.yml nginx.conf .env.example web/Dockerfile web/next.config.ts
git commit -m "feat: add Docker and nginx production deployment config"
```

---

## What's Next (Phases 2-4)

**Phase 2: Library & Upload UI** (separate plan)
- Song card component (with YouTube thumbnails, status, raga pills)
- Library grid + list views with toggle
- Drag-and-drop on library
- Upload page (file tab, YouTube tab, form fields)
- Job submission and polling (adapt existing jobs.py)
- Processing status on cards

**Phase 3: Results & Editor** (separate plan)
- Results page hero, raga context, YouTube embed
- Synced pitch player with stem toggles (port existing Plotly setup)
- Karaoke transcription view
- Deep analysis section (histograms, transition matrix, candidates)
- Transcript editor page (port from current transcription_editor.js)

**Phase 4: Community Features** (separate plan)
- Explore feed with sort/filter/search
- Fork flow (UI for forking + re-analysis)
- Timestamped comments with pitch plot markers
- Public/private visibility toggle
- View counting
