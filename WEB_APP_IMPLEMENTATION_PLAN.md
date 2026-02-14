# Raga Detection Web Application - Implementation Plan

**Target:** Professional musicians, educators, researchers (100-500 users)  
**Timeline:** 3-4 month MVP  
**Platform:** Web app + Cross-platform mobile (React Native/Flutter)  
**Storage:** Persistent (Firebase/Supabase)  
**Auth:** Social login (Google/Apple)

---

## Architecture Overview

### High-Level Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                            │
├──────────────────────┬──────────────────────────────────────┤
│  Web Frontend        │  Mobile App                          │
│  (React + TypeScript)│  (React Native / Flutter)            │
│  - Drag & drop UI    │  - Native file picker                │
│  - Interactive plots │  - Offline caching                   │
│  - Real-time updates │  - Push notifications                │
└──────────────────────┴──────────────────────────────────────┘
                            ↓ ↑ (REST/GraphQL + WebSocket)
┌─────────────────────────────────────────────────────────────┐
│                     API GATEWAY                             │
│  (Node.js/Bun + Hono OR FastAPI)                            │
│  - Authentication (Firebase Auth / Supabase Auth)           │
│  - Rate limiting (Redis)                                    │
│  - Job submission & status polling                          │
│  - Static file serving (reports/audio)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                   MESSAGE QUEUE                             │
│  (BullMQ + Redis OR Cloud Tasks)                            │
│  - Job prioritization (stem sep → pitch → analysis)         │
│  - Progress tracking                                        │
│  - Retry/failure handling                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
┌───────────────────────────────────────────────────────────┐
│                  WORKER POOL                              │
│  (Python + Celery OR Temporal)                            │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Stem Separation  │  │ Pitch Extraction │               │
│  │ (GPU-accelerated)│  │ (SwiftF0 + CPU)  │               │
│  │ - Demucs         │  │                  │               │
│  │ - Spleeter       │  └──────────────────┘               │
│  └──────────────────┘                                     │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Raga Detection   │  │ Report Generator │               │
│  │ (CPU)            │  │ (CPU)            │               │
│  └──────────────────┘  └──────────────────┘               │
└───────────────────────────────────────────────────────────┘
                            ↓ ↑
┌───────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                          │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Database         │  │ Object Storage   │               │
│  │ (Firebase/       │  │ (S3/GCS/R2)      │               │
│  │  Supabase)       │  │ - Audio files    │               │
│  │ - User profiles  │  │ - Stems          │               │
│  │ - Job metadata   │  │ - HTML reports   │               │
│  │ - Analysis cache │  │ - Pitch CSVs     │               │
│  └──────────────────┘  └──────────────────┘               │
└───────────────────────────────────────────────────────────┘
```

---

## Technology Stack Recommendations

### Frontend (Web)

**Framework:** React 18+ with TypeScript  
**Why:** Mature ecosystem, excellent TypeScript support, large talent pool

**Key Libraries:**

- **UI Framework:** shadcn/ui (accessible, customizable components)
- **Styling:** Tailwind CSS (rapid development, consistent design)
- **State Management:** Zustand or Jotai (lighter than Redux, sufficient for this scale)
- **Data Fetching:** TanStack Query (formerly React Query) - handles caching, retries, polling
- **Charts:** Plotly.js (already used in reports) + Recharts (for dashboards)
- **Audio:** Wavesurfer.js or Tone.js (waveform visualization, playback sync)
- **File Upload:** Uppy (handles chunked uploads, retries, drag-drop)
- **Notifications:** React Hot Toast

**Project Structure:**
```
web-app/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn components
│   │   ├── audio/          # AudioPlayer, Waveform
│   │   ├── upload/         # FileUploader, DropZone
│   │   └── visualizations/ # PitchPlot, Histogram, etc.
│   ├── features/
│   │   ├── auth/           # Login, signup flows
│   │   ├── analysis/       # Analysis submission, results
│   │   └── library/        # User's analysis history
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # API client, utilities
│   └── stores/             # Zustand stores
├── public/
└── vite.config.ts          # Using Vite for fast builds
```

---

### Mobile App

**Framework:** **React Native (Expo)** - RECOMMENDED  
**Alternative:** Flutter (if team has Dart expertise)

**Why React Native with Expo:**

- Share ~70% code with web (business logic, API calls, state management)
- Expo provides managed workflow (easier deployment, OTA updates)
- Better TypeScript integration than Flutter
- Native modules for audio processing available
- Faster iteration for MVP

**Key Libraries:**

- **Navigation:** React Navigation 6
- **UI:** React Native Paper or Tamagui (cross-platform primitives)
- **Audio:** expo-av or react-native-track-player
- **File Picker:** expo-document-picker
- **Notifications:** expo-notifications
- **Storage:** expo-secure-store (tokens) + AsyncStorage (cache)
- **State:** Same as web (Zustand + TanStack Query)

**Shared Code Strategy:**
```
packages/
├── shared/                  # Shared TypeScript packages
│   ├── api-client/         # API calls, types
│   ├── auth/               # Auth logic
│   ├── utils/              # Pure functions
│   └── types/              # Shared TypeScript types
├── web/                    # Web app
└── mobile/                 # React Native app
```

---

### Backend API

**Option A: FastAPI (Python) - RECOMMENDED FOR MVP**  
**Why:**
- Your existing pipeline is Python-based
- Minimal refactoring needed
- FastAPI is async, performant, has auto-generated OpenAPI docs
- Easy to add WebSocket support for real-time updates
- Pydantic for request validation (matches your dataclasses)

**Structure:**
```python

backend/
├── app/
│   ├── main.py                # FastAPI app entry
│   ├── api/
│   │   ├── auth.py           # /auth/* endpoints
│   │   ├── jobs.py           # /jobs/* (submit, status, cancel)
│   │   ├── results.py        # /results/* (fetch reports, audio)
│   │   └── websocket.py      # Real-time job updates
│   ├── core/
│   │   ├── config.py         # Environment vars, secrets
│   │   ├── auth.py           # Firebase/Supabase token verification
│   │   └── tasks.py          # Celery task definitions
│   ├── models/               # Pydantic models (request/response)
│   ├── services/
│   │   ├── storage.py        # S3/GCS upload/download
│   │   └── queue.py          # Job queue interface
│   └── raga_pipeline/        # Your existing pipeline (imported as module)
├── tests/
└── requirements.txt
```

**Option B: Node.js (Bun + Hono)**  
**Why:** If you want unified TypeScript across frontend/backend

- Bun is extremely fast (Python-level speed for I/O)
- Hono is minimal, performant, similar to FastAPI
- Call Python pipeline via subprocesses or gRPC
- Easier for frontend devs to contribute

**Trade-off:** More complex to integrate Python pipeline, better for long-term if you want to rewrite core logic in Rust/Go.

**Recommendation for MVP:** Start with FastAPI, migrate to Node/Bun later if needed.

---

### Job Queue & Workers

**Option A: Celery + Redis (Python-native)**  
**Pros:**

- Mature, battle-tested
- Works seamlessly with existing Python code
- Rich monitoring (Flower dashboard)

**Cons:**

- Heavier footprint
- Requires separate Redis instance

**Option B: BullMQ (Node.js) + Redis**  
**Pros:**

- Excellent TypeScript support
- Modern API, better observability
- Lighter than Celery

**Cons:**

- Python workers need to communicate via HTTP/gRPC

**Option C: Cloud-Native (AWS SQS + Lambda / GCP Tasks + Cloud Run)**  
**Pros:**

- Fully managed, scales to zero
- No Redis instance to maintain

**Cons:**

- Vendor lock-in
- Cold starts (mitigated with reserved concurrency)

**Recommendation:** Start with **Celery + Redis** for MVP (simpler integration), migrate to cloud tasks if scaling demands it.

---

### Database

**Option A: Supabase (PostgreSQL + Realtime + Auth + Storage)**  
**RECOMMENDED for MVP**

**Why:**

- All-in-one: Database, Auth, Storage, Realtime subscriptions
- Generous free tier (500MB storage, 2GB bandwidth, 50K monthly active users)
- PostgreSQL under the hood (can self-host later)
- Auto-generated REST + GraphQL APIs
- Built-in Row Level Security (RLS)

**Schema:**

```sql
-- Users (managed by Supabase Auth)
-- No need to create users table, use auth.users

-- Jobs
CREATE TABLE jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  status TEXT NOT NULL, -- 'queued' | 'processing' | 'completed' | 'failed'
  mode TEXT NOT NULL,   -- 'detect' | 'analyze'
  config JSONB,         -- Pipeline configuration
  progress INTEGER DEFAULT 0, -- 0-100
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE
);

-- Results (denormalized for fast access)
CREATE TABLE results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  detected_raga TEXT,
  detected_tonic TEXT,
  confidence FLOAT,
  candidates JSONB,     -- Top 10 candidates
  report_url TEXT,      -- Link to HTML report in storage
  audio_urls JSONB,     -- {original, vocals, accompaniment}
  metadata JSONB,       -- Duration, pitch range, etc.
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Usage tracking (for quotas/analytics)
CREATE TABLE usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  processing_seconds INTEGER,
  audio_duration_seconds INTEGER,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_jobs_user_id ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_results_job_id ON results(job_id);
CREATE INDEX idx_results_user_id ON results(user_id);
```

**Option B: Firebase (Firestore + Auth + Storage)**  
**Pros:**

- Excellent mobile SDK integration
- Realtime listeners out of the box
- Offline support

**Cons:**

- NoSQL (less powerful queries)
- More expensive at scale
- Vendor lock-in

**Recommendation:** **Supabase** for better SQL support, easier data export, and PostgreSQL's rich ecosystem.

---

### Object Storage

**Option A: Cloudflare R2**  
**RECOMMENDED**

**Why:**

- S3-compatible API (easy migration)
- **Zero egress fees** (huge cost savings for serving audio/reports)
- 10GB free storage
- Fast global CDN

**Structure:**

```
raga-detection-bucket/
├── audio/
│   └── {user_id}/
│       └── {job_id}/
│           ├── original.mp3
│           ├── vocals.mp3
│           └── accompaniment.mp3
├── reports/
│   └── {user_id}/
│       └── {job_id}/
│           ├── detection_report.html
│           ├── analysis_report.html
│           └── assets/
│               ├── histogram_vocals.png
│               └── pitch_plot.png
└── cache/
    └── {job_id}/
        ├── vocals_pitch_data.csv
        └── candidates.csv
```

**Option B: Supabase Storage**  
**Pros:**

- Integrated with Supabase auth (RLS policies built-in)
- Simpler setup

**Cons:**

- More expensive egress
- Smaller free tier (1GB)

**Recommendation:** Use **Supabase Storage for user profile images**, **Cloudflare R2 for large files** (audio, reports). Configure CORS for direct browser upload to R2.

---

## Infrastructure & Deployment

### Hosting Strategy

**For 100-500 users with stem separation workload:**

#### API Server (Stateless)

- **Platform:** Render, Railway, or Fly.io (for MVP)
- **Specs:** 2 CPU, 4GB RAM (sufficient for API + queue management)
- **Cost:** ~$20-40/month
- **Scaling:** Horizontal (add instances as needed)

#### Worker Instances (Python + GPU for Demucs)

**Challenge:** Stem separation is GPU-intensive. Options:

**Option A: Dedicated GPU VPS (RECOMMENDED for MVP)**

- **Provider:** RunPod, Vast.ai, or Lambda Labs
- **Specs:** 1x NVIDIA T4 or RTX 3060 (10GB+ VRAM), 4 CPU cores, 16GB RAM
- **Cost:** ~$0.30-0.50/hour = ~$220-360/month (if running 24/7)
- **Cost Optimization:** 
  - Auto-shutdown when queue is empty (save ~50% if usage is bursty)
  - Use spot instances (save ~70%, accept occasional evictions)
- **Queue:** Pull jobs from Redis/SQS when instance is running

**Setup:**

```bash
# Worker instance startup script
docker run \
  -e REDIS_URL=$REDIS_URL \
  -e SUPABASE_URL=$SUPABASE_URL \
  -e SUPABASE_KEY=$SUPABASE_KEY \
  -e R2_ACCESS_KEY=$R2_ACCESS_KEY \
  --gpus all \
  raga-detection-worker:latest \
  celery -A app.core.tasks worker --loglevel=info --concurrency=1
```

**Option B: Serverless GPU (Modal, Banana.dev, Replicate)**

- **Pros:** Pay per execution, scales to zero
- **Cons:** Cold starts (10-30s), more expensive per job (~$0.50-1.00 per analysis)
- **When to use:** If usage is very sporadic (<10 jobs/day)

**Option C: Cloud GPU Instances (AWS EC2 P3, GCP A2)**

- **Pros:** Fully managed, auto-scaling
- **Cons:** Expensive (~$3-5/hour for P3 instances)
- **When to use:** If you have consistent high volume (50+ jobs/day)

**Recommendation for MVP:** Start with **RunPod GPU instance (on-demand)** + auto-shutdown script. Monitor usage and migrate to serverless or cloud GPUs if pattern changes.

#### CPU Workers (Pitch Extraction, Analysis)

- **Platform:** Same as API (Render/Railway)
- **Specs:** 2 CPU, 2GB RAM per worker
- **Scaling:** 2-3 workers initially
- **Cost:** ~$15-25/month per worker

#### Redis (Queue + Cache)

- **Option A:** Upstash Redis (serverless, generous free tier)
- **Option B:** Railway Redis (managed, $5/month)
- **Recommendation:** Upstash for MVP

**Total MVP Infrastructure Cost Estimate:**

- API Server: $30/month
- GPU Worker: $250/month (on-demand with auto-shutdown)
- CPU Workers (2x): $40/month
- Redis: Free (Upstash)
- Supabase: Free tier
- Cloudflare R2: Free tier initially
- **Total: ~$320/month** (handles 100-500 jobs/month comfortably)

---

### Deployment Workflow

**Infrastructure as Code:** Use Terraform or Pulumi

```hcl
# terraform/main.tf
resource "render_service" "api" {
  name   = "raga-detection-api"
  type   = "web"
  repo   = "github.com/yourusername/raga-detection"
  branch = "main"
  env_vars = {
    DATABASE_URL = var.supabase_url
    REDIS_URL    = var.redis_url
    R2_BUCKET    = var.r2_bucket
  }
}

resource "runpod_pod" "gpu_worker" {
  name       = "raga-worker-gpu"
  gpu_type   = "NVIDIA RTX A4000"
  image      = "yourdockerhub/raga-worker:latest"
  env_vars   = {
    REDIS_URL = var.redis_url
  }
  auto_shutdown_idle_minutes = 10
}
```

**CI/CD:**

- **GitHub Actions** for automated testing + deployment
- **Docker** for consistent environments
- **Sentry** for error tracking
- **Logflare** or **Axiom** for log aggregation

---

## API Design

### REST Endpoints

**Authentication:** All endpoints require `Authorization: Bearer <firebase_token>`

#### Job Management

```typescript
POST /api/v1/jobs
Content-Type: multipart/form-data

FormData:
  audio: File (required)
  mode: 'detect' | 'analyze' (required)
  config: JSON (optional) {
    source_type?: 'mixed' | 'vocal' | 'instrumental'
    vocalist_gender?: 'male' | 'female'
    tonic?: string (e.g., "C#" or "C,D#")
    raga?: string
    separator?: 'demucs' | 'spleeter'
    // ... other pipeline config
  }

Response: {
  job_id: UUID
  status: 'queued'
  estimated_duration_seconds: 180
}
```

```typescript
GET /api/v1/jobs/:job_id

Response: {
  job_id: UUID
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: 0-100
  current_step: 'stem_separation' | 'pitch_extraction' | 'analysis' | 'report_generation'
  created_at: ISO timestamp
  estimated_completion: ISO timestamp
  error?: string
}
```

```typescript
DELETE /api/v1/jobs/:job_id

Response: 204 No Content
```

#### Results

```typescript
GET /api/v1/results/:job_id

Response: {
  job_id: UUID
  detected_raga: string
  detected_tonic: string
  confidence: number
  candidates: Array<{raga: string, tonic: string, score: number}>
  report_url: string (presigned URL, expires in 24h)
  audio_urls: {
    original: string
    vocals: string
    accompaniment: string
  }
  metadata: {
    duration: number
    processing_time: number
    pitch_range: {min: number, max: number}
  }
}
```

```typescript
GET /api/v1/results
Query params:
  limit?: number (default 20)
  offset?: number (default 0)
  sort?: 'created_at' | 'confidence' (default 'created_at')
  order?: 'asc' | 'desc' (default 'desc')

Response: {
  results: Array<ResultSummary>
  total: number
  has_more: boolean
}
```

#### User Library

```typescript
GET /api/v1/library
Response: {
  analyses: Array<{
    job_id: UUID
    filename: string
    raga: string
    tonic: string
    created_at: string
    thumbnail_url: string
  }>
  total_analyses: number
  storage_used_mb: number
}
```

---

### WebSocket for Real-Time Updates

```typescript
// Client connects
ws://api.example.com/ws?token=<firebase_token>

// Server sends job updates
{
  type: 'job_update'
  job_id: UUID
  status: 'processing'
  progress: 45
  current_step: 'pitch_extraction'
}

// Client subscribes to specific jobs
{
  type: 'subscribe'
  job_ids: [UUID, UUID]
}
```

**Implementation (FastAPI):**
```python
from fastapi import WebSocket
from app.core.auth import verify_token

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    user = await verify_token(token)
    
    # Subscribe to Redis pub/sub for user's jobs
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"user:{user.id}:jobs")
    
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_json(json.loads(message["data"]))
    except WebSocketDisconnect:
        await pubsub.unsubscribe(f"user:{user.id}:jobs")
```

---

## Worker Implementation

### Task Pipeline

```python
# app/core/tasks.py
from celery import Celery, chain
from raga_pipeline import PipelineConfig, separate_stems, extract_pitch, RagaScorer

celery_app = Celery('raga_detection', broker=os.getenv('REDIS_URL'))

@celery_app.task(bind=True)
def process_audio(self, job_id: str, audio_url: str, config: dict):
    """
    Main task that orchestrates the pipeline.
    Updates progress via Redis pub/sub.
    """
    try:
        # 1. Download audio from R2
        self.update_state(state='PROGRESS', meta={'progress': 5, 'step': 'download'})
        audio_path = download_from_storage(audio_url, f"/tmp/{job_id}.mp3")
        
        # 2. Stem separation (40% of total time)
        self.update_state(state='PROGRESS', meta={'progress': 10, 'step': 'stem_separation'})
        vocals_path, accomp_path = separate_stems(
            audio_path=audio_path,
            output_dir=f"/tmp/{job_id}",
            engine=config.get('separator', 'demucs')
        )
        self.update_state(state='PROGRESS', meta={'progress': 50})
        
        # Upload stems to R2
        vocals_url = upload_to_storage(vocals_path, f"{job_id}/vocals.mp3")
        accomp_url = upload_to_storage(accomp_path, f"{job_id}/accompaniment.mp3")
        
        # 3. Pitch extraction (20% of total time)
        self.update_state(state='PROGRESS', meta={'progress': 55, 'step': 'pitch_extraction'})
        pitch_data_vocals = extract_pitch(vocals_path, ...)
        pitch_data_accomp = extract_pitch(accomp_path, ...)
        self.update_state(state='PROGRESS', meta={'progress': 70})
        
        # 4. Analysis (20% of total time)
        self.update_state(state='PROGRESS', meta={'progress': 75, 'step': 'analysis'})
        if config['mode'] == 'detect':
            results = detect_mode_analysis(pitch_data_vocals, pitch_data_accomp, config)
        else:
            results = analyze_mode_analysis(pitch_data_vocals, config)
        self.update_state(state='PROGRESS', meta={'progress': 90})
        
        # 5. Generate report (10% of total time)
        self.update_state(state='PROGRESS', meta={'progress': 92, 'step': 'report_generation'})
        report_path = generate_html_report(results)
        report_url = upload_to_storage(report_path, f"{job_id}/report.html")
        
        # 6. Save to database
        save_results_to_supabase(job_id, {
            'detected_raga': results.detected_raga,
            'detected_tonic': results.detected_tonic,
            'candidates': results.candidates,
            'report_url': report_url,
            'audio_urls': {
                'original': audio_url,
                'vocals': vocals_url,
                'accompaniment': accomp_url
            }
        })
        
        # 7. Notify via Redis pub/sub
        redis_client.publish(
            f"user:{user_id}:jobs",
            json.dumps({'job_id': job_id, 'status': 'completed', 'progress': 100})
        )
        
        # Cleanup temp files
        cleanup_temp_files(job_id)
        
        return {'status': 'completed', 'report_url': report_url}
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        update_job_status(job_id, 'failed', error=str(e))
        raise
```

### Error Handling & Retries

```python
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 1 minute
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 10 minutes
    retry_jitter=True
)
def process_audio_with_retry(self, job_id: str, audio_url: str, config: dict):
    try:
        return process_audio(self, job_id, audio_url, config)
    except StemSeparationError as e:
        # Don't retry if audio is corrupted
        if "corrupted" in str(e).lower():
            raise self.retry(exc=e, countdown=0, max_retries=0)
        raise
```

---

## Frontend Implementation Details

### File Upload Flow

```typescript
// src/features/analysis/components/UploadPane.tsx
import Uppy from '@uppy/core'
import { Dashboard } from '@uppy/react'
import AwsS3 from '@uppy/aws-s3'

export function UploadPane() {
  const uppy = useUppy(() => {
    return new Uppy({
      restrictions: {
        maxFileSize: 100 * 1024 * 1024, // 100MB
        allowedFileTypes: ['.mp3', '.wav', '.flac', '.m4a']
      }
    }).use(AwsS3, {
      // Get presigned URL from your API
      getUploadParameters: async (file) => {
        const response = await api.post('/upload/presigned-url', {
          filename: file.name,
          content_type: file.type
        })
        return {
          method: 'PUT',
          url: response.data.upload_url,
          fields: {},
          headers: {
            'Content-Type': file.type
          }
        }
      }
    })
  })

  uppy.on('upload-success', (file, response) => {
    // Submit job after upload
    submitAnalysisJob({
      audio_url: response.uploadURL,
      mode: selectedMode,
      config: formValues
    })
  })

  return <Dashboard uppy={uppy} />
}
```

### Real-Time Progress Updates

```typescript
// src/hooks/useJobProgress.ts
import { useEffect, useState } from 'react'
import { useWebSocket } from './useWebSocket'

export function useJobProgress(jobId: string) {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('queued')
  const [currentStep, setCurrentStep] = useState('')
  
  const { subscribe } = useWebSocket()
  
  useEffect(() => {
    const unsubscribe = subscribe<JobUpdate>('job_update', (data) => {
      if (data.job_id === jobId) {
        setProgress(data.progress)
        setStatus(data.status)
        setCurrentStep(data.current_step)
      }
    })
    
    return unsubscribe
  }, [jobId])
  
  return { progress, status, currentStep }
}
```

### Interactive Report Viewer

```typescript
// src/features/analysis/components/ReportViewer.tsx
import DOMPurify from 'isomorphic-dompurify'

export function ReportViewer({ reportUrl }: { reportUrl: string }) {
  const [htmlContent, setHtmlContent] = useState('')
  
  useEffect(() => {
    // Fetch and sanitize HTML
    fetch(reportUrl)
      .then(res => res.text())
      .then(html => {
        const sanitized = DOMPurify.sanitize(html, {
          ALLOW_TAGS: ['div', 'span', 'canvas', 'audio', 'script'],
          ALLOW_ATTR: ['id', 'class', 'src', 'controls', 'data-*']
        })
        setHtmlContent(sanitized)
      })
  }, [reportUrl])
  
  return (
    <div className="report-container">
      <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
    </div>
  )
}
```

**Alternative:** Rebuild reports as React components (better UX, more control)

---

## Mobile App Implementation

### Project Setup (React Native + Expo)

```bash
npx create-expo-app raga-detection-mobile --template
cd raga-detection-mobile
npx expo install expo-av expo-document-picker expo-file-system expo-notifications
npm install @tanstack/react-query zustand
```

### File Upload from Mobile

```typescript
// mobile/src/screens/UploadScreen.tsx
import * as DocumentPicker from 'expo-document-picker'
import * as FileSystem from 'expo-file-system'

async function pickAndUploadAudio() {
  // Pick file
  const result = await DocumentPicker.getDocumentAsync({
    type: 'audio/*',
    copyToCacheDirectory: true
  })
  
  if (result.type === 'cancel') return
  
  // Get presigned URL
  const { upload_url, job_id } = await api.post('/upload/presigned-url', {
    filename: result.name,
    content_type: result.mimeType
  })
  
  // Upload to R2 using FileSystem
  const uploadResponse = await FileSystem.uploadAsync(upload_url, result.uri, {
    httpMethod: 'PUT',
    uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
    headers: { 'Content-Type': result.mimeType }
  })
  
  if (uploadResponse.status === 200) {
    // Submit job
    await api.post('/jobs', { job_id, mode: 'detect', config: {} })
    navigation.navigate('JobProgress', { job_id })
  }
}
```

### Push Notifications (Job Complete)

```typescript
// mobile/src/services/notifications.ts
import * as Notifications from 'expo-notifications'

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
})

export async function setupNotifications() {
  const { status } = await Notifications.requestPermissionsAsync()
  if (status !== 'granted') return
  
  const token = (await Notifications.getExpoPushTokenAsync()).data
  
  // Send token to backend
  await api.post('/users/push-token', { token })
}

// Backend sends notification via Expo Push API when job completes
```

### Offline Support

```typescript
// mobile/src/services/offline.ts
import AsyncStorage from '@react-native-async-storage/async-storage'
import NetInfo from '@react-native-community/netinfo'

const offlineQueue: QueuedJob[] = []

export async function submitJobOffline(jobData: JobData) {
  const isConnected = await NetInfo.fetch().then(state => state.isConnected)
  
  if (!isConnected) {
    // Save to local queue
    offlineQueue.push({ ...jobData, timestamp: Date.now() })
    await AsyncStorage.setItem('offline_queue', JSON.stringify(offlineQueue))
    
    // Show toast: "Job will be submitted when online"
    return { queued_offline: true }
  }
  
  return api.post('/jobs', jobData)
}

// Listen for network reconnection
NetInfo.addEventListener(state => {
  if (state.isConnected && offlineQueue.length > 0) {
    processOfflineQueue()
  }
})
```

---

## Security Considerations

### Authentication

**Firebase Auth / Supabase Auth:**

```typescript
// Web: Sign in with Google
import { signInWithPopup, GoogleAuthProvider } from 'firebase/auth'

async function signInWithGoogle() {
  const provider = new GoogleAuthProvider()
  const result = await signInWithPopup(auth, provider)
  const token = await result.user.getIdToken()
  
  // Store token, use in API requests
  localStorage.setItem('auth_token', token)
}
```

**Backend token verification:**

```python
from firebase_admin import auth as firebase_auth

async def verify_token(token: str) -> User:
    try:
        decoded = firebase_auth.verify_id_token(token)
        user_id = decoded['uid']
        
        # Fetch or create user in your database
        user = await db.get_or_create_user(user_id, decoded['email'])
        return user
    except Exception as e:
        raise HTTPException(401, "Invalid token")
```

### API Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/jobs")
@limiter.limit("10/hour")  # 10 job submissions per hour per IP
async def create_job(request: Request, ...):
    pass
```

### File Upload Security

- **Validate file types** (check magic bytes, not just extension)
- **Scan for malware** (use ClamAV or VirusTotal API)
- **Limit file size** (enforce 100MB max)
- **Generate random filenames** (prevent directory traversal)
- **Use presigned URLs** (client uploads directly to R2, never through your API)

```python
import magic

def validate_audio_file(file_path: str) -> bool:
    mime = magic.from_file(file_path, mime=True)
    allowed_types = ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/mp4']
    return mime in allowed_types
```

### Data Privacy

- **Encrypt audio at rest** (use R2's server-side encryption)
- **Auto-delete old files** (set lifecycle rules: delete after 30 days)
- **User data export** (GDPR compliance: provide endpoint to download all user data)
- **Row Level Security** (Supabase RLS ensures users only see their own data)

```sql
-- Supabase RLS policy
CREATE POLICY "Users can only view their own jobs"
  ON jobs FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can only insert their own jobs"
  ON jobs FOR INSERT
  WITH CHECK (auth.uid() = user_id);
```

---

## MVP Feature Checklist

### Phase 1: Core Backend (Weeks 1-4)

- [ ] FastAPI setup with auth (Firebase/Supabase)
- [ ] Job queue (Celery + Redis)
- [ ] Worker integration (existing pipeline as Celery tasks)
- [ ] Supabase schema + RLS policies
- [ ] R2 storage integration
- [ ] WebSocket for real-time updates
- [ ] REST API endpoints (jobs, results, library)
- [ ] Error handling + logging (Sentry)
- [ ] GPU worker deployment (RunPod)

### Phase 2: Web Frontend (Weeks 3-6)
\
- [ ] React + TypeScript + Vite setup
- [ ] Auth UI (sign in with Google/Apple)
- [ ] File upload with progress (Uppy)
- [ ] Analysis config form (detect vs analyze mode, options)
- [ ] Job progress page with real-time updates
- [ ] Report viewer (embedded HTML or rebuilt as React)
- [ ] User library (list past analyses)
- [ ] Responsive design (mobile-first)
- [ ] Dark mode

### Phase 3: Mobile App (Weeks 5-8)

- [ ] Expo app setup
- [ ] Shared API client package
- [ ] Auth screens (Google/Apple sign-in)
- [ ] File picker + upload
- [ ] Job submission form
- [ ] Progress tracking with push notifications
- [ ] Results viewer (simplified report)
- [ ] Offline queue
- [ ] iOS TestFlight build
- [ ] Android internal testing build

### Phase 4: Polish & Launch (Weeks 9-12)

- [ ] Performance optimization (lazy loading, code splitting)
- [ ] Error boundaries + fallbacks
- [ ] User onboarding flow
- [ ] Help documentation
- [ ] Analytics (Posthog or Mixpanel)
- [ ] Load testing (artillery.io)
- [ ] Beta testing with 10-20 users
- [ ] Bug fixes from beta feedback
- [ ] Production deployment
- [ ] App Store / Play Store submission

---

## Post-MVP Roadmap

### Performance Optimizations

- **Caching:** Cache frequently analyzed ragas/tonics
- **Batch processing:** Process multiple files from same album together
- **Model optimization:** Quantize Demucs model (INT8) for faster inference

### Advanced Features

- **Collaborative analysis:** Share reports with collaborators
- **Comparison mode:** Compare 2+ recordings side-by-side
- **Playlist analysis:** Analyze entire concerts/albums
- **Export to MIDI:** Convert transcribed notes to MIDI file
- **Educational mode:** Interactive tutorials on raga theory
- **Social features:** Community raga database contributions

### Scaling Considerations

- **CDN for static assets** (Cloudflare Pages)
- **Database read replicas** (Supabase supports this)
- **Horizontal worker scaling** (Kubernetes for workers)
- **Multi-region deployment** (if serving global audience)
- **GraphQL API** (replace REST for more efficient mobile queries)

---

## Cost Breakdown (Monthly, MVP Scale)

| Service | Provider | Usage | Cost |
|---------|----------|-------|------|
| API Server | Render/Railway | 2 CPU, 4GB RAM | $30 |
| GPU Worker | RunPod | RTX 3060, ~50% uptime | $250 |
| CPU Workers (2x) | Render | 2 CPU, 2GB RAM each | $40 |
| Redis | Upstash | Free tier | $0 |
| Database | Supabase | Free tier | $0 |
| Storage | Cloudflare R2 | 50GB, 500GB egress | $1.50 |
| Auth | Firebase/Supabase | Included | $0 |
| Domain | Cloudflare | .com | $10 |
| Monitoring | Sentry | Free tier | $0 |
| **Total** | | | **~$330/month** |

**At scale (500 users, 1000 jobs/month):**

- GPU worker: $500-700 (full-time)
- Storage: $10-20 (200GB)
- Database: $25 (Pro plan for better performance)
- Workers: $100 (4-5 instances)
- **Total: ~$650-850/month**

---

## Development Timeline (3-4 Months)

### Month 1: Backend Foundation

- Week 1: FastAPI setup, Supabase schema, R2 integration
- Week 2: Celery workers, pipeline integration
- Week 3: WebSocket, real-time updates
- Week 4: Deploy to Render + RunPod, end-to-end testing

### Month 2: Web Frontend

- Week 5: React setup, auth, routing
- Week 6: Upload flow, job submission
- Week 7: Results viewer, library
- Week 8: Polish, responsive design

### Month 3: Mobile App

- Week 9: Expo setup, shared packages
- Week 10: Auth, upload, job submission
- Week 11: Results viewer, offline support
- Week 12: Push notifications, TestFlight build

### Month 4: Testing & Launch

- Week 13: Beta testing, bug fixe
- Week 14: Performance optimization
- Week 15: Documentation, onboarding
- Week 16: Production launch, monitoring

---

## Open Questions for You

1. **Branding:** Do you have a name/domain for the app?
2. **User research:** Have you talked to potential users? What are their pain points?
3. **Unique value:** What makes this better than transcription services like YouTube auto-captions or proprietary tools?
4. **Collaboration:** Will this be a solo project or do you plan to hire/partner with developers?
5. **Monetization:** Even if not immediate, any thoughts on long-term business model? (Freemium, education licensing, API access?)
6. **Data retention:** Should users be able to delete their data? Auto-delete after X days?
7. **Accuracy vs speed trade-off:** Would users prefer faster results with lower accuracy, or wait longer for better quality?
8. **Educational content:** Should the app include learning resources (raga reference pages, theory guides)?

---

## Next Steps

1. **Validate assumptions:** Interview 5-10 potential users (musicians, teachers, students)
2. **Set up infrastructure:** Reserve domain, create Supabase project, Cloudflare account
3. **Prototype API:** Build minimal FastAPI with 1 endpoint (submit job) + 1 worker
4. **Design UI mockups:** Sketch main screens (upload → progress → results)
5. **Choose tech stack:** Finalize React vs Next.js, React Native vs Flutter
6. **Build MVP backend:** Follow Phase 1 checklist above
7. **Test with real users:** Get feedback early and often

Let me know your thoughts on any of these recommendations, and I can refine the plan further!
