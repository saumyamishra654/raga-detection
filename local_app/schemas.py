from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    mode: str = Field(..., description="preprocess | detect | analyze")
    params: Dict[str, Any] = Field(default_factory=dict)
    extra_args: List[str] = Field(default_factory=list)


class BatchJobRequest(BaseModel):
    input_dir: str
    output_dir: Optional[str] = None
    mode: str = Field(default="auto", description="auto | detect")
    ground_truth: Optional[str] = None
    silent: bool = True


class ArtifactInfo(BaseModel):
    name: str
    path: str
    exists: bool
    url: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    mode: str
    status: str
    progress: float
    message: str
    cancel_requested: bool = False
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    argv: List[str] = Field(default_factory=list)
    artifacts: List[ArtifactInfo] = Field(default_factory=list)


class JobLogsResponse(BaseModel):
    job_id: str
    logs: List[str]


class EditableNote(BaseModel):
    id: str
    start: float
    end: float
    pitch_midi: float
    pitch_hz: Optional[float] = None
    raw_pitch_midi: Optional[float] = None
    snapped_pitch_midi: Optional[float] = None
    corrected_pitch_midi: Optional[float] = None
    rendered_pitch_midi: Optional[float] = None
    confidence: float = 0.8
    energy: float = 0.0
    sargam: str = ""
    pitch_class: Optional[int] = None


class EditablePhrase(BaseModel):
    id: str
    start: float = 0.0
    end: float = 0.0
    note_ids: List[str] = Field(default_factory=list)


class TranscriptionEditPayload(BaseModel):
    notes: List[EditableNote] = Field(default_factory=list)
    phrases: List[EditablePhrase] = Field(default_factory=list)


class TranscriptionEditVersionInfo(BaseModel):
    version_id: str
    created_at: str
    note_count: int
    phrase_count: int
    json_url: Optional[str] = None
    csv_url: Optional[str] = None
    report_url: Optional[str] = None
    source_report_url: Optional[str] = None


class TranscriptionEditVersionsResponse(BaseModel):
    versions: List[TranscriptionEditVersionInfo] = Field(default_factory=list)
    latest_version_id: Optional[str] = None
    default_selection: str = "original"
    default_report_url: Optional[str] = None


class TranscriptionEditBaseResponse(BaseModel):
    ready: bool
    requires_rerun: bool = False
    detail: Optional[str] = None
    payload: Optional[TranscriptionEditPayload] = None
    tonic: Optional[int] = None
    sargam_options: List[Dict[str, Any]] = Field(default_factory=list)


class TranscriptionEditVersionResponse(BaseModel):
    has_version: bool
    version: Optional[TranscriptionEditVersionInfo] = None
    payload: Optional[TranscriptionEditPayload] = None


class TranscriptionEditSaveResponse(BaseModel):
    save_mode: str = "created"
    version: TranscriptionEditVersionInfo
    versions: List[TranscriptionEditVersionInfo] = Field(default_factory=list)
    default_selection: str = "original"
    default_report_url: Optional[str] = None
    payload: TranscriptionEditPayload
