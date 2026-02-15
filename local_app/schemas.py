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
