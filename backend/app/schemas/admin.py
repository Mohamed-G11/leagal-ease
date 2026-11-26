# app/schemas/admin.py  (updated — rich BI schemas)
from datetime import datetime

from pydantic import BaseModel

# ── User management ───────────────────────────────────────────────────────────

class AdminUserResponse(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime
    procedure_count: int = 0

    class Config:
        from_attributes = True


class AdminUpdateUser(BaseModel):
    is_active: bool | None = None
    is_admin: bool | None = None


# ── Procedure type management ─────────────────────────────────────────────────

class AdminStepCreate(BaseModel):
    order: int
    title: str
    description: str | None = None
    documents_required: str | None = None
    estimated_days: int | None = None


class AdminStepUpdate(BaseModel):
    order: int | None = None
    title: str | None = None
    description: str | None = None
    documents_required: str | None = None
    estimated_days: int | None = None


class AdminProcedureTypeCreate(BaseModel):
    name: str
    description: str | None = None
    icon: str | None = None
    estimated_days: int | None = None
    steps: list[AdminStepCreate] = []


class AdminProcedureTypeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    icon: str | None = None
    estimated_days: int | None = None


# ── BI schemas ────────────────────────────────────────────────────────────────

class TimeSeriesPoint(BaseModel):
    label: str      # "Jan 2026", "Feb 2026", ...
    value: int


class StepFunnelItem(BaseModel):
    step_order: int
    step_title: str
    total: int
    completed: int
    blocked: int
    pending: int
    in_progress: int
    completion_rate: float   # 0–100


class DocumentTypeStat(BaseModel):
    content_type: str
    label: str               # "PDF", "Image", "Word", "Other"
    count: int
    pct: float


class UserActivityStat(BaseModel):
    username: str
    procedure_count: int
    completed_count: int
    total_steps_done: int
    total_docs_uploaded: int
    last_active: datetime | None = None


class StepStat(BaseModel):
    step_title: str
    procedure_type: str
    total: int
    completed: int
    blocked: int
    avg_days_to_complete: float | None = None


class ProcedureTypeStat(BaseModel):
    id: int
    name: str
    total_started: int
    total_completed: int
    avg_completion_pct: float
    avg_days_to_complete: float | None = None


class StatusDistribution(BaseModel):
    completed: int
    in_progress: int
    pending: int
    blocked: int


class GlobalStats(BaseModel):
    # ── KPIs ──
    total_users: int
    active_users: int
    total_procedures: int
    completed_procedures: int
    total_documents_uploaded: int
    completion_rate: float
    avg_days_to_complete: float | None = None

    # ── Existing ──
    procedure_type_stats: list[ProcedureTypeStat]
    most_blocked_steps: list[StepStat]

    # ── BI ──
    users_over_time: list[TimeSeriesPoint]
    procedures_over_time: list[TimeSeriesPoint]
    step_funnels: list[StepFunnelItem]          # flat list, frontend groups by type
    document_type_stats: list[DocumentTypeStat]
    top_users: list[UserActivityStat]
    status_distribution: StatusDistribution
