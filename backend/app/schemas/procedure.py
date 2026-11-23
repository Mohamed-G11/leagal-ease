# app/schemas/procedure.py  (updated — adds due_date + notification schemas)
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class StepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    blocked = "blocked"


class NotificationType(str, Enum):
    deadline_3days = "deadline_3days"
    deadline_today = "deadline_today"
    deadline_passed = "deadline_passed"
    step_completed = "step_completed"
    procedure_done = "procedure_done"


# ── Document ──────────────────────────────────────────────────────────────────

class StepDocumentResponse(BaseModel):
    id: int
    step_progress_id: int
    original_filename: str
    content_type: str
    file_size: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ── Step template ─────────────────────────────────────────────────────────────

class ProcedureStepBase(BaseModel):
    order: int
    title: str
    description: str | None = None
    documents_required: str | None = None
    estimated_days: int | None = None


class ProcedureStepResponse(ProcedureStepBase):
    id: int
    procedure_type_id: int

    class Config:
        from_attributes = True


# ── Procedure type ────────────────────────────────────────────────────────────

class ProcedureTypeResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    icon: str | None = None
    estimated_days: int | None = None
    steps: list[ProcedureStepResponse] = []

    class Config:
        from_attributes = True


# ── Step progress ─────────────────────────────────────────────────────────────

class UserStepProgressResponse(BaseModel):
    id: int
    step_id: int
    status: StepStatus
    notes: str | None = None
    completed_at: datetime | None = None
    due_date: datetime | None = None          # Phase 4
    step: ProcedureStepResponse
    documents: list[StepDocumentResponse] = []

    class Config:
        from_attributes = True


class UpdateStepStatus(BaseModel):
    status: StepStatus
    notes: str | None = None


class UpdateStepDueDate(BaseModel):
    due_date: datetime | None = None          # null = clear the deadline


# ── User procedure ────────────────────────────────────────────────────────────

class StartProcedureRequest(BaseModel):
    procedure_type_id: int
    title: str | None = None


class UserProcedureResponse(BaseModel):
    id: int
    procedure_type_id: int
    title: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    is_active: bool
    procedure_type: ProcedureTypeResponse
    step_progress: list[UserStepProgressResponse] = []
    completion_percentage: float | None = None
    current_step_order: int | None = None

    class Config:
        from_attributes = True


# ── Notifications ─────────────────────────────────────────────────────────────

class NotificationResponse(BaseModel):
    id: int
    type: NotificationType
    title: str
    message: str
    is_read: bool
    user_procedure_id: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class NotificationSummary(BaseModel):
    """Lightweight summary returned in the navbar badge."""
    total: int
    unread: int
    items: list[NotificationResponse]
