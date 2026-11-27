# app/v1/endpoints/admin.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_admin
from app.core.errors import ERROR_RESPONSES
from app.crud import admin as crud_admin
from app.schemas.admin import (
    AdminProcedureTypeCreate,
    AdminProcedureTypeUpdate,
    AdminStepCreate,
    AdminStepUpdate,
    AdminUpdateUser,
    AdminUserResponse,
    GlobalStats,
)
from app.schemas.procedure import ProcedureStepResponse, ProcedureTypeResponse
from app.services.rag import rag_service

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Served model ──────────────────────────────────────────────────────────────

class ReloadResult(BaseModel):
    changed: bool
    source: str | None
    version: str | None
    config: str | None
    message: str | None  # why the registry champion is not the one being served


# A plain `def`: loading a pipeline blocks, so FastAPI must run this in its thread pool.
@router.post("/rag/reload", response_model=ReloadResult, responses=ERROR_RESPONSES,
             summary="Serve the current champion")
def reload_model(_: object = Depends(get_current_admin)):
    """Call after moving the champion alias in the model registry. No restart or redeploy is needed."""
    before = rag_service.model_info()
    message = rag_service.reload()
    after = rag_service.model_info()
    return ReloadResult(changed=after != before, message=message, **after)


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=GlobalStats)
def global_stats(
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    return crud_admin.get_global_stats(db)


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users", response_model=list[AdminUserResponse])
def list_users(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    users = crud_admin.list_users(db, skip=skip, limit=limit)
    result = []
    for u in users:
        count = crud_admin.get_user_procedure_count(db, u.id)
        result.append(AdminUserResponse(
            id=u.id,
            email=u.email,
            username=u.username,
            is_active=bool(u.is_active) if u.is_active is not None else True,
            is_admin=bool(u.is_admin) if u.is_admin is not None else False,
            created_at=u.created_at,
            procedure_count=count,
        ))
    return result


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
def update_user(
    user_id: int,
    body: AdminUpdateUser,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    user = crud_admin.update_user(db, user_id, body)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    count = crud_admin.get_user_procedure_count(db, user.id)
    return AdminUserResponse(
        id=user.id, email=user.email, username=user.username,
        is_active=user.is_active, is_admin=user.is_admin,
        created_at=user.created_at, procedure_count=count,
    )


# ── Procedure types ───────────────────────────────────────────────────────────

@router.post("/procedure-types", response_model=ProcedureTypeResponse, status_code=201)
def create_procedure_type(
    body: AdminProcedureTypeCreate,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    return crud_admin.create_procedure_type(db, body)


@router.patch("/procedure-types/{type_id}", response_model=ProcedureTypeResponse)
def update_procedure_type(
    type_id: int,
    body: AdminProcedureTypeUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    pt = crud_admin.update_procedure_type(db, type_id, body)
    if not pt:
        raise HTTPException(status_code=404, detail="Procedure type not found")
    return pt


@router.delete("/procedure-types/{type_id}", status_code=204)
def delete_procedure_type(
    type_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    if not crud_admin.delete_procedure_type(db, type_id):
        raise HTTPException(status_code=404, detail="Procedure type not found")


# ── Steps ─────────────────────────────────────────────────────────────────────

@router.post("/procedure-types/{type_id}/steps", response_model=ProcedureStepResponse, status_code=201)
def add_step(
    type_id: int,
    body: AdminStepCreate,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    step = crud_admin.add_step(db, type_id, body)
    if not step:
        raise HTTPException(status_code=404, detail="Procedure type not found")
    return step


@router.patch("/steps/{step_id}", response_model=ProcedureStepResponse)
def update_step(
    step_id: int,
    body: AdminStepUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    step = crud_admin.update_step(db, step_id, body)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")
    return step


@router.delete("/steps/{step_id}", status_code=204)
def delete_step(
    step_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(get_current_admin),
):
    if not crud_admin.delete_step(db, step_id):
        raise HTTPException(status_code=404, detail="Step not found")
