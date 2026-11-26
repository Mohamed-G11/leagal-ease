# app/main.py  (final — all routers registered)
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import SessionLocal
from app.crud.notification import check_deadlines
from app.v1.endpoints import (
    admin,
    ai,
    analytics,
    auth,
    collaboration,
    documents,
    feedback,
    notifications,
    procedure,
    upload,
)


def run_deadline_check():
    db = SessionLocal()
    try:
        check_deadlines(db)
        print("[scheduler] Deadline check completed.")
    finally:
        db.close()


scheduler = BackgroundScheduler()
scheduler.add_job(run_deadline_check, trigger="cron", hour=8, minute=0,
                  id="deadline_check", replace_existing=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="LegalEase Tunisia API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("uploads").mkdir(exist_ok=True)

app.include_router(auth.router,          prefix="/api/v1")
app.include_router(procedure.router,     prefix="/api/v1")
app.include_router(documents.router,     prefix="/api/v1")
app.include_router(notifications.router, prefix="/api/v1")
app.include_router(admin.router,         prefix="/api/v1")
app.include_router(ai.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(collaboration.router, prefix="/api/v1")




@app.get("/")
def root():
    return {"message": "Welcome to the LegalEase Tunisia API"}


@app.middleware("http")
async def log_requests(request, call_next):
    print(f"  {request.method} {request.url.path}")
    response = await call_next(request)
    return response
