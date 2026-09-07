"""
backend/main.py
===============
Main FastAPI application entry point for the CycloCast platform.
Serves REST endpoints, mounts satellite imagery, and manages database lifecycle.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database import engine, Base, SessionLocal
from backend.models import *  # Ensure all models are registered with Base
from backend.services.seed_data import seed_database
from backend.routers import cyclones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger("cyclocast_backend")

# Project root and output directory for satellite overlays
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EUMETSAT_OUTPUT_DIR = PROJECT_ROOT / "eumetsat_output"
EUMETSAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def init_db():
    """Initialize tables and run seeder."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()

# Auto-initialize on startup
init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event: ensure database schema and seed data are ready."""
    init_db()
    yield
    logger.info("CycloCast backend shutting down.")


app = FastAPI(
    title="CycloCast Platform API",
    description="Tropical Cyclone Pattern Identification, Classification & Multimodal Prediction System (SIH 26070)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local React/Vite development and dashboard embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount satellite imagery directory as static assets
app.mount("/api/satellite", StaticFiles(directory=str(EUMETSAT_OUTPUT_DIR)), name="satellite_images")

# Register routers: support both /api/cyclones and /cyclones directly
app.include_router(cyclones.router, prefix="/api")
app.include_router(cyclones.router, prefix="")


@app.post("/api/forecast/live")
@app.post("/forecast/live")
def trigger_forecast_live_alias(db=Depends(cyclones.get_db)):
    """Convenience alias route matching frontend trigger."""
    return cyclones.trigger_live_forecast(db=db)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CycloCast Backend API",
        "version": "1.0.0",
        "database": str(engine.url),
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
