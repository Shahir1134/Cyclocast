"""
backend/database.py
===================
Database connection manager supporting PostgreSQL/PostGIS with an automatic
fallback to SQLite with standard GeoJSON geometries for portable, zero-setup
evaluation on any machine.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read DATABASE_URL or default to portable SQLite database
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    db_file = PROJECT_ROOT / "cyclocast.db"
    DATABASE_URL = f"sqlite:///{db_file}"
    logger.info("DATABASE_URL not set. Defaulting to local SQLite: %s", DATABASE_URL)
else:
    logger.info("Connecting to provided DATABASE_URL: %s", DATABASE_URL)

# Configure engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
