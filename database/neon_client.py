"""
database/neon_client.py
=======================
Neon PostgreSQL connection pool with pgvector support.
"""

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator

from config.settings import NEON_DATABASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """Initialize and return the connection pool (singleton)."""
    global _pool
    if _pool is None:
        if not NEON_DATABASE_URL:
            raise ValueError(
                "NEON_DATABASE_URL is not set. Please add it to your .env file."
            )
        logger.info("Creating Neon PostgreSQL connection pool...")
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=NEON_DATABASE_URL,
        )
        logger.info("Neon connection pool ready.")
    return _pool


@contextmanager
def get_connection() -> Generator:
    """Context manager to get a DB connection from the pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(cursor_factory=RealDictCursor) -> Generator:
    """Context manager for a DB cursor."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("Neon connection pool closed.")


def test_connection() -> bool:
    """Test the database connection."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT version();")
            row = cur.fetchone()
            logger.info(f"DB Connected: {row['version'][:50]}...")
            return True
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return False
