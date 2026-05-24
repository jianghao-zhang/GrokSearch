from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "timed_out"}
DEFAULT_TIMEOUT_S = 900
MAX_TIMEOUT_S = 7200


def now_s() -> float:
    return time.time()


def job_db_path() -> Path:
    configured = os.getenv("GROK_SEARCH_JOB_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "grok-search" / "jobs.sqlite"


def job_log_dir() -> Path:
    configured = os.getenv("GROK_SEARCH_JOB_LOG_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "grok-search" / "jobs"


def normalize_timeout(timeout_s: int | None) -> int:
    if timeout_s is None or timeout_s <= 0:
        return DEFAULT_TIMEOUT_S
    return max(30, min(int(timeout_s), MAX_TIMEOUT_S))


def new_job_id() -> str:
    return f"gs_{uuid.uuid4().hex[:20]}"


def _connect() -> sqlite3.Connection:
    path = job_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            query TEXT NOT NULL,
            platform TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            effective_model TEXT NOT NULL DEFAULT '',
            extra_sources INTEGER NOT NULL DEFAULT 0,
            timeout_s INTEGER NOT NULL DEFAULT 900,
            created_at REAL NOT NULL,
            started_at REAL,
            updated_at REAL NOT NULL,
            completed_at REAL,
            pid INTEGER,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            content TEXT,
            sources_json TEXT,
            sources_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            timing_json TEXT,
            log_path TEXT
        )
        """
    )
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("sources_json", "timing_json"):
        raw = data.pop(key, None)
        out_key = "sources" if key == "sources_json" else "timing"
        if raw:
            try:
                data[out_key] = json.loads(raw)
            except json.JSONDecodeError:
                data[out_key] = [] if out_key == "sources" else {}
        else:
            data[out_key] = [] if out_key == "sources" else {}
    if data.get("started_at") and data.get("completed_at"):
        data["duration_s"] = round(float(data["completed_at"]) - float(data["started_at"]), 2)
    elif data.get("started_at"):
        data["running_for_s"] = round(now_s() - float(data["started_at"]), 2)
    data["cancel_requested"] = bool(data.get("cancel_requested"))
    return data


def create_search_job(
    *,
    query: str,
    platform: str = "",
    model: str = "",
    effective_model: str = "",
    extra_sources: int = 0,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    ts = now_s()
    job_id = new_job_id()
    timeout = normalize_timeout(timeout_s)
    log_path = job_log_dir() / f"{job_id}.log"
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO search_jobs (
                job_id, status, query, platform, model, effective_model,
                extra_sources, timeout_s, created_at, updated_at, sources_json,
                timing_json, log_path
            )
            VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, '[]', '{}', ?)
            """,
            (
                job_id,
                query,
                platform,
                model,
                effective_model,
                int(extra_sources),
                timeout,
                ts,
                ts,
                str(log_path),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM search_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) or {}


def get_search_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM search_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def list_search_jobs(limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM search_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(row) or {} for row in rows]


def claim_job(job_id: str, pid: int) -> bool:
    ts = now_s()
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE search_jobs
            SET status = 'running', pid = ?, started_at = COALESCE(started_at, ?),
                updated_at = ?
            WHERE job_id = ? AND status = 'queued' AND cancel_requested = 0
            """,
            (pid, ts, ts, job_id),
        )
        conn.commit()
        return cur.rowcount == 1


def heartbeat_job(job_id: str) -> bool:
    ts = now_s()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE search_jobs SET updated_at = ? WHERE job_id = ? AND status = 'running'",
            (ts, job_id),
        )
        conn.commit()
        row = conn.execute("SELECT cancel_requested FROM search_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def complete_job(
    job_id: str,
    *,
    content: str,
    sources: list[dict],
    timing: dict[str, Any],
) -> None:
    ts = now_s()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE search_jobs
            SET status = 'succeeded', updated_at = ?, completed_at = ?,
                content = ?, sources_json = ?, sources_count = ?,
                timing_json = ?, error = NULL
            WHERE job_id = ?
            """,
            (
                ts,
                ts,
                content,
                json.dumps(sources, ensure_ascii=False),
                len(sources),
                json.dumps(timing, ensure_ascii=False),
                job_id,
            ),
        )
        conn.commit()


def fail_job(job_id: str, *, status: str, error: str, timing: dict[str, Any] | None = None) -> None:
    if status not in {"failed", "cancelled", "timed_out"}:
        status = "failed"
    ts = now_s()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE search_jobs
            SET status = ?, updated_at = ?, completed_at = ?,
                error = ?, timing_json = ?
            WHERE job_id = ?
            """,
            (
                status,
                ts,
                ts,
                error[:4000],
                json.dumps(timing or {}, ensure_ascii=False),
                job_id,
            ),
        )
        conn.commit()


def spawn_worker(job_id: str) -> int:
    job_log_dir().mkdir(parents=True, exist_ok=True)
    log_path = job_log_dir() / f"{job_id}.log"
    env = os.environ.copy()
    src_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_root) if not existing_pythonpath else f"{src_root}{os.pathsep}{existing_pythonpath}"
    with open(log_path, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "grok_search.job_worker", job_id],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(Path.cwd()),
            close_fds=True,
            start_new_session=True,
        )
    return int(proc.pid)


def request_cancel(job_id: str) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM search_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return {"job_id": job_id, "status": "not_found", "cancelled": False}
        job = _row_to_dict(row) or {}
        if job.get("status") in TERMINAL_STATUSES:
            return {"job_id": job_id, "status": job.get("status"), "cancelled": False}
        conn.execute(
            "UPDATE search_jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
            (now_s(), job_id),
        )
        conn.commit()

    pid = job.get("pid")
    killed = False
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            killed = False
        except PermissionError:
            killed = False

    fail_job(job_id, status="cancelled", error="cancel requested")
    return {"job_id": job_id, "status": "cancelled", "cancelled": True, "terminated_process": killed}
