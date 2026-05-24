from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

from .jobs import claim_job, complete_job, fail_job, get_search_job, heartbeat_job
from .runner import run_search


HEARTBEAT_INTERVAL_S = 15.0


async def _run_with_heartbeat(job: dict[str, Any]) -> None:
    job_id = str(job["job_id"])
    started = time.perf_counter()
    timeout_s = int(job.get("timeout_s") or 900)
    deadline = started + timeout_s
    task = asyncio.create_task(
        run_search(
            str(job.get("query") or ""),
            str(job.get("platform") or ""),
            str(job.get("effective_model") or job.get("model") or ""),
            int(job.get("extra_sources") or 0),
            suppress_grok_errors=False,
        )
    )

    try:
        while not task.done():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                task.cancel()
                raise TimeoutError(f"search job exceeded timeout_s={timeout_s}")
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=min(HEARTBEAT_INTERVAL_S, remaining))
            except asyncio.TimeoutError:
                cancel_requested = heartbeat_job(job_id)
                if cancel_requested:
                    task.cancel()
                    raise asyncio.CancelledError("search job cancelled")

        result = await task
        elapsed = time.perf_counter() - started
        complete_job(
            job_id,
            content=result.content,
            sources=result.sources,
            timing={
                "total_s": round(elapsed, 2),
                "timeout_s": timeout_s,
                "model": result.model,
            },
        )
    except asyncio.CancelledError as exc:
        task.cancel()
        fail_job(
            job_id,
            status="cancelled",
            error=str(exc) or "search job cancelled",
            timing={"total_s": round(time.perf_counter() - started, 2), "timeout_s": timeout_s},
        )
        raise
    except TimeoutError as exc:
        task.cancel()
        fail_job(
            job_id,
            status="timed_out",
            error=str(exc),
            timing={"total_s": round(time.perf_counter() - started, 2), "timeout_s": timeout_s},
        )
    except Exception as exc:
        task.cancel()
        fail_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            timing={"total_s": round(time.perf_counter() - started, 2), "timeout_s": timeout_s},
        )


async def main_async(job_id: str) -> int:
    job = get_search_job(job_id)
    if not job:
        return 2
    if job.get("cancel_requested"):
        fail_job(job_id, status="cancelled", error="cancel requested before start")
        return 0
    if not claim_job(job_id, os.getpid()):
        return 1
    job = get_search_job(job_id)
    if not job:
        return 2
    try:
        await _run_with_heartbeat(job)
    except asyncio.CancelledError:
        return 130
    return 0


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m grok_search.job_worker <job_id>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main_async(sys.argv[1])))


if __name__ == "__main__":
    main()
