"""huey worker entry point (JOBS_MODE=huey).

Run with:  huey_consumer api.worker.huey --workers 1
The queue lives in SQLite on the shared data volume, so no Redis is needed.
"""

from __future__ import annotations

from huey import SqliteHuey

from api.config import get_settings
from api.observability import configure_logging

_settings = get_settings()
_settings.data_dir.mkdir(parents=True, exist_ok=True)
configure_logging(_settings.log_level, _settings.log_json)
huey = SqliteHuey(name="smart-book-reader", filename=str(_settings.data_dir / "queue.db"))


@huey.task(retries=1, retry_delay=10)  # type: ignore[untyped-decorator]
def process_job(job_id: str) -> None:
    from api.jobs import run_processing

    run_processing(job_id)
