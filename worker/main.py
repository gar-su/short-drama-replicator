"""Local worker: poll GitHub for pending tasks, execute replication, update status."""
from __future__ import annotations

import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from apscheduler.schedulers.background import BackgroundScheduler

_EXISTING_PROJECT = os.environ.get(
    "SHORT_DRAMA_PROJECT",
    str(Path.home() / "Downloads" / "short-drama-automation" / "short-drama-automation"),
)
if _EXISTING_PROJECT not in sys.path:
    sys.path.insert(0, _EXISTING_PROJECT)

from pipeline import get_logger, setup_logging

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CST = timezone(timedelta(hours=8))


def _load_dotenv() -> None:
    """Load .env file from project root into os.environ."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def _setup_file_logging(log_dir: Path) -> None:
    """Append a rotating file handler to the root logger for persistent logs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "worker.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)
    logger.info("File logging to %s", log_dir / "worker.log")


def _load_config() -> dict[str, str | float | Path]:
    required = ["GITHUB_TOKEN", "GITHUB_REPO", "NETSHORT_TOKEN"]
    config: dict[str, str | float | Path] = {}
    for key in required:
        value = os.environ.get(key, "")
        if not value:
            raise RuntimeError(f"Missing required env var: {key}")
        config[key.lower().replace("github_", "")] = value
    config["netshort_token"] = os.environ["NETSHORT_TOKEN"]
    config["author"] = os.environ.get("AUTHOR", "KAHUI")
    config["endings_dir"] = os.environ.get("ENDINGS_DIR", "")
    config["min_free_gb"] = float(os.environ.get("MIN_FREE_GB", "10"))
    # replicate() overall deadline
    config["max_run_seconds"] = float(os.environ.get("TASK_MAX_RUN_SECONDS", "21600"))
    # how long a running task may be silent before it is requeued
    config["requeue_after_seconds"] = float(os.environ.get("TASK_REQUEUE_AFTER_SECONDS", "28800"))
    config["notify_webhook_url"] = os.environ.get("NOTIFY_WEBHOOK_URL", "")
    config["log_dir"] = Path(os.environ.get("LOG_DIR", str(_PROJECT_ROOT / "logs")))
    # Iteration phase: keep output clips locally, never upload to the platform.
    # Flip SKIP_UPLOAD=0 in .env when the pipeline is stable enough to ship.
    config["skip_upload"] = os.environ.get("SKIP_UPLOAD", "1") not in ("0", "false", "False", "")
    config["worker_id"] = f"{socket.gethostname()}:{os.getpid()}"
    return config


def _disk_ok(min_free_gb: float) -> bool:
    """True if the storage volume has at least min_free_gb free bytes."""
    from worker.replicator import disk_free_bytes

    free = disk_free_bytes()
    if free < min_free_gb * 1e9:
        logger.warning("Disk low: %.1f GB free (min %.0f GB), refusing new tasks", free / 1e9, min_free_gb)
        return False
    return True


def _notify_failure(webhook_url: str, material_name: str, error: str) -> None:
    if not webhook_url:
        return
    try:
        httpx.post(
            webhook_url,
            json={"text": f"[short-drama-replicator] 任务失败: {material_name}\n{str(error)[:500]}"},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Notify failed: %s", e)


def poll_and_execute() -> None:
    """Check for pending tasks and execute one if found."""
    config = _load_config()
    client = httpx.Client(timeout=30)

    from worker.github_task import list_tasks, move_task, requeue_stale_running
    from worker.replicator import replicate

    repo = config["repo"]
    token = config["token"]

    # Requeue tasks whose worker died mid-run (stale running leases)
    try:
        requeued = requeue_stale_running(
            client, repo, token, int(config["requeue_after_seconds"])
        )
        if requeued:
            logger.info("Requeued %d stale running task(s): %s", len(requeued), requeued)
    except Exception as e:
        logger.error("Requeue check failed: %s", e)

    try:
        pending = list_tasks(client, repo, token, "pending")
    except Exception as e:
        logger.error("Failed to list pending tasks: %s", e)
        client.close()
        return

    if not pending:
        logger.debug("No pending tasks")
        client.close()
        return

    if not _disk_ok(config["min_free_gb"]):
        client.close()
        return

    task = pending[0]
    material_name = task["material_name"]
    logger.info("Processing task: %s", material_name)

    try:
        started_at = datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S+08:00")
        move_task(
            client, repo, token, material_name, "pending", "running",
            {"started_at": started_at, "worker_id": config["worker_id"]},
        )
    except Exception as e:
        logger.error("Failed to move task to running: %s", e)
        client.close()
        return

    finished_at = datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S+08:00")
    try:
        dub_filter = task.get("dub_filter", "ai")
        if dub_filter not in ("ai", "human", "both"):
            logger.warning("Unknown dub_filter=%r on task, falling back to 'ai'", dub_filter)
            dub_filter = "ai"
        target_langs = task.get("target_langs") or None
        banner_font = task.get("banner_font") or None
        lang_fonts = task.get("lang_fonts") or None
        summary_enabled = bool(task.get("summary_enabled", True))
        upload_folder_id = task.get("upload_folder_id") or None
        result = replicate(
            material_name=material_name,
            netshort_token=config["netshort_token"],
            author=config["author"],
            endings_dir=config["endings_dir"],
            max_run_seconds=config["max_run_seconds"],
            dub_filter=dub_filter,
            target_langs=target_langs,
            banner_font=banner_font,
            lang_fonts=lang_fonts,
            summary_enabled=summary_enabled,
            upload_folder_id=upload_folder_id,
            skip_upload=config["skip_upload"],
        )
        result["material_name"] = material_name
        result["finished_at"] = finished_at
        move_task(client, repo, token, material_name, "running", "done", result)
        logger.info("Task %s completed successfully", material_name)
    except Exception as e:
        logger.error("Task %s failed: %s", material_name, e)
        move_task(
            client, repo, token, material_name, "running", "failed",
            {"material_name": material_name, "finished_at": finished_at, "error": str(e)},
        )
        _notify_failure(config["notify_webhook_url"], material_name, str(e))

    client.close()


def main() -> None:
    _load_dotenv()
    setup_logging()
    config = _load_config()  # Validate config early
    _setup_file_logging(config["log_dir"])

    # Independent storage/cache cleanup: startup + daily (decoupled from task success)
    from worker.replicator import cleanup_storage

    try:
        summary = cleanup_storage()
        logger.info("Startup cleanup: %s", summary)
    except Exception as e:
        logger.error("Startup cleanup failed: %s", e)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        poll_and_execute,
        "interval",
        minutes=5,
        id="poll_tasks",
        replace_existing=True,
        next_run_time=datetime.now(),
    )
    scheduler.add_job(
        cleanup_storage,
        "cron",
        hour=4,
        minute=0,
        id="daily_cleanup",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Worker started, polling every 5 minutes")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
