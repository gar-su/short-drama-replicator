"""Local worker: poll GitHub for pending tasks, execute replication, update status."""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta
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

_CST = timezone(timedelta(hours=8))


def _load_config() -> dict[str, str]:
    required = ["GITHUB_TOKEN", "GITHUB_REPO", "NETSHORT_TOKEN"]
    config = {}
    for key in required:
        value = os.environ.get(key, "")
        if not value:
            raise RuntimeError(f"Missing required env var: {key}")
        config[key.lower().replace("github_", "")] = value
    config["netshort_token"] = os.environ["NETSHORT_TOKEN"]
    config["author"] = os.environ.get("AUTHOR", "KAHUI")
    config["endings_dir"] = os.environ.get("ENDINGS_DIR", "")
    return config


def poll_and_execute() -> None:
    """Check for pending tasks and execute one if found."""
    config = _load_config()
    client = httpx.Client(timeout=30)

    from worker.github_task import list_tasks, move_task

    repo = config["repo"]
    token = config["token"]

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

    task = pending[0]
    material_name = task["material_name"]
    logger.info("Processing task: %s", material_name)

    try:
        move_task(client, repo, token, material_name, "pending", "running")
    except Exception as e:
        logger.error("Failed to move task to running: %s", e)
        client.close()
        return

    from worker.replicator import replicate

    finished_at = datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S+08:00")
    try:
        result = replicate(
            material_name=material_name,
            netshort_token=config["netshort_token"],
            author=config["author"],
            endings_dir=config["endings_dir"],
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

    client.close()


def main() -> None:
    setup_logging()
    _load_config()  # Validate config early

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        poll_and_execute,
        "interval",
        minutes=5,
        id="poll_tasks",
        replace_existing=True,
        next_run_time=datetime.now(),
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
