"""GitHub task file read/write via REST API."""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx

_CST = timezone(timedelta(hours=8))


def _api_base(repo: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents"


def _api_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _get_file_content(
    client: httpx.Client, repo: str, token: str, path: str
) -> tuple[str | None, str | None]:
    """Get file content and SHA. Returns (content_str, sha) or (None, None) if not found."""
    url = f"{_api_base(repo)}/{path}"
    resp = client.get(url, headers=_api_headers(token))
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def _put_file(
    client: httpx.Client,
    repo: str,
    token: str,
    path: str,
    content: str,
    message: str,
    sha: str | None = None,
) -> dict[str, Any]:
    """Create or update a file. Returns API response data."""
    url = f"{_api_base(repo)}/{path}"
    body: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        body["sha"] = sha
    resp = client.put(url, headers=_api_headers(token), json=body)
    resp.raise_for_status()
    return resp.json()


def _delete_file(
    client: httpx.Client,
    repo: str,
    token: str,
    path: str,
    sha: str,
    message: str,
) -> None:
    url = f"{_api_base(repo)}/{path}"
    resp = client.request(
        "DELETE", url, headers=_api_headers(token),
        json={"message": message, "sha": sha},
    )
    resp.raise_for_status()


def list_tasks(client: httpx.Client, repo: str, token: str, status: str) -> list[dict[str, Any]]:
    """List all task files in tasks/{status}/ directory."""
    url = f"{_api_base(repo)}/tasks/{status}"
    resp = client.get(url, headers=_api_headers(token))
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    tasks = []
    for item in resp.json():
        if item["name"].endswith(".json"):
            content, _ = _get_file_content(client, repo, token, item["path"])
            if content:
                tasks.append(json.loads(content))
    return tasks


def create_task(client: httpx.Client, repo: str, token: str, material_name: str) -> dict[str, Any]:
    """Create a new pending task."""
    path = f"tasks/pending/{material_name}.json"
    task = {
        "material_name": material_name,
        "created_at": datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "status": "pending",
    }
    content = json.dumps(task, ensure_ascii=False, indent=2)
    _put_file(client, repo, token, path, content, f"new task: {material_name}")
    return task


def move_task(
    client: httpx.Client,
    repo: str,
    token: str,
    material_name: str,
    from_status: str,
    to_status: str,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Move a task file from one status directory to another, optionally updating content."""
    old_path = f"tasks/{from_status}/{material_name}.json"
    content, sha = _get_file_content(client, repo, token, old_path)
    if content is None:
        raise FileNotFoundError(f"Task not found: {old_path}")
    task = json.loads(content)
    task["status"] = to_status
    if extra_fields:
        task.update(extra_fields)
    new_content = json.dumps(task, ensure_ascii=False, indent=2)
    new_path = f"tasks/{to_status}/{material_name}.json"
    _put_file(client, repo, token, new_path, new_content, f"move to {to_status}: {material_name}")
    _delete_file(client, repo, token, old_path, sha, f"remove from {from_status}: {material_name}")


def get_task(
    client: httpx.Client, repo: str, token: str, status: str, material_name: str
) -> dict[str, Any] | None:
    path = f"tasks/{status}/{material_name}.json"
    content, _ = _get_file_content(client, repo, token, path)
    if content is None:
        return None
    return json.loads(content)
