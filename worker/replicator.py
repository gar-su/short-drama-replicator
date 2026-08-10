"""Replication engine: clone a viral clip into other languages."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_EXISTING_PROJECT = os.environ.get(
    "SHORT_DRAMA_PROJECT",
    str(Path.home() / "Downloads" / "short-drama-automation" / "short-drama-automation"),
)
if _EXISTING_PROJECT not in sys.path:
    sys.path.insert(0, _EXISTING_PROJECT)

from fetcher.netshort_client import NetshortClient
from fetcher.netshort_uploader import (
    STSToken,
    bind_material,
    create_folder,
    get_sts_token,
    upload_to_oss,
)
from pipeline import get_logger
from video.editor import VideoEditor

logger = get_logger(__name__)

from PIL import Image, ImageFont, ImageStat
import imagehash

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from faster_whisper import WhisperModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COUNTER_FILE = Path.home() / ".config" / "short-drama-replicator" / "clip_counter.json"
_ASR_CACHE_DIR = Path.home() / ".config" / "short-drama-replicator" / "asr_cache"
_WHISPER_CACHE_DIR = Path.home() / ".cache" / "short-drama-replicator" / "whisper_asr"
_STORAGE_DIR = _PROJECT_ROOT / "storage"
_EPISODE_CACHE_DIR = _STORAGE_DIR / "episodes"
_ENDING_CACHE_DIR = _STORAGE_DIR / "ending_cache"
_RETENTION_DAYS = 7
_CACHE_TTL_SECONDS = _RETENTION_DAYS * 86400

# Work dir of the currently running replicate() (if any). Cleanup skips it so
# a scheduled cleanup can never delete files of an in-flight task.
_ACTIVE_WORK_DIR: str | None = None

# Serializes clip_counter.json read-modify-write across parallel target threads.
_SEQ_LOCK = threading.Lock()


def _retry(func, *args, max_attempts: int = 3, **kwargs) -> Any:
    """Call func with retry on transient errors (5xx, connection, timeout)."""
    import httpx as _httpx
    last_err = ""
    for attempt in range(max_attempts):
        if attempt > 0:
            wait = 2 ** attempt
            logger.warning("Retry %d/%d for %s (waiting %ds)",
                           attempt + 1, max_attempts,
                           getattr(func, "__name__", str(func)), wait)
            time.sleep(wait)
        try:
            return func(*args, **kwargs)
        except _httpx.HTTPStatusError as e:
            last_err = str(e)
            if e.response.status_code < 500:
                raise
        except (_httpx.ConnectError, _httpx.ReadTimeout,
                _httpx.RemoteProtocolError) as e:
            last_err = str(e)
        except Exception as e:
            last_err = str(e)
            logger.warning("Non-HTTP error in %s: %s",
                           getattr(func, "__name__", str(func)), e)
    raise RuntimeError(f"Failed after {max_attempts} attempts: {last_err}")


_whisper_model: WhisperModel | None = None
_whisper_model_name: str = ""


def _get_whisper_model(language: str = "") -> WhisperModel:
    global _whisper_model, _whisper_model_name
    model_name = "tiny.en" if language[:2].lower() == "en" else "tiny"
    if _whisper_model is None or _whisper_model_name != model_name:
        logger.info("Loading whisper model: %s", model_name)
        if model_name == "tiny":
            # hf-mirror doesn't cache the multilingual tiny model,
            # must download from huggingface.co directly
            os.environ.pop("HF_ENDPOINT", None)
        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _whisper_model_name = model_name
    return _whisper_model


def _whisper_cache_path(short_play_id: str, episode: int) -> str:
    return str(_WHISPER_CACHE_DIR / f"{short_play_id}_ep{episode}.json")


def _cache_is_fresh(path: str, ttl_seconds: float) -> bool:
    """True if cache file exists and is newer than ttl_seconds."""
    if not os.path.exists(path):
        return False
    return time.time() - os.path.getmtime(path) < ttl_seconds


def _invalidate_whisper_cache(short_play_id: str, episode: int) -> None:
    cache_path = _whisper_cache_path(short_play_id, episode)
    if os.path.exists(cache_path):
        os.remove(cache_path)


def _episode_video_path(short_play_id: str, episode: int) -> str:
    """Persistent cross-run episode cache path."""
    return str(_EPISODE_CACHE_DIR / short_play_id / f"ep_{episode}.mp4")


def _curl_resume_download(
    url: str, output_path: str, authorization: str, timeout: int = 300,
) -> None:
    """Download via curl with resume and built-in retries.

    Keeps partial files so a dropped connection resumes instead of restarting.
    curl's --retry-all-errors absorbs transient DNS/connection failures
    internally, avoiding the expensive URL re-fetch + restart path.
    """
    import shutil
    curl_bin = shutil.which("curl") or "curl"
    result = subprocess.run(  # noqa: S603
        [
            curl_bin, "-f", "-L", "-C", "-",
            "--connect-timeout", "30",
            "--max-time", str(timeout),
            "--retry", "5", "--retry-all-errors", "--retry-delay", "3",
            "--speed-time", "20", "--speed-limit", "2048",
            "-H", f"authorization: {authorization}",
            "-o", output_path, url,
        ],
        capture_output=True, text=True, timeout=timeout + 30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Download failed (exit {result.returncode}): "
            f"{result.stderr[:300] or result.stdout[:300]}"
        )


def _download_episode_with_retry(
    client: NetshortClient, short_play_id: str, episode: int, video_path: str,
    pick_voucher, max_attempts: int = 3, timeout: int = 300,
) -> None:
    """Download episode with fresh URL on each retry (auth_key may expire).

    Partial downloads are kept so a retry resumes instead of restarting.
    Raises FileNotFoundError if the episode doesn't exist (404).
    """
    import httpx as _httpx
    authorization = client._headers.get("authorization", "")
    last_err = ""
    for attempt in range(max_attempts):
        if attempt > 0:
            logger.warning("Re-fetching URL for %s ep %d (attempt %d/%d)",
                           short_play_id, episode, attempt + 1, max_attempts)
            time.sleep(3)
        try:
            ep_data = client.get_episode(short_play_id, episode)
            url = pick_voucher(ep_data["episodeVoucherVos"])
            _curl_resume_download(url, video_path, authorization, timeout=timeout)
            return
        except _httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(f"Episode {episode} not found (404)")
            last_err = str(e)
        except Exception as e:
            last_err = str(e)
            if "not found on page" in str(e).lower():
                raise FileNotFoundError(f"Episode {episode} not found") from e
        # keep partial file so the next attempt resumes
    raise RuntimeError(last_err or "Episode download failed")


def _ensure_episode_video(
    client: NetshortClient, short_play_id: str, episode: int,
) -> str:
    """Download (if not cached) and return the episode video path.

    Refreshing a video invalidates its whisper transcript, which is stale
    relative to the new file.
    """
    video_path = _episode_video_path(short_play_id, episode)
    if not _cache_is_fresh(video_path, _CACHE_TTL_SECONDS):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        if os.path.exists(video_path):
            os.remove(video_path)
        _download_episode_with_retry(
            client, short_play_id, episode, video_path, _pick_whisper_voucher,
        )
        _invalidate_whisper_cache(short_play_id, episode)
    return video_path


def _whisper_transcribe_episode(
    client: NetshortClient, short_play_id: str, episode: int, work_dir: str,
    language: str = "",
) -> list[dict[str, Any]]:
    cache_path = _whisper_cache_path(short_play_id, episode)
    video_path = _episode_video_path(short_play_id, episode)
    if _cache_is_fresh(cache_path, _CACHE_TTL_SECONDS) and _cache_is_fresh(video_path, _CACHE_TTL_SECONDS):
        return json.loads(Path(cache_path).read_text(encoding="utf-8"))

    video_path = _ensure_episode_video(client, short_play_id, episode)

    model = _get_whisper_model(language[:2].lower())
    segments, _ = _retry(model.transcribe, video_path, max_attempts=2)
    seg_list = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_path).write_text(json.dumps(seg_list, ensure_ascii=False), encoding="utf-8")
    return seg_list


def _lis_non_decreasing(vals: list[int]) -> list[int]:
    """Longest non-decreasing subsequence; returns indices into vals."""
    n = len(vals)
    if not n:
        return []
    dp = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if vals[j] <= vals[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    best = max(range(n), key=lambda i: dp[i])
    idxs = []
    i = best
    while i != -1:
        idxs.append(i)
        i = prev[i]
    return idxs[::-1]


def _line_coverage(
    source_lines: list[str], all_subs: list[dict[str, Any]], ratio_thr: float = 0.5
) -> tuple[float, int, int]:
    """Fraction of material lines matched, in order, to the source stream.

    For each material line, take its best-matching source segment (anchor), then
    the longest non-decreasing subsequence of anchors = how many material lines
    the drama contains as a contiguous clip. Robust to whisper segmentation
    drift between the material and episode transcripts, which breaks the old
    whole-text sliding-window match (see 2026-08-05 612/1837 failures).

    Returns (coverage, first_anchor_idx, last_anchor_idx).
    """
    n = len(source_lines)
    if not n or not all_subs:
        return 0.0, -1, -1
    anchors: list[int] = []
    for line in source_lines:
        lt = line.lower()
        best_j, best_r = -1, 0.0
        if lt:
            for j, s in enumerate(all_subs):
                r = SequenceMatcher(None, lt, s["text"].lower()).ratio()
                if r > best_r:
                    best_r, best_j = r, j
        anchors.append(best_j if best_r >= ratio_thr else -1)
    matched = [a for a in anchors if a >= 0]
    if not matched:
        return 0.0, -1, -1
    lis = _lis_non_decreasing(matched)
    first_idx = matched[lis[0]]
    last_idx = matched[lis[-1]]
    return len(lis) / n, first_idx, last_idx


def _episode_for_global_ms(
    all_subs: list[dict[str, Any]], global_ms: int
) -> int:
    """Episode containing a global time, from the ordered segment stream."""
    prev_ep = all_subs[0]["ep"]
    for s in all_subs:
        if s["global_ms"] > global_ms:
            return prev_ep
        prev_ep = s["ep"]
    return prev_ep


def _batch_search_clip_position(
    client: NetshortClient,
    drama_id: str,
    source_lines: list[str],
    source_first_seg_start_ms: int,
    source_duration_ms: int,
    work_dir: str,
    source_lang: str = "",
    deadline: float = 0,
) -> tuple[tuple[int, int, int, int] | None, float]:
    BATCH = 10
    COVERAGE_THR = 0.6
    MAX_EPISODES = 80
    n = len(source_lines)

    all_subs: list[dict[str, Any]] = []
    cumulative = 0
    parallel_downloads = max(1, int(os.environ.get("PARALLEL_DOWNLOADS", "4")))

    def _accept(cov: float, span: int) -> bool:
        return cov >= COVERAGE_THR and span <= 1.5 * n

    def _build_position(cov: float, first_idx: int, last_idx: int) -> tuple[tuple[int, int, int, int], float]:
        first = all_subs[first_idx]
        ep_start_global = _get_episode_start_global(all_subs, first["ep"])
        clip_start_global = first["global_ms"] - source_first_seg_start_ms
        # The END is NOT derived here. The material = drama content + an
        # appended 片尾 (not drama), and compressed materials drift from the
        # source timeline, so start + source_duration only centers the hook
        # search and picks the episode range; Stage 3 frame-matches the hook to
        # pin the real cut end. Do NOT anchor on the last LIS line: a 片尾 CTA
        # line can false-anchor to a later episode (2222: "Hurry up" -> E9),
        # which inflates the episode range and mis-centers the hook window.
        clip_end_global = clip_start_global + source_duration_ms
        # local_* are offsets into the concat stream (which starts at ep_start),
        # so both are relative to ep_start's global start; end_ep is derived from
        # the clip END time, not the last anchor, so a false trailing anchor
        # (612 last anchor lands in ep4 while the clip ends in ep3) can't
        # inflate the episode range.
        end_ep = _episode_for_global_ms(all_subs, clip_end_global)
        local_start_ms = clip_start_global - ep_start_global
        local_end_ms = clip_end_global - ep_start_global
        logger.info(
            "Found clip: E%d-E%d, local %dms-%dms (%.1fs, lead_in=%dms)",
            first["ep"], end_ep, local_start_ms, local_end_ms,
            (local_end_ms - local_start_ms) / 1000,
            source_first_seg_start_ms,
        )
        return (local_start_ms, local_end_ms, first["ep"], end_ep), cov

    for batch_start in range(1, MAX_EPISODES + 1, BATCH):
        if deadline and time.time() > deadline:
            raise RuntimeError("Task exceeded time limit during clip location")
        batch_end = min(batch_start + BATCH, MAX_EPISODES + 1)
        batch_added = 0

        # Parallel-download the batch's episodes: the CDN throttles per
        # connection (~2 Mbps), so concurrency multiplies aggregate bandwidth.
        # Transcription stays serial — whisper is CPU-bound and the shared
        # model singleton isn't safe for concurrent transcribe().
        eps = list(range(batch_start, batch_end))
        missing: set[int] = set()
        failed: set[int] = set()
        with ThreadPoolExecutor(max_workers=min(parallel_downloads, len(eps))) as pool:
            futures = {
                pool.submit(_ensure_episode_video, client, drama_id, ep): ep
                for ep in eps
            }
            for fut in as_completed(futures):
                ep = futures[fut]
                try:
                    fut.result()
                except FileNotFoundError:
                    missing.add(ep)  # episode doesn't exist → drama ends here
                except Exception:
                    logger.warning("Episode %d download failed, skipping", ep, exc_info=True)
                    failed.add(ep)

        first_missing = min(missing) if missing else batch_end
        for ep in eps:
            if ep >= first_missing or ep in failed:
                continue
            try:
                segs = _whisper_transcribe_episode(client, drama_id, ep, work_dir, source_lang)
            except Exception:
                logger.warning("Episode %d transcription failed, skipping", ep, exc_info=True)
                continue  # transient failure, try next episode
            for s in segs:
                all_subs.append({
                    "ep": ep,
                    "start_ms": int(s["start"] * 1000),
                    "end_ms": int(s["end"] * 1000),
                    "global_ms": cumulative + int(s["start"] * 1000),
                    "text": s["text"],
                })
            if segs:
                cumulative += int(segs[-1]["end"] * 1000) + 5000
            batch_added += 1

        # Don't evaluate until we have a meaningful amount of the source stream.
        # Drift lets several material lines collapse onto one source segment, so
        # half the material's line count is enough to judge.
        if len(all_subs) < max(5, n // 2):
            if batch_added == 0:
                break
            continue

        cov, first_idx, last_idx = _line_coverage(source_lines, all_subs)
        span = (last_idx - first_idx + 1) if first_idx >= 0 else 0
        logger.info(
            "Batch E%d-E%d: %d segs, coverage=%.3f span=%d",
            batch_start, batch_end - 1, len(all_subs), cov, span,
        )
        if _accept(cov, span):
            return _build_position(cov, first_idx, last_idx)

        if batch_added == 0:
            break

    # Not found with confidence
    cov, first_idx, last_idx = _line_coverage(source_lines, all_subs)
    span = (last_idx - first_idx + 1) if first_idx >= 0 else 0
    if _accept(cov, span):
        return _build_position(cov, first_idx, last_idx)
    logger.warning("Could not locate clip with confidence: coverage=%.3f", cov)
    return None, cov


def _get_episode_start_global(all_subs: list[dict[str, Any]], ep: int) -> int:
    for s in all_subs:
        if s["ep"] == ep:
            return s["global_ms"] - s["start_ms"]
    return 0


_CLARITY_RANK = {"1080p": 3, "720p": 2, "540p": 1}

_VC_VENV = Path.home() / ".local" / "share" / "short-drama-automation" / "vc-venv"
_VC_BIN = _VC_VENV / "bin" / "videocaptioner"
if not _VC_BIN.exists():
    _resolved = shutil.which("videocaptioner")
    if _resolved:
        _VC_BIN = Path(_resolved)


# ---------------------------------------------------------------------------
# Material search
# ---------------------------------------------------------------------------

def search_material(client: NetshortClient, material_name: str) -> dict[str, Any]:
    """Search for a material by name. Returns material info dict."""
    resp = client._client.post(
        "/batchput/material/search",
        json={"searchType": 1, "search": material_name, "tagIds": []},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 200 or not data.get("data", {}).get("materials"):
        raise RuntimeError(f"Material not found: {material_name}")
    mat = data["data"]["materials"][0]
    resource = mat["resource"]
    return {
        "name": resource["name"],
        "url": resource["url"],
        "language": resource.get("language", ""),
        "videoId": resource.get("videoId", mat.get("videoId", "")),
        "folderId": mat["folderId"],
        "shortPlayName": mat.get("shortPlayName", ""),
        "widthAndHigh": resource.get("widthAndHigh", ""),
    }


def search_dubbed_dramas(
    client: NetshortClient, short_play_name: str, source_language: str
) -> list[dict[str, Any]]:
    """Find all language versions of a drama, excluding the source language.

    Some dramas use a bare library name (all languages returned directly),
    others use "{name}(AI配音)" as the dubbed library name.

    All library-name variants are queried and merged (dedup by shortPlayId), so
    both the human-dubbed series ("（配音）"/"(Dubbed)", P-remark) and the
    AI-dubbed series ("（AI配音）"/"(AI Dubbed)") are returned together. Each
    target is tagged with dub_type = "ai"|"human"; the caller picks which to
    keep via dub_filter.
    """
    source_prefix = source_language[:2].lower()
    search_names = [
        short_play_name,
        f"{short_play_name}(AI配音)",
        f"{short_play_name}（AI配音）",
        f"{short_play_name}(AI Dubbed)",
    ]

    seen: dict[str, dict[str, Any]] = {}
    for search_name in search_names:
        resp = client._client.get(
            "/video/shortPlay/pageList",
            params={
                "pageNum": 1,
                "pageSize": 200,
                "isCheck": 0,
                "shortPlayLibraryName": search_name,
            },
        )
        resp.raise_for_status()
        for r in resp.json().get("rows", []):
            if r.get("language", "")[:2].lower() == source_prefix:
                continue
            if not r.get("remark", "").upper().startswith("P"):
                continue
            sid = str(r["shortPlayId"])
            if sid in seen:
                continue
            library_name = r.get("shortPlayLibraryName", "") or ""
            library_alias = r.get("shortPlayLibraryAlias", "") or ""
            seen[sid] = {
                "shortPlayId": sid,
                "language": r.get("language", ""),
                "shortPlayName": r.get("shortPlayName", ""),
                "remark": r.get("remark", ""),
                "library_name": library_name,
                "library_alias": library_alias,
                "dub_type": "ai" if "AI" in (library_name + " " + library_alias).upper() else "human",
            }

    result = list(seen.values())
    if result:
        ai_count = sum(1 for t in result if t["dub_type"] == "ai")
        logger.info(
            "Found %d target languages for '%s' (%d AI-dubbed, %d human-dubbed)",
            len(result), short_play_name, ai_count, len(result) - ai_count)
    else:
        logger.info("No dubbed target languages found for '%s'", short_play_name)
    return result


def _filter_dubbed_targets(
    targets: list[dict[str, Any]], dub_filter: str
) -> list[dict[str, Any]]:
    """Keep only the target series type chosen for this task: ai|human|both."""
    if dub_filter == "both":
        return targets
    wanted = "ai" if dub_filter == "ai" else "human"
    kept = [t for t in targets if t["dub_type"] == wanted]
    if kept:
        logger.info("dub_filter=%s: kept %d of %d target(s)",
                    dub_filter, len(kept), len(targets))
    else:
        logger.warning(
            "dub_filter=%s: no %s-dubbed target(s) for this drama (%d candidate(s) "
            "found); task will fail downstream",
            dub_filter, wanted, len(targets))
    return kept


def _find_source_drama(
    client: NetshortClient, library_name: str, source_lang: str, source_text: str,
    work_dir: str, preferred_remark: str = "",
) -> dict[str, Any] | None:
    """Find the original (non-dubbed) drama whose audio matches the material.

    Prefers candidates whose remark matches preferred_remark (same drama version).
    """
    source_prefix = source_lang[:2].lower()
    search_names = [
        library_name,
        f"{library_name}(AI配音)",
        f"{library_name}（AI配音）",
        f"{library_name}(AI Dubbed)",
    ]
    candidates: list[dict[str, Any]] = []
    for search_name in search_names:
        resp = client._client.get(
            "/video/shortPlay/pageList",
            params={
                "pageNum": 1, "pageSize": 200, "isCheck": 0,
                "shortPlayLibraryName": search_name,
            },
        )
        resp.raise_for_status()
        for r in resp.json().get("rows", []):
            lang = r.get("language", "")
            remark = r.get("remark", "")
            if not remark.upper().startswith("P"):
                candidates.append({
                    "shortPlayId": r["shortPlayId"],
                    "language": lang,
                    "shortPlayName": r.get("shortPlayName", ""),
                    "remark": remark,
                })

    if not candidates:
        return None

    # Priority: same drama version (exact remark) + language > same version + EN
    # > any language match > any EN > same version > first available
    same = [c for c in candidates if c["remark"] == preferred_remark] if preferred_remark else []
    same_en = [c for c in same if c["language"][:2].lower() == "en"]
    same_match = [c for c in same if c["language"][:2].lower() == source_prefix]
    en = [c for c in candidates if c["language"][:2].lower() == "en"]
    match = [c for c in candidates if c["language"][:2].lower() == source_prefix]
    for c in (same_match or same_en or match or en or same or candidates):
        logger.info("Found source drama: %s (%s, remark=%s)",
                     c["shortPlayId"], c["language"], c["remark"])
        return c
    return None


# ---------------------------------------------------------------------------
# ASR + VTT timestamp matching
# ---------------------------------------------------------------------------

def _asr_cache_path(short_play_id: str, episode: int) -> str:
    return str(_ASR_CACHE_DIR / short_play_id / f"ep_{episode}.srt")


def _asr_transcribe_episode(
    client: NetshortClient, short_play_id: str, episode: int, work_dir: str
) -> list[dict[str, Any]]:
    """Deprecated: use _whisper_transcribe_episode() instead. Kept for test compatibility."""
    cache_path = _asr_cache_path(short_play_id, episode)
    if _cache_is_fresh(cache_path, _CACHE_TTL_SECONDS):
        logger.info("Using cached ASR for %s ep %d", short_play_id, episode)
        with open(cache_path, encoding="utf-8") as f:
            srt_content = f.read()
    else:
        logger.info("ASR transcribing %s ep %d ...", short_play_id, episode)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        ep_data = client.get_episode(short_play_id, episode)
        voucher_url = _pick_best_voucher(ep_data["episodeVoucherVos"])
        video_path = os.path.join(work_dir, f"asr_ep_{episode}.mp4")
        client.download_file(voucher_url, video_path)
        srt_content = _asr_transcribe(video_path, cache_path)
        os.remove(video_path)

    import srt as srt_module
    subs = list(srt_module.parse(srt_content))
    return [
        {"index": i, "start_ms": int(sub.start.total_seconds() * 1000),
         "end_ms": int(sub.end.total_seconds() * 1000), "text": sub.content.replace("\n", " "),
         "episode": episode}
        for i, sub in enumerate(subs)
    ]


def _asr_transcribe(video_path: str, output_srt: str) -> str:
    """Deprecated: bijian ASR. Use faster-whisper via _get_whisper_model() instead. Kept for test compatibility."""
    result = subprocess.run(
        [str(_VC_BIN), "transcribe", video_path, "--asr", "bijian", "-o", output_srt],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ASR failed: {result.stderr[:200]}")
    with open(output_srt, encoding="utf-8") as f:
        return f.read()


def _srt_to_plain_text(srt_content: str) -> str:
    """Extract plain text from SRT content."""
    import srt
    subs = list(srt.parse(srt_content))
    return " ".join(sub.content.replace("\n", " ") for sub in subs)


def _vtt_to_subtitle_list(vtt_text: str) -> list[dict[str, Any]]:
    """Parse VTT text into list of {index, start_ms, end_ms, text}."""
    from fetcher.vtt_parser import parse_vtt
    lines = parse_vtt(vtt_text)
    return [
        {"index": i, "start_ms": line.start_ms, "end_ms": line.end_ms, "text": line.text}
        for i, line in enumerate(lines)
    ]


def _find_best_match_window(
    asr_text: str, vtt_subs: list[dict[str, Any]], is_start: bool
) -> int:
    """Find the VTT subtitle index that best matches the start or end of ASR text."""
    from difflib import SequenceMatcher

    snippet = asr_text[:200] if is_start else asr_text[-200:]

    best_idx = 0
    best_ratio = 0.0
    for sub in vtt_subs:
        ratio = SequenceMatcher(None, snippet.lower(), sub["text"].lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = sub["index"]

    logger.debug(
        "Best %s match at VTT[%d] (ratio=%.2f): '%.80s'",
        "start" if is_start else "end",
        best_idx,
        best_ratio,
        vtt_subs[best_idx]["text"],
    )
    return best_idx


def get_source_timestamps(
    client: NetshortClient, material_url: str, source_short_play_id: str, work_dir: str
) -> tuple[int, int]:
    """Deprecated: use _batch_search_clip_position() instead. Kept for test compatibility."""
    material_path = os.path.join(work_dir, "source_material.mp4")
    client.download_file(material_url, material_path)

    source_drama = client.get_short_play(source_short_play_id)
    pay_point = source_drama["payPoint"]

    # Try VTT subtitles first (fast text matching)
    all_subs: list[dict[str, Any]] = []
    for ep in range(1, pay_point):
        try:
            url = client.get_subtitle_url(source_short_play_id, ep)
            vtt_text = client.download_text(url)
            subs = _vtt_to_subtitle_list(vtt_text)
            for sub in subs:
                sub["episode"] = ep
            all_subs.extend(subs)
        except RuntimeError:
            continue

    if all_subs:
        logger.info("Using VTT subtitles: %d lines across %d episodes", len(all_subs), pay_point)
        srt_path = os.path.join(work_dir, "source_asr.srt")
        srt_content = _asr_transcribe(material_path, srt_path)
        asr_text = _srt_to_plain_text(srt_content)
        logger.info("ASR transcript length: %d chars", len(asr_text))
        start_idx = _find_best_match_window(asr_text, all_subs, is_start=True)
        end_idx = _find_best_match_window(asr_text, all_subs, is_start=False)
        start_ms = all_subs[start_idx]["start_ms"]
        end_ms = all_subs[min(end_idx + 1, len(all_subs) - 1)]["end_ms"]
        logger.info("Source timestamps (VTT): %dms - %dms (%.1fs)", start_ms, end_ms, (end_ms - start_ms) / 1000)
        return start_ms, end_ms

    # Fall back to frame matching on source drama episodes
    logger.info("No VTT available, using frame matching on %d episodes...", pay_point)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", material_path],
        capture_output=True, text=True,
    )
    source_duration_ms = int(float(probe.stdout.strip()) * 1000)

    # Use a frame from 5 seconds in (not frame 0, which may be a transition)
    ref_frame_ms = min(5000, source_duration_ms // 4)
    ref_first_hash = _compute_dhash(_extract_frame_ref(material_path, ref_frame_ms, work_dir, "ref"))

    # Also get hash from 10s before end (not the very last frame)
    ref_end_ms = max(0, source_duration_ms - 10000)
    ref_last_hash = _compute_dhash(_extract_frame_ref(material_path, ref_end_ms, work_dir, "ref_end"))

    if ref_first_hash is None or ref_last_hash is None:
        raise RuntimeError("Failed to compute reference frame hashes")

    # Phase 1: find start frame in any episode
    search_interval = 3000
    found_start_ms: int | None = None
    found_ep: int | None = None
    ep_paths: dict[int, str] = {}

    for ep in range(1, pay_point):
        logger.info("Searching episode %d/%d for start frame...", ep, pay_point)
        try:
            ep_data = client.get_episode(source_short_play_id, ep)
            voucher_url = _pick_best_voucher(ep_data["episodeVoucherVos"])
            ep_path = os.path.join(work_dir, f"src_ep_{ep}.mp4")
            client.download_file(voucher_url, ep_path)
            ep_paths[ep] = ep_path

            probe_ep = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ep_path],
                capture_output=True, text=True,
            )
            ep_duration_ms = int(float(probe_ep.stdout.strip()) * 1000)

            first_match = _find_best_frame_match(
                ep_path, [(ref_frame_ms, ref_first_hash)], 0, ep_duration_ms, search_interval, work_dir,
            )
            if first_match is not None and first_match[2] <= 9:
                found_start_ms = first_match[0] - first_match[1]
                found_ep = ep
                logger.info("Found start frame in episode %d at %dms (dist=%.1f)",
                          ep, found_start_ms, first_match[2])
                break
        except Exception as e:
            logger.error("Episode %d search failed: %s", ep, e)

    if found_start_ms is None:
        _cleanup_episodes(ep_paths)
        raise RuntimeError("Could not locate clip start in any source drama episode")

    # Phase 2: estimated end = start + clip_duration, search nearby for last frame
    estimated_end_ms = found_start_ms + source_duration_ms
    logger.info("Estimated end: %dms (start=%d + duration=%d)", estimated_end_ms, found_start_ms, source_duration_ms)

    # Build cumulative episode offsets to map global timestamps to episodes
    cumulative = 0
    ep_boundaries: list[tuple[int, int, int]] = []  # (ep, global_start, global_end)
    for ep in range(1, pay_point):
        if ep in ep_paths:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ep_paths[ep]],
                capture_output=True, text=True,
            )
            dur = int(float(probe.stdout.strip()) * 1000)
            ep_boundaries.append((ep, cumulative, cumulative + dur))
            cumulative += dur
        elif ep > found_ep:
            # Haven't downloaded future episodes yet
            ep_data = client.get_episode(source_short_play_id, ep)
            voucher_url = _pick_best_voucher(ep_data["episodeVoucherVos"])
            ep_path = os.path.join(work_dir, f"src_ep_{ep}.mp4")
            client.download_file(voucher_url, ep_path)
            ep_paths[ep] = ep_path
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ep_path],
                capture_output=True, text=True,
            )
            dur = int(float(probe.stdout.strip()) * 1000)
            ep_boundaries.append((ep, cumulative, cumulative + dur))
            cumulative += dur
        else:
            # Earlier episodes we didn't download — approximate with average
            dur = int(cumulative / max(found_ep - 1, 1))
            ep_boundaries.append((ep, cumulative, cumulative + dur))
            cumulative += dur

    # Find which episode contains estimated_end_ms
    end_ep = found_ep
    for ep, gs, ge in ep_boundaries:
        if gs <= estimated_end_ms < ge:
            end_ep = ep
            break

    # Search for end frame near estimated position
    search_window = 60000  # ±60 seconds
    for ep, gs, ge in ep_boundaries:
        if ep < end_ep:
            continue
        ep_path = ep_paths.get(ep)
        if ep_path is None:
            continue
        search_start = max(0, estimated_end_ms - gs - search_window)
        search_end = min(ge - gs, estimated_end_ms - gs + search_window)
        if search_end <= 0:
            continue
        logger.info("Searching end frame in episode %d [%d, %d]ms...", ep, search_start, search_end)
        last_match = _find_best_frame_match(
            ep_path, [(ref_end_ms, ref_last_hash)], search_start, search_end, 1000, work_dir,
        )
        if last_match is not None and last_match[2] <= 12:
            actual_end_ms = gs + last_match[0] + (source_duration_ms - ref_end_ms)
            _cleanup_episodes(ep_paths)
            logger.info("Source timestamps (frame match): %dms - %dms, dist=%.1f",
                      found_start_ms, actual_end_ms, last_match[2])
            return found_start_ms, actual_end_ms

    # Fallback: use estimated end
    logger.warning("Could not match end frame precisely, using estimated end")
    _cleanup_episodes(ep_paths)
    return found_start_ms, estimated_end_ms


def _cleanup_episodes(paths: dict[int, str]) -> None:
    for p in paths.values():
        try:
            os.remove(p)
        except OSError:
            pass


def _extract_frame_ref(video_path: str, timestamp_ms: int, work_dir: str, tag: str) -> str:
    """Extract a reference frame and return its path."""
    out = os.path.join(work_dir, f"{tag}.png")
    _extract_frame(video_path, timestamp_ms, out)
    return out


def _extract_ref_candidates(
    video_path: str,
    start_ms: int,
    end_ms: int,
    step_ms: int,
    work_dir: str,
    tag: str,
    max_candidates: int = 4,
    reverse: bool = False,
) -> list[tuple[int, imagehash.ImageHash]]:
    """Extract up to max_candidates distinct dHash refs from [start_ms, end_ms].

    Near-duplicate frames (static shots / a black tail) are skipped so the
    matcher tries a diverse set and keeps whichever candidate yields a confident
    match. A single fixed ref is fragile: it can be a black or cover frame that
    hashes near-identically across many moments of the dubbed drama (the 2222
    case: every sampled point 9-19 apart, no real valley).

    `reverse=True` walks the window backward from end_ms so the refs land on the
    tail of the range instead of its head — used for the hook, where the frames
    that matter sit right before the appended 片尾.

    Returns [(timestamp_ms, hash), ...]."""
    cands: list[tuple[int, imagehash.ImageHash]] = []
    if reverse:
        ts, end, step = end_ms, start_ms, -step_ms
    else:
        ts, end, step = start_ms, end_ms, step_ms
    while (ts <= end) if not reverse else (ts >= end):
        if len(cands) >= max_candidates:
            break
        frame_path = os.path.join(work_dir, f"{tag}_cand_{ts}.png")
        if _extract_frame(video_path, ts, frame_path):
            h = _compute_dhash(frame_path)
            if os.path.exists(frame_path):
                os.remove(frame_path)
            if h is not None and all((h - prev) >= 8 for _, prev in cands):
                cands.append((ts, h))
        ts += step
    return cands


# ---------------------------------------------------------------------------
# Frame extraction + dHash matching
# ---------------------------------------------------------------------------

def _extract_frame(video_path: str, timestamp_ms: int, output_path: str) -> bool:
    """Extract a single frame from video at timestamp_ms."""
    ts = timestamp_ms / 1000.0
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path, "-vframes", "1", "-q:v", "2", output_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and os.path.exists(output_path)


def _compute_dhash(image_path: str) -> imagehash.ImageHash | None:
    """Compute dHash for a frame image."""
    try:
        img = Image.open(image_path)
        return imagehash.dhash(img)
    except Exception:
        return None


def _find_best_frame_match(
    video_path: str,
    refs: list[tuple[int, imagehash.ImageHash]],
    search_start_ms: int,
    search_end_ms: int,
    interval_ms: int,
    work_dir: str,
    max_distance: int = 18,
    gap_ratio: float = 0.7,
) -> tuple[int, int, float] | None:
    """Search [search_start_ms, search_end_ms] for the best match against any
    ref in `refs` (a list of (src_timestamp_ms, hash)).

    Each concat frame is extracted once and compared to every ref, so trying
    several candidate refs costs no extra extractions. Coarse-scans every
    interval_ms, picks the (ref, ts) with the lowest distance, then refines
    that spot at 100ms. A match is accepted only when its dHash distance is <=
    max_distance and clearly below the runner-up (best <= gap_ratio *
    runner-up); otherwise None so the caller falls back to the ASR position.
    Short-drama frames repeat heavily, so a weak/borderline hit (2222 case:
    every sample ~9-19 apart, no real valley) must not be trusted as the clip
    boundary.

    Returns (timestamp_ms, ref_timestamp_ms, distance) or None.
    """
    def _scan(points: list[int], ref_hashes: list[imagehash.ImageHash]) -> list[list[int]]:
        res = [[-1, 999, 999] for _ in ref_hashes]
        for ts in points:
            frame_path = os.path.join(work_dir, f"frame_{ts}.png")
            if not _extract_frame(video_path, ts, frame_path):
                continue
            frame_hash = _compute_dhash(frame_path)
            if os.path.exists(frame_path):
                os.remove(frame_path)
            if frame_hash is None:
                continue
            for i, ref_hash in enumerate(ref_hashes):
                distance = ref_hash - frame_hash
                if distance < res[i][1]:
                    res[i][2] = res[i][1]
                    res[i][1], res[i][0] = distance, ts
                elif distance < res[i][2]:
                    res[i][2] = distance
        return res

    coarse_points = list(range(search_start_ms, search_end_ms, interval_ms))
    if coarse_points and coarse_points[-1] < search_end_ms:
        coarse_points.append(search_end_ms)
    coarse = _scan(coarse_points, [h for _, h in refs])
    cand = min(range(len(refs)), key=lambda i: coarse[i][1])
    best_ts, best_dist, second_dist = coarse[cand]
    if best_ts < 0:
        return None

    # Refine the winning candidate at 100ms resolution within +/- one interval.
    fine_lo = max(search_start_ms, best_ts - interval_ms)
    fine_hi = min(search_end_ms, best_ts + interval_ms)
    fine_points = list(range(fine_lo, fine_hi, 100))
    if fine_points and fine_points[-1] < fine_hi:
        fine_points.append(fine_hi)
    fine = _scan(fine_points, [refs[cand][1]])
    if fine[0][0] >= 0:
        best_ts, best_dist = fine[0][0], fine[0][1]

    if best_dist > max_distance:
        return None
    if second_dist < 999 and best_dist > gap_ratio * second_dist:
        return None
    return best_ts, refs[cand][0], float(best_dist)


def _find_hook_end(
    video_path: str,
    refs: list[tuple[int, imagehash.ImageHash]],
    search_start_ms: int,
    search_end_ms: int,
    interval_ms: int,
    work_dir: str,
    max_distance: int = 18,
    gap_ratio: float = 0.7,
) -> tuple[int, int, float] | None:
    """Locate the hook's tail inside [search_start_ms, search_end_ms].

    The material is drama content + an appended 片尾 (not drama). The hook is
    the silent visual beat after the material's last dialogue line; `refs` are
    frames of that section, each tagged with its source timestamp. Every concat
    frame is extracted once and compared to every ref; each ref keeps its best
    (dist, ts) and runner-up. A ref is confident when best_dist <= max_distance
    and best <= gap_ratio * runner-up (scene repeats must not masquerade as the
    hook). Among confident refs the one with the LARGEST match timestamp is the
    hook's last position in the concat; its match ts is the hook's tail.
    Returns (match_ts, ref_ts, dist) or None.
    """
    def _scan(points: list[int], ref_hashes: list[imagehash.ImageHash]) -> list[list[int]]:
        res = [[-1, 999, 999] for _ in ref_hashes]
        for ts in points:
            frame_path = os.path.join(work_dir, f"hook_{ts}.png")
            if not _extract_frame(video_path, ts, frame_path):
                continue
            frame_hash = _compute_dhash(frame_path)
            if os.path.exists(frame_path):
                os.remove(frame_path)
            if frame_hash is None:
                continue
            for i, ref_hash in enumerate(ref_hashes):
                distance = ref_hash - frame_hash
                if distance < res[i][1]:
                    res[i][2] = res[i][1]
                    res[i][1], res[i][0] = distance, ts
                elif distance < res[i][2]:
                    res[i][2] = distance
        return res

    coarse_points = list(range(search_start_ms, search_end_ms, interval_ms))
    if coarse_points and coarse_points[-1] < search_end_ms:
        coarse_points.append(search_end_ms)
    coarse = _scan(coarse_points, [h for _, h in refs])

    # Pick the confident ref whose MATCH is the latest in the concat: the hook
    # tail is the drama content's last position, so among the refs that pass
    # the confidence gates the one with the largest hit_ts is the hook's end.
    # (Refs from the 片尾 don't match the concat at all; if one does, e.g. a
    # highlight-reel frame, preferring the largest match position rather than
    # the largest ref timestamp keeps a repeat of an early scene from winning.)
    best_idx = -1
    best_hit_ts = -1
    for i, (ref_ts, _) in enumerate(refs):
        hit_ts, hit_dist, second_dist = coarse[i]
        if hit_ts < 0:
            continue
        if hit_dist > max_distance:
            continue
        if second_dist < 999 and hit_dist > gap_ratio * second_dist:
            continue
        if hit_ts > best_hit_ts:
            best_hit_ts, best_idx = hit_ts, i
    if best_idx < 0:
        return None

    best_hit_ts, best_dist = coarse[best_idx][0], coarse[best_idx][1]
    fine_lo = max(search_start_ms, best_hit_ts - interval_ms)
    fine_hi = min(search_end_ms, best_hit_ts + interval_ms)
    fine_points = list(range(fine_lo, fine_hi, 100))
    if fine_points and fine_points[-1] < fine_hi:
        fine_points.append(fine_hi)
    fine = _scan(fine_points, [refs[best_idx][1]])
    if fine[0][0] >= 0 and fine[0][1] <= max_distance:
        best_hit_ts, best_dist = fine[0][0], fine[0][1]
    return best_hit_ts, refs[best_idx][0], float(best_dist)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _pick_best_voucher(vouchers: list[dict[str, Any]]) -> str:
    ranked = sorted(
        vouchers,
        key=lambda v: (
            _CLARITY_RANK.get(v.get("playClarity", ""), 0),
            1 if v.get("codec") == "h264" else 0,
        ),
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("No video vouchers available")
    return str(ranked[0]["playVoucher"])


def _pick_whisper_voucher(vouchers: list[dict[str, Any]]) -> str:
    """Pick lowest clarity voucher for whisper ASR (audio only)."""
    valid = [v for v in vouchers if v.get("playClarity") and v.get("codec")]
    if not valid:
        raise RuntimeError("No video vouchers available")
    ranked = sorted(valid, key=lambda v: _CLARITY_RANK.get(v.get("playClarity", ""), 0))
    return str(ranked[0]["playVoucher"])


_ENDING_LANG_MAP = {
    "ja": "jp",
    "ko": "kr",
    "vi": "vn",
}


def _pick_ending(endings_dir: str, language: str, width: int = 1080) -> str | None:
    """Pick a random ending video for the given language and resolution."""
    if not endings_dir or not os.path.isdir(endings_dir):
        return None
    # Select subdirectory by resolution: endings/{width}p/
    res_dir = os.path.join(endings_dir, f"{width}p")
    if not os.path.isdir(res_dir):
        res_dir = endings_dir  # fallback to flat directory
    lang_prefix = language[:2].lower()
    lookup = lang_prefix
    if lookup in _ENDING_LANG_MAP:
        lookup = _ENDING_LANG_MAP[lookup]
    endings = [
        f for f in os.listdir(res_dir)
        if f.lower().startswith(lookup) and f.endswith(".mp4")
    ]
    if not endings:
        return None
    return os.path.join(res_dir, random.choice(endings))


def _get_video_props(path: str) -> dict:
    """Get video properties using ffprobe (JSON)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-of", "json", path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        streams = data.get("streams", [{}])
        if streams:
            s = streams[0]
            return {
                "codec": s.get("codec_name", ""),
                "width": s.get("width", 0),
                "height": s.get("height", 0),
                "fps": s.get("r_frame_rate", ""),
            }
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _prepare_ending_for_concat(ending_path: str, video_path: str, work_dir: str) -> str | None:
    """Re-encode ending to match clip codec/resolution/fps if needed.
    Returns path to compatible ending, or None if aspect ratios differ.
    """
    v1 = _get_video_props(video_path)
    v2 = _get_video_props(ending_path)
    if not v1 or not v2:
        logger.warning("Could not probe video properties, assuming incompatible")
        return None

    codec_match = v1.get("codec") == v2.get("codec")
    resolution_match = v1["width"] == v2["width"] and v1["height"] == v2["height"]
    fps_match = v1.get("fps") == v2.get("fps")

    if codec_match and resolution_match and fps_match:
        return ending_path

    # Check aspect ratio before re-encoding
    def _same_aspect(w1: int, h1: int, w2: int, h2: int) -> bool:
        from math import gcd
        g1 = gcd(w1, h1)
        g2 = gcd(w2, h2)
        return (w1 // g1, h1 // g1) == (w2 // g2, h2 // g2)

    if not _same_aspect(v1["width"], v1["height"], v2["width"], v2["height"]):
        logger.warning("Aspect ratio mismatch: clip=%dx%d ending=%dx%d, skipping ending",
                       v1["width"], v1["height"], v2["width"], v2["height"])
        return None

    # Check pre-computed ending cache first
    cache_key = f"{v1['width']}x{v1['height']}_{v1.get('fps', '0').replace('/', '_')}_{v1.get('codec', 'none')}"
    cache_dir = _ENDING_CACHE_DIR / cache_key
    ending_name = Path(ending_path).name
    cached_path = cache_dir / ending_name
    if cached_path.exists():
        logger.info("Using cached ending: %s", cached_path)
        return str(cached_path)

    # Re-encode ending to match clip, then save to cache atomically (parallel
    # targets may encode the same ending concurrently; temp+rename avoids a
    # torn cache entry being served by a later exists() check).
    cache_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg infers the output muxer from the file extension, so the temp name
    # must end in .mp4 (a bare .tmp suffix makes ffmpeg bail with "Unable to
    # choose an output format" -> ending silently dropped on every run).
    tmp_path = cache_dir / f".{ending_name}.{uuid.uuid4().hex[:8]}.tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", ending_path,
        "-vf", f"scale={v1['width']}:{v1['height']}:force_original_aspect_ratio=decrease,"
               f"pad={v1['width']}:{v1['height']}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
    ]
    if v1.get("fps"):
        cmd.extend(["-r", v1["fps"]])
    cmd.append(str(tmp_path))
    logger.info("Re-encoding ending to %dx%d @%s", v1["width"], v1["height"], v1.get("fps", "?"))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        logger.error("Failed to re-encode ending: %s", result.stderr[:200] if result.stderr else "unknown")
        return None
    os.replace(tmp_path, cached_path)
    return str(cached_path)


# ---------------------------------------------------------------------------
# Subtitle burn (target drama's own VTT) + title/summary banner (ASS)
# ---------------------------------------------------------------------------

_BANNER_FONT_CACHE: dict[str, str] = {}

# User-pickable banner fonts (frontend chips send the CSS family name).
# The MADE fonts live in the repo fonts/ dir; the rest are Windows system fonts.
_FONT_PICKER = {
    "Microsoft YaHei": "C:/Windows/Fonts/msyh.ttc",
    "SimHei": "C:/Windows/Fonts/simhei.ttf",
    "SimSun": "C:/Windows/Fonts/simsun.ttc",
    "Microsoft JhengHei": "C:/Windows/Fonts/msjh.ttc",
    "DengXian": "C:/Windows/Fonts/Deng.ttf",
    "Malgun Gothic": "C:/Windows/Fonts/malgun.ttf",
    "Yu Gothic": "C:/Windows/Fonts/YuGothM.ttc",
    "Arial": "C:/Windows/Fonts/arial.ttf",
    "Times New Roman": "C:/Windows/Fonts/times.ttf",
    "Verdana": "C:/Windows/Fonts/verdana.ttf",
    "Georgia": "C:/Windows/Fonts/georgia.ttf",
    "Oliver": str(_PROJECT_ROOT / "fonts" / "oliver" / "Oliver-Regular.ttf"),
    "Mellow": str(_PROJECT_ROOT / "fonts" / "mellow" / "MADEMellowPERSONALUSE-Regular.otf"),
    "Awelier": str(_PROJECT_ROOT / "fonts" / "awelier" / "MADEAwelierPERSONALUSE-Regular.otf"),
    "Blossom": str(_PROJECT_ROOT / "fonts" / "blossom" / "Blossom.ttf"),
    "Dokdo": str(_PROJECT_ROOT / "fonts" / "rixdokdo" / "Dokdo-Regular.ttf"),
    "YaoTi": str(_PROJECT_ROOT / "fonts" / "yaoti" / "方正姚体_GBK.ttf"),
}


# msyh.ttc (the fallback below) covers latin/CJK + their diacritics but has NO
# glyphs for Hangul/Thai/Arabic/Devanagari/Bengali/Burmese/Vietnamese
# precomposed chars (probed via fontTools cmap). Those scripts need their own
# font or the banner/summary renders as tofu boxes. lang2 -> default font file.
_LANG_DEFAULT_FONT = {
    "ko": "C:/Windows/Fonts/malgun.ttf",    # Hangul
    "th": "C:/Windows/Fonts/tahoma.ttf",    # Thai
    "vi": "C:/Windows/Fonts/tahoma.ttf",    # Vietnamese precomposed diacritics
    "ar": "C:/Windows/Fonts/arial.ttf",     # Arabic (RTL)
    "hi": "C:/Windows/Fonts/nirmala.ttf",   # Devanagari
    "bn": "C:/Windows/Fonts/nirmala.ttf",   # Bengali
    "my": "C:/Windows/Fonts/mmrtext.ttf",   # Burmese
}

# lang2 -> the script it needs, for scripts NOT every picker font can render.
# Latin (de/es/fr/it/id/ms/pt/ru/tr/tl) is intentionally absent: every picker
# font renders latin, so the user's pick always wins there.
_LANG_SCRIPT = {
    "ko": "hangul", "th": "thai", "vi": "vietnamese",
    "ar": "arabic", "hi": "devanagari", "bn": "devanagari", "my": "burmese",
    "zh": "cjk", "ja": "cjk",
}

# Scripts each user-pickable font can actually render (probed via fontTools
# cmap). A picked font lacking a target's script is ignored so we never burn
# tofu — e.g. Arial for a Chinese target, or SimHei for a Thai one.
_PICKER_SCRIPT_COVERAGE = {
    "Microsoft YaHei": {"cjk"},
    "SimHei": {"cjk"},
    "SimSun": {"cjk"},
    "Microsoft JhengHei": {"cjk"},
    "DengXian": {"cjk"},
    "Yu Gothic": {"cjk"},
    "Malgun Gothic": {"cjk", "hangul"},
    "Arial": {"arabic", "vietnamese"},
    "Times New Roman": {"arabic", "vietnamese"},
    "Verdana": {"vietnamese"},
    "Dokdo": {"hangul"},
    "YaoTi": {"cjk"},
}

# Per-language banner font lists (one language may use several fonts, from the
# NetShort font-picker screenshots). First family that exists AND covers the
# language's script wins as the default; "" is a sentinel meaning "use the
# legacy per-language default" (e.g. ja/ko get their native fonts first, hi/
# th/ar fall through to system fonts for Devanagari/Thai/Arabic). The MADE
# fonts (Oliver/Mellow/Awelier/Blossom) are latin-only, so they are listed
# only for latin languages.
_LANG_FONTS: dict[str, list[str]] = {
    "hi": [""],
    "id": ["Mellow", "Blossom", "Awelier", "Georgia", ""],
    "tr": ["Mellow", "Awelier", "Georgia", ""],
    "de": ["Oliver", "Awelier", "Blossom", "Georgia", ""],
    "it": ["Oliver", "Awelier", "Blossom", "Georgia", ""],
    "fr": ["Oliver", "Awelier", "Georgia", "Blossom", ""],
    "ja": [""],
    "th": [""],
    "zh_tw": ["YaoTi", ""],
    "en": ["Verdana", "Awelier", "Blossom", "Georgia", ""],
    "pt": ["Awelier", "Blossom", "Georgia", ""],
    "es": ["Awelier", "Blossom", "Georgia", ""],
    "vi": ["", "Awelier"],
    "ar": [""],
    "ko": ["Dokdo", ""],
    "ms": ["Mellow", "Blossom", "Awelier", "Georgia", ""],
}


def _resolve_banner_font(family: str | None, lang: str) -> str:
    """Map a user-picked CSS family to a font file for one language.

    '' means "use the per-language default". A picked font that lacks the
    target's script (e.g. SimHei for Thai) falls back so the burn never shows
    tofu boxes.
    """
    if not family:
        return ""
    path = _FONT_PICKER.get(family, "")
    if not (path and os.path.exists(path)):
        return ""
    lang2 = (lang or "").split("_")[0].lower()
    script = _LANG_SCRIPT.get(lang2)
    if script is not None and script not in _PICKER_SCRIPT_COVERAGE.get(family, ()):
        return ""
    return path


_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

_LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "id": "Indonesian", "ms": "Malay", "th": "Thai", "vi": "Vietnamese",
    "es": "Spanish", "pt": "Portuguese", "fr": "French", "it": "Italian",
    "de": "German", "ru": "Russian", "ar": "Arabic", "tr": "Turkish",
    "hi": "Hindi", "tl": "Tagalog", "my": "Burmese", "bn": "Bengali",
}

# Scripts whose words are space-separated; used to pick the summary max line
# width (Latin ~24 chars fit 1080px at fontsize 46, compact scripts ~12).
def _banner_font_for(lang: str) -> str:
    """Resolve the per-language default font.

    Priority: BANNER_FONT_DIR override -> first _LANG_FONTS entry that exists
    and covers the language's script -> _LANG_DEFAULT_FONT -> system CJK font.
    A "" entry in _LANG_FONTS means "no preferred font, use the system default".
    """
    lang_lower = (lang or "").lower()
    lang2 = lang_lower.split("_")[0]
    if lang_lower in _BANNER_FONT_CACHE:
        return _BANNER_FONT_CACHE[lang_lower]
    font = ""
    font_dir = os.environ.get("BANNER_FONT_DIR", "") or ""
    if font_dir:
        if not os.path.isabs(font_dir):
            font_dir = str(_PROJECT_ROOT / font_dir)
        for ext in (".ttf", ".ttc", ".otf"):
            cand = os.path.join(font_dir, f"{lang2}{ext}")
            if os.path.exists(cand):
                font = cand
                break
    if not font:
        for family in _LANG_FONTS.get(lang_lower, _LANG_FONTS.get(lang2, [])):
            if not family:
                break  # sentinel: fall through to the system default
            if _resolve_banner_font(family, lang):
                font = _FONT_PICKER[family]
                break
    if not font:
        font = _LANG_DEFAULT_FONT.get(lang2, "")
        if font and not os.path.exists(font):
            font = ""
    if not font:
        for cand in (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ):
            if os.path.exists(cand):
                font = cand
                break
    _BANNER_FONT_CACHE[lang_lower] = font
    return font


def _font_family(font_path: str) -> str:
    """Best-effort family name for a font file (used as ASS Fontname)."""
    if not font_path or not os.path.exists(font_path):
        return "Microsoft YaHei"
    try:
        return ImageFont.truetype(font_path, 20).getname()[0]
    except Exception:
        return "Microsoft YaHei"


def _clean_title(name: str) -> str:
    """Strip parenthetical content (（AI配音）/(AI Dubbed)…) and trim."""
    return re.sub(r"[（(][^（()）]*[）)]", "", name or "").strip()


def _doubao_api_key() -> str | None:
    """Resolve the doubao API key: env DOUBAO_API_KEY, else the shared
    short-drama-automation models.json registry (provider='doubao')."""
    key = (os.environ.get("DOUBAO_API_KEY", "") or "").strip()
    if key:
        return key
    reg = Path.home() / ".config" / "short-drama-automation" / "models.json"
    try:
        data = json.loads(reg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in data.values():
        if (entry or {}).get("provider") == "doubao":
            return (entry.get("api_key") or "").strip() or None
    return None


def _doubao_summarize(transcript: str, lang: str) -> str | None:
    """Summarize the source transcript in the target language via doubao.

    Returns the summary text (newline-separated), or None when no API key is
    configured or the call fails — the caller then burns only the title.
    """
    api_key = _doubao_api_key()
    if not api_key:
        logger.info("No doubao API key configured, skipping summary burn")
        return None
    import httpx as _httpx
    model = os.environ.get("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215").strip()
    lang2 = (lang or "").split("_")[0].lower()
    friendly = _LANG_NAMES.get(lang2, lang2.upper())
    system = (
        "You summarize a dialogue transcript into a concise plot summary "
        "burned into the bottom of a vertical short video. Rules: output only "
        "the summary text — no title, no bullets, no numbering, no quotes; "
        "write entirely in " + friendly + "; aim for about 100 characters. "
        "Line breaks are not important; write one short paragraph."
    )

    def _call() -> str:
        resp = _httpx.post(
            _DOUBAO_BASE_URL + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Dialogue transcript:\n{transcript[:8000]}"},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
                # seed-2-0-pro reasons by default (~100s on a summary call);
                # disabling thinking makes it answer in seconds.
                "thinking": {"type": "disabled"},
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return content.strip()

    try:
        text = _retry(_call, max_attempts=2)
    except Exception as e:
        logger.warning("doubao summary failed, skipping: %s", e)
        return None
    if not text:
        return None
    # Models sometimes emit a literal "\N" (or backslash-n) instead of a real
    # newline; normalize both to real paragraph breaks. Keep the paragraphs —
    # burn-time width wrapping breaks long CJK runs against the real font, so a
    # whole paragraph must not be collapsed into one space-joined blob that
    # reads as a single unbreakable "word".
    text = (text or "").replace("\\N", "\n").replace("\\n", "\n")
    paras = [" ".join(p.split()) for p in text.split("\n")]
    paras = [p for p in paras if p]
    flat = "\n".join(paras)
    budget = 110
    if len(flat) > budget:
        cut = flat.rfind(" ", 0, budget + 1)
        flat = flat[:cut if cut > 0 else budget].rstrip()
    logger.info("doubao summary for %s: %d chars", lang, len(flat))
    return flat


def _wrap_to_width(text: str, font_path: str, fontsize: int, max_width: int) -> str:
    """Re-wrap text so no rendered line exceeds max_width px (20px side margins).

    Measures real glyph widths via PIL TrueType metrics — accurate for
    proportional Latin scripts where char-count wrapping under/overflows.
    Space-separated scripts wrap at word boundaries, and a single token wider
    than the line (a long CJK run next to a space-separated word) is hard-cut
    per character so it can never overflow the frame. Preserves explicit
    \\n paragraph breaks. Falls back to an estimated width when the font cannot
    be loaded.
    """
    font = None
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, fontsize)
    except Exception:
        font = None
    max_w = max(40, int(max_width))

    def _px(s: str) -> float:
        if font is not None:
            return max(1.0, float(font.getlength(s)))
        return max(1.0, len(s) * fontsize * 0.55)

    out: list[str] = []
    for para in (text or "").split("\n"):
        line = para.strip()
        if not line:
            continue
        cur = ""
        for w in line.split():
            if cur and _px(f"{cur} {w}") > max_w:
                out.append(cur)
                cur = ""
            if _px(w) > max_w:
                # A single token is wider than the line (e.g. a long CJK run
                # that sits next to a space-separated word, which the old
                # word-branch treated as unbreakable and let overflow the
                # frame): hard-cut it char by char.
                run = ""
                for ch in w:
                    if run and _px(run + ch) > max_w:
                        out.append(run)
                        run = ch
                    else:
                        run += ch
                cur = run
            else:
                cur = w if not cur else f"{cur} {w}"
        if cur:
            out.append(cur)
    return "\n".join(out)


def _escape_ass(text: str) -> str:
    """Escape ASS dialogue text: backslashes first, then newlines as line breaks."""
    return text.replace("\\", "\\\\").replace("\n", "\\N")


def _banner_to_ass(
    title: str,
    summary: str,
    width: int,
    height: int,
    font_family: str,
    font_path: str,
    work_dir: str,
    dur_ms: int,
) -> str:
    """ASS overlaying the title (top) and summary (bottom) for the whole clip.

    PlayRes must equal the actual video size — libass scales text down when the
    script resolution exceeds the frame, making text near-invisible. Fontsizes
    are scaled from a 1080x1920 design so proportions hold across resolutions.
    """
    s = max(height, 1) / 1920.0
    title_fs = max(16, int(round(72 * s)))
    summary_fs = max(12, int(round(46 * s)))
    margin_top = max(16, int(round(80 * s)))
    margin_bottom = max(24, int(round(60 * s)))
    outline = max(1, int(round(2 * s)))
    shadow = 1
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: TitleTop,{font_family},{title_fs},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},8,0,0,{margin_top},1",
        f"Style: SummaryBottom,{font_family},{summary_fs},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,0,0,{margin_bottom},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    dur = _ms_to_ass(max(1, int(dur_ms)))
    max_w = max(40, int(width) - 2 * 20)
    title_wrapped = _wrap_to_width(title or "", font_path, title_fs, max_w)
    summary_wrapped = _wrap_to_width(summary or "", font_path, summary_fs, max_w)
    if title_wrapped.strip():
        lines.append(f"Dialogue: 0,0:00:00.00,{dur},TitleTop,,0,0,0,,{_escape_ass(title_wrapped.strip())}")
    if summary_wrapped.strip():
        lines.append(f"Dialogue: 0,0:00:00.00,{dur},SummaryBottom,,0,0,0,,{_escape_ass(summary_wrapped.strip())}")
    ass_path = os.path.join(work_dir, f"banner_{uuid.uuid4().hex[:8]}.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return ass_path


def _ms_to_ass(ms: int) -> str:
    """Milliseconds to ASS timestamp H:MM:SS.cc."""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, c = divmod(rem, 1000)
    return f"{h}:{m:02d}:{s:02d}.{c // 10:02d}"


def _ms_to_srt(ms: int) -> str:
    """Milliseconds to SRT timestamp HH:MM:SS,mmm."""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _srt_to_ass(srt_path: str, work_dir: str, font_family: str,
                width: int = 1080, height: int = 1920) -> str:
    """Convert an SRT to an ASS whose PlayRes matches the target frame.

    PlayRes must equal the actual video size: libass scales text down when the
    script resolution is larger than the video, making subtitles near-invisible
    on small portrait frames (e.g. 608x1080). Fontsize/margins are scaled from
    the 1080x1920 design so proportions stay constant across resolutions.
    """
    import srt as _srt
    with open(srt_path, encoding="utf-8") as f:
        subs = list(_srt.parse(f.read()))
    ass_path = os.path.join(work_dir, f"subs_{uuid.uuid4().hex[:8]}.ass")
    s = max(height, 1) / 1920.0
    fs = max(12, int(round(36 * s)))
    mv = max(40, int(round(110 * s)))
    mlr = max(30, int(round(80 * s)))
    outline = max(1, int(round(3 * s)))
    shadow = max(0, int(round(1 * s)))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_family},{fs},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{mlr},{mlr},{mv},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for sub in subs:
        text = sub.content.replace("\n", "\\N")
        t0 = int(round(sub.start.total_seconds() * 1000))
        t1 = int(round(sub.end.total_seconds() * 1000))
        lines.append(
            f"Dialogue: 0,{_ms_to_ass(t0)},{_ms_to_ass(t1)},Default,,0,0,0,,{text}"
        )
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return ass_path


def _probe_duration_ms(path: str) -> int:
    """ffprobe format=duration in milliseconds."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return int(float(probe.stdout.strip()) * 1000)
    except (ValueError, TypeError):
        return 0


def _fetch_clip_subtitles(
    client: NetshortClient,
    target_id: str,
    ep_start: int,
    ep_end: int,
    ep_paths: list[str],
    precise_start_ms: int,
    precise_end_ms: int,
    tgt_work: str,
) -> str | None:
    """Fetch the target drama's own subtitles and slice to the clip window.

    Returns a clip-relative SRT path, or None when the drama has no subtitle
    file (get_subtitle_url raises RuntimeError) or nothing falls in the window.
    VTT timestamps are ms since each episode video start; add cumulative concat
    offsets (ep_paths are already in concat order) to get global timestamps.
    """
    all_lines: list[dict[str, Any]] = []
    offset_ms = 0
    for ep, ep_path in zip(range(ep_start, ep_end + 1), ep_paths):
        try:
            url = client.get_subtitle_url(target_id, ep)
            subs = _vtt_to_subtitle_list(client.download_text(url))
        except RuntimeError:
            logger.info("No subtitles for %s episode %d, skipping burn", target_id, ep)
            return None
        for s in subs:
            all_lines.append({
                "start": offset_ms + s["start_ms"],
                "end": offset_ms + s["end_ms"],
                "text": s["text"],
            })
        offset_ms += _probe_duration_ms(ep_path)

    if not all_lines:
        return None
    clip_dur_ms = precise_end_ms - precise_start_ms
    if clip_dur_ms <= 0:
        return None
    kept: list[tuple[int, int, str]] = []
    for line in all_lines:
        if line["end"] <= precise_start_ms or line["start"] >= precise_end_ms:
            continue
        t0 = max(0, line["start"] - precise_start_ms)
        t1 = min(clip_dur_ms, line["end"] - precise_start_ms)
        if t1 - t0 >= 80:
            kept.append((t0, t1, line["text"]))
    if not kept:
        return None
    srt_path = os.path.join(tgt_work, f"clip_{target_id}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (t0, t1, text) in enumerate(kept, 1):
            f.write(f"{idx}\n{_ms_to_srt(t0)} --> {_ms_to_srt(t1)}\n{text}\n\n")
    logger.info("%s: %d subtitle lines within clip window -> %s",
                target_id, len(kept), srt_path)
    return srt_path


def _burn_overlays(
    clip_path: str,
    work_dir: str,
    subtitle_srt: str | None,
    banner_title: str | None,
    banner_summary: str | None,
    lang: str,
    banner_font_path: str | None = None,
) -> None:
    """Burn subtitle ASS + top-title/bottom-summary banner ASS in one pass.

    Re-encodes but keeps WxH / fps / h264 / yuv420p so the ending cache key
    (WxH_fps_codec) and -c copy concat in editor.concat still hold.
    """
    props = _get_video_props(clip_path)
    height = props.get("height", 0) if props else 0
    if not height:
        logger.warning("Cannot probe clip for overlay burn, skipping")
        return

    font_path = banner_font_path or _banner_font_for(lang)
    # Copy the chosen font into a small local dir so libass finds it quickly
    # instead of scanning all of C:/Windows/Fonts.
    font_dir = os.path.join(work_dir, "fonts")
    os.makedirs(font_dir, exist_ok=True)
    local_font = os.path.join(font_dir, os.path.basename(font_path))
    if font_path and not os.path.exists(local_font):
        shutil.copy2(font_path, local_font)
    family = _font_family(local_font)
    width, height = int(props["width"]), int(props["height"])

    vf: list[str] = []
    if subtitle_srt and os.path.exists(subtitle_srt):
        ass_path = _srt_to_ass(subtitle_srt, work_dir, family, width, height)
        # Drive-letter colons break the ass filter even when escaped into a
        # filtergraph (the graph parser consumes the backslash, then the option
        # parser re-splits on ':'), so run ffmpeg with cwd=work_dir and use
        # relative paths in the filter values instead.
        vf.append(f"ass={os.path.basename(ass_path)}:fontsdir=fonts")

    if (banner_title or "").strip() or (banner_summary or "").strip():
        dur_ms = _probe_duration_ms(clip_path)
        banner_ass = _banner_to_ass(
            banner_title or "", banner_summary or "",
            width, height, family, local_font, work_dir, dur_ms,
        )
        vf.append(f"ass={os.path.basename(banner_ass)}:fontsdir=fonts")

    if not vf:
        return
    fps = props.get("fps", "") if props else ""
    tmp_path = os.path.join(work_dir, f"burn_{uuid.uuid4().hex[:8]}.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", os.path.basename(clip_path),
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
    ]
    if fps:
        cmd.extend(["-r", fps])
    cmd.append(os.path.basename(tmp_path))
    logger.info("Burning overlays: %s", ", ".join(vf)[:200])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=work_dir)
    if result.returncode != 0:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        logger.error("Overlay burn failed: %s", (result.stderr or "")[-400:])
        return
    os.replace(tmp_path, clip_path)


def cut_and_assemble(
    editor: VideoEditor,
    target_video_path: str,
    start_ms: int,
    end_ms: int,
    endings_dir: str,
    target_language: str,
    output_path: str,
    work_dir: str,
    subtitle_srt: str | None = None,
    banner_title: str | None = None,
    banner_summary: str | None = None,
    banner_font_path: str | None = None,
) -> None:
    """Cut segment from target video, convert to portrait if needed, append ending."""
    temp_cut = os.path.join(work_dir, f"cut_{uuid.uuid4().hex[:8]}.mp4")
    duration_sec = (end_ms - start_ms) / 1000
    _retry(
        lambda: subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{start_ms / 1000:.3f}",
             "-i", target_video_path,
             "-t", f"{duration_sec:.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-c:a", "aac",
             temp_cut],
            capture_output=True, text=True, check=True,
        ),
        max_attempts=2,
    )

    # If landscape, convert to vertical 1080×1920
    _ensure_vertical(temp_cut, work_dir)

    # Burn the target drama's subtitles + top title / bottom summary banner.
    # Re-encode keeps 1080x1920/fps/h264/yuv420p so ending-cache + -c copy hold.
    if subtitle_srt or banner_title or banner_summary:
        _burn_overlays(temp_cut, work_dir, subtitle_srt, banner_title,
                       banner_summary, target_language,
                       banner_font_path=banner_font_path)

    # Probe cut video width to pick matching ending resolution (e.g. 1080p)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width", "-of", "csv=p=0", temp_cut],
        capture_output=True, text=True,
    )
    try:
        width = int(probe.stdout.strip())
    except (ValueError, TypeError):
        width = 1080

    ending = _pick_ending(endings_dir, target_language, width)
    if ending:
        compatible_ending = _prepare_ending_for_concat(ending, temp_cut, work_dir)
        if compatible_ending:
            editor.concat([temp_cut, compatible_ending], output_path)
            os.remove(temp_cut)
        else:
            # Ending skipped: temp_cut is renamed into the final file, so it no
            # longer exists to be cleaned up here (a stray os.remove below would
            # raise WinError 2 and mark a good output as failed).
            logger.warning("Ending incompatible with clip, skipping")
            os.rename(temp_cut, output_path)
    else:
        logger.warning("No ending found for %s (%dp)", target_language, width)
        os.rename(temp_cut, output_path)


def _ensure_vertical(video_path: str, work_dir: str) -> None:
    """If landscape (width > height), convert to vertical 1080x1920 in place."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    try:
        w_str, h_str = probe.stdout.strip().split(",")
        w, h = int(w_str), int(h_str)
    except (ValueError, TypeError):
        return

    if h >= w:
        return  # already portrait or square

    logger.info("Converting landscape %dx%d to vertical 1080x1920", w, h)
    tmp_path = os.path.join(work_dir, f"vertical_{uuid.uuid4().hex[:8]}.mp4")
    _retry(
        lambda: subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-c:a", "copy", tmp_path],
            capture_output=True, text=True, check=True,
        ),
        max_attempts=2,
    )
    os.remove(video_path)
    os.rename(tmp_path, video_path)
    logger.info("Converted to vertical")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _next_monthly_seq() -> int:
    with _SEQ_LOCK:
        month_key = date.today().strftime("%Y%m")
        _COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _COUNTER_FILE.exists():
            try:
                data = json.loads(_COUNTER_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}
        seq = int(data.get(month_key, 0)) + 1
        data[month_key] = seq
        _COUNTER_FILE.write_text(json.dumps(data))
        return seq


def _build_replicate_filename(
    seq: int,
    author: str,
    target_lang: str,
    drama_code: str,
    ep_start: int,
    ep_end: int,
    start_ms: int,
    end_ms: int,
) -> str:
    today = date.today().strftime("%y%m%d")
    lang = target_lang[:2].upper()
    return f"{seq}_{author}_YP_{lang}_{drama_code}_{ep_start}-{ep_end}_{start_ms}_{end_ms}_{today}_AI复刻.mp4"


def _upload_single_clip_replicate(
    clip_path: str,
    sts_token: STSToken,
    drama_language: str,
    drama_id: str,
) -> dict[str, Any]:
    """Upload one clip to OSS."""
    oss_key = f"material/{Path(clip_path).stem}.mp4"
    url = upload_to_oss(clip_path, sts_token, oss_key)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", clip_path],
        capture_output=True, text=True,
    )
    w, h = probe.stdout.strip().split(",") if probe.stdout.strip() else ("720", "1280")

    dur_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip_path],
        capture_output=True, text=True,
    )
    duration_ms = int(float(dur_probe.stdout.strip()) * 1000)
    file_size_mb = Path(clip_path).stat().st_size // (1024 * 1024)

    hh = duration_ms // 3_600_000
    mm = (duration_ms % 3_600_000) // 60_000
    ss = (duration_ms % 60_000) // 1000

    return {
        "name": Path(clip_path).stem,
        "widthAndHigh": f"{w} * {h}",
        "format": "mp4",
        "size": f"{file_size_mb}MB",
        "language": drama_language,
        "url": url,
        "videoId": drama_id,
        "videoTime": f"{hh:02d}:{mm:02d}:{ss:02d}",
    }


def upload_replicated_clips(
    client: NetshortClient,
    parent_folder_id: int,
    replicate_folder_name: str,
    clip_files: list[dict[str, Any]],
    target_languages: list[str],
    drama_language: str,
    drama_id: str,
) -> list[dict[str, Any]]:
    """Upload replicated clips to OSS and bind to replicate folder."""
    sts_token = _retry(get_sts_token, client)
    replicate_folder_id = _retry(create_folder, replicate_folder_name, parent_folder_id, client)

    resources: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_clip = {}
        for clip_info in clip_files:
            future_to_clip[
                pool.submit(_upload_single_clip_replicate, clip_info["path"], sts_token, drama_language, drama_id)
            ] = clip_info

        for future in as_completed(future_to_clip):
            clip_info = future_to_clip[future]
            try:
                resource = future.result()
                resources.append(resource)
                results.append({"lang": clip_info["lang"], "file": os.path.basename(clip_info["path"]), "status": "ok"})
            except Exception as e:
                logger.error("Upload failed for %s: %s", clip_info["path"], e)
                results.append({"lang": clip_info["lang"], "file": None, "status": "failed", "error": str(e)})

    if resources:
        for i in range(0, len(resources), 20):
            batch = resources[i : i + 20]
            _retry(bind_material, replicate_folder_id, batch, client)

    return results


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def disk_free_bytes() -> int:
    """Free bytes on the storage volume."""
    _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(_STORAGE_DIR).free


def cleanup_storage(
    max_age_days: int = _RETENTION_DAYS,
    exclude_dir: str | None = None,
) -> dict[str, int]:
    """Delete stale run dirs and expired caches.

    Independent of replicate() so cleanup still runs when tasks fail.
    - Per-run work dirs: removed wholesale once older than max_age_days.
    - Shared caches (episodes, ending_cache) and transcript caches: individual
      files expired by _CACHE_TTL_SECONDS.
    - The currently active work dir (if any) is never touched.
    Returns a summary dict of removed runs/files.
    """
    if exclude_dir is None:
        exclude_dir = _ACTIVE_WORK_DIR
    cutoff = time.time() - max_age_days * 86400
    removed_runs = 0
    removed_files = 0

    def _expire_dir(directory: Path) -> None:
        nonlocal removed_files
        for f in directory.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    removed_files += 1
                except OSError:
                    pass
        for d in sorted(
            (p for p in directory.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                d.rmdir()
            except OSError:
                pass

    if _STORAGE_DIR.exists():
        for entry in _STORAGE_DIR.iterdir():
            if not entry.is_dir():
                continue
            if exclude_dir and os.path.abspath(entry) == os.path.abspath(exclude_dir):
                continue
            if entry.name in ("episodes", "ending_cache"):
                _expire_dir(entry)
                continue
            if entry.stat().st_mtime < cutoff:
                logger.info("Cleaning up old storage: %s", entry.name)
                shutil.rmtree(entry, ignore_errors=True)
                removed_runs += 1

    for cache_dir in (_WHISPER_CACHE_DIR, _ASR_CACHE_DIR):
        _expire_dir(Path(cache_dir))

    if removed_runs or removed_files:
        logger.info("Cleanup done: removed %d run dir(s), %d cached file(s)", removed_runs, removed_files)
    return {"runs": removed_runs, "files": removed_files}


def _summarize_target_errors(results: list[dict[str, Any]], max_distinct: int = 3) -> str:
    """Condense per-target failures into the most common error strings.

    The task-level error message is the only thing surfaced on the board, so a
    bare count ("16 language(s) failed") hides why. This appends the dominant
    per-target error(s) with counts, e.g. errors: 'Episode 1 not found' x16.
    """
    if not results:
        return ""
    counts = Counter((r.get("error") or "unknown").strip() or "unknown" for r in results)
    parts = [
        f"'{err}' x{n}" if n > 1 else f"'{err}'"
        for err, n in counts.most_common(max_distinct)
    ]
    return "; errors: " + ", ".join(parts)


def replicate(
    material_name: str,
    netshort_token: str,
    author: str,
    endings_dir: str,
    skip_upload: bool = False,
    max_run_seconds: float = 0,
    resume_manifest: str | None = None,
    dub_filter: str = "ai",
    target_langs: list[str] | None = None,
    banner_font: str | None = None,
    lang_fonts: dict[str, str] | None = None,
    summary_enabled: bool = True,
) -> dict[str, Any]:
    """Replicate a viral clip into dubbed language versions.

    Uses faster-whisper ASR + text matching to locate the clip position
    in the source drama, then applies the same episode range and local
    timestamps to all target languages (identical scene structure).

    Set skip_upload=True to keep output clips locally without uploading.
    max_run_seconds > 0 enforces an overall deadline; exceeding it raises.
    resume_manifest: optional JSON file mapping target shortPlayId -> existing
    final clip path, so targets whose output already exists are reused instead
    of re-processed (e.g. resuming a run that was killed mid-Stage-3).
    dub_filter: which dubbed series type to produce — "ai" (default, AI-dubbed
    only), "human" (human-dubbed only), or "both" (all dubbed versions).
    target_langs: optional subset of language codes to produce (2-letter prefix
    or full "{lang}_{REGION}" form). Empty/None = all available targets.
    banner_font: CSS family name of a user-picked banner font; empty/unknown
    falls back to the per-language default.
    lang_fonts: optional {lang -> CSS family} overrides, e.g. {"de_DE": "Oliver"}.
    A matching override wins over banner_font for that language; keys may be a
    full "{lang}_{REGION}" code or a 2-letter prefix.
    summary_enabled: when False, skip the doubao plot summary (title still
    burns); requires BANNER_BURN_ENABLED=1 too.
    """
    client = NetshortClient(token=netshort_token)
    editor = VideoEditor()

    # Resolve endings_dir to absolute path (might be relative in .env)
    if endings_dir and not os.path.isabs(endings_dir):
        endings_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, endings_dir))

    material = _retry(search_material, client, material_name)
    source_lang = material["language"]
    folder_id = material["folderId"]
    source_url = material["url"]

    # Resolve library name via the drama this material belongs to
    source_drama_info = _retry(client.get_short_play, material["videoId"])
    library_name = source_drama_info.get("shortPlayLibraryName") or source_drama_info.get("shortPlayName", "")

    # Reject P-series materials at input: a P-remark means the material itself
    # was cut from an AI-dubbed version, which is not a valid replication input
    # (only original, non-dubbed materials are).
    remark = source_drama_info.get("remark", "") or ""
    if remark.upper().startswith("P"):
        raise RuntimeError(
            f"Material '{material_name}' comes from AI-dubbed version "
            f"(remark='{remark}'); P-series (AI-dubbed) materials are not "
            f"supported as replication input"
        )

    logger.info("Source: %s (lang=%s, library=%s)", material_name, source_lang, library_name)

    # Resume support: reuse final clips produced by a previous (e.g. killed) run
    # for targets whose output already exists, so the re-run only processes the
    # remaining targets. Map key = target shortPlayId (unique; language+remark
    # pairs are not, e.g. zh_CN vs zh_TW can share a remark code).
    resume_map: dict[str, str] = {}
    if resume_manifest:
        with open(resume_manifest, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                resume_map[str(k)] = v
        logger.info("Resume manifest: reusing %d already-produced target clip(s)",
                    len(resume_map))

    # Create persistent work directory (cleanup now runs independently, see cleanup_storage)
    run_id = f"{material_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir = str(_STORAGE_DIR / run_id)
    concat_dir = os.path.join(work_dir, "concat")
    output_dir = os.path.join(work_dir, "output")
    for d in (concat_dir, output_dir):
        os.makedirs(d, exist_ok=True)

    deadline = time.time() + max_run_seconds if max_run_seconds and max_run_seconds > 0 else 0

    def _check_deadline(where: str) -> None:
        if deadline and time.time() > deadline:
            raise RuntimeError(f"Task exceeded {max_run_seconds:.0f}s time limit at {where}")

    global _ACTIVE_WORK_DIR
    _ACTIVE_WORK_DIR = work_dir

    logger.info("Work dir: %s", work_dir)

    try:
        # 1. Download source material and transcribe with faster-whisper
        logger.info("[Stage 1/4] Downloading source material...")
        source_material_path = os.path.join(work_dir, "source_material.mp4")
        _retry(client.download_file, source_url, source_material_path, max_attempts=3)

        # Always use multilingual tiny for source — we don't trust the label
        model = _get_whisper_model("")
        src_segs, info = _retry(model.transcribe, source_material_path, max_attempts=2)
        detected_lang = info.language
        logger.info("Whisper detected language: %s (label=%s, prob=%.2f)",
                     detected_lang, source_lang, info.language_probability)
        src_list = list(src_segs)
        src_lines = [(s.start, s.text.strip()) for s in src_list]
        source_text = " ".join(t for _, t in src_lines)
        source_seg_count = len(src_lines)
        source_first_seg_start_ms = int(src_lines[0][0] * 1000) if src_lines else 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", source_material_path],
            capture_output=True, text=True,
        )
        source_duration_ms = int(float(probe.stdout.strip()) * 1000)
        logger.info("[Stage 1/4] Source ASR: %d segments, %d chars, lead_in=%dms dur=%dms",
                     source_seg_count, len(source_text),
                     source_first_seg_start_ms, source_duration_ms)

        # Target languages: exclude the whisper-detected source language, not
        # the material label — the label is often wrong (e.g. a pt_PT label on
        # English audio), which would otherwise produce a target in the real
        # source language. Falls back to the label if detection gave nothing.
        effective_source_lang = detected_lang or source_lang
        targets = _retry(search_dubbed_dramas, client, library_name, effective_source_lang)
        if dub_filter not in ("ai", "human", "both"):
            logger.warning("Unknown dub_filter=%r, falling back to 'ai'", dub_filter)
            dub_filter = "ai"
        targets = _filter_dubbed_targets(targets, dub_filter)
        if target_langs:
            wanted = {x.strip().lower() for x in target_langs if x}
            targets = [
                t for t in targets
                if t["language"].split("_")[0].lower() in wanted
                or t["language"].lower() in wanted
            ]
            if not targets:
                logger.warning("target_langs=%r matched no dubbed targets", target_langs)
        max_targets = int(os.environ.get("MAX_TARGETS", "0"))
        if max_targets > 0 and len(targets) > max_targets:
            logger.info("MAX_TARGETS=%d, processing first %d of %d target languages",
                        max_targets, max_targets, len(targets))
            targets = targets[:max_targets]

        # 2. Find clip position in source drama via batch download + text matching
        logger.info("[Stage 2/4] Locating clip in source drama...")
        if not targets:
            raise RuntimeError(f"No dubbed versions found for '{library_name}'")
        for t in targets[:3]:
            logger.info("Target: %s %s (PA545=%s)", t["language"], t["shortPlayId"], t["remark"])
        if len(targets) > 3:
            logger.info("  ... and %d more target languages", len(targets) - 3)
        target_languages = [t["language"] for t in targets]
        logger.info("Target languages: %s", target_languages)

        # Use a single source drama matching the whisper-detected language.
        # The material's language label and videoId can both point at the wrong
        # drama (e.g. a th_TH label on English audio), so trust the detected
        # language instead of scanning multiple candidates.
        own_remark = source_drama_info.get("remark", "")
        source_drama = _retry(_find_source_drama, client, library_name, detected_lang,
                              source_text, work_dir, own_remark)
        if source_drama is None:
            raise RuntimeError(
                f"No source drama found for detected language '{detected_lang}' "
                f"in library '{library_name}'"
            )
        logger.info("Source drama: %s (%s, remark=%s)",
                     source_drama["shortPlayId"], source_drama["language"],
                     source_drama.get("remark", ""))

        source_lines = [t for _, t in src_lines]
        best_position, coverage = _batch_search_clip_position(
            client, source_drama["shortPlayId"], source_lines,
            source_first_seg_start_ms, source_duration_ms, work_dir, detected_lang,
            deadline=deadline,
        )
        if best_position is None:
            raise RuntimeError(
                f"Could not locate clip in source drama {source_drama['shortPlayId']} "
                f"(coverage={coverage:.2f})"
            )
        logger.info("Selected source drama: %s (%s), coverage=%.3f",
                     source_drama["shortPlayId"], source_drama["language"], coverage)

        local_start_ms, local_end_ms, ep_start, ep_end = best_position
        logger.info(
            "Clip position: E%d-E%d, local %dms-%dms (%.1fs)",
            ep_start, ep_end, local_start_ms, local_end_ms,
            (local_end_ms - local_start_ms) / 1000,
        )

        # Extract reference frames from source material for frame matching.
        # Use several distinct head/tail frames so the matcher picks whichever
        # yields a confident match (a single fixed ref is fragile: it may be a
        # black/cover frame that hashes near-identically across the dubbed
        # drama). Each candidate keeps its source timestamp so a match hit can
        # be converted back to a clip offset (a ref at src_ts sits at
        # local_start + src_ts in the concat, not at local_start itself).
        head_refs = _extract_ref_candidates(
            source_material_path,
            min(500, source_duration_ms // 4),
            min(4000, source_duration_ms // 2),
            500, work_dir, "ref_first",
        )
        if not head_refs:
            raise RuntimeError("Failed to extract reference frame hashes from source material")
        logger.info("Material head ref frames: %s", [t for t, _ in head_refs])

        SEARCH_WINDOW_MS = 10_000
        FRAME_INTERVAL_MS = 1000
        # The material = drama content + an appended 片尾 (not part of the
        # drama). Its tail, the "hook", is the drama's final visual beat; the
        # 片尾 is NOT in the drama, so its frames never match the concat and
        # only the hook anchors the end. The last whisper line includes the 片尾
        # CTA (2222: "click and watch" lines run to the very end), so instead of
        # trusting dialogue timing we sample refs from a fixed tail window
        # [dur-20000, dur-2000]: that reaches the drama content end for 片尾s up
        # to ~18s, whatever the hook's length. Frames past the content end (片尾)
        # simply don't match, and _find_hook_end takes the latest confident hit.
        HOOK_REF_TAIL_MS = 20000   # how far back from material end to sample
        HOOK_REF_MARGIN_MS = 2000  # skip the very last frames (deep in 片尾)
        HOOK_WINDOW_MS = 25000     # ± window around start+source_duration to scan
        HOOK_END_TAIL_MS = 1000    # silent beat after the last matched hook ref
        hook_refs = []
        if source_duration_ms > 30000:
            # Walk BACKWARD from the material end: the hook's decisive frames
            # are the last distinct drama-content frames right before the 片尾
            # (the tail can be a long slow/still sequence whose head frames
            # would otherwise crowd out the tail under the 4-candidate cap).
            hook_refs = _extract_ref_candidates(
                source_material_path,
                max(0, source_duration_ms - HOOK_REF_TAIL_MS),
                source_duration_ms - HOOK_REF_MARGIN_MS,
                500, work_dir, "ref_hook",
                max_candidates=12, reverse=True,
            )
        logger.info("Material hook ref frames: %s", [t for t, _ in hook_refs])

        # 3. Process each target — frame match within ±10s of expected position
        logger.info("[Stage 3/4] Processing %d target languages...", len(targets))
        results: list[dict[str, Any]] = []
        clip_files: list[dict[str, Any]] = []

        def process_target(target: dict[str, Any]) -> dict[str, Any]:
            """Cut and assemble one target language. Returns clip info."""
            target_lang = target["language"]
            target_id = target["shortPlayId"]
            target_remark = target["remark"] or "XX"

            resume_path = resume_map.get(str(target_id))
            if resume_path and os.path.exists(resume_path):
                logger.info("%s: resume — reusing existing clip %s",
                            target_lang, os.path.basename(resume_path))
                return {"path": resume_path, "lang": target_lang}

            logger.info("Processing %s (%s)...", target_lang, target_id)

            # Per-target scratch dir: frame-match temp files use fixed names
            # (frame_{ts}.png), so concurrent targets must not share a dir.
            # Key by target_id, not language: dub_filter="both" produces an AI
            # and a human target of the same language (different shortPlayId).
            tgt_work = os.path.join(work_dir, "scratch", target_id)
            os.makedirs(tgt_work, exist_ok=True)

            ep_paths = _download_target_episodes(client, target_id, ep_start, ep_end)
            concat_path = os.path.join(concat_dir, f"target_{target_id}.mp4")
            _concat_videos(ep_paths, concat_path)

            # Frame match: find precise start within ±10s of expected position
            # Probe video duration to cap search range
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", concat_path],
                capture_output=True, text=True,
            )
            try:
                video_dur_ms = int(float(probe.stdout.strip()) * 1000)
            except (ValueError, TypeError):
                video_dur_ms = local_end_ms + 30000

            search_start = max(0, local_start_ms - SEARCH_WINDOW_MS)
            search_end = min(video_dur_ms - 500, local_start_ms + SEARCH_WINDOW_MS)
            start_match = _find_best_frame_match(
                concat_path, head_refs, search_start, search_end,
                FRAME_INTERVAL_MS, tgt_work,
            )
            if start_match is None:
                logger.warning("%s: start frame not matched, using ASR position", target_lang)
                precise_start_ms = local_start_ms
            else:
                match_start_ms, ref_start_ms, start_dist = start_match
                # The matched frame sits at clip_start + ref_start_ms in the
                # concat, so subtract the ref's own source timestamp to get the
                # clip boundary itself.
                precise_start_ms = match_start_ms - ref_start_ms
                logger.info("%s: start ref@%dms (dist=%.1f) -> clip start %dms",
                            target_lang, match_start_ms, start_dist, precise_start_ms)

            # Frame match the end: the material's drama content ends with a
            # silent "hook" beat; the appended 片尾 is not drama content and
            # never anchors. The hook window is centered on start + source_dur
            # (the material-length position), NOT on a text anchor — the last
            # LIS line can false-anchor to a later episode (2222: 片尾 "Hurry
            # up" -> E9), which would push the window past the real hook. The
            # hook refs are the material's last ~18s, so their concat matches
            # always fall within ±25s of start + source_duration (compression
            # offsets run a few to ~20s); the concat-end cap bounds the window.
            # The 612-era blanket skip of end matching was because it matched
            # the 片尾 itself, which repeats wrongly; matching the hook (drama
            # content) avoids that.
            coarse_end_ms = precise_start_ms + (local_end_ms - local_start_ms)
            hook_match = None
            if hook_refs:
                hook_match = _find_hook_end(
                    concat_path, hook_refs,
                    max(0, min(video_dur_ms - 500, coarse_end_ms - HOOK_WINDOW_MS)),
                    min(video_dur_ms - 100, coarse_end_ms + HOOK_WINDOW_MS),
                    FRAME_INTERVAL_MS, tgt_work,
                )
            if hook_match is not None:
                hook_match_ts, hook_ref_ts, hook_dist = hook_match
                precise_end_ms = hook_match_ts + HOOK_END_TAIL_MS
                logger.info("%s: hook ref@%dms (dist=%.1f) -> clip end %dms",
                            target_lang, hook_match_ts, hook_dist, precise_end_ms)
            else:
                precise_end_ms = coarse_end_ms + HOOK_END_TAIL_MS
                logger.warning(
                    "%s: no confident hook match, using material-length end %dms "
                    "(coarse %dms + tail %dms)", target_lang, precise_end_ms,
                    coarse_end_ms, HOOK_END_TAIL_MS,
                )
            if precise_end_ms >= video_dur_ms - 100:
                logger.warning("%s: derived end %dms past concat (%dms), capping",
                               target_lang, precise_end_ms, video_dur_ms)
                precise_end_ms = video_dur_ms - 100
            logger.info("%s: clip end %dms (drama span %dms, start %dms)",
                        target_lang, precise_end_ms, precise_end_ms - precise_start_ms,
                        precise_start_ms)

            seq = _next_monthly_seq()
            filename = _build_replicate_filename(
                seq, author, target_lang, target_remark,
                ep_start, ep_end, precise_start_ms, precise_end_ms,
            )
            output_path = os.path.join(output_dir, filename)
            # Burn the drama's own subtitles only if it has a subtitle file
            # (get_subtitle_url raises RuntimeError otherwise -> no burn).
            subtitle_srt = None
            if os.environ.get("SUBTITLE_BURN_ENABLED", "1") == "1":
                try:
                    subtitle_srt = _fetch_clip_subtitles(
                        client, target_id, ep_start, ep_end, ep_paths,
                        precise_start_ms, precise_end_ms, tgt_work,
                    )
                except Exception as e:
                    logger.warning("%s: subtitle fetch failed, burning without subtitles: %s",
                                   target_lang, e)
            # Top banner: the target drama's own name, parentheticals stripped.
            # Bottom banner: doubao plot summary of the whole source clip, in the
            # target language. Summary is optional (no key / failure -> title only).
            banner_title = _clean_title(target.get("shortPlayName", ""))
            banner_summary = None
            if summary_enabled and os.environ.get("BANNER_BURN_ENABLED", "1") == "1":
                banner_summary = _doubao_summarize(source_text, target_lang)
            # Per-language: a lang_fonts override wins, then the global pick;
            # '' -> per-language default; a picked font lacking the target's
            # script (e.g. SimHei for Thai) is ignored to avoid tofu.
            family = ((lang_fonts or {}).get(target_lang)
                      or (lang_fonts or {}).get(target_lang.split("_")[0])
                      or banner_font)
            banner_font_path = _resolve_banner_font(family, target_lang)
            cut_and_assemble(
                editor, concat_path,
                precise_start_ms, precise_end_ms,
                endings_dir, target_lang,
                output_path, tgt_work,
                subtitle_srt, banner_title or None, banner_summary,
                banner_font_path=banner_font_path,
            )
            logger.info("%s done: %s", target_lang, filename)
            return {"path": output_path, "lang": target_lang}

        # Process all targets in parallel. Each target downloads its 2-3
        # episodes at ~2 Mbps/connection (the CDN throttles per connection, not
        # per pipe), so running targets concurrently multiplies aggregate
        # bandwidth. Per-target scratch dirs keep frame-match temps isolated.
        parallel = max(1, min(int(os.environ.get("PARALLEL_TARGETS", "6")), len(targets)))
        logger.info("Processing %d targets with %d parallel workers", len(targets), parallel)
        _check_deadline("stage3")

        def _process_with_retry(target: dict[str, Any]) -> dict[str, Any]:
            last_err = ""
            for attempt in range(2):
                try:
                    return process_target(target)
                except Exception as e:
                    last_err = str(e)
                    logger.warning("Failed for %s (attempt %d/2): %s",
                                   target["language"], attempt + 1, last_err)
                    if attempt == 0:
                        time.sleep(5)
            raise RuntimeError(last_err or "unknown error")

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            fut_map = {pool.submit(_process_with_retry, t): t for t in targets}
            for fut in as_completed(fut_map):
                target = fut_map[fut]
                _check_deadline("stage3")
                try:
                    clip_files.append(fut.result())
                except Exception as e:
                    results.append({
                        "lang": target["language"], "file": None,
                        "status": "failed", "error": str(e),
                    })

        if not clip_files:
            raise RuntimeError(
                "No clip replicated successfully"
                + (f"; {len(results)} language(s) failed" if results else "")
                + _summarize_target_errors(results)
            )

        # 4. Upload all clips (or skip for development)
        logger.info("[Stage 4/4] %d clips ready%s",
                     len(clip_files),
                     " (upload skipped)" if skip_upload else "")
        if clip_files and not skip_upload:
            replicate_folder_name = f"{material_name}_复刻"
            upload_results = upload_replicated_clips(
                client, folder_id, replicate_folder_name,
                clip_files, target_languages,
                targets[0]["language"] if targets else source_lang,
                targets[0]["shortPlayId"] if targets else material["videoId"],
            )
            results.extend(upload_results)
        elif clip_files:
            logger.info("Upload skipped (skip_upload=True), files kept in %s", output_dir)
            for cf in clip_files:
                results.append({"lang": cf["lang"], "file": os.path.basename(cf["path"]), "status": "local_only"})

        return {
            "source_lang": source_lang,
            "detected_lang": detected_lang,
            "target_langs": target_languages,
            "results": results,
        }

    finally:
        _ACTIVE_WORK_DIR = None
        # work files retained for _RETENTION_DAYS, cleaned by cleanup_storage


def _download_target_episodes(
    client: NetshortClient, target_id: str, ep_start: int, ep_end: int
) -> list[str]:
    """Download target episodes ep_start..ep_end in parallel, return paths.

    Uses persistent cross-run cache. Stale (>_CACHE_TTL_SECONDS) cached episodes
    are re-downloaded and their whisper transcripts invalidated so a refreshed
    video never pairs with a stale transcript.
    """
    def _download(ep: int) -> str:
        ep_path = _episode_video_path(target_id, ep)
        if not _cache_is_fresh(ep_path, _CACHE_TTL_SECONDS):
            os.makedirs(os.path.dirname(ep_path), exist_ok=True)
            if os.path.exists(ep_path):
                os.remove(ep_path)
            _download_episode_with_retry(
                client, target_id, ep, ep_path, _pick_best_voucher,
            )
            _invalidate_whisper_cache(target_id, ep)
        return ep_path

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(_download, range(ep_start, ep_end + 1)))


def _concat_videos(paths: list[str], output: str) -> None:
    """Concatenate video files with re-encoding to ensure A/V sync."""
    import shutil as _shutil
    if len(paths) == 1:
        _shutil.copy2(paths[0], output)
        return
    concat_list = output + ".txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{p.replace(chr(92), chr(47))}'\n")
    _retry(
        lambda: subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-c:a", "aac", output],
            capture_output=True, text=True, check=True,
        ),
        max_attempts=2,
    )
    os.remove(concat_list)
