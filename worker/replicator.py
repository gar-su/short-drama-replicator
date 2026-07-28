"""Replication engine: clone a viral clip into other languages."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
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

from PIL import Image
import imagehash

_COUNTER_FILE = Path.home() / ".config" / "short-drama-replicator" / "clip_counter.json"
_ASR_CACHE_DIR = Path.home() / ".config" / "short-drama-replicator" / "asr_cache"

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
    """
    candidates: list[dict[str, Any]] = []
    search_names = [
        short_play_name,
        f"{short_play_name}(AI配音)",
        f"{short_play_name}（AI配音）",
        f"{short_play_name}(AI Dubbed)",
    ]

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
        rows = resp.json().get("rows", [])
        if rows:
            for r in rows:
                candidates.append({
                    "shortPlayId": r["shortPlayId"],
                    "language": r.get("language", ""),
                    "shortPlayName": r.get("shortPlayName", ""),
                    "remark": r.get("remark", ""),
                })
            break

    source_prefix = source_language[:2].lower()
    result = [c for c in candidates if c["language"][:2].lower() != source_prefix]
    logger.info("Found %d target languages (filtered from %d total)", len(result), len(candidates))
    return result


# ---------------------------------------------------------------------------
# ASR + VTT timestamp matching
# ---------------------------------------------------------------------------

def _asr_cache_path(short_play_id: str, episode: int) -> str:
    return str(_ASR_CACHE_DIR / short_play_id / f"ep_{episode}.srt")


def _asr_transcribe_episode(
    client: NetshortClient, short_play_id: str, episode: int, work_dir: str
) -> list[dict[str, Any]]:
    """ASR transcribe a drama episode, caching the result. Returns subtitle list."""
    cache_path = _asr_cache_path(short_play_id, episode)
    if os.path.exists(cache_path):
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
    """Transcribe video to SRT using videocaptioner. Returns SRT text."""
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
    """ASR the source material, match against source drama subtitles (VTT or ASR cached)."""
    material_path = os.path.join(work_dir, "source_material.mp4")
    client.download_file(material_url, material_path)

    srt_path = os.path.join(work_dir, "source_asr.srt")
    srt_content = _asr_transcribe(material_path, srt_path)
    asr_text = _srt_to_plain_text(srt_content)
    logger.info("ASR transcript length: %d chars", len(asr_text))

    source_drama = client.get_short_play(source_short_play_id)
    pay_point = source_drama["payPoint"]

    # Try VTT subtitles first
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
    else:
        logger.info("No VTT available, building ASR cache for %d episodes...", pay_point)
        for ep in range(1, pay_point):
            try:
                ep_subs = _asr_transcribe_episode(client, source_short_play_id, ep, work_dir)
                all_subs.extend(ep_subs)
            except Exception as e:
                logger.error("ASR failed for ep %d: %s", ep, e)

    if not all_subs:
        raise RuntimeError("No subtitles (VTT or ASR) available for source drama")

    start_idx = _find_best_match_window(asr_text, all_subs, is_start=True)
    end_idx = _find_best_match_window(asr_text, all_subs, is_start=False)

    start_ms = all_subs[start_idx]["start_ms"]
    end_ms = all_subs[min(end_idx + 1, len(all_subs) - 1)]["end_ms"]

    logger.info("Source timestamps: %dms - %dms (%.1fs)", start_ms, end_ms, (end_ms - start_ms) / 1000)
    return start_ms, end_ms


# ---------------------------------------------------------------------------
# Frame extraction + dHash matching
# ---------------------------------------------------------------------------

def _extract_frame(video_path: str, timestamp_ms: int, output_path: str) -> bool:
    """Extract a single frame from video at given timestamp."""
    ts = timestamp_ms / 1000.0
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path, "-vframes", "1", "-q:v", "2", output_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def _compute_dhash(image_path: str) -> imagehash.ImageHash | None:
    """Compute dHash of an image file. Returns None on failure."""
    try:
        img = Image.open(image_path)
        return imagehash.dhash(img)
    except Exception as e:
        logger.warning("Failed to compute dHash for %s: %s", image_path, e)
        return None


def _find_best_frame_match(
    video_path: str,
    ref_hash: imagehash.ImageHash,
    search_start_ms: int,
    search_end_ms: int,
    interval_ms: int,
    work_dir: str,
) -> tuple[int, float] | None:
    """Search for the frame that best matches ref_hash in the given range."""
    best_ts = -1
    best_distance = 999
    max_distance = 256

    for ts in range(search_start_ms, search_end_ms + interval_ms, interval_ms):
        frame_path = os.path.join(work_dir, f"frame_{ts}.png")
        if not _extract_frame(video_path, ts, frame_path):
            continue
        frame_hash = _compute_dhash(frame_path)
        if os.path.exists(frame_path):
            os.remove(frame_path)
        if frame_hash is None:
            continue
        distance = ref_hash - frame_hash
        if distance < best_distance:
            best_distance = distance
            best_ts = ts

    if best_ts < 0:
        return None
    similarity = 1.0 - (best_distance / max_distance)
    return best_ts, similarity


def match_frame_positions(
    source_material_path: str,
    target_video_path: str,
    source_start_ms: int,
    source_end_ms: int,
    search_window_ms: int = 30_000,
    sample_interval_ms: int = 1000,
    similarity_threshold: float = 0.85,
    work_dir: str = "",
) -> tuple[int | None, int | None]:
    """Match first and last frames of source material against target video."""
    wd = work_dir or tempfile.mkdtemp(prefix="frame_match_")

    first_frame_path = os.path.join(wd, "ref_first.png")
    last_frame_path = os.path.join(wd, "ref_last.png")

    if not _extract_frame(source_material_path, 0, first_frame_path):
        raise RuntimeError("Failed to extract first frame from source material")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", source_material_path],
        capture_output=True, text=True,
    )
    source_duration_ms = int(float(probe.stdout.strip()) * 1000)
    last_frame_ts = max(0, source_duration_ms - 500)
    if not _extract_frame(source_material_path, last_frame_ts, last_frame_path):
        raise RuntimeError("Failed to extract last frame from source material")

    ref_first_hash = _compute_dhash(first_frame_path)
    ref_last_hash = _compute_dhash(last_frame_path)
    if ref_first_hash is None or ref_last_hash is None:
        raise RuntimeError("Failed to compute reference frame hashes")

    first_search_start = max(0, source_start_ms - search_window_ms)
    first_search_end = source_start_ms + search_window_ms
    logger.info("Searching first frame in [%d, %d]ms", first_search_start, first_search_end)
    first_match = _find_best_frame_match(
        target_video_path, ref_first_hash,
        first_search_start, first_search_end, sample_interval_ms, wd,
    )

    last_search_start = max(0, source_end_ms - search_window_ms)
    last_search_end = source_end_ms + search_window_ms
    logger.info("Searching last frame in [%d, %d]ms", last_search_start, last_search_end)
    last_match = _find_best_frame_match(
        target_video_path, ref_last_hash,
        last_search_start, last_search_end, sample_interval_ms, wd,
    )

    if first_match is None or last_match is None:
        return None, None

    first_ts, first_sim = first_match
    last_ts, last_sim = last_match
    logger.info("Frame match: first@%dms (sim=%.3f), last@%dms (sim=%.3f)", first_ts, first_sim, last_ts, last_sim)

    if first_sim < similarity_threshold or last_sim < similarity_threshold:
        logger.warning("Frame similarity below threshold (%.3f)", similarity_threshold)
        return None, None

    return first_ts, last_ts


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


_ENDING_LANG_MAP = {
    "ja": "jp",
    "ko": "kr",
    "vi": "vn",
}


def _pick_ending(endings_dir: str, language: str) -> str | None:
    """Pick a random ending video for the given language."""
    if not endings_dir or not os.path.isdir(endings_dir):
        return None
    lang_prefix = language[:2].lower()
    lookup = lang_prefix
    if lookup in _ENDING_LANG_MAP:
        lookup = _ENDING_LANG_MAP[lookup]
    endings = [
        f for f in os.listdir(endings_dir)
        if f.lower().startswith(lookup) and f.endswith(".mp4")
    ]
    if not endings:
        return None
    return os.path.join(endings_dir, random.choice(endings))


def cut_and_assemble(
    editor: VideoEditor,
    target_video_path: str,
    start_ms: int,
    end_ms: int,
    endings_dir: str,
    target_language: str,
    output_path: str,
    work_dir: str,
) -> None:
    """Cut segment from target video and append ending. Subtitles are already burned in."""
    temp_cut = os.path.join(work_dir, f"cut_{uuid.uuid4().hex[:8]}.mp4")
    editor.cut(target_video_path, temp_cut, start_ms, end_ms)

    ending = _pick_ending(endings_dir, target_language)
    if ending:
        editor.concat([temp_cut, ending], output_path)
        os.remove(temp_cut)
    else:
        os.rename(temp_cut, output_path)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _next_monthly_seq() -> int:
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
    sts_token = get_sts_token(client)
    replicate_folder_id = create_folder(replicate_folder_name, parent_folder_id, client)

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
            bind_material(replicate_folder_id, batch, client)

    return results


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def replicate(
    material_name: str,
    netshort_token: str,
    author: str,
    endings_dir: str,
) -> dict[str, Any]:
    """Main entry: replicate a clip into all available languages."""
    client = NetshortClient(token=netshort_token)
    editor = VideoEditor()

    material = search_material(client, material_name)
    source_lang = material["language"]
    source_short_play_id = material["videoId"]
    folder_id = material["folderId"]
    source_url = material["url"]
    short_play_name = material["shortPlayName"]

    logger.info("Source: %s (lang=%s, drama=%s)", material_name, source_lang, short_play_name)

    targets = search_dubbed_dramas(client, short_play_name, source_lang)
    if not targets:
        raise RuntimeError(f"No dubbed versions found for '{short_play_name}'")

    target_languages = [t["language"] for t in targets]
    logger.info("Target languages: %s", target_languages)

    work_dir = tempfile.mkdtemp(prefix="replicate_")
    try:
        source_start_ms, source_end_ms = get_source_timestamps(
            client, source_url, source_short_play_id, work_dir
        )

        source_material_path = os.path.join(work_dir, "source_material.mp4")
        if not os.path.exists(source_material_path):
            client.download_file(source_url, source_material_path)

        results: list[dict[str, Any]] = []
        clip_files: list[dict[str, Any]] = []
        episodes_dir = os.path.join(work_dir, "episodes")
        os.makedirs(episodes_dir, exist_ok=True)

        for target in targets:
            target_lang = target["language"]
            target_id = target["shortPlayId"]
            target_remark = target["remark"] or "XX"

            logger.info("Processing %s (%s)...", target_lang, target_id)

            try:
                episode = client.get_episode(target_id, 1)
                voucher_url = _pick_best_voucher(episode["episodeVoucherVos"])
                target_video_path = os.path.join(episodes_dir, f"{target_lang}_{target_id}_ep1.mp4")
                client.download_file(voucher_url, target_video_path)

                fine_start_ms, fine_end_ms = match_frame_positions(
                    source_material_path, target_video_path,
                    source_start_ms, source_end_ms, work_dir=work_dir,
                )

                if fine_start_ms is None:
                    logger.warning("%s: frame match failed, using source timestamps", target_lang)
                    fine_start_ms = source_start_ms
                    fine_end_ms = source_end_ms

                seq = _next_monthly_seq()
                filename = _build_replicate_filename(
                    seq, author, target_lang, target_remark,
                    1, 1, fine_start_ms, fine_end_ms,
                )
                output_path = os.path.join(work_dir, filename)
                cut_and_assemble(
                    editor, target_video_path,
                    fine_start_ms, fine_end_ms,
                    endings_dir, target_lang,
                    output_path, work_dir,
                )

                clip_files.append({"path": output_path, "lang": target_lang})

            except Exception as e:
                logger.error("Failed for %s: %s", target_lang, e)
                results.append({"lang": target_lang, "file": None, "status": "failed", "error": str(e)})

        if clip_files:
            replicate_folder_name = f"{material_name}_复刻"
            upload_results = upload_replicated_clips(
                client, folder_id, replicate_folder_name,
                clip_files, target_languages,
                targets[0]["language"] if targets else source_lang,
                targets[0]["shortPlayId"] if targets else source_short_play_id,
            )
            results.extend(upload_results)

        return {
            "source_lang": source_lang,
            "target_langs": target_languages,
            "results": results,
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
