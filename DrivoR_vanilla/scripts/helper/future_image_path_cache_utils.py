"""Utilities for caching NAVSIM future camera image path availability."""

from __future__ import annotations

import gzip
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CACHE_FILE_NAME = "future_image_paths.gz"
DRIVOR_CAMERAS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"]
ALL_CAMERAS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2"]


def dump_cache(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data, f)


def load_cache(path: Path) -> Dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def parse_scene_filter_log_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.is_file():
        return None

    log_names: List[str] = []
    in_log_names = False
    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("log_names:"):
            in_log_names = True
            continue
        if in_log_names and stripped.endswith(":") and not stripped.startswith("- "):
            break
        if in_log_names and stripped.startswith("- "):
            log_names.append(stripped[2:].strip().strip("'\""))

    return log_names or None


def iter_log_files(
    navsim_log_path: Path,
    scene_filter_yaml: Optional[Path],
    log_names: Optional[List[str]],
    max_logs: Optional[int],
) -> List[Path]:
    if log_names is None:
        log_names = parse_scene_filter_log_names(scene_filter_yaml)

    if log_names is None:
        files = sorted(navsim_log_path.glob("*.pkl"))
    else:
        files = [navsim_log_path / f"{name}.pkl" for name in log_names]
        files = [path for path in files if path.is_file()]

    if max_logs is not None:
        files = files[:max_logs]
    return files


def camera_names_from_set(camera_set: str) -> List[str]:
    if camera_set == "all":
        return ALL_CAMERAS
    if camera_set == "drivor":
        return DRIVOR_CAMERAS
    raise ValueError(f"Unsupported camera_set={camera_set!r}")


def camera_path_status(frame: Dict, sensor_blobs_path: Path, camera_names: Iterable[str]) -> Dict[str, Dict]:
    status: Dict[str, Dict] = {}
    cameras = frame.get("cams", {})
    for camera_name in camera_names:
        relative_path = cameras.get(camera_name, {}).get("data_path")
        exists = bool(relative_path) and (sensor_blobs_path / relative_path).is_file()
        status[camera_name] = {
            "relative_path": relative_path,
            "exists": bool(exists),
        }
    return status


def make_cache_record(
    log_name: str,
    window_start: int,
    window: List[Dict],
    sensor_blobs_path: Path,
    camera_names: List[str],
    num_history_frames: int,
    future_offsets: int,
) -> Dict:
    t_index = num_history_frames - 1
    t_frame = window[t_index]
    record = {
        "log_name": log_name,
        "window_start": window_start,
        "t_index_in_window": t_index,
        "t_token": t_frame.get("token"),
        "future": {},
    }

    for offset in range(1, future_offsets + 1):
        future_frame = window[t_index + offset]
        camera_status = camera_path_status(future_frame, sensor_blobs_path, camera_names)
        num_existing = sum(1 for item in camera_status.values() if item["exists"])
        record["future"][f"t+{offset}"] = {
            "token": future_frame.get("token"),
            "timestamp": future_frame.get("timestamp"),
            "num_existing_cameras": num_existing,
            "num_cameras": len(camera_names),
            "all_cameras_present": num_existing == len(camera_names),
            "cameras": camera_status,
        }

    return record


def update_summary_from_record(totals: Counter, per_offset: Dict[str, Counter], record: Dict) -> None:
    totals["windows_kept"] += 1
    for offset_key, future_record in record["future"].items():
        num_cameras = future_record["num_cameras"]
        num_existing = future_record["num_existing_cameras"]
        per_offset[offset_key]["frames"] += 1
        per_offset[offset_key]["camera_paths"] += num_cameras
        per_offset[offset_key]["existing_camera_paths"] += num_existing
        per_offset[offset_key]["missing_camera_paths"] += num_cameras - num_existing

        if num_existing == num_cameras:
            per_offset[offset_key]["frames_all_cameras_present"] += 1
        elif num_existing == 0:
            per_offset[offset_key]["frames_all_cameras_missing"] += 1
        else:
            per_offset[offset_key]["frames_partial_cameras_present"] += 1
