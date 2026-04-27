#!/usr/bin/env python3
"""Summarize NAVSIM future camera image availability with cumulative offsets."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DRIVOR_CAMERAS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"]
ALL_CAMERAS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2"]


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


def frame_camera_status(frame: Dict, sensor_blobs_path: Path, camera_names: Iterable[str]) -> Dict:
    cameras = frame.get("cams", {})
    per_camera = {}
    for camera_name in camera_names:
        relative_path = cameras.get(camera_name, {}).get("data_path")
        exists = bool(relative_path) and (sensor_blobs_path / relative_path).is_file()
        per_camera[camera_name] = exists

    num_existing = sum(1 for exists in per_camera.values() if exists)
    return {
        "num_existing": num_existing,
        "num_cameras": len(per_camera),
        "all_present": num_existing == len(per_camera),
        "all_missing": num_existing == 0,
    }


def update_frame_summary(counter: Counter, status: Dict, prefix: str = "frames") -> None:
    counter[prefix] += 1
    counter["camera_paths"] += status["num_cameras"]
    counter["existing_camera_paths"] += status["num_existing"]
    counter["missing_camera_paths"] += status["num_cameras"] - status["num_existing"]
    if status["all_present"]:
        counter[f"{prefix}_all_cameras_present"] += 1
    elif status["all_missing"]:
        counter[f"{prefix}_all_cameras_missing"] += 1
    else:
        counter[f"{prefix}_partial_cameras_present"] += 1


def _summarize_log(args: Dict) -> Dict:
    log_file = Path(args["log_file"])
    sensor_blobs_path = Path(args["sensor_blobs_path"])
    camera_names = args["camera_names"]
    num_history_frames = args["num_history_frames"]
    future_offsets = args["future_offsets"]
    window_size = args["window_size"]
    frame_interval = args["frame_interval"]
    require_route = args["require_route"]

    totals = Counter()
    current = Counter()
    independent: Dict[str, Counter] = defaultdict(Counter)
    cumulative: Dict[str, Counter] = defaultdict(Counter)
    cumulative_from_t: Dict[str, Counter] = defaultdict(Counter)
    log_counts = Counter()

    frames = pickle.load(open(log_file, "rb"))
    max_start = len(frames) - window_size + 1
    log_counts["num_frames"] = len(frames)

    for window_start in range(0, max(0, max_start), frame_interval):
        window = frames[window_start : window_start + window_size]
        if len(window) < window_size:
            continue

        totals["windows"] += 1
        log_counts["windows"] += 1
        t_frame = window[num_history_frames - 1]
        if require_route and len(t_frame.get("roadblock_ids", [])) == 0:
            totals["filtered_no_route"] += 1
            log_counts["filtered_no_route"] += 1
            continue

        totals["windows_kept"] += 1

        t_status = frame_camera_status(t_frame, sensor_blobs_path, camera_names)
        update_frame_summary(current, t_status)
        if t_status["all_present"]:
            log_counts["t_all_present"] += 1
        elif t_status["all_missing"]:
            log_counts["t_all_missing"] += 1
        else:
            log_counts["t_partial_present"] += 1

        prefix_all_present = True
        prefix_existing_camera_paths = 0
        prefix_camera_paths = 0
        prefix_from_t_all_present = t_status["all_present"]
        prefix_from_t_existing_camera_paths = t_status["num_existing"]
        prefix_from_t_camera_paths = t_status["num_cameras"]

        for offset in range(1, future_offsets + 1):
            offset_key = f"t+{offset}"
            future_frame = window[num_history_frames - 1 + offset]
            status = frame_camera_status(future_frame, sensor_blobs_path, camera_names)

            update_frame_summary(independent[offset_key], status)
            if status["all_present"]:
                log_counts[f"{offset_key}_all_present_independent"] += 1
            elif status["all_missing"]:
                log_counts[f"{offset_key}_all_missing_independent"] += 1
            else:
                log_counts[f"{offset_key}_partial_present_independent"] += 1

            prefix_all_present = prefix_all_present and status["all_present"]
            prefix_existing_camera_paths += status["num_existing"]
            prefix_camera_paths += status["num_cameras"]
            prefix_from_t_all_present = prefix_from_t_all_present and status["all_present"]
            prefix_from_t_existing_camera_paths += status["num_existing"]
            prefix_from_t_camera_paths += status["num_cameras"]

            cumulative[offset_key]["frames"] += 1
            cumulative[offset_key]["prefix_camera_paths"] += prefix_camera_paths
            cumulative[offset_key]["existing_prefix_camera_paths"] += prefix_existing_camera_paths
            cumulative[offset_key]["missing_prefix_camera_paths"] += prefix_camera_paths - prefix_existing_camera_paths
            if prefix_all_present:
                cumulative[offset_key]["frames_all_previous_offsets_present"] += 1
                log_counts[f"{offset_key}_all_previous_offsets_present"] += 1
            else:
                cumulative[offset_key]["frames_any_previous_offset_missing"] += 1
                log_counts[f"{offset_key}_any_previous_offset_missing"] += 1

            cumulative_from_t[offset_key]["frames"] += 1
            cumulative_from_t[offset_key]["prefix_camera_paths"] += prefix_from_t_camera_paths
            cumulative_from_t[offset_key]["existing_prefix_camera_paths"] += prefix_from_t_existing_camera_paths
            cumulative_from_t[offset_key]["missing_prefix_camera_paths"] += (
                prefix_from_t_camera_paths - prefix_from_t_existing_camera_paths
            )
            if prefix_from_t_all_present:
                cumulative_from_t[offset_key]["frames_t_through_offset_present"] += 1
                log_counts[f"{offset_key}_t_through_offset_present"] += 1
            else:
                cumulative_from_t[offset_key]["frames_t_through_offset_missing"] += 1
                log_counts[f"{offset_key}_t_through_offset_missing"] += 1

    return {
        "totals": dict(totals),
        "current_frame": dict(current),
        "per_offset_independent": {key: dict(value) for key, value in independent.items()},
        "per_offset_cumulative": {key: dict(value) for key, value in cumulative.items()},
        "per_offset_cumulative_from_t": {key: dict(value) for key, value in cumulative_from_t.items()},
        "per_log": {log_file.stem: dict(log_counts)},
    }


def _merge_summaries(items: List[Dict]) -> Dict:
    totals = Counter()
    current = Counter()
    independent: Dict[str, Counter] = defaultdict(Counter)
    cumulative: Dict[str, Counter] = defaultdict(Counter)
    cumulative_from_t: Dict[str, Counter] = defaultdict(Counter)
    per_log: Dict[str, Counter] = defaultdict(Counter)

    for item in items:
        totals.update(item["totals"])
        current.update(item["current_frame"])
        for offset_key, offset_counts in item["per_offset_independent"].items():
            independent[offset_key].update(offset_counts)
        for offset_key, offset_counts in item["per_offset_cumulative"].items():
            cumulative[offset_key].update(offset_counts)
        for offset_key, offset_counts in item["per_offset_cumulative_from_t"].items():
            cumulative_from_t[offset_key].update(offset_counts)
        for log_name, log_counts in item["per_log"].items():
            per_log[log_name].update(log_counts)

    return {
        "totals": dict(totals),
        "current_frame": dict(current),
        "per_offset_independent": {key: dict(value) for key, value in sorted(independent.items())},
        "per_offset_cumulative": {key: dict(value) for key, value in sorted(cumulative.items())},
        "per_offset_cumulative_from_t": {key: dict(value) for key, value in sorted(cumulative_from_t.items())},
        "per_log": [{"log_name": key, **dict(value)} for key, value in sorted(per_log.items())],
    }


def summarize_dataset(args: argparse.Namespace) -> Dict:
    camera_names = camera_names_from_set(args.camera_set)
    log_files = iter_log_files(args.navsim_log_path, args.scene_filter_yaml, args.log_name, args.max_logs)
    worker_args = [
        {
            "log_file": str(log_file),
            "sensor_blobs_path": str(args.sensor_blobs_path),
            "camera_names": camera_names,
            "num_history_frames": args.num_history_frames,
            "future_offsets": args.future_offsets,
            "window_size": args.window_size,
            "frame_interval": args.frame_interval,
            "require_route": args.require_route,
        }
        for log_file in log_files
    ]

    if args.num_workers <= 1:
        partials = [_summarize_log(item) for item in worker_args]
    else:
        partials = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_summarize_log, item) for item in worker_args]
            for future in as_completed(futures):
                partials.append(future.result())

    summary = _merge_summaries(partials)
    summary.update(
        {
            "navsim_log_path": str(args.navsim_log_path),
            "sensor_blobs_path": str(args.sensor_blobs_path),
            "scene_filter_yaml": str(args.scene_filter_yaml) if args.scene_filter_yaml else None,
            "camera_set": args.camera_set,
            "camera_names": camera_names,
            "num_history_frames": args.num_history_frames,
            "future_offsets": args.future_offsets,
            "window_size": args.window_size,
            "frame_interval": args.frame_interval,
            "require_route": args.require_route,
            "num_logs": len(log_files),
            "cumulative_definition": (
                "per_offset_cumulative[t+k] counts a window as present only if every future "
                "offset t+1..t+k has all selected camera images present."
            ),
            "cumulative_from_t_definition": (
                "per_offset_cumulative_from_t[t+k] counts a window as present only if the current "
                "frame t and every future offset t+1..t+k have all selected camera images present."
            ),
        }
    )

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(summary, indent=2))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--scene-filter-yaml", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--log-name", action="append", default=None)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--camera-set", choices=["drivor", "all"], default="drivor")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--future-offsets", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--frame-interval", type=int, default=14)
    parser.set_defaults(require_route=True)
    parser.add_argument("--require-route", dest="require_route", action="store_true")
    parser.add_argument("--no-require-route", dest="require_route", action="store_false")
    return parser.parse_args()


def main() -> None:
    summary = summarize_dataset(parse_args())
    print(
        json.dumps(
            {
                "num_logs": summary["num_logs"],
                "totals": summary["totals"],
                "current_frame": summary["current_frame"],
                "per_offset_cumulative": summary["per_offset_cumulative"],
                "per_offset_cumulative_from_t": summary["per_offset_cumulative_from_t"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
