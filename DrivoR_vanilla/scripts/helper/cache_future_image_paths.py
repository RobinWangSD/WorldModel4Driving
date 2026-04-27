#!/usr/bin/env python3
"""Cache NAVSIM t+1...t+N future camera image paths per scene token."""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from future_image_path_cache_utils import (
    CACHE_FILE_NAME,
    camera_names_from_set,
    dump_cache,
    iter_log_files,
    make_cache_record,
)


def _cache_log(args: Dict) -> Dict:
    log_file = Path(args["log_file"])
    cache_path = Path(args["cache_path"])
    sensor_blobs_path = Path(args["sensor_blobs_path"])
    camera_names = args["camera_names"]
    num_history_frames = args["num_history_frames"]
    future_offsets = args["future_offsets"]
    window_size = args["window_size"]
    frame_interval = args["frame_interval"]
    require_route = args["require_route"]
    force = args["force"]

    stats = {
        "log_name": log_file.stem,
        "num_frames": 0,
        "windows": 0,
        "filtered_no_route": 0,
        "cached": 0,
        "skipped_existing": 0,
    }
    frames = pickle.load(open(log_file, "rb"))
    stats["num_frames"] = len(frames)
    max_start = len(frames) - window_size + 1

    for window_start in range(0, max(0, max_start), frame_interval):
        window = frames[window_start : window_start + window_size]
        if len(window) < window_size:
            continue

        stats["windows"] += 1
        t_frame = window[num_history_frames - 1]
        if require_route and len(t_frame.get("roadblock_ids", [])) == 0:
            stats["filtered_no_route"] += 1
            continue

        t_token = t_frame.get("token")
        record_path = cache_path / log_file.stem / t_token / CACHE_FILE_NAME
        if record_path.is_file() and not force:
            stats["skipped_existing"] += 1
            continue

        record = make_cache_record(
            log_name=log_file.stem,
            window_start=window_start,
            window=window,
            sensor_blobs_path=sensor_blobs_path,
            camera_names=camera_names,
            num_history_frames=num_history_frames,
            future_offsets=future_offsets,
        )
        dump_cache(record_path, record)
        stats["cached"] += 1

    return stats


def cache_future_image_paths(args: argparse.Namespace) -> Dict:
    camera_names = camera_names_from_set(args.camera_set)
    log_files = iter_log_files(args.navsim_log_path, args.scene_filter_yaml, args.log_name, args.max_logs)
    args.cache_path.mkdir(parents=True, exist_ok=True)

    worker_args = [
        {
            "log_file": str(log_file),
            "cache_path": str(args.cache_path),
            "sensor_blobs_path": str(args.sensor_blobs_path),
            "camera_names": camera_names,
            "num_history_frames": args.num_history_frames,
            "future_offsets": args.future_offsets,
            "window_size": args.window_size,
            "frame_interval": args.frame_interval,
            "require_route": args.require_route,
            "force": args.force,
        }
        for log_file in log_files
    ]

    per_log: List[Dict] = []
    if args.num_workers <= 1:
        iterator = (_cache_log(item) for item in worker_args)
        per_log = list(tqdm(iterator, total=len(worker_args), desc="Caching future image paths"))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_cache_log, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching future image paths"):
                per_log.append(future.result())
        per_log = sorted(per_log, key=lambda item: item["log_name"])

    totals: Dict[str, int] = {}
    for item in per_log:
        for key, value in item.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value

    metadata = {
        "cache_file_name": CACHE_FILE_NAME,
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
        "totals": totals,
        "per_log": per_log,
    }
    metadata_path = args.cache_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--scene-filter-yaml", type=Path, default=None)
    parser.add_argument("--log-name", action="append", default=None, help="Specific log stem to cache. Can be repeated.")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--camera-set", choices=["drivor", "all"], default="drivor")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--future-offsets", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(require_route=True)
    parser.add_argument("--require-route", dest="require_route", action="store_true")
    parser.add_argument("--no-require-route", dest="require_route", action="store_false")
    return parser.parse_args()


def main() -> None:
    metadata = cache_future_image_paths(parse_args())
    print(json.dumps({
        "cache_file_name": metadata["cache_file_name"],
        "num_logs": metadata["num_logs"],
        "totals": metadata["totals"],
    }, indent=2))


if __name__ == "__main__":
    main()
