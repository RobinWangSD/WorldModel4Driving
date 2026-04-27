#!/usr/bin/env python3
"""Summarize NAVSIM future camera image path availability directly from dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from future_image_path_cache_utils import (
    camera_names_from_set,
    iter_log_files,
    make_cache_record,
    update_summary_from_record,
)


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
    per_offset: Dict[str, Counter] = defaultdict(Counter)
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

        record = make_cache_record(
            log_name=log_file.stem,
            window_start=window_start,
            window=window,
            sensor_blobs_path=sensor_blobs_path,
            camera_names=camera_names,
            num_history_frames=num_history_frames,
            future_offsets=future_offsets,
        )
        update_summary_from_record(totals, per_offset, record)

        for offset_key, future_record in record["future"].items():
            num_existing = future_record["num_existing_cameras"]
            num_cameras = future_record["num_cameras"]
            if num_existing == num_cameras:
                log_counts[f"{offset_key}_all_present"] += 1
            elif num_existing == 0:
                log_counts[f"{offset_key}_all_missing"] += 1
            else:
                log_counts[f"{offset_key}_partial_present"] += 1

    return {
        "totals": dict(totals),
        "per_offset": {key: dict(value) for key, value in per_offset.items()},
        "per_log": {log_file.stem: dict(log_counts)},
    }


def _merge_summaries(items: List[Dict]) -> Dict:
    totals = Counter()
    per_offset: Dict[str, Counter] = defaultdict(Counter)
    per_log: Dict[str, Counter] = defaultdict(Counter)

    for item in items:
        totals.update(item["totals"])
        for offset_key, offset_counts in item["per_offset"].items():
            per_offset[offset_key].update(offset_counts)
        for log_name, log_counts in item["per_log"].items():
            per_log[log_name].update(log_counts)

    return {
        "totals": dict(totals),
        "per_offset": {key: dict(value) for key, value in sorted(per_offset.items())},
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
        iterator = (_summarize_log(item) for item in worker_args)
        partials = list(tqdm(iterator, total=len(worker_args), desc="Summarizing dataset"))
    else:
        partials = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_summarize_log, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing dataset"):
                partials.append(future.result())

    summary = _merge_summaries(partials)
    summary.update({
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
    })

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
    parser.add_argument("--log-name", action="append", default=None, help="Specific log stem to summarize. Can be repeated.")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--camera-set", choices=["drivor", "all"], default="drivor")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--future-offsets", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.set_defaults(require_route=True)
    parser.add_argument("--require-route", dest="require_route", action="store_true")
    parser.add_argument("--no-require-route", dest="require_route", action="store_false")
    return parser.parse_args()


def main() -> None:
    summary = summarize_dataset(parse_args())
    print(json.dumps({
        "num_logs": summary["num_logs"],
        "totals": summary["totals"],
        "per_offset": summary["per_offset"],
    }, indent=2))


if __name__ == "__main__":
    main()
