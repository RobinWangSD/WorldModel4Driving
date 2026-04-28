from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_NAVSIM_LOG_PATH = Path("/avl-west/navsim/trainval_navsim_logs/trainval")
DEFAULT_SENSOR_BLOBS_PATH = Path("/avl-west/navsim/trainval_sensor_blobs/trainval")
DEFAULT_OUTPUT_PATH = Path("/hugsim-storage/navsim_trainval_camera_offset_availability.csv")
DEFAULT_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")

NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 10
FRAME_INTERVAL = 1
HAS_ROUTE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count independent per-offset 4-camera availability for NAVSIM trainval window tokens."
    )
    parser.add_argument("--navsim-log-path", type=Path, default=DEFAULT_NAVSIM_LOG_PATH)
    parser.add_argument("--sensor-blobs-path", type=Path, default=DEFAULT_SENSOR_BLOBS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-offset", type=int, default=8)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--max-logs", type=int, default=None)
    return parser.parse_args()


def iter_windows(frames: Sequence[Dict[str, Any]], num_frames: int) -> Iterable[List[Dict[str, Any]]]:
    for start_idx in range(0, len(frames), FRAME_INTERVAL):
        window = list(frames[start_idx:start_idx + num_frames])
        if len(window) == num_frames:
            yield window


def has_valid_route(window: Sequence[Dict[str, Any]], current_idx: int) -> bool:
    if not HAS_ROUTE:
        return True
    return len(window[current_idx].get("roadblock_ids", [])) > 0


def has_all_camera_files(
    frame: Dict[str, Any],
    cameras: Sequence[str],
    sensor_blobs_path: Path,
    availability_cache: Dict[str, bool],
) -> bool:
    token = frame.get("token")
    cache_key = str(token) if token is not None else str(id(frame))
    if cache_key in availability_cache:
        return availability_cache[cache_key]

    frame_cameras = frame.get("cams", {})
    available = True
    for camera_name in cameras:
        camera = frame_cameras.get(camera_name)
        if not isinstance(camera, dict):
            available = False
            break

        data_path = camera.get("data_path")
        if not data_path or not (sensor_blobs_path / data_path).is_file():
            available = False
            break

    availability_cache[cache_key] = available
    return available


def label_offset(offset: int) -> str:
    return "t" if offset == 0 else f"t+{offset}"


def collect_counts(args: argparse.Namespace) -> tuple[int, List[int], int]:
    if args.max_offset < 0:
        raise ValueError("--max-offset must be non-negative")
    if args.max_offset > NUM_FUTURE_FRAMES:
        raise ValueError(f"--max-offset must be <= {NUM_FUTURE_FRAMES} for the 14-frame NAVSIM window")

    log_files = sorted(args.navsim_log_path.glob("*.pkl"))
    if args.max_logs is not None:
        log_files = log_files[:args.max_logs]

    total_eligible_tokens = 0
    counts = [0 for _ in range(args.max_offset + 1)]
    current_idx = NUM_HISTORY_FRAMES - 1
    num_frames = NUM_HISTORY_FRAMES + NUM_FUTURE_FRAMES
    availability_cache: Dict[str, bool] = {}

    print(f"Reading {len(log_files)} log files from {args.navsim_log_path}")
    for log_idx, log_pickle_path in enumerate(log_files, start=1):
        with open(log_pickle_path, "rb") as fp:
            frames = pickle.load(fp)

        for window in iter_windows(frames, num_frames):
            if not has_valid_route(window, current_idx):
                continue

            total_eligible_tokens += 1
            for offset in range(args.max_offset + 1):
                frame = window[current_idx + offset]
                if has_all_camera_files(frame, args.cameras, args.sensor_blobs_path, availability_cache):
                    counts[offset] += 1

        if log_idx % 100 == 0:
            print(f"Processed {log_idx}/{len(log_files)} logs; eligible tokens so far: {total_eligible_tokens}")

    return total_eligible_tokens, counts, len(log_files)


def write_csv(args: argparse.Namespace, total_eligible_tokens: int, counts: Sequence[int]) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cameras = "|".join(args.cameras)
    fieldnames = [
        "offset_label",
        "offset",
        "total_eligible_tokens",
        "available_tokens",
        "availability_ratio",
        "cameras",
        "navsim_log_path",
        "sensor_blobs_path",
    ]

    with open(args.output_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for offset, available_tokens in enumerate(counts):
            ratio = available_tokens / total_eligible_tokens if total_eligible_tokens else 0.0
            writer.writerow(
                {
                    "offset_label": label_offset(offset),
                    "offset": offset,
                    "total_eligible_tokens": total_eligible_tokens,
                    "available_tokens": available_tokens,
                    "availability_ratio": f"{ratio:.6f}",
                    "cameras": cameras,
                    "navsim_log_path": str(args.navsim_log_path),
                    "sensor_blobs_path": str(args.sensor_blobs_path),
                }
            )


def main() -> None:
    args = parse_args()
    total_eligible_tokens, counts, log_count = collect_counts(args)
    write_csv(args, total_eligible_tokens, counts)

    print(f"Wrote {args.output_path}")
    print(f"Processed logs: {log_count}")
    print(f"Total eligible NAVSIM window tokens: {total_eligible_tokens}")
    for offset, available_tokens in enumerate(counts):
        ratio = available_tokens / total_eligible_tokens if total_eligible_tokens else 0.0
        print(f"{label_offset(offset)}: {available_tokens} ({ratio:.6f})")


if __name__ == "__main__":
    main()
