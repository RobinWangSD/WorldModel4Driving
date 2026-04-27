#!/usr/bin/env python3
"""Utilities for generating latent-learning NAVSIM scene filters."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm


DRIVOR_CAMERAS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"]
ALL_CAMERAS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2"]

POOL_SCENE_FILTER_TOKENS = "scene_filter_tokens"
POOL_RAW_WINDOWS = "raw_windows"
RULE_TPLUS1 = "tplus1"
RULE_ANY_1_8 = "any_1_8"

_WORKER_CONFIG: Dict = {}


@dataclass(frozen=True)
class SceneFilterSpec:
    """Minimal SceneFilter fields needed to reproduce NAVSIM windowing."""

    target: str
    convert: str
    num_history_frames: int
    num_future_frames: int
    frame_interval: int
    has_route: bool
    max_scenes: Optional[int]
    log_names: List[str]
    tokens: Optional[List[str]]

    @property
    def num_frames(self) -> int:
        return self.num_history_frames + self.num_future_frames


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_scalar(value: str):
    value = value.split("#", 1)[0].strip()
    if value in {"null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        return _strip_quotes(value)


def parse_scene_filter_yaml(path: Path) -> SceneFilterSpec:
    """Parse the subset of SceneFilter YAML used by NAVSIM configs."""

    scalars: Dict[str, object] = {}
    lists: Dict[str, List[str]] = {}
    current_list_key: Optional[str] = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("- "):
            if current_list_key is None:
                continue
            lists.setdefault(current_list_key, []).append(_strip_quotes(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            current_list_key = None
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            current_list_key = key
            lists.setdefault(key, [])
        else:
            current_list_key = None
            scalars[key] = _parse_scalar(value)

    num_history_frames = int(scalars.get("num_history_frames", 4))
    num_future_frames = int(scalars.get("num_future_frames", 10))
    parsed_frame_interval = scalars.get("frame_interval", None)
    frame_interval = (
        num_history_frames + num_future_frames if parsed_frame_interval is None else int(parsed_frame_interval)
    )

    log_names_value = lists.get("log_names", scalars.get("log_names"))
    if log_names_value is None:
        raise ValueError(f"{path} must define log_names for latent filter generation.")
    log_names = list(log_names_value)

    if "tokens" in lists:
        tokens = lists["tokens"]
    else:
        tokens_value = scalars.get("tokens")
        tokens = None if tokens_value is None else list(tokens_value)

    return SceneFilterSpec(
        target=str(scalars.get("_target_", "navsim.common.dataclasses.SceneFilter")),
        convert=str(scalars.get("_convert_", "all")),
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=frame_interval,
        has_route=bool(scalars.get("has_route", True)),
        max_scenes=scalars.get("max_scenes"),
        log_names=log_names,
        tokens=tokens,
    )


def camera_names_from_set(camera_set: str) -> List[str]:
    if camera_set == "drivor":
        return DRIVOR_CAMERAS
    if camera_set == "all":
        return ALL_CAMERAS
    raise ValueError(f"Unsupported camera_set={camera_set!r}")


def frame_has_all_cameras(frame: Dict, sensor_blobs_path: Path, camera_names: Iterable[str]) -> bool:
    cameras = frame.get("cams", {})
    for camera_name in camera_names:
        relative_path = cameras.get(camera_name, {}).get("data_path")
        if not relative_path or not (sensor_blobs_path / relative_path).is_file():
            return False
    return True


def future_rule_matches(
    window: Sequence[Dict],
    t_index: int,
    sensor_blobs_path: Path,
    camera_names: Sequence[str],
    future_rule: str,
    future_offsets: int,
) -> bool:
    if future_rule == RULE_TPLUS1:
        return frame_has_all_cameras(window[t_index + 1], sensor_blobs_path, camera_names)

    if future_rule == RULE_ANY_1_8:
        return any(
            frame_has_all_cameras(window[t_index + offset], sensor_blobs_path, camera_names)
            for offset in range(1, future_offsets + 1)
        )

    raise ValueError(f"Unsupported future_rule={future_rule!r}")


def _scan_log(
    log_file: Path,
    sensor_blobs_path: Path,
    spec: SceneFilterSpec,
    camera_names: Sequence[str],
    candidate_pool: str,
    future_rule: str,
    future_offsets: int,
    input_tokens: Optional[Set[str]],
) -> Dict:
    stats = Counter()
    selected_tokens: List[str] = []
    selected_token_set: Set[str] = set()

    stats["logs"] = 1
    if not log_file.is_file():
        stats["missing_logs"] = 1
        return {"log_name": log_file.stem, "selected_tokens": selected_tokens, "stats": dict(stats)}

    frames = pickle.load(open(log_file, "rb"))
    stats["num_frames"] = len(frames)
    max_start = len(frames) - spec.num_frames + 1
    t_index = spec.num_history_frames - 1

    for window_start in range(0, max(0, max_start), spec.frame_interval):
        window = frames[window_start : window_start + spec.num_frames]
        if len(window) < spec.num_frames:
            continue

        stats["windows"] += 1
        t_frame = window[t_index]
        if spec.has_route and len(t_frame.get("roadblock_ids", [])) == 0:
            stats["filtered_no_route"] += 1
            continue

        t_token = t_frame.get("token")
        if candidate_pool == POOL_SCENE_FILTER_TOKENS:
            if input_tokens is None:
                raise ValueError("scene_filter_tokens candidate pool requires input tokens.")
            if t_token not in input_tokens:
                stats["filtered_not_in_input_tokens"] += 1
                continue
        elif candidate_pool != POOL_RAW_WINDOWS:
            raise ValueError(f"Unsupported candidate_pool={candidate_pool!r}")

        stats["candidates"] += 1
        if t_token in selected_token_set:
            stats["duplicate_candidate_tokens"] += 1
            continue

        if not frame_has_all_cameras(t_frame, sensor_blobs_path, camera_names):
            stats["filtered_current_missing"] += 1
            continue

        if not future_rule_matches(window, t_index, sensor_blobs_path, camera_names, future_rule, future_offsets):
            stats["filtered_future_missing"] += 1
            continue

        selected_tokens.append(t_token)
        selected_token_set.add(t_token)
        stats["selected_tokens"] += 1

    return {"log_name": log_file.stem, "selected_tokens": selected_tokens, "stats": dict(stats)}


def _init_worker(config: Dict) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _scan_log_worker(log_file: str) -> Dict:
    return _scan_log(log_file=Path(log_file), **_WORKER_CONFIG)


def scan_scene_filter(
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    spec: SceneFilterSpec,
    camera_names: Sequence[str],
    candidate_pool: str,
    future_rule: str,
    future_offsets: int,
    max_logs: Optional[int],
    num_workers: int,
) -> Tuple[List[str], List[str], Dict[str, int], List[Dict]]:
    if future_offsets < 1:
        raise ValueError("future_offsets must be >= 1")
    if future_offsets > spec.num_future_frames:
        raise ValueError(
            f"future_offsets={future_offsets} exceeds num_future_frames={spec.num_future_frames}"
        )
    if spec.frame_interval != 1:
        raise ValueError(
            f"Expected DrivoR/navtrain frame_interval=1, got {spec.frame_interval}. "
            "Use an input SceneFilter with frame_interval: 1."
        )
    if spec.num_history_frames != 4 or spec.num_future_frames != 10:
        raise ValueError(
            "Expected DrivoR/navtrain num_history_frames=4 and num_future_frames=10, "
            f"got {spec.num_history_frames} and {spec.num_future_frames}."
        )

    input_tokens = set(spec.tokens) if spec.tokens is not None else None
    if candidate_pool == POOL_SCENE_FILTER_TOKENS and input_tokens is None:
        raise ValueError("Input SceneFilter must define tokens for navtrain-token filtering.")

    log_names = spec.log_names[:max_logs] if max_logs is not None else spec.log_names
    log_files = [navsim_log_path / f"{log_name}.pkl" for log_name in log_names]
    worker_config = {
        "sensor_blobs_path": sensor_blobs_path,
        "spec": spec,
        "camera_names": list(camera_names),
        "candidate_pool": candidate_pool,
        "future_rule": future_rule,
        "future_offsets": future_offsets,
        "input_tokens": input_tokens,
    }

    if num_workers <= 1:
        results = [
            _scan_log(log_file=log_file, **worker_config)
            for log_file in tqdm(log_files, desc="Scanning logs")
        ]
    else:
        results: List[Optional[Dict]] = [None] * len(log_files)
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(worker_config,)) as executor:
            future_to_index = {
                executor.submit(_scan_log_worker, str(log_file)): index
                for index, log_file in enumerate(log_files)
            }
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Scanning logs"):
                results[future_to_index[future]] = future.result()
        results = [result for result in results if result is not None]

    selected_by_log: Dict[str, List[str]] = {}
    selected_tokens: List[str] = []
    seen_tokens: Set[str] = set()
    totals = Counter()
    per_log: List[Dict] = []

    for result in results:
        log_name = result["log_name"]
        stats = Counter(result["stats"])
        tokens_for_log = []
        for token in result["selected_tokens"]:
            if token in seen_tokens:
                totals["duplicate_selected_tokens_across_logs"] += 1
                continue
            selected_tokens.append(token)
            tokens_for_log.append(token)
            seen_tokens.add(token)

        if tokens_for_log:
            selected_by_log[log_name] = tokens_for_log

        totals.update(stats)
        per_log.append({"log_name": log_name, **dict(stats), "selected_tokens": len(tokens_for_log)})

    selected_log_names = [log_name for log_name in log_names if log_name in selected_by_log]
    totals["selected_logs"] = len(selected_log_names)
    totals["selected_tokens_unique"] = len(selected_tokens)

    return selected_log_names, selected_tokens, dict(totals), per_log


def _format_yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append_yaml_list(lines: List[str], key: str, values: Sequence[str]) -> None:
    if not values:
        lines.append(f"{key}: []")
        return
    lines.append(f"{key}:")
    for value in values:
        escaped = str(value).replace("'", "''")
        lines.append(f"  - '{escaped}'")


def write_scene_filter_yaml(output_path: Path, spec: SceneFilterSpec, log_names: Sequence[str], tokens: Sequence[str]) -> None:
    lines = [
        f"_target_: {spec.target}",
        f"_convert_: '{spec.convert}'",
        f"num_history_frames: {spec.num_history_frames}",
        f"num_future_frames: {spec.num_future_frames}",
        f"frame_interval: {spec.frame_interval}",
        f"has_route: {_format_yaml_scalar(spec.has_route)}",
        f"max_scenes: {_format_yaml_scalar(spec.max_scenes)}",
    ]
    _append_yaml_list(lines, "log_names", log_names)
    _append_yaml_list(lines, "tokens", tokens)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def build_arg_parser(description: str, default_output_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--scene-filter-yaml", type=Path, required=True)
    parser.add_argument("--output-yaml", type=Path, default=None)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--camera-set", choices=["drivor", "all"], default="drivor")
    parser.add_argument("--future-offsets", type=int, default=8)
    parser.set_defaults(default_output_name=default_output_name)
    return parser


def run_generation(
    args: argparse.Namespace,
    candidate_pool: str,
    future_rule: str,
) -> Dict:
    spec = parse_scene_filter_yaml(args.scene_filter_yaml)
    camera_names = camera_names_from_set(args.camera_set)
    output_yaml = args.output_yaml or (args.scene_filter_yaml.parent / args.default_output_name)

    selected_log_names, selected_tokens, totals, per_log = scan_scene_filter(
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        spec=spec,
        camera_names=camera_names,
        candidate_pool=candidate_pool,
        future_rule=future_rule,
        future_offsets=args.future_offsets,
        max_logs=args.max_logs,
        num_workers=args.num_workers,
    )
    write_scene_filter_yaml(output_yaml, spec, selected_log_names, selected_tokens)

    summary = {
        "output_yaml": str(output_yaml),
        "candidate_pool": candidate_pool,
        "future_rule": future_rule,
        "camera_set": args.camera_set,
        "camera_names": camera_names,
        "future_offsets": args.future_offsets,
        "scene_filter_yaml": str(args.scene_filter_yaml),
        "navsim_log_path": str(args.navsim_log_path),
        "sensor_blobs_path": str(args.sensor_blobs_path),
        "totals": totals,
        "per_log": per_log,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "per_log"}, indent=2))
    return summary


def main(
    description: str,
    default_output_name: str,
    candidate_pool: str,
    future_rule: str,
) -> None:
    parser = build_arg_parser(description, default_output_name)
    run_generation(parser.parse_args(), candidate_pool=candidate_pool, future_rule=future_rule)
