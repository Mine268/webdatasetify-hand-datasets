"""
Scan exported WebDataset shards and summarize camera-space root joint distributions.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import os.path as osp
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_ROOT = "/mnt/qnap/data/datasets/webdatasets2"
DEFAULT_STAT_ROOT = "statistic"
DEFAULT_MANIFEST_PATH = "manifest.json"
ROOT_JOINT_INDEX = 0
HISTOGRAM_BINS = 128
SUMMARY_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
RELEVANT_SUFFIXES = (
    "joint_3d_valid.npy",
    "joint_valid.npy",
    "joint_cam.npy",
    "handedness.json",
    "data_source.json",
    "source_split.json",
)


@dataclass
class SplitAccumulator:
    dataset_name: str
    split_name: str
    rel_split_dir: str
    split_dir: str
    tar_paths: List[str]
    num_samples: int = 0
    num_frames: int = 0
    num_valid_root_frames: int = 0
    samples_with_valid_root: int = 0
    samples_missing_joint_cam: int = 0
    samples_missing_valid_mask: int = 0
    tar_error_count: int = 0
    tar_errors: List[Dict[str, str]] = field(default_factory=list)
    handedness_counts: Counter = field(default_factory=Counter)
    data_source_counts: Counter = field(default_factory=Counter)
    source_split_counts: Counter = field(default_factory=Counter)
    clip_lengths: List[int] = field(default_factory=list)
    root_positions_chunks: List[np.ndarray] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize root joint camera-space distributions from WebDataset shards."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stat-root", default=DEFAULT_STAT_ROOT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--hist-bins", type=int, default=HISTOGRAM_BINS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescan splits even if statistic outputs already exist.",
    )
    return parser.parse_args()


def _discover_split_dirs(dataset_root: str) -> List[str]:
    split_dirs: List[str] = []
    for current_root, _dirs, files in os.walk(dataset_root):
        if any(name.endswith(".tar") for name in files):
            split_dirs.append(current_root)
    split_dirs.sort()
    return split_dirs


def _build_split_accumulator(dataset_root: str, split_dir: str) -> SplitAccumulator:
    rel_split_dir = osp.relpath(split_dir, dataset_root)
    rel_parts = rel_split_dir.split(os.sep)
    dataset_name = rel_parts[0]
    split_name = "/".join(rel_parts[1:]) if len(rel_parts) > 1 else "__root__"
    tar_paths = sorted(
        osp.join(split_dir, name)
        for name in os.listdir(split_dir)
        if name.endswith(".tar")
    )
    return SplitAccumulator(
        dataset_name=dataset_name,
        split_name=split_name,
        rel_split_dir=rel_split_dir,
        split_dir=split_dir,
        tar_paths=tar_paths,
    )


def _match_relevant_suffix(member_name: str) -> Tuple[Optional[str], Optional[str]]:
    for suffix in RELEVANT_SUFFIXES:
        suffix_token = f".{suffix}"
        if member_name.endswith(suffix_token):
            return suffix, member_name[: -len(suffix_token)]
    return None, None


def _load_member_payload(member_name: str, payload: bytes) -> Any:
    if member_name.endswith(".npy"):
        with io.BytesIO(payload) as buffer:
            return np.load(buffer, allow_pickle=False)
    if member_name.endswith(".json"):
        return json.loads(payload.decode("utf-8"))
    raise ValueError(f"Unsupported payload type: {member_name}")


def _coerce_frame_array(
    value: np.ndarray, tail_shape: Tuple[int, ...], field_name: str
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == len(tail_shape):
        arr = arr[None]
    if arr.ndim != len(tail_shape) + 1:
        raise ValueError(f"{field_name} expected ndim {len(tail_shape) + 1}, got {arr.ndim}")
    if tuple(arr.shape[1:]) != tail_shape:
        raise ValueError(
            f"{field_name} expected tail shape {tail_shape}, got {tuple(arr.shape[1:])}"
        )
    return arr.astype(np.float32, copy=False)


def _normalize_valid_mask(
    fields: Dict[str, Any], num_frames: int
) -> Tuple[np.ndarray, bool]:
    valid = fields.get("joint_3d_valid.npy")
    if valid is None:
        valid = fields.get("joint_valid.npy")
    if valid is None:
        return np.ones((num_frames,), dtype=bool), True

    valid_arr = np.asarray(valid)
    if valid_arr.ndim == 1:
        if valid_arr.shape[0] == 21:
            valid_arr = valid_arr[None]
        elif num_frames == 1:
            valid_arr = valid_arr.reshape(1, -1)
        else:
            raise ValueError(
                f"joint valid expected frame axis for {num_frames} frames, got shape {valid_arr.shape}"
            )
    if valid_arr.ndim != 2:
        raise ValueError(f"joint valid expected ndim 2, got {valid_arr.ndim}")
    if valid_arr.shape[0] != num_frames:
        raise ValueError(
            f"joint valid expected {num_frames} frames, got {valid_arr.shape[0]}"
        )
    if valid_arr.shape[1] <= ROOT_JOINT_INDEX:
        raise ValueError(
            f"joint valid expected at least {ROOT_JOINT_INDEX + 1} joints, got {valid_arr.shape[1]}"
        )
    return valid_arr[:, ROOT_JOINT_INDEX] > 0.5, False


def _finalize_sample(sample_key: str, fields: Dict[str, Any], acc: SplitAccumulator) -> None:
    joint_cam = fields.get("joint_cam.npy")
    if joint_cam is None:
        acc.samples_missing_joint_cam += 1
        return

    joint_cam_arr = _coerce_frame_array(joint_cam, (21, 3), "joint_cam.npy")
    num_frames = int(joint_cam_arr.shape[0])
    root_positions = joint_cam_arr[:, ROOT_JOINT_INDEX, :]
    valid_mask, missing_valid_mask = _normalize_valid_mask(fields, num_frames)

    acc.num_samples += 1
    acc.num_frames += num_frames
    acc.clip_lengths.append(num_frames)

    handedness = str(fields.get("handedness.json", "unknown"))
    data_source = str(fields.get("data_source.json", "unknown"))
    source_split = str(fields.get("source_split.json", "unknown"))
    acc.handedness_counts[handedness] += 1
    acc.data_source_counts[data_source] += 1
    acc.source_split_counts[source_split] += 1

    if missing_valid_mask:
        acc.samples_missing_valid_mask += 1

    valid_root_positions = np.asarray(root_positions[valid_mask], dtype=np.float32)
    acc.num_valid_root_frames += int(valid_root_positions.shape[0])
    if valid_root_positions.shape[0] > 0:
        acc.samples_with_valid_root += 1
        acc.root_positions_chunks.append(valid_root_positions)


def _scan_tar_file(tar_path: str, acc: SplitAccumulator) -> None:
    current_key: Optional[str] = None
    current_fields: Dict[str, Any] = {}

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue

            suffix, sample_key = _match_relevant_suffix(member.name)
            if suffix is None or sample_key is None:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()

            if current_key is None:
                current_key = sample_key
            elif sample_key != current_key:
                _finalize_sample(current_key, current_fields, acc)
                current_key = sample_key
                current_fields = {}

            current_fields[suffix] = _load_member_payload(member.name, payload)

    if current_key is not None:
        _finalize_sample(current_key, current_fields, acc)


def _safe_float(value: np.ndarray) -> float:
    return float(np.asarray(value).item())


def _summarize_axis(values: np.ndarray, bins: int) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "percentiles": {str(p): None for p in SUMMARY_PERCENTILES},
            "histogram": {"counts": [], "bin_edges": []},
        }

    hist_counts, bin_edges = np.histogram(values, bins=bins)
    return {
        "count": int(values.size),
        "min": _safe_float(values.min()),
        "max": _safe_float(values.max()),
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std()),
        "percentiles": {
            str(p): _safe_float(np.percentile(values, p)) for p in SUMMARY_PERCENTILES
        },
        "histogram": {
            "counts": hist_counts.astype(np.int64).tolist(),
            "bin_edges": bin_edges.astype(np.float64).tolist(),
        },
    }


def _summarize_root_positions(root_positions: np.ndarray, bins: int) -> Dict[str, Any]:
    if root_positions.size == 0:
        return {
            "xyz_axes": {
                "x_mm": _summarize_axis(np.array([], dtype=np.float32), bins),
                "y_mm": _summarize_axis(np.array([], dtype=np.float32), bins),
                "z_mm": _summarize_axis(np.array([], dtype=np.float32), bins),
            },
            "norm_mm": _summarize_axis(np.array([], dtype=np.float32), bins),
        }

    norms = np.linalg.norm(root_positions, axis=1)
    return {
        "xyz_axes": {
            "x_mm": _summarize_axis(root_positions[:, 0], bins),
            "y_mm": _summarize_axis(root_positions[:, 1], bins),
            "z_mm": _summarize_axis(root_positions[:, 2], bins),
        },
        "norm_mm": _summarize_axis(norms, bins),
    }


def _to_builtin_counter(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _split_output_paths(stat_root: str, rel_split_dir: str) -> Tuple[str, str, str]:
    split_output_dir = osp.join(stat_root, rel_split_dir)
    summary_path = osp.join(split_output_dir, "summary.json")
    root_positions_path = osp.join(split_output_dir, "root_positions.npy")
    return split_output_dir, summary_path, root_positions_path


def _load_existing_summary(summary_path: str, root_positions_path: str) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)
    summary["summary_path"] = summary_path
    summary["root_positions_path"] = root_positions_path
    return summary


def _scan_split(
    acc: SplitAccumulator, stat_root: str, bins: int, overwrite: bool = False
) -> Dict[str, Any]:
    split_output_dir, summary_path, root_positions_path = _split_output_paths(
        stat_root, acc.rel_split_dir
    )
    if (
        not overwrite
        and osp.isfile(summary_path)
        and osp.isfile(root_positions_path)
    ):
        print(f"[skip] {acc.rel_split_dir}", flush=True)
        return _load_existing_summary(summary_path, root_positions_path)

    print(f"[scan] {acc.rel_split_dir} ({len(acc.tar_paths)} tars)", flush=True)
    for tar_idx, tar_path in enumerate(acc.tar_paths, start=1):
        print(
            f"  [tar {tar_idx:03d}/{len(acc.tar_paths):03d}] {osp.basename(tar_path)}",
            flush=True,
        )
        try:
            _scan_tar_file(tar_path, acc)
        except Exception as ex:
            acc.tar_error_count += 1
            if len(acc.tar_errors) < 10:
                acc.tar_errors.append({"tar_path": tar_path, "error": str(ex)})

    if acc.root_positions_chunks:
        root_positions = np.concatenate(acc.root_positions_chunks, axis=0).astype(
            np.float32, copy=False
        )
    else:
        root_positions = np.zeros((0, 3), dtype=np.float32)

    os.makedirs(split_output_dir, exist_ok=True)
    np.save(root_positions_path, root_positions)

    clip_lengths = np.asarray(acc.clip_lengths, dtype=np.int64)
    summary: Dict[str, Any] = {
        "dataset_name": acc.dataset_name,
        "split_name": acc.split_name,
        "rel_split_dir": acc.rel_split_dir.replace(os.sep, "/"),
        "split_dir": acc.split_dir,
        "num_tars": len(acc.tar_paths),
        "num_samples": int(acc.num_samples),
        "num_frames": int(acc.num_frames),
        "num_valid_root_frames": int(acc.num_valid_root_frames),
        "valid_root_frame_ratio": (
            float(acc.num_valid_root_frames / acc.num_frames) if acc.num_frames > 0 else 0.0
        ),
        "samples_with_valid_root": int(acc.samples_with_valid_root),
        "samples_missing_joint_cam": int(acc.samples_missing_joint_cam),
        "samples_missing_valid_mask": int(acc.samples_missing_valid_mask),
        "tar_error_count": int(acc.tar_error_count),
        "tar_errors": acc.tar_errors,
        "handedness_counts": _to_builtin_counter(acc.handedness_counts),
        "data_source_counts": _to_builtin_counter(acc.data_source_counts),
        "source_split_counts": _to_builtin_counter(acc.source_split_counts),
        "clip_length": {
            "count": int(clip_lengths.size),
            "min": int(clip_lengths.min()) if clip_lengths.size > 0 else None,
            "max": int(clip_lengths.max()) if clip_lengths.size > 0 else None,
            "mean": float(clip_lengths.mean()) if clip_lengths.size > 0 else None,
            "median": float(np.median(clip_lengths)) if clip_lengths.size > 0 else None,
        },
        "root_position_summary": _summarize_root_positions(root_positions, bins),
        "root_positions_path": root_positions_path,
    }
    _write_json(summary_path, summary)
    summary["summary_path"] = summary_path
    print(
        f"[done] {acc.rel_split_dir} samples={summary['num_samples']} "
        f"valid_root_frames={summary['num_valid_root_frames']}",
        flush=True,
    )
    return summary


def _aggregate_dataset_summaries(split_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for split_summary in split_summaries:
        grouped[split_summary["dataset_name"]].append(split_summary)

    dataset_summaries: List[Dict[str, Any]] = []
    for dataset_name in sorted(grouped.keys()):
        splits = sorted(grouped[dataset_name], key=lambda item: item["rel_split_dir"])
        handedness_counts: Counter = Counter()
        data_source_counts: Counter = Counter()
        source_split_counts: Counter = Counter()
        num_tars = 0
        num_samples = 0
        num_frames = 0
        num_valid_root_frames = 0
        tar_error_count = 0
        for split in splits:
            handedness_counts.update(split["handedness_counts"])
            data_source_counts.update(split["data_source_counts"])
            source_split_counts.update(split["source_split_counts"])
            num_tars += int(split["num_tars"])
            num_samples += int(split["num_samples"])
            num_frames += int(split["num_frames"])
            num_valid_root_frames += int(split["num_valid_root_frames"])
            tar_error_count += int(split["tar_error_count"])

        dataset_summaries.append(
            {
                "dataset_name": dataset_name,
                "num_splits": len(splits),
                "num_tars": num_tars,
                "num_samples": num_samples,
                "num_frames": num_frames,
                "num_valid_root_frames": num_valid_root_frames,
                "valid_root_frame_ratio": (
                    float(num_valid_root_frames / num_frames) if num_frames > 0 else 0.0
                ),
                "tar_error_count": tar_error_count,
                "handedness_counts": _to_builtin_counter(handedness_counts),
                "data_source_counts": _to_builtin_counter(data_source_counts),
                "source_split_counts": _to_builtin_counter(source_split_counts),
                "splits": splits,
            }
        )
    return dataset_summaries


def _build_manifest(
    dataset_root: str,
    stat_root: str,
    split_summaries: Sequence[Dict[str, Any]],
    all_split_rel_dirs: Sequence[str],
    status: str,
) -> Dict[str, Any]:
    dataset_summaries = _aggregate_dataset_summaries(split_summaries)
    total_tars = sum(int(item["num_tars"]) for item in split_summaries)
    total_samples = sum(int(item["num_samples"]) for item in split_summaries)
    total_frames = sum(int(item["num_frames"]) for item in split_summaries)
    total_valid_root_frames = sum(
        int(item["num_valid_root_frames"]) for item in split_summaries
    )
    completed_split_dirs = [str(item["rel_split_dir"]) for item in split_summaries]
    completed_split_set = set(completed_split_dirs)
    pending_split_dirs = [
        rel_split_dir
        for rel_split_dir in all_split_rel_dirs
        if rel_split_dir not in completed_split_set
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_root": dataset_root,
        "stat_root": stat_root,
        "num_datasets": len(dataset_summaries),
        "num_splits_discovered": len(all_split_rel_dirs),
        "num_splits_completed": len(split_summaries),
        "num_splits_pending": len(pending_split_dirs),
        "completed_split_dirs": completed_split_dirs,
        "pending_split_dirs": pending_split_dirs,
        "num_tars": int(total_tars),
        "num_samples": int(total_samples),
        "num_frames": int(total_frames),
        "num_valid_root_frames": int(total_valid_root_frames),
        "valid_root_frame_ratio": (
            float(total_valid_root_frames / total_frames) if total_frames > 0 else 0.0
        ),
        "datasets": dataset_summaries,
    }


def main() -> None:
    args = parse_args()
    dataset_root = osp.abspath(args.dataset_root)
    stat_root = osp.abspath(args.stat_root)
    manifest_path = osp.abspath(args.manifest_path)

    if not osp.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    split_dirs = _discover_split_dirs(dataset_root)
    all_split_rel_dirs = [osp.relpath(split_dir, dataset_root).replace(os.sep, "/") for split_dir in split_dirs]
    split_summaries: List[Dict[str, Any]] = []

    for split_dir in split_dirs:
        acc = _build_split_accumulator(dataset_root, split_dir)
        split_summary = _scan_split(
            acc, stat_root, args.hist_bins, overwrite=args.overwrite
        )
        split_summaries.append(split_summary)
        manifest = _build_manifest(
            dataset_root=dataset_root,
            stat_root=stat_root,
            split_summaries=split_summaries,
            all_split_rel_dirs=all_split_rel_dirs,
            status="running",
        )
        _write_json(manifest_path, manifest)

    manifest = _build_manifest(
        dataset_root=dataset_root,
        stat_root=stat_root,
        split_summaries=split_summaries,
        all_split_rel_dirs=all_split_rel_dirs,
        status="completed",
    )
    _write_json(manifest_path, manifest)

    print(f"Saved statistics to {stat_root}", flush=True)
    print(f"Saved manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
