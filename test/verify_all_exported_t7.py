import copy
import glob
import argparse
import json
import os
import os.path as osp
import random
import sys
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

ROOT_CANDIDATES = [
    "/mnt/qnap/data/datasets/webdatasets2",
    "/mnt/qnap/data/datasets/webdataset2",
]
DEFAULT_OUTPUT_ROOT = "temp/verify_exported_t7"
DEFAULT_NUM_FRAMES = 7
DEFAULT_SAMPLES_PER_SPLIT = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_SHUFFLE_BUFFER = 0
DEVICE = torch.device("cpu")


def _resolve_dataset_root() -> str:
    for path in ROOT_CANDIDATES:
        if osp.isdir(path):
            return path
    raise FileNotFoundError(f"Failed to find any dataset root from {ROOT_CANDIDATES}")


def _discover_split_dirs(dataset_root: str) -> List[str]:
    split_dirs: List[str] = []
    for current_root, _dirs, files in os.walk(dataset_root):
        if any(name.endswith(".tar") for name in files):
            split_dirs.append(current_root)
    split_dirs.sort()
    return split_dirs


def _infer_data_source(split_dir: str) -> str:
    parts = osp.normpath(split_dir).split(os.sep)
    if "InterHand2.6M" in parts:
        return "ih26m"
    if "DexYCB" in parts:
        return "dexycb"
    if "HO3D_v3" in parts:
        return "ho3d"
    if "HOT3D" in parts:
        return "hot3d"
    if "COCO-WholeBody" in parts:
        return "coco_wholebody"
    if "FreiHAND" in parts:
        return "freihand"
    if "RHD" in parts:
        return "rhd"
    if "MTC" in parts:
        return "mtc"
    if "AssemblyHands" in parts:
        return "assemblyhands"
    return "unknown"


def _relative_split_name(dataset_root: str, split_dir: str) -> str:
    rel = osp.relpath(split_dir, dataset_root)
    return rel.replace(os.sep, "__")


def _terminal_split_name(split_dir: str) -> str:
    return osp.basename(split_dir)


def _seed_for(rel_name: str, sample_index: int) -> int:
    return abs(hash((rel_name, sample_index, "verify_t7"))) % (2 ** 31)


def _write_summary(summary: Dict) -> None:
    output_root = summary["output_root"]
    os.makedirs(output_root, exist_ok=True)
    with open(osp.join(output_root, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def _build_overview(output_dir: str, samples: List[Dict]) -> None:
    rows: List[np.ndarray] = []
    for sample in samples:
        sample_dir = sample["sample_dir"]
        img_paths = [
            osp.join(sample_dir, "origin", "hand_bbox-joint_img.png"),
            osp.join(sample_dir, "origin", "joint_cam_reproj.png"),
            osp.join(sample_dir, "processed", "bbox-joint_img.png"),
        ]
        imgs = [cv2.imread(path) for path in img_paths]
        imgs = [img for img in imgs if img is not None]
        if len(imgs) == 0:
            continue
        row = np.concatenate(imgs, axis=1)
        header = np.zeros((40, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"sample={sample['sample_index']:02d} frame={sample['frame_index']} key={sample['sample_key']}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate([header, row], axis=0))

    if len(rows) == 0:
        return

    max_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_width:
            pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded_rows.append(row)

    overview = np.concatenate(padded_rows, axis=0)
    cv2.imwrite(osp.join(output_dir, "overview.jpg"), overview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify exported WebDatasets with old verify logic")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--samples-per-split", type=int, default=DEFAULT_SAMPLES_PER_SPLIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.append(osp.abspath("src"))

    from dataloader_v2 import get_dataloader
    from preprocess_v2 import preprocess_batch
    from verify_wds_v2 import verify_origin_sample, verify_processed_sample

    dataset_root = args.dataset_root or _resolve_dataset_root()
    split_dirs = _discover_split_dirs(dataset_root)
    os.makedirs(args.output_root, exist_ok=True)

    summary: Dict[str, object] = {
        "dataset_root": dataset_root,
        "output_root": args.output_root,
        "num_frames": args.num_frames,
        "samples_per_split": args.samples_per_split,
        "splits": [],
    }

    for split_dir in split_dirs:
        tar_pattern = osp.join(split_dir, "*.tar")
        urls = sorted(glob.glob(tar_pattern))
        rel_name = _relative_split_name(dataset_root, split_dir)
        output_dir = osp.join(args.output_root, rel_name)
        os.makedirs(output_dir, exist_ok=True)

        split_summary: Dict[str, object] = {
            "split_dir": split_dir,
            "tar_pattern": tar_pattern,
            "num_tars": len(urls),
            "data_source": _infer_data_source(split_dir),
            "source_split": _terminal_split_name(split_dir),
            "status": "pending",
            "samples": [],
        }

        try:
            loader = get_dataloader(
                urls,
                num_frames=args.num_frames,
                stride=1,
                batch_size=args.batch_size,
                num_workers=0,
                shardshuffle=0,
                shuffle_buffer=args.shuffle_buffer,
                default_data_source=split_summary["data_source"],
                default_source_split=split_summary["source_split"],
            )
            batch_origin = next(iter(loader))
        except StopIteration:
            split_summary["status"] = "skipped_no_t_sample"
            summary["splits"].append(split_summary)
            _write_summary(summary)
            continue
        except Exception as ex:
            split_summary["status"] = "failed_loader"
            split_summary["error"] = str(ex)
            summary["splits"].append(split_summary)
            _write_summary(summary)
            continue

        available = len(batch_origin["__key__"])
        sample_count = min(args.samples_per_split, available)
        split_summary["available_batch_samples"] = available

        for sample_index in range(sample_count):
            rng = random.Random(_seed_for(rel_name, sample_index))
            frame_index = rng.randrange(args.num_frames)
            sample_dir = osp.join(output_dir, f"sample_{sample_index:02d}")
            origin_dir = osp.join(sample_dir, "origin")
            processed_dir = osp.join(sample_dir, "processed")
            os.makedirs(sample_dir, exist_ok=True)

            verify_origin_sample(batch_origin, origin_dir, sample_index, frame_index)

            batch_for_preprocess = copy.deepcopy(batch_origin)
            batch_processed, trans_2d_mat = preprocess_batch(
                batch_for_preprocess,
                (256, 256),
                1.1,
                (0.9, 1.1),
                (0.8, 1.1),
                (float(torch.pi) / 12.0),
                False,
                DEVICE,
            )
            verify_processed_sample(
                batch_processed,
                batch_origin,
                trans_2d_mat,
                processed_dir,
                sample_index,
                frame_index,
            )

            split_summary["samples"].append(
                {
                    "sample_index": sample_index,
                    "frame_index": frame_index,
                    "sample_dir": sample_dir,
                    "sample_key": batch_origin["__key__"][sample_index],
                }
            )

        split_summary["status"] = (
            "ok" if sample_count == args.samples_per_split else "ok_insufficient_batch"
        )
        _build_overview(output_dir, split_summary["samples"])
        summary["splits"].append(split_summary)
        _write_summary(summary)

    print(f"Saved verification outputs to {args.output_root}")


if __name__ == "__main__":
    main()
