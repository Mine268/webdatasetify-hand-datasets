import json
import math
import multiprocessing
import os
import os.path as osp
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import webdataset as wds

try:
    from .utils import RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER, reorder_joints
except ImportError:
    from utils import RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER, reorder_joints


RHD_ROOT = os.environ.get("RHD_ROOT", "/mnt/qnap/data/datasets/rhd/RHD_published_v2")
SPLIT = os.environ.get("SPLIT", "train")
assert SPLIT in {"train", "evaluation"}, f"Unsupported SPLIT={SPLIT}"
SPLIT_TO_DIR = {
    "train": "training",
    "evaluation": "evaluation",
}
OUTPUT_PATTERN = f"rhd_{SPLIT}_wds_output/rhd_{SPLIT}-worker{{worker_id}}-%06d.tar"
MAX_COUNT = 100000
MAX_SIZE = 3 * 1024 * 1024 * 1024
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEBUG_MAX_SAMPLES = int(os.environ.get("DEBUG_MAX_SAMPLES", "0"))

NUMPY_KEYS = [
    "hand_bbox",
    "joint_img",
    "joint_hand_bbox",
    "joint_cam",
    "joint_rel",
    "joint_2d_valid",
    "joint_3d_valid",
    "joint_valid",
    "mano_pose",
    "mano_shape",
    "has_mano",
    "mano_valid",
    "has_intr",
    "timestamp",
    "focal",
    "princpt",
]

HAND_TO_INDEX = {
    "left": slice(0, 21),
    "right": slice(21, 42),
}
HAND_TO_MASK_IDS = {
    "left": tuple(range(2, 18)),
    "right": tuple(range(18, 34)),
}

os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)

_ANNO_ALL = None


def get_split_dir() -> str:
    return SPLIT_TO_DIR[SPLIT]


def get_ann_path() -> str:
    split_dir = get_split_dir()
    return osp.join(RHD_ROOT, split_dir, f"anno_{split_dir}.pickle")


def load_annotations() -> Dict[int, Dict[str, np.ndarray]]:
    global _ANNO_ALL
    if _ANNO_ALL is None:
        with open(get_ann_path(), "rb") as fp:
            _ANNO_ALL = pickle.load(fp)
    return _ANNO_ALL


def get_color_relpath(sample_id: int) -> str:
    return osp.join(get_split_dir(), "color", f"{sample_id:05d}.png")


def get_mask_abspath(sample_id: int) -> str:
    return osp.join(RHD_ROOT, get_split_dir(), "mask", f"{sample_id:05d}.png")


def get_color_abspath(sample_id: int) -> str:
    return osp.join(RHD_ROOT, get_color_relpath(sample_id))


def _mask_to_bbox(mask: np.ndarray, handedness: str) -> Optional[np.ndarray]:
    hand_mask = np.isin(mask, HAND_TO_MASK_IDS[handedness])
    ys, xs = np.where(hand_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _joints_to_bbox(
    joint_img: np.ndarray,
    joint_valid: np.ndarray,
    image_width: int,
    image_height: int,
    expand_ratio: float = 1.2,
) -> np.ndarray:
    pts = joint_img[joint_valid > 0.5]
    if pts.shape[0] == 0:
        raise ValueError("No valid 2D joints available for bbox construction")

    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.5 * expand_ratio, 1.0)
    half_h = max((y2 - y1) * 0.5 * expand_ratio, 1.0)

    bbox = np.array(
        [
            np.clip(cx - half_w, 0.0, image_width - 1.0),
            np.clip(cy - half_h, 0.0, image_height - 1.0),
            np.clip(cx + half_w, 0.0, image_width - 1.0),
            np.clip(cy + half_h, 0.0, image_height - 1.0),
        ],
        dtype=np.float32,
    )
    return bbox


def _expand_bbox_xyxy(
    bbox: np.ndarray,
    image_width: int,
    image_height: int,
    expand_ratio: float = 1.2,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.5 * expand_ratio, 1.0)
    half_h = max((y2 - y1) * 0.5 * expand_ratio, 1.0)
    out = np.array(
        [
            np.clip(cx - half_w, 0.0, image_width - 1.0),
            np.clip(cy - half_h, 0.0, image_height - 1.0),
            np.clip(cx + half_w, 0.0, image_width - 1.0),
            np.clip(cy + half_h, 0.0, image_height - 1.0),
        ],
        dtype=np.float32,
    )
    return out


def _has_hand(ann: Dict[str, np.ndarray], handedness: str) -> bool:
    hand_slice = HAND_TO_INDEX[handedness]
    joint_2d_valid = ann["uv_vis"][hand_slice, 2] > 0.5
    return bool(np.any(joint_2d_valid))


def build_tasks() -> List[Tuple[int, str]]:
    annotations = load_annotations()
    tasks: List[Tuple[int, str]] = []
    for sample_id in sorted(annotations.keys()):
        ann = annotations[sample_id]
        for handedness in ("left", "right"):
            if _has_hand(ann, handedness):
                tasks.append((sample_id, handedness))
                if DEBUG_MAX_SAMPLES > 0 and len(tasks) >= DEBUG_MAX_SAMPLES:
                    return tasks[:DEBUG_MAX_SAMPLES]
    return tasks


def process_single_sample(sample_id: int, handedness: str) -> Dict[str, object]:
    annotations = load_annotations()
    ann = annotations[sample_id]
    hand_slice = HAND_TO_INDEX[handedness]

    image_path = get_color_abspath(sample_id)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_height, image_width = image.shape[:2]

    mask = cv2.imread(get_mask_abspath(sample_id), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {get_mask_abspath(sample_id)}")

    success, encoded_img = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise RuntimeError(f"Failed to encode image: {image_path}")

    joint_cam = np.asarray(ann["xyz"][hand_slice], dtype=np.float32) * 1e3
    joint_img = np.asarray(ann["uv_vis"][hand_slice, :2], dtype=np.float32)
    joint_2d_valid = np.asarray(ann["uv_vis"][hand_slice, 2] > 0.5, dtype=np.float32)
    joint_3d_valid = np.ones((21,), dtype=np.float32)
    joint_valid = joint_3d_valid.copy()
    K = np.asarray(ann["K"], dtype=np.float32)

    bbox = _mask_to_bbox(mask, handedness)
    bbox_source = "mask"
    if bbox is None:
        bbox = _joints_to_bbox(joint_img, joint_2d_valid, image_width, image_height)
        bbox_source = "joint_visible"
    else:
        bbox = _expand_bbox_xyxy(bbox, image_width, image_height)

    joint_hand_bbox = joint_img - bbox[None, :2]
    joint_rel = joint_cam - joint_cam[:1]

    joint_img = reorder_joints(joint_img, RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_hand_bbox = reorder_joints(
        joint_hand_bbox, RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )
    joint_cam = reorder_joints(joint_cam, RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_rel = reorder_joints(joint_rel, RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_2d_valid = reorder_joints(
        joint_2d_valid[:, None], RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0]
    joint_3d_valid = reorder_joints(
        joint_3d_valid[:, None], RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0]
    joint_valid = reorder_joints(
        joint_valid[:, None], RHD_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0]

    focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
    princpt = np.array([K[0, 2], K[1, 2]], dtype=np.float32)
    visible_joint_count = int(np.sum(joint_2d_valid > 0.5))

    return {
        "img_path": get_color_relpath(sample_id),
        "img_bytes": encoded_img.tobytes(),
        "handedness": handedness,
        "hand_bbox": bbox.astype(np.float32),
        "joint_img": joint_img.astype(np.float32),
        "joint_hand_bbox": joint_hand_bbox.astype(np.float32),
        "joint_cam": joint_cam.astype(np.float32),
        "joint_rel": joint_rel.astype(np.float32),
        "joint_2d_valid": joint_2d_valid.astype(np.float32),
        "joint_3d_valid": joint_3d_valid.astype(np.float32),
        "joint_valid": joint_valid.astype(np.float32),
        "mano_pose": np.zeros((48,), dtype=np.float32),
        "mano_shape": np.zeros((10,), dtype=np.float32),
        "has_mano": np.float32(0.0),
        "mano_valid": np.float32(0.0),
        "has_intr": np.float32(1.0),
        "timestamp": np.float32(0.0),
        "focal": focal,
        "princpt": princpt,
        "source_index": {
            "sample_id": int(sample_id),
            "handedness": handedness,
        },
        "additional_desc": {
            "dataset": "rhd",
            "sample_id": int(sample_id),
            "bbox_source": bbox_source,
            "visible_joint_count": visible_joint_count,
            "image_size": [int(image_width), int(image_height)],
        },
    }


def process_batch(tasks: List[Tuple[int, str]], worker_id: int) -> int:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)
    processed_count = 0

    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for sample_id, handedness in tasks:
            try:
                frame = process_single_sample(sample_id, handedness)
            except Exception as ex:
                print(
                    f"[Worker {worker_id}] sample_id={sample_id} hand={handedness} error: {ex}"
                )
                continue

            key_str = f"{sample_id:05d}_{handedness}_{SPLIT}_rhd"
            wds_sample = {
                "__key__": key_str,
                "imgs_path.json": json.dumps([frame["img_path"]]),
                "img_bytes.pickle": pickle.dumps([frame["img_bytes"]]),
                "handedness.json": json.dumps(frame["handedness"]),
                "additional_desc.json": json.dumps([frame["additional_desc"]]),
                "data_source.json": json.dumps("rhd"),
                "source_split.json": json.dumps(SPLIT),
                "source_index.json": json.dumps([frame["source_index"]]),
                "intr_type.json": json.dumps("synthetic"),
            }
            for key in NUMPY_KEYS:
                wds_sample[f"{key}.npy"] = np.stack([frame[key]])
            sink.write(wds_sample)
            processed_count += 1

    return processed_count


def main() -> None:
    print(f"Loading RHD annotations for split={SPLIT} ...")
    load_annotations()

    tasks = build_tasks()
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No valid RHD hand samples found.")
        return

    worker_count = max(NUM_WORKERS, 1)
    chunk_size = math.ceil(total_tasks / worker_count)
    chunks = [tasks[i : i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    print(f"Total RHD hand samples: {total_tasks}")
    print(f"Starting {len(chunks)} workers processing ~{chunk_size} samples each ...")

    process_args = [(chunk, worker_id) for worker_id, chunk in enumerate(chunks)]
    if len(chunks) == 1:
        results = [process_batch(chunks[0], 0)]
    else:
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            results = pool.starmap(process_batch, process_args)

    print(f"All done! Total RHD clips processed: {sum(results)}")


if __name__ == "__main__":
    main()
