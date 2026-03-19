import json
import math
import os
import os.path as osp
import pickle
import multiprocessing
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import webdataset as wds
from pycocotools.coco import COCO
from tqdm import tqdm


COCO_ROOT = os.environ.get("COCO_ROOT", "/mnt/qnap/data/datasets/coco2017")
SPLIT = os.environ.get("SPLIT", "train")  # train val
assert SPLIT in {"train", "val"}, f"Unsupported SPLIT={SPLIT}"
OUTPUT_PATTERN = (
    f"coco_wholebody_{SPLIT}_wds_output/coco_wholebody_{SPLIT}-worker{{worker_id}}-%06d.tar"
)
MAX_COUNT = 100000
MAX_SIZE = 3 * 1024 * 1024 * 1024
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEBUG_MAX_HANDS = int(os.environ.get("DEBUG_MAX_HANDS", "0"))

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

SPLIT_TO_ANN = {
    "train": "coco_wholebody_train_v1.0.json",
    "val": "coco_wholebody_val_v1.0.json",
}

IMAGE_SUBDIR_CANDIDATES = {
    "train": ["train/images", "train2017"],
    "val": ["valid/images", "val2017"],
}

os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)


def resolve_image_relpath(file_name: str) -> str:
    for subdir in IMAGE_SUBDIR_CANDIDATES[SPLIT]:
        full_path = osp.join(COCO_ROOT, subdir, file_name)
        if osp.isfile(full_path):
            return osp.join(subdir, file_name)
    raise FileNotFoundError(f"Failed to resolve image path for {file_name}")


def get_virtual_intrinsics(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    focal_scalar = float(max(width, height))
    focal = np.array([focal_scalar, focal_scalar], dtype=np.float32)
    princpt = np.array([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)
    return focal, princpt


def xywh_to_xyxy(box_xywh: List[float], width: int, height: int) -> Optional[np.ndarray]:
    if box_xywh is None or len(box_xywh) != 4:
        return None
    x, y, w, h = [float(v) for v in box_xywh]
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width - 1), x + w)
    y2 = min(float(height - 1), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_from_valid_keypoints(joint_img: np.ndarray, joint_valid: np.ndarray) -> Optional[np.ndarray]:
    valid_mask = joint_valid > 0.5
    if not np.any(valid_mask):
        return None
    pts = joint_img[valid_mask]
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def build_tasks(coco: COCO, image_map: Dict[int, Dict]) -> List[Tuple[int, str]]:
    tasks: List[Tuple[int, str]] = []
    for ann_id in tqdm(coco.anns.keys(), ncols=80, desc="Building COCO-WB tasks"):
        ann = coco.anns[ann_id]
        if ann.get("iscrowd", 0):
            continue
        image_info = image_map.get(ann["image_id"])
        if image_info is None:
            continue

        for handedness in ("left", "right"):
            if not bool(ann.get(f"{handedness}hand_valid", False)):
                continue
            tasks.append((ann_id, handedness))
            if DEBUG_MAX_HANDS > 0 and len(tasks) >= DEBUG_MAX_HANDS:
                return tasks[:DEBUG_MAX_HANDS]

    return tasks


def process_single_annot(coco: COCO, image_map: Dict[int, Dict], ann_id: int, handedness: str):
    ann = coco.anns[ann_id]
    image_info = image_map[ann["image_id"]]
    width = int(image_info["width"])
    height = int(image_info["height"])

    image_relpath = resolve_image_relpath(image_info["file_name"])
    image_abspath = osp.join(COCO_ROOT, image_relpath)
    img = cv2.imread(image_abspath)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_abspath}")

    success, encoded_img_arr = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise RuntimeError(f"Failed to encode image: {image_abspath}")
    img_bytes = encoded_img_arr.tobytes()

    kpts = np.asarray(ann[f"{handedness}hand_kpts"], dtype=np.float32).reshape(21, 3)
    joint_img = kpts[:, :2].astype(np.float32)
    joint_2d_valid = (kpts[:, 2] > 0).astype(np.float32)
    if np.sum(joint_2d_valid) <= 0:
        return None

    hand_bbox = xywh_to_xyxy(ann.get(f"{handedness}hand_box"), width, height)
    bbox_source = "annotation"
    if hand_bbox is None:
        hand_bbox = bbox_from_valid_keypoints(joint_img, joint_2d_valid)
        bbox_source = "keypoints"
    if hand_bbox is None:
        return None

    joint_hand_bbox = (joint_img - hand_bbox[None, :2]).astype(np.float32)
    joint_cam = np.zeros((21, 3), dtype=np.float32)
    joint_rel = np.zeros((21, 3), dtype=np.float32)
    joint_3d_valid = np.zeros((21,), dtype=np.float32)
    joint_valid = joint_2d_valid.copy()
    mano_pose = np.zeros((48,), dtype=np.float32)
    mano_shape = np.zeros((10,), dtype=np.float32)
    has_mano = np.float32(0.0)
    mano_valid = np.float32(0.0)
    focal, princpt = get_virtual_intrinsics(width, height)
    has_intr = np.float32(1.0)
    timestamp = np.float32(0.0)

    source_index = {
        "image_id": int(ann["image_id"]),
        "ann_id": int(ann["id"]),
        "handedness": handedness,
    }
    additional_desc = {
        "dataset": "coco_wholebody",
        "image_id": int(ann["image_id"]),
        "ann_id": int(ann["id"]),
        "is_2d_only": True,
        "bbox_source": bbox_source,
        "orig_file_name": image_info["file_name"],
    }

    return {
        "img_path": image_relpath,
        "img_bytes": img_bytes,
        "handedness": handedness,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_hand_bbox": joint_hand_bbox,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_2d_valid": joint_2d_valid,
        "joint_3d_valid": joint_3d_valid,
        "joint_valid": joint_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "has_mano": has_mano,
        "mano_valid": mano_valid,
        "has_intr": has_intr,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
        "source_index": source_index,
        "additional_desc": additional_desc,
    }


def process_task_batch(batch_tasks: List[Tuple[int, str]], worker_id: int):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    ann_path = osp.join(COCO_ROOT, "annotations", SPLIT_TO_ANN[SPLIT])
    coco = COCO(ann_path)
    image_map = coco.imgs

    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)
    processed_count = 0
    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for local_idx, (ann_id, handedness) in enumerate(batch_tasks):
            try:
                frame = process_single_annot(coco, image_map, ann_id, handedness)
            except Exception as ex:
                print(f"[Worker {worker_id}] ann_id={ann_id} hand={handedness} error: {ex}")
                continue
            if frame is None:
                continue

            key_str = f"{frame['source_index']['image_id']}_{ann_id}_{handedness}_{SPLIT}_coco_wb"
            wds_sample = {
                "__key__": key_str,
                "imgs_path.json": json.dumps([frame["img_path"]]),
                "img_bytes.pickle": pickle.dumps([frame["img_bytes"]]),
                "handedness.json": json.dumps(frame["handedness"]),
                "additional_desc.json": json.dumps([frame["additional_desc"]]),
                "data_source.json": json.dumps("coco_wholebody"),
                "source_split.json": json.dumps(SPLIT),
                "source_index.json": json.dumps([frame["source_index"]]),
                "intr_type.json": json.dumps("fixed_virtual"),
            }
            for key in NUMPY_KEYS:
                wds_sample[f"{key}.npy"] = np.stack([frame[key]])
            sink.write(wds_sample)
            processed_count += 1

    return processed_count


def main():
    ann_path = osp.join(COCO_ROOT, "annotations", SPLIT_TO_ANN[SPLIT])
    print(f"Loading COCO-WholeBody annotations from {ann_path}")
    coco = COCO(ann_path)
    image_map = coco.imgs

    tasks = build_tasks(coco, image_map)
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No valid hand samples found.")
        return

    worker_count = max(NUM_WORKERS, 1)
    chunk_size = math.ceil(total_tasks / worker_count)
    chunks = [tasks[i:i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    print(f"Total Hand Samples: {total_tasks}")
    print(f"Starting {worker_count} workers processing ~{chunk_size} samples each...")

    process_args = [(chunks[i], i) for i in range(len(chunks))]
    if worker_count <= 1:
        results = [process_task_batch(chunks[0], 0)]
    else:
        with multiprocessing.Pool(processes=worker_count) as pool:
            results = pool.starmap(process_task_batch, process_args)

    print(f"All done! Total samples processed: {sum(results)}")


if __name__ == "__main__":
    main()
