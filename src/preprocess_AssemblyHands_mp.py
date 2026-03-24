import json
import math
import multiprocessing
import os
import os.path as osp
from typing import Dict, List, Tuple

import cv2
import numpy as np
import webdataset as wds

try:
    from .utils import IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER, reorder_joints
except ImportError:
    from utils import IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER, reorder_joints


AH_ROOT = os.environ.get("ASSEMBLYHANDS_ROOT", "/mnt/qnap/data/datasets/AssemblyHands")
SPLIT = os.environ.get("SPLIT", "train")
assert SPLIT == "train", (
    f"AssemblyHands export currently supports SPLIT=train only; got SPLIT={SPLIT}"
)
SPLIT_TO_ANN_DIR = {
    "train": "train",
}
OUTPUT_PATTERN = (
    f"assemblyhands_{SPLIT}_wds_output/assemblyhands_{SPLIT}-worker{{worker_id}}-%06d.tar"
)
MAX_COUNT = 100000
MAX_SIZE = 3 * 1024 * 1024 * 1024
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEBUG_MAX_CLIPS = int(os.environ.get("DEBUG_MAX_CLIPS", os.environ.get("DEBUG_MAX_SAMPLES", "0")))
BBOX_EXPAND_RATIO = float(os.environ.get("BBOX_EXPAND_RATIO", "1.2"))
FRAME_GAP = int(os.environ.get("FRAME_GAP", "2"))
MAX_CLIP_LEN = int(os.environ.get("MAX_CLIP_LEN", "256"))

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
    "right": slice(0, 21),
    "left": slice(21, 42),
}

os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)

_EGO_DATA = None
_JOINT_3D = None
_CALIB = None


def get_ann_dir() -> str:
    return osp.join(AH_ROOT, "annotations", SPLIT_TO_ANN_DIR[SPLIT])


def _ego_data_path() -> str:
    return osp.join(get_ann_dir(), f"assemblyhands_{SPLIT}_ego_data_v1-1.json")


def _joint_3d_path() -> str:
    return osp.join(get_ann_dir(), f"assemblyhands_{SPLIT}_joint_3d_v1-1.json")


def _ego_calib_path() -> str:
    return osp.join(get_ann_dir(), f"assemblyhands_{SPLIT}_ego_calib_v1-1.json")


def load_ego_data() -> Dict:
    global _EGO_DATA
    if _EGO_DATA is None:
        with open(_ego_data_path(), "r") as fp:
            _EGO_DATA = json.load(fp)
    return _EGO_DATA


def load_joint_3d() -> Dict:
    global _JOINT_3D
    if _JOINT_3D is None:
        with open(_joint_3d_path(), "r") as fp:
            _JOINT_3D = json.load(fp)["annotations"]
    return _JOINT_3D


def load_calib() -> Dict:
    global _CALIB
    if _CALIB is None:
        with open(_ego_calib_path(), "r") as fp:
            _CALIB = json.load(fp)["calibration"]
    return _CALIB

def _resolve_file_name(file_name: str) -> str:
    path1 = osp.join(AH_ROOT, file_name)
    if osp.isfile(path1):
        return file_name

    raise FileNotFoundError(f"Failed to resolve AssemblyHands image path: {file_name}")


def _resolve_camera_name(camera_name: str, calib_seq: Dict) -> str:
    intrinsics = calib_seq["intrinsics"]
    if camera_name in intrinsics:
        return camera_name
    for candidate in intrinsics.keys():
        if candidate.startswith(camera_name):
            return candidate
    raise KeyError(camera_name)


def _expand_bbox_xyxy(
    bbox: np.ndarray,
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.5 * expand_ratio, 1.0)
    half_h = max((y2 - y1) * 0.5 * expand_ratio, 1.0)
    return np.array(
        [
            np.clip(cx - half_w, 0.0, image_width - 1.0),
            np.clip(cy - half_h, 0.0, image_height - 1.0),
            np.clip(cx + half_w, 0.0, image_width - 1.0),
            np.clip(cy + half_h, 0.0, image_height - 1.0),
        ],
        dtype=np.float32,
    )


def build_sequences() -> List[List[Tuple[int, str, int]]]:
    ego_data = load_ego_data()
    images = {img["id"]: img for img in ego_data["images"]}
    grouped: Dict[Tuple[str, str, str], List[Tuple[int, str, int]]] = {}
    for ann_idx, ann in enumerate(ego_data["annotations"]):
        image_info = images[ann["image_id"]]
        for handedness in ("right", "left"):
            if ann["bbox"][handedness] is None:
                continue
            group_key = (image_info["seq_name"], image_info["camera"], handedness)
            grouped.setdefault(group_key, []).append(
                (ann_idx, handedness, int(image_info["frame_idx"]))
            )

    clips: List[List[Tuple[int, str, int]]] = []
    for items in grouped.values():
        items.sort(key=lambda x: x[2])
        start = 0
        for i in range(1, len(items) + 1):
            is_break = i == len(items) or (items[i][2] - items[i - 1][2]) != FRAME_GAP
            if not is_break:
                continue

            segment = items[start:i]
            if MAX_CLIP_LEN > 0 and len(segment) > MAX_CLIP_LEN:
                for j in range(0, len(segment), MAX_CLIP_LEN):
                    clips.append(segment[j : j + MAX_CLIP_LEN])
            else:
                clips.append(segment)
            start = i

    if DEBUG_MAX_CLIPS > 0:
        return clips[:DEBUG_MAX_CLIPS]
    return clips


def process_single_sample(ann_idx: int, handedness: str) -> Dict[str, object]:
    ego_data = load_ego_data()
    joint_3d_all = load_joint_3d()
    calib_all = load_calib()

    ann = ego_data["annotations"][ann_idx]
    image_info = ego_data["images"][ann["image_id"]]
    file_name = _resolve_file_name(image_info["file_name"])
    image_abspath = osp.join(AH_ROOT, file_name)
    image = cv2.imread(image_abspath)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_abspath}")
    image_height, image_width = image.shape[:2]

    success, encoded_img = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise RuntimeError(f"Failed to encode image: {image_abspath}")

    hand_slice = HAND_TO_INDEX[handedness]
    keypoints_2d = np.asarray(ann["keypoints"], dtype=np.float32).reshape(42, 3)
    joint_2d_valid_ann = np.asarray(ann["joint_valid"], dtype=np.float32)
    frame_str = f"{int(ann['frame_id']):06d}"
    seq_name = image_info["seq_name"]
    camera_name = image_info["camera"]
    calib_camera_name = _resolve_camera_name(camera_name, calib_all[seq_name])

    joint_world = np.asarray(
        joint_3d_all[seq_name][frame_str]["world_coord"], dtype=np.float32
    ).reshape(42, 3)
    joint_3d_valid_ann = np.asarray(
        joint_3d_all[seq_name][frame_str]["joint_valid"], dtype=np.float32
    )

    extr = np.asarray(
        calib_all[seq_name]["extrinsics"][frame_str][calib_camera_name], dtype=np.float32
    )
    K = np.asarray(calib_all[seq_name]["intrinsics"][calib_camera_name], dtype=np.float32)
    R = extr[:, :3]
    t = extr[:, 3:]
    joint_cam = (R @ joint_world.T + t).T
    joint_img_annot = keypoints_2d[hand_slice, :2].astype(np.float32)
    joint_img_proj_h = joint_cam @ K.T
    joint_img_proj = (joint_img_proj_h[:, :2] / joint_img_proj_h[:, 2:]).astype(np.float32)
    joint_2d_valid = joint_2d_valid_ann[hand_slice].astype(np.float32)
    joint_3d_valid = joint_3d_valid_ann[hand_slice]
    joint_valid = (joint_2d_valid * joint_3d_valid).astype(np.float32)

    annot_range = float(np.ptp(joint_img_annot))
    annot_abs_max = float(np.max(np.abs(joint_img_annot)))
    use_projected_2d = annot_range < 5.0 or annot_abs_max > 1e4
    joint_img = joint_img_proj if use_projected_2d else joint_img_annot

    bbox = np.asarray(ann["bbox"][handedness], dtype=np.float32)
    bbox = _expand_bbox_xyxy(bbox, image_width, image_height, BBOX_EXPAND_RATIO)
    joint_hand_bbox = joint_img - bbox[None, :2]
    joint_cam = joint_cam[hand_slice].astype(np.float32)
    joint_rel = (joint_cam - joint_cam[-1:]).astype(np.float32)

    joint_img = reorder_joints(joint_img, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_hand_bbox = reorder_joints(
        joint_hand_bbox, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
    )
    joint_cam = reorder_joints(joint_cam, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_rel = reorder_joints(joint_rel, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_2d_valid = reorder_joints(
        joint_2d_valid[:, None], IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0].astype(np.float32)
    joint_3d_valid = reorder_joints(
        joint_3d_valid[:, None], IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0].astype(np.float32)
    joint_valid = reorder_joints(
        joint_valid[:, None], IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0].astype(np.float32)

    focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
    princpt = np.array([K[0, 2], K[1, 2]], dtype=np.float32)
    timestamp = np.float32(float(image_info["timestamp"]) * 1000.0)

    additional_desc = {
        "dataset": "assemblyhands",
        "camera": camera_name,
        "calib_camera": calib_camera_name,
        "seq_name": seq_name,
        "frame_idx": int(image_info["frame_idx"]),
        "frame_id": int(ann["frame_id"]),
        "image_id": int(image_info["id"]),
        "rectified_ego": True,
        "bbox_source": "annotation",
        "joint_img_source": "projected_3d" if use_projected_2d else "annotation_2d",
    }

    source_index = {
        "image_id": int(image_info["id"]),
        "ann_id": int(ann["id"]),
        "seq_name": seq_name,
        "camera": camera_name,
        "calib_camera": calib_camera_name,
        "frame_idx": int(image_info["frame_idx"]),
        "frame_id": int(ann["frame_id"]),
        "handedness": handedness,
    }

    return {
        "img_path": file_name,
        "img_bytes": encoded_img.tobytes(),
        "handedness": handedness,
        "hand_bbox": bbox.astype(np.float32),
        "joint_img": joint_img.astype(np.float32),
        "joint_hand_bbox": joint_hand_bbox.astype(np.float32),
        "joint_cam": joint_cam.astype(np.float32),
        "joint_rel": joint_rel.astype(np.float32),
        "joint_2d_valid": joint_2d_valid,
        "joint_3d_valid": joint_3d_valid,
        "joint_valid": joint_valid,
        "mano_pose": np.zeros((48,), dtype=np.float32),
        "mano_shape": np.zeros((10,), dtype=np.float32),
        "has_mano": np.float32(0.0),
        "mano_valid": np.float32(0.0),
        "has_intr": np.float32(1.0),
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
        "source_index": source_index,
        "additional_desc": additional_desc,
    }


def process_batch(clips: List[List[Tuple[int, str, int]]], worker_id: int) -> int:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)
    processed_count = 0

    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for clip in clips:
            clip_frames = []
            clip_descs = []
            valid_clip = True

            for ann_idx, handedness, _frame_idx in clip:
                try:
                    frame = process_single_sample(ann_idx, handedness)
                except Exception as ex:
                    print(
                        f"[Worker {worker_id}] ann_idx={ann_idx} hand={handedness} error: {ex}"
                    )
                    valid_clip = False
                    break
                clip_frames.append(frame)
                clip_descs.append(frame["additional_desc"])

            if not valid_clip or len(clip_frames) == 0:
                continue

            source_index = clip_frames[0]["source_index"]
            key_str = (
                f"{source_index['seq_name']}_{source_index['frame_id']:06d}_"
                f"{source_index['camera']}_{clip_frames[0]['handedness']}_{SPLIT}_ah"
            )
            wds_sample = {
                "__key__": key_str,
                "imgs_path.json": json.dumps([frame["img_path"] for frame in clip_frames]),
                "img_bytes.pickle": pickle.dumps([frame["img_bytes"] for frame in clip_frames]),
                "handedness.json": json.dumps(clip_frames[0]["handedness"]),
                "additional_desc.json": json.dumps(clip_descs),
                "data_source.json": json.dumps("assemblyhands"),
                "source_split.json": json.dumps(SPLIT),
                "source_index.json": json.dumps([frame["source_index"] for frame in clip_frames]),
                "intr_type.json": json.dumps("real"),
            }
            for key in NUMPY_KEYS:
                wds_sample[f"{key}.npy"] = np.stack([frame[key] for frame in clip_frames])
            sink.write(wds_sample)
            processed_count += 1

    return processed_count


def main() -> None:
    os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)
    print(f"Loading AssemblyHands annotations for split={SPLIT} ...")
    load_ego_data()
    load_joint_3d()
    load_calib()
    print(
        f"AssemblyHands temporal packing: frame_gap={FRAME_GAP}, "
        f"max_clip_len={MAX_CLIP_LEN}, timestamp_unit=ms"
    )

    clips = build_sequences()
    total_clips = len(clips)
    if total_clips == 0:
        print("No valid AssemblyHands hand samples found.")
        return

    worker_count = max(NUM_WORKERS, 1)
    chunk_size = math.ceil(total_clips / worker_count)
    chunks = [clips[i : i + chunk_size] for i in range(0, total_clips, chunk_size)]

    print(f"Total AssemblyHands clips: {total_clips}")
    print(f"Starting {len(chunks)} workers processing ~{chunk_size} clips each ...")

    process_args = [(chunk, worker_id) for worker_id, chunk in enumerate(chunks)]
    if len(chunks) == 1:
        results = [process_batch(chunks[0], 0)]
    else:
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            results = pool.starmap(process_batch, process_args)

    print(f"All done! Total AssemblyHands clips processed: {sum(results)}")


if __name__ == "__main__":
    import pickle

    main()
