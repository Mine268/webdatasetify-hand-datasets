import json
import math
import multiprocessing
import os
import os.path as osp
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
import webdataset as wds

MTC_ROOT = os.environ.get("MTC_ROOT", "/mnt/qnap/data/datasets/mtc/a4_release")
SPLIT = os.environ.get("SPLIT", "train")
assert SPLIT in {"train", "test"}, f"Unsupported SPLIT={SPLIT}"
SPLIT_TO_KEY = {
    "train": "training_data",
    "test": "testing_data",
}
OUTPUT_PATTERN = f"mtc_{SPLIT}_wds_output/mtc_{SPLIT}-worker{{worker_id}}-%06d.tar"
MAX_COUNT = 100000
MAX_SIZE = 3 * 1024 * 1024 * 1024
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEBUG_MAX_CLIPS = int(os.environ.get("DEBUG_MAX_CLIPS", os.environ.get("DEBUG_MAX_SAMPLES", "0")))
MIN_VISIBLE_RATIO = float(os.environ.get("MIN_VISIBLE_RATIO", "0.8"))
MIN_VISIBLE_JOINTS = max(1, int(math.ceil(21 * MIN_VISIBLE_RATIO)))
BBOX_EXPAND_RATIO = float(os.environ.get("BBOX_EXPAND_RATIO", "1.2"))
FRAME_GAP = int(os.environ.get("FRAME_GAP", "5"))
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

os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)

_ANNOTATIONS = None
_CAMERAS = None


def load_annotations() -> List[Dict]:
    global _ANNOTATIONS
    if _ANNOTATIONS is None:
        with open(osp.join(MTC_ROOT, "annotation.pkl"), "rb") as fp:
            data = pickle.load(fp)
        _ANNOTATIONS = data[SPLIT_TO_KEY[SPLIT]]
    return _ANNOTATIONS


def load_cameras() -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    global _CAMERAS
    if _CAMERAS is None:
        with open(osp.join(MTC_ROOT, "camera_data.pkl"), "rb") as fp:
            _CAMERAS = pickle.load(fp)
    return _CAMERAS


def hand_key_to_handedness(hand_key: str) -> str:
    return "left" if hand_key == "left_hand" else "right"


def camera_id_to_name(cam_info: Dict) -> str:
    return str(cam_info["name"])


def get_image_relpath(seq_name: str, frame_str: str, camera_name: str) -> str:
    return osp.join("hdImgs", seq_name, frame_str, f"{camera_name}_{frame_str}.jpg")


def get_image_abspath(seq_name: str, frame_str: str, camera_name: str) -> str:
    return osp.join(MTC_ROOT, get_image_relpath(seq_name, frame_str, camera_name))


def _project_with_distortion(
    joint_cam: np.ndarray,
    K: np.ndarray,
    dist_coef: np.ndarray,
) -> np.ndarray:
    pts_2d, _ = cv2.projectPoints(
        joint_cam.astype(np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        K.astype(np.float32),
        dist_coef.astype(np.float32),
    )
    return pts_2d[:, 0, :].astype(np.float32)


def _project_pinhole(
    joint_cam: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    proj = joint_cam @ K.T
    return (proj[:, :2] / proj[:, 2:]).astype(np.float32)


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


def _build_bbox_from_joints(
    joint_img: np.ndarray,
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> np.ndarray:
    x1, y1 = np.min(joint_img, axis=0)
    x2, y2 = np.max(joint_img, axis=0)
    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    return _expand_bbox_xyxy(bbox, image_width, image_height, expand_ratio)


def build_sequences() -> List[List[Tuple[int, str, int, int]]]:
    annotations = load_annotations()
    grouped: Dict[Tuple[str, int, str, int], List[Tuple[int, str, int, int]]] = {}
    for sample_idx, sample in enumerate(annotations):
        for hand_key in ("left_hand", "right_hand"):
            if hand_key not in sample:
                continue
            for camera_id in sorted(sample[hand_key]["2D"].keys()):
                inside = np.asarray(
                    sample[hand_key]["2D"][camera_id]["insideImg"], dtype=np.int32
                )
                occluded = np.asarray(
                    sample[hand_key]["2D"][camera_id]["occluded"], dtype=np.int32
                )
                visible = (inside > 0) & (occluded == 0)
                if int(np.sum(visible)) < MIN_VISIBLE_JOINTS:
                    continue
                group_key = (
                    sample["seqName"],
                    int(sample["id"]),
                    hand_key,
                    int(camera_id),
                )
                grouped.setdefault(group_key, []).append(
                    (sample_idx, hand_key, int(camera_id), int(sample["frame_str"]))
                )

    clips: List[List[Tuple[int, str, int, int]]] = []
    for items in grouped.values():
        items.sort(key=lambda x: x[3])
        start = 0
        for i in range(1, len(items) + 1):
            is_break = i == len(items) or (items[i][3] - items[i - 1][3]) != FRAME_GAP
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


def process_single_sample(
    sample_idx: int,
    hand_key: str,
    camera_id: int,
) -> Dict[str, object]:
    annotations = load_annotations()
    cameras = load_cameras()

    sample = annotations[sample_idx]
    seq_name = sample["seqName"]
    frame_str = sample["frame_str"]
    hand = sample[hand_key]
    handedness = hand_key_to_handedness(hand_key)
    cam_info = cameras[seq_name][camera_id]
    camera_name = camera_id_to_name(cam_info)
    image_abspath = get_image_abspath(seq_name, frame_str, camera_name)

    image = cv2.imread(image_abspath)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_abspath}")
    undistorted = cv2.undistort(
        image,
        cam_info["K"].astype(np.float32),
        cam_info["distCoef"].astype(np.float32),
        None,
        cam_info["K"].astype(np.float32),
    )
    image_height, image_width = undistorted.shape[:2]

    success, encoded_img = cv2.imencode(
        ".webp", undistorted, [cv2.IMWRITE_WEBP_QUALITY, 100]
    )
    if not success:
        raise RuntimeError(f"Failed to encode image: {image_abspath}")

    joint_world_cm = np.asarray(hand["landmarks"], dtype=np.float32).reshape(21, 3)
    R = cam_info["R"].astype(np.float32)
    t = cam_info["t"].astype(np.float32).reshape(3, 1)
    K = cam_info["K"].astype(np.float32)
    dist = cam_info["distCoef"].astype(np.float32)

    joint_cam_cm = (R @ joint_world_cm.T + t).T
    joint_img = _project_pinhole(joint_cam_cm, K)
    joint_img_dist = _project_with_distortion(joint_cam_cm, K, dist)

    hand_2d_meta = hand["2D"][camera_id]
    inside = np.asarray(hand_2d_meta["insideImg"], dtype=np.float32)
    occluded = np.asarray(hand_2d_meta["occluded"], dtype=np.float32)
    overlap_flag = float(hand_2d_meta["overlap"])

    visible = (inside > 0.5) & (occluded < 0.5)
    if int(np.sum(visible)) < MIN_VISIBLE_JOINTS:
        raise ValueError(
            f"Not enough visible joints for sample_idx={sample_idx}, hand={hand_key}, cam={camera_id}"
        )

    joint_2d_valid = visible.astype(np.float32)
    joint_3d_valid = np.ones((21,), dtype=np.float32)
    joint_valid = joint_2d_valid.copy()

    bbox = _build_bbox_from_joints(joint_img, image_width, image_height, BBOX_EXPAND_RATIO)
    joint_hand_bbox = joint_img - bbox[None, :2]
    joint_cam_mm = (joint_cam_cm * 10.0).astype(np.float32)
    joint_rel = (joint_cam_mm - joint_cam_mm[:1]).astype(np.float32)

    focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
    princpt = np.array([K[0, 2], K[1, 2]], dtype=np.float32)
    frame_idx = int(frame_str)
    timestamp = np.float32(frame_idx * (1000.0 / 30.0))

    additional_desc = {
        "dataset": "mtc",
        "seq_name": seq_name,
        "frame_str": frame_str,
        "camera_id": int(camera_id),
        "camera_name": camera_name,
        "overlap_flag": overlap_flag,
        "occluded_joint_count": int(np.sum(occluded > 0.5)),
        "inside_joint_count": int(np.sum(inside > 0.5)),
        "visible_joint_count": int(np.sum(visible)),
        "undistorted_from_hd": True,
        "distCoef": dist.astype(np.float32).tolist(),
        "projected_distorted_first_joint": joint_img_dist[0].astype(np.float32).tolist(),
    }

    source_index = {
        "sample_idx": int(sample_idx),
        "seq_name": seq_name,
        "frame_str": frame_str,
        "camera_id": int(camera_id),
        "camera_name": camera_name,
        "handedness": handedness,
        "person_id": int(sample["id"]),
    }

    return {
        "img_path": get_image_relpath(seq_name, frame_str, camera_name),
        "img_bytes": encoded_img.tobytes(),
        "handedness": handedness,
        "hand_bbox": bbox.astype(np.float32),
        "joint_img": joint_img.astype(np.float32),
        "joint_hand_bbox": joint_hand_bbox.astype(np.float32),
        "joint_cam": joint_cam_mm.astype(np.float32),
        "joint_rel": joint_rel.astype(np.float32),
        "joint_2d_valid": joint_2d_valid.astype(np.float32),
        "joint_3d_valid": joint_3d_valid.astype(np.float32),
        "joint_valid": joint_valid.astype(np.float32),
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


def process_batch(clips: List[List[Tuple[int, str, int, int]]], worker_id: int) -> int:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)
    processed_count = 0

    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for clip in clips:
            clip_frames = []
            clip_descs = []
            valid_clip = True

            for sample_idx, hand_key, camera_id, _frame_idx in clip:
                try:
                    frame = process_single_sample(sample_idx, hand_key, camera_id)
                except Exception as ex:
                    print(
                        f"[Worker {worker_id}] sample_idx={sample_idx} hand={hand_key} camera_id={camera_id} error: {ex}"
                    )
                    valid_clip = False
                    break
                clip_frames.append(frame)
                clip_descs.append(frame["additional_desc"])

            if not valid_clip or len(clip_frames) == 0:
                continue

            source_index = clip_frames[0]["source_index"]
            key_str = (
                f"{source_index['seq_name']}_{source_index['frame_str']}_"
                f"{source_index['camera_name']}_{source_index['handedness']}_{SPLIT}_mtc"
            )
            wds_sample = {
                "__key__": key_str,
                "imgs_path.json": json.dumps([frame["img_path"] for frame in clip_frames]),
                "img_bytes.pickle": pickle.dumps([frame["img_bytes"] for frame in clip_frames]),
                "handedness.json": json.dumps(clip_frames[0]["handedness"]),
                "additional_desc.json": json.dumps(clip_descs),
                "data_source.json": json.dumps("mtc"),
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
    print(f"Loading MTC annotations for split={SPLIT} ...")
    load_annotations()
    load_cameras()
    print(
        f"MTC filter: visible joints >= {MIN_VISIBLE_JOINTS}/21 "
        f"(ratio={MIN_VISIBLE_RATIO:.2f})"
    )
    print(f"MTC temporal packing: frame_gap={FRAME_GAP}, max_clip_len={MAX_CLIP_LEN}")

    clips = build_sequences()
    total_clips = len(clips)
    if total_clips == 0:
        print("No valid MTC hand-camera samples found.")
        return

    worker_count = max(NUM_WORKERS, 1)
    chunk_size = math.ceil(total_clips / worker_count)
    chunks = [clips[i : i + chunk_size] for i in range(0, total_clips, chunk_size)]

    print(f"Total MTC clips: {total_clips}")
    print(f"Starting {len(chunks)} workers processing ~{chunk_size} clips each ...")

    process_args = [(chunk, worker_id) for worker_id, chunk in enumerate(chunks)]
    if len(chunks) == 1:
        results = [process_batch(chunks[0], 0)]
    else:
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            results = pool.starmap(process_batch, process_args)

    print(f"All done! Total MTC clips processed: {sum(results)}")


if __name__ == "__main__":
    main()
