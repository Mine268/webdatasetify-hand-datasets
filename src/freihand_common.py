import json
import os
import os.path as osp
from functools import lru_cache
from typing import Any, Dict, Tuple

import cv2
import numpy as np

try:
    from .utils import TARGET_JOINTS_CONNECTION, TARGET_JOINTS_ORDER, reorder_joints
except ImportError:
    from utils import TARGET_JOINTS_CONNECTION, TARGET_JOINTS_ORDER, reorder_joints


FREIHAND_ROOT = os.environ.get("FREIHAND_ROOT", "/mnt/qnap/data/datasets/Freihand")
FREIHAND_JOINTS_ORDER = TARGET_JOINTS_ORDER
FREIHAND_HANDEDNESS = "right"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def _split_to_prefix(split: str) -> str:
    if split == "train":
        return "training"
    if split == "evaluation":
        return "evaluation"
    raise ValueError(f"Unsupported split: {split}")


@lru_cache(maxsize=2)
def load_split_annotations(split: str) -> Dict[str, Any]:
    prefix = _split_to_prefix(split)
    annotations = {}
    for key in ("K", "xyz", "mano", "scale"):
        path = osp.join(FREIHAND_ROOT, f"{prefix}_{key}.json")
        with open(path, "r", encoding="utf-8") as fp:
            annotations[key] = json.load(fp)
    return annotations


@lru_cache(maxsize=2)
def get_split_layout(split: str) -> Dict[str, int]:
    annotations = load_split_annotations(split)
    prefix = _split_to_prefix(split)
    rgb_dir = osp.join(FREIHAND_ROOT, prefix, "rgb")
    rgb_count = len(os.listdir(rgb_dir))
    unique_count = len(annotations["K"])

    if split == "train":
        if rgb_count % unique_count != 0:
            raise ValueError(
                f"Training rgb count {rgb_count} is not divisible by unique count {unique_count}"
            )
        version_count = rgb_count // unique_count
        mask_count = len(os.listdir(osp.join(FREIHAND_ROOT, prefix, "mask")))
        if mask_count != unique_count:
            raise ValueError(
                f"Training mask count {mask_count} does not match unique count {unique_count}"
            )
    else:
        version_count = 1
        if rgb_count != unique_count:
            raise ValueError(
                f"Evaluation rgb count {rgb_count} does not match annotation count {unique_count}"
            )

    return {
        "unique_count": unique_count,
        "rgb_count": rgb_count,
        "version_count": version_count,
    }


def get_image_count(split: str) -> int:
    return get_split_layout(split)["rgb_count"]


def map_image_index(split: str, image_idx: int) -> Tuple[int, int]:
    layout = get_split_layout(split)
    if image_idx < 0 or image_idx >= layout["rgb_count"]:
        raise IndexError(
            f"image_idx={image_idx} is outside [0, {layout['rgb_count'] - 1}] for split={split}"
        )

    if split == "train":
        sample_idx = image_idx % layout["unique_count"]
        version_idx = image_idx // layout["unique_count"]
        return sample_idx, version_idx

    return image_idx, 0


def get_version_name(version_idx: int) -> str:
    return f"version_{version_idx}"


def get_image_relpath(split: str, image_idx: int) -> str:
    prefix = _split_to_prefix(split)
    return osp.join(prefix, "rgb", f"{image_idx:08d}.jpg")


def get_image_abspath(split: str, image_idx: int) -> str:
    return osp.join(FREIHAND_ROOT, get_image_relpath(split, image_idx))


def get_mask_relpath(sample_idx: int) -> str:
    return osp.join("training", "mask", f"{sample_idx:08d}.jpg")


def get_mask_abspath(sample_idx: int) -> str:
    return osp.join(FREIHAND_ROOT, get_mask_relpath(sample_idx))


def load_image(split: str, image_idx: int) -> np.ndarray:
    img_path = get_image_abspath(split, image_idx)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img


def encode_image_to_webp(img: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise RuntimeError("Failed to encode FreiHAND image to WebP")
    return encoded.tobytes()


def project_points(K: np.ndarray, joint_cam: np.ndarray) -> np.ndarray:
    homog = joint_cam @ K.T
    return homog[:, :2] / np.clip(homog[:, 2:], 1e-8, None)


def compute_bbox_xyxy(
    joint_img: np.ndarray,
    expand_ratio: float = 1.2,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> np.ndarray:
    x1 = float(np.min(joint_img[:, 0]))
    y1 = float(np.min(joint_img[:, 1]))
    x2 = float(np.max(joint_img[:, 0]))
    y2 = float(np.max(joint_img[:, 1]))

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.5 * expand_ratio, 1.0)
    half_h = max((y2 - y1) * 0.5 * expand_ratio, 1.0)

    x1 = np.clip(cx - half_w, 0.0, image_width - 1.0)
    y1 = np.clip(cy - half_h, 0.0, image_height - 1.0)
    x2 = np.clip(cx + half_w, 0.0, image_width - 1.0)
    y2 = np.clip(cy + half_h, 0.0, image_height - 1.0)

    if x2 <= x1:
        x2 = min(float(image_width - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_height - 1), y1 + 1.0)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _extract_scale_scalar(value: Any) -> float:
    value_arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if value_arr.size == 0:
        raise ValueError("Empty FreiHAND scale value")
    return float(value_arr[0])


def build_sample_record(split: str, image_idx: int) -> Dict[str, Any]:
    annotations = load_split_annotations(split)
    sample_idx, version_idx = map_image_index(split, image_idx)

    K = np.asarray(annotations["K"][sample_idx], dtype=np.float32)
    joint_cam = np.asarray(annotations["xyz"][sample_idx], dtype=np.float32) * 1e3
    mano = np.asarray(annotations["mano"][sample_idx], dtype=np.float32).reshape(-1)
    sample_scale = _extract_scale_scalar(annotations["scale"][sample_idx])

    if joint_cam.shape != (21, 3):
        raise ValueError(
            f"Expected FreiHAND joint_cam shape (21, 3), got {joint_cam.shape} for sample {sample_idx}"
        )
    if mano.shape[0] != 61:
        raise ValueError(
            f"Expected FreiHAND mano shape (61,), got {mano.shape} for sample {sample_idx}"
        )

    joint_img = project_points(K, joint_cam)
    hand_bbox = compute_bbox_xyxy(joint_img)
    joint_hand_bbox = joint_img - hand_bbox[None, :2]
    joint_rel = joint_cam - joint_cam[:1]
    joint_valid = np.ones((21,), dtype=np.float32)

    joint_img = reorder_joints(joint_img, FREIHAND_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_hand_bbox = reorder_joints(
        joint_hand_bbox, FREIHAND_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )
    joint_cam = reorder_joints(joint_cam, FREIHAND_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_rel = reorder_joints(joint_rel, FREIHAND_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_valid = reorder_joints(
        joint_valid[:, None], FREIHAND_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )[:, 0]

    focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
    princpt = np.array([K[0, 2], K[1, 2]], dtype=np.float32)
    mano_pose = mano[:48].astype(np.float32)
    mano_shape = mano[48:58].astype(np.float32)
    mano_uv_root = mano[58:60].astype(np.float32)
    mano_scale = np.float32(mano[60])

    source_index = {
        "sample_idx": int(sample_idx),
        "image_idx": int(image_idx),
        "version_idx": int(version_idx),
        "version_name": get_version_name(version_idx),
    }

    additional_desc = {
        "dataset": "freihand",
        "annotation_idx": int(sample_idx),
        "version_idx": int(version_idx),
        "version_name": get_version_name(version_idx),
        "is_green_screen": bool(split == "train" and version_idx == 0),
        "mano_uv_root": mano_uv_root.tolist(),
        "mano_scale": float(mano_scale),
        "sample_scale": float(sample_scale),
        "joint_order": list(TARGET_JOINTS_ORDER),
        "is_right_hand_dataset": True,
    }
    if split == "train":
        additional_desc["mask_path"] = get_mask_relpath(sample_idx)

    return {
        "img_path": get_image_relpath(split, image_idx),
        "handedness": FREIHAND_HANDEDNESS,
        "hand_bbox": hand_bbox.astype(np.float32),
        "joint_img": joint_img.astype(np.float32),
        "joint_hand_bbox": joint_hand_bbox.astype(np.float32),
        "joint_cam": joint_cam.astype(np.float32),
        "joint_rel": joint_rel.astype(np.float32),
        "joint_2d_valid": joint_valid.astype(np.float32),
        "joint_3d_valid": joint_valid.astype(np.float32),
        "joint_valid": joint_valid.astype(np.float32),
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "has_mano": np.float32(1.0),
        "mano_valid": np.float32(1.0),
        "has_intr": np.float32(1.0),
        "timestamp": np.float32(0.0),
        "focal": focal,
        "princpt": princpt,
        "source_index": source_index,
        "additional_desc": additional_desc,
    }


__all__ = [
    "FREIHAND_ROOT",
    "FREIHAND_JOINTS_ORDER",
    "FREIHAND_HANDEDNESS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "TARGET_JOINTS_CONNECTION",
    "TARGET_JOINTS_ORDER",
    "build_sample_record",
    "encode_image_to_webp",
    "get_image_abspath",
    "get_image_count",
    "get_image_relpath",
    "get_mask_abspath",
    "get_mask_relpath",
    "get_split_layout",
    "get_version_name",
    "load_image",
    "load_split_annotations",
    "map_image_index",
]
