import numpy as np
from typing import *
from collections import defaultdict
import torch


IH26M_RJOINTS_ORDER = (
    "Thumb_4",
    "Thumb_3",
    "Thumb_2",
    "Thumb_1",
    "Index_4",
    "Index_3",
    "Index_2",
    "Index_1",
    "Middle_4",
    "Middle_3",
    "Middle_2",
    "Middle_1",
    "Ring_4",
    "Ring_3",
    "Ring_2",
    "Ring_1",
    "Pinky_4",
    "Pinky_3",
    "Pinky_2",
    "Pinky_1",
    "Wrist",
)

HO3D_JOINTS_ORDER = (
    "Wrist",
    "Index_1",
    "Index_2",
    "Index_3",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
    "Thumb_4",
    "Index_4",
    "Middle_4",
    "Ring_4",
    "Pinky_4",
)

MANO_JOINTS_ORDER = (
    "Wrist",
    "Index_1",
    "Index_2",
    "Index_3",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
)

HOT3D_JOINTS_ORDER = (
    "Thumb_4",
    "Index_4",
    "Middle_4",
    "Ring_4",
    "Pinky_4",
    "Wrist",
    "Thumb_2",
    "Thumb_3",
    "Index_1",
    "Index_2",
    "Index_3",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Thumb_1",
)

TARGET_JOINTS_ORDER = (
    "Wrist",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
    "Thumb_4",
    "Index_1",
    "Index_2",
    "Index_3",
    "Index_4",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Middle_4",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Ring_4",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Pinky_4",
)

TARGET_JOINTS_CONNECTION = [
    (0, 1),  # Wrist -> Thumb_1
    (0, 5),  # Wrist -> Index_1
    (0, 9),  # Wrist -> Middle_1
    (0, 13),  # Wrist -> Ring_1
    (0, 17),  # Wrist -> Pinky_1
    # Thumb
    (1, 2),  # Thumb_1 -> Thumb_2
    (2, 3),  # Thumb_2 -> Thumb_3
    (3, 4),  # Thumb_3 -> Thumb_4
    # Index
    (5, 6),  # Index_1 -> Index_2
    (6, 7),  # Index_2 -> Index_3
    (7, 8),  # Index_3 -> Index_4
    # Middle
    (9, 10),  # Middle_1 -> Middle_2
    (10, 11),  # Middle_2 -> Middle_3
    (11, 12),  # Middle_3 -> Middle_4
    # Ring
    (13, 14),  # Ring_1 -> Ring_2
    (14, 15),  # Ring_2 -> Ring_3
    (15, 16),  # Ring_3 -> Ring_4
    # Pinky
    (17, 18),  # Pinky_1 -> Pinky_2
    (18, 19),  # Pinky_2 -> Pinky_3
    (19, 20),  # Pinky_3 -> Pinky_4
]


def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.0
    width = xmax - xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin + ymax) / 2.0
    height = ymax - ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None

    return bbox


def process_bbox(bbox, img_width, img_height, do_sanitize=True, extend_ratio=1.25):
    if do_sanitize:
        bbox = sanitize_bbox(bbox, img_width, img_height)
        if bbox is None:
            return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    aspect_ratio = 512 / 384  # from interhand2.6m code
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * extend_ratio
    bbox[3] = h * extend_ratio
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0

    bbox = bbox.astype(np.float32)
    return bbox


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


_JOINT_REORDER_CACHE = defaultdict(dict)


def reorder_joints(
    joints: np.ndarray, origin: List[str], target: List[str]
) -> np.ndarray:
    """
    Reorder the joints from origin order to target order. Optimized for fast reordering.
    (NumPy Version)

    Args:
        joints (np.ndarray): Input joint data of shape [..., J, D] where J is number of joints
        origin: List of joint names in the original order (length J)
        target: List of joint names in the target order (length J)

    Returns:
        np.ndarray: Reordered joint data of shape [..., J, D]

    Raises:
        TypeError: If origin or target are not lists/tuples
        ValueError: If origin and target lists have different lengths or contain different joints
    """
    if not isinstance(origin, (list, tuple)) or not isinstance(target, (list, tuple)):
        raise TypeError("Joint orders must be lists/tuples")

    cache_key = (tuple(origin), tuple(target))

    if cache_key not in _JOINT_REORDER_CACHE:
        if len(origin) != len(target):
            raise ValueError("Origin and target joint lists must have same length")
        if set(origin) != set(target):
            raise ValueError("Origin and target joint lists must contain same joints")

        origin_map = {name: idx for idx, name in enumerate(origin)}
        try:
            indices = [origin_map[name] for name in target]
        except KeyError as e:
            raise ValueError(f"Missing joint in mapping: {e}")

        # 使用 np.array 存储索引
        _JOINT_REORDER_CACHE[cache_key] = np.array(indices, dtype=np.int64)

    # 使用 np.take 进行指定维度的索引选择
    # axis=-2 对应 PyTorch 中的 index_select(dim=-2)
    return np.take(joints, _JOINT_REORDER_CACHE[cache_key], axis=-2)
