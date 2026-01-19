import os
import os.path as osp
import sys
import pickle
import yaml
import json
from tqdm import tqdm
import math
import multiprocessing
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import webdataset as wds
from .utils import *

sys.path.append("../hot3d/hot3d")  # HOT3D 代码的路径

from dataset_api import Hot3dDataProvider
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.calibration import LINEAR
from data_loaders.loader_hand_poses import Handedness, HandPose3dCollection
from data_loaders.pytorch3d_rotation.rotation_conversions import (  # @manual
    matrix_to_axis_angle,
)
from typing import Any, Optional
from projectaria_tools.core.calibration import CameraCalibration
import matplotlib.patches as patches


HOT3D_ROOT = "/mnt/qnap/data/datasets/hot3d/"
SPLIT = os.environ.get("SPLIT", "train")  # train test
TRAIN_SET = [
    "P0001",
    "P0002",
    "P0003",
    "P0007",
    "P0009",
    "P0010",
    "P0011",
    "P0012",
    "P0013",
    "P0014",
    "P0015",
    "P0017",
    "P0018",
    "P0019",
    "P0021",
]
TEST_SET = ["P0004", "P0005", "P0006", "P0008", "P0016", "P0020"]
OUTPUT_PATTERN = f"hot3d_{SPLIT}_wds_output/hot3d_{SPLIT}-worker{{worker_id}}-%06d.tar"
MAX_COUNT = 100000  # 已修改：大幅增加数量限制，让切割主要由 MAX_SIZE 决定
MAX_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
NUM_WORKERS = 16  # 建议设置为 CPU 核心数 - 2

GAP_THRESHOLD = 1 * 1e3 * 1e3 * 1e3  # 1s
INVALID_THRESHOLD = 16

# 定义需要堆叠成 Numpy 数组的字段
NUMPY_KEYS = [
    "hand_bbox",
    "joint_img",
    "joint_hand_bbox",
    "joint_cam",
    "joint_rel",
    "joint_valid",
    "mano_pose",
    "mano_shape",
    "mano_valid",
    "timestamp",
    "focal",
    "princpt",
]

os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)

# mano 信息
mano_hand_model_path = "./models/mano/"
mano_hand_model = MANOHandModel(mano_hand_model_path)
mano_pca_path = os.path.join(mano_hand_model_path, "mano_lr_pca.npz")
_mano_tmp = np.load(mano_pca_path)
mano_pca_comps = {k: _mano_tmp[k] for k in _mano_tmp.files}
_mano_tmp.close()


def add_clip(clip, clips):
    if len(clip) > 0:
        clips.append(clip.copy())
        clip.clear()


def prepare_data():
    clips = []
    for sequence_name in os.listdir(HOT3D_ROOT):
        if sequence_name == "assets":
            continue
        if sequence_name[:len(TRAIN_SET[0])] not in TRAIN_SET:
            continue

        sequence_path = os.path.join(HOT3D_ROOT, sequence_name)

        # 初始化data_provider
        hot3d_data_provider = Hot3dDataProvider(
            sequence_folder=sequence_path,
            object_library=None,
            mano_hand_model=mano_hand_model,
        )
        device_data_provider = (
            hot3d_data_provider.device_data_provider
        )  # 图像数据、设备标定信息
        device_pose_provider = (
            hot3d_data_provider.device_pose_data_provider
        )  # 相机姿态
        hand_data_provider = hot3d_data_provider.mano_hand_data_provider  # 手部数据
        timestamps = (
            device_data_provider.get_sequence_timestamps()
        )  # 时间戳，ns，不连续（对应每帧的timestamp？）
        stream_id = StreamId("214-1")  # 使用RGB相机的数据流

        if device_pose_provider is None or hand_data_provider is None:
            continue  # 整个序列都无效

        for handedness in ["right", "left"]:
            prev_ts_ns = None
            curr_clip = []
            for timestamp_ns in tqdm(timestamps, ncols=50):
                if prev_ts_ns is not None and timestamp_ns - prev_ts_ns >= GAP_THRESHOLD:
                    if curr_clip != []:
                        clips.append(curr_clip.copy())
                    prev_ts_ns = timestamp_ns
                    curr_clip.clear()

                # 加载当前帧相机姿态
                headset_pose3d_with_dt = None
                headset_pose3d = None
                headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp_ns,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
                if (
                    headset_pose3d_with_dt is None
                    or headset_pose3d_with_dt.pose3d is None
                ):
                    continue
                headset_pose3d = headset_pose3d_with_dt.pose3d

                # 加载当前帧手部姿态
                hand_poses_with_dt = None
                hand_data = None
                hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp_ns,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
                if (
                    hand_poses_with_dt is None
                    or hand_poses_with_dt.pose3d_collection is None
                ):
                    continue
                hand_data = hand_poses_with_dt.pose3d_collection

                # 处理手部姿态
                # 相机参数
                [T_device_camera, linear_camera_intrinsic] = (
                    device_data_provider.get_online_camera_calibration(
                        stream_id, timestamp_ns, camera_model=LINEAR
                    )
                )
                T_world_camera = headset_pose3d.T_world_device @ T_device_camera
                focal = linear_camera_intrinsic.get_focal_lengths()
                princpt = linear_camera_intrinsic.get_principal_point()

                # 单手数据
                handedness_key = Handedness.Right if handedness == "right" else Handedness.Left
                if handedness_key not in hand_data.poses:
                    continue
                hand_pose_data = hand_data.poses[handedness_key]
                landmarks_world = hand_data_provider.get_hand_landmarks(
                    hand_pose_data
                )  # 关键点（世界，m）
                landmarks_world = landmarks_world.squeeze(0).numpy()

                # 过滤无效标注，如果落在图像中的点太少，则认为标注无效
                invalid_count = sum(
                    int(linear_camera_intrinsic.project(T_world_camera.inverse() @ x) is None)
                    for x in landmarks_world
                )
                if invalid_count >= INVALID_THRESHOLD:
                    continue

                joint_cam = landmarks_world
                joint_cam_mm = joint_cam * 1000.0  # m -> mm

                # HOT3D 没有 thumb1 关键点
                # 将 wrist 和 thumb2 的中点作为 thumb1
                thumb1_cam_mm = (joint_cam_mm[5] + joint_cam_mm[6]) / 2
                joint_cam_mm = np.concatenate(
                    [joint_cam_mm, thumb1_cam_mm[np.newaxis, :]], axis=0
                )

                # 手动投影
                extr = T_world_camera.inverse().to_matrix().T
                joint_cam_homo = np.concatenate(
                    [joint_cam, np.ones_like(joint_cam[:, :1])], axis=-1
                )
                joint_cam_homo = joint_cam_homo @ extr
                joint_cam = joint_cam_homo[..., :-1] / joint_cam_homo[..., -1:]
                u = focal[0] * joint_cam[..., 0] / joint_cam[..., 2] + princpt[0]
                v = focal[1] * joint_cam[..., 1] / joint_cam[..., 2] + princpt[1]

                joint_img = np.stack([u, v], axis=-1) # [j,2]
                thumb1_joint_img = (joint_img[5] + joint_img[6]) / 2
                joint_img = np.concatenate(
                    [joint_img, thumb1_joint_img[np.newaxis, :]], axis=0
                )

                # 包围盒
                xmin, ymin = joint_img.min(axis=0)
                xmax, ymax = joint_img.max(axis=0)
                hand_bbox = np.array(
                    [xmin, ymin, xmax, ymax], dtype=np.float32
                )

                T_world_wrist = hand_pose_data.wrist_pose
                T_camera_wrist = T_world_camera.inverse() @ T_world_wrist
                wrist_rot_mat = T_camera_wrist.to_matrix()[:3, :3]
                root_axis_angle = matrix_to_axis_angle(
                    torch.from_numpy(wrist_rot_mat)
                ).numpy()
                pca_pose = np.array(hand_pose_data.joint_angles)  # (15,)
                num_pose_coeffs = pca_pose.shape[0]
                axis_angle = (
                    pca_pose @ mano_pca_comps[handedness][:num_pose_coeffs]
                )  # (45,)
                mano_pose = np.concatenate([root_axis_angle, axis_angle])
                mano_shape = hand_data_provider._mano_shape_params
                mano_valid = True

                frame = {
                    "sequence_name": sequence_name,
                    "timestamp_ns": timestamp_ns,
                    "handedness": handedness,  # flip？
                    # 3D信息
                    "joint_cam": joint_cam_mm,  # (J,3), 单位 mm
                    "joint_rel": joint_cam_mm - joint_cam_mm[5],
                    # 2D信息（正畸）
                    "joint_img": joint_img,  # (J,2)
                    "bbox_tight": hand_bbox,  # (4,)
                    "joint_bbox_img": joint_img - hand_bbox[:2],
                    # MANO信息
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                    "mano_valid": mano_valid,
                    # 内参（正畸）
                    "focal": focal,
                    "princpt": princpt,
                }

                curr_clip.append(frame)
                prev_ts_ns = timestamp_ns

            if curr_clip != []:
                clips.append(curr_clip)

        break

    return clips


def process_single_annot(sample, idx: int, device_data_provider):
    """
    Args:
        idx: 该样本在序列中的位置，用于时间戳的计算
    """
    # timestamp
    timestamp = sample["timestamp_ns"]

    # img_path
    sequence_name = sample["sequence_name"]
    img_path = os.path.join(sequence_name, "recording.vrs")

    # img_bytes
    stream_id = StreamId("214-1")
    img = device_data_provider.get_undistorted_image(timestamp, stream_id)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success, img_bytes = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise Exception(f"Failed to encode the image: {osp.join(HOT3D_ROOT, img_path)}")

    # handedness
    handedness = sample["handedness"]  # right, left

    # joint_img, hand_bbox, joint_hand_bbox
    joint_img = sample["joint_img"]  # [J,2]
    hand_bbox = sample["bbox_tight"]
    joint_hand_bbox = sample["joint_bbox_img"]

    # joint_cam, joint_rel
    joint_cam = sample["joint_cam"]  # [J,3] 毫米单位
    joint_rel = sample["joint_rel"]

    # reorder joints
    joint_img = reorder_joints(joint_img, HOT3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_hand_bbox = reorder_joints(
        joint_hand_bbox, HOT3D_JOINTS_ORDER, TARGET_JOINTS_ORDER
    )
    joint_cam = reorder_joints(joint_cam, HOT3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_rel = reorder_joints(joint_rel, HOT3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)

    # joint_valid: HOT3D 没有 thumb_1
    # thumb_1 无效
    joint_valid = np.ones_like(joint_cam[:, 0])
    joint_valid[1] = 0

    # mano_pose, mano_shape, mano_valid
    mano_pose = sample["mano_pose"]  # [48]
    mano_shape = sample["mano_shape"]
    mano_valid = sample["mano_valid"]

    # focal, princpt
    focal = sample["focal"]
    princpt = sample["princpt"]

    return {
        "img_path": img_path,  # 不需要存入最终 Tensor
        "img_bytes": img_bytes,
        "handedness": handedness,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_hand_bbox": joint_hand_bbox,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_valid": joint_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "mano_valid": mano_valid,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
    }


# === 多线程处理函数 ===
def process_sequence_batch(batch_seqs, worker_id):
    # 全是有效帧
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # 构造属于当前 Worker 的文件名模式
    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)

    processed_count = 0

    # 开启ShardWriter
    # batch_seqs 有 n 个 clip，每个 annots 是一个 clip，即连续帧的数据
    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for cx, annots in enumerate(batch_seqs):
            # 处理单个 clip
            clip_frames = []
            clip_descs = []
            valid_clip = True

            # 每个 clip 属于同一个序列，初始化 data_provider
            sequence_name = annots[0]["sequence_name"]
            sequence_path = os.path.join(HOT3D_ROOT, sequence_name)
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=sequence_path,
                object_library=None,
                mano_hand_model=mano_hand_model,
            )

            device_data_provider = hot3d_data_provider.device_data_provider

            for i in range(len(annots)):  # 逐帧处理 clip
                sample = annots[i]
                try:
                    processed_frame = process_single_annot(
                        sample, i, device_data_provider
                    )
                    clip_descs.append({})
                    clip_frames.append(processed_frame)
                except Exception as ex:
                    print(f"[Worker {worker_id}] Error: {ex}")
                    valid_clip = False
                    break

            if not valid_clip or len(clip_frames) == 0:
                continue

            # === 数据写入wds ===
            key_str = f"{worker_id}_{cx}_hot3d"

            # 1. Pickle: List[bytes] -> bytes
            img_bytes_pickle = pickle.dumps([f["img_bytes"] for f in clip_frames])

            # 2. JSON: dict/list/str -> str (Writer 会自动 encode 为 utf-8)
            imgs_path_json = json.dumps([v["img_path"] for v in clip_frames])
            handedness_json = json.dumps(clip_frames[0]["handedness"])
            desc_json = json.dumps(clip_descs)

            wds_sample = {
                "__key__": key_str,
                "imgs_path.json": imgs_path_json,
                "img_bytes.pickle": img_bytes_pickle,
                "handedness.json": handedness_json,
                "additional_desc.json": desc_json,
            }

            # 3. Numpy: ShardWriter 默认支持 .npy 自动处理 (np.save logic)
            for k in NUMPY_KEYS:
                data_list = [f[k] for f in clip_frames]
                wds_sample[f"{k}.npy"] = np.stack(data_list)

            sink.write(wds_sample)
            processed_count += 1

    return processed_count


def main():
    clips = prepare_data()
    total_seqs = len(clips)
    chunk_size = math.ceil(total_seqs / NUM_WORKERS)
    chunks = [clips[i : i + chunk_size] for i in range(0, total_seqs, chunk_size)]

    print(f"Total Sequences: {total_seqs}")
    print(f"Starting {NUM_WORKERS} workers processing ~{chunk_size} sequences each...")

    process_args = []
    for i in range(len(chunks)):
        process_args.append((chunks[i], i))

    # 将 clips 切成 chunks，每个 chunk 包含 n 个 clip，每个线程处理一个 chunk
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.starmap(process_sequence_batch, process_args)

    print(f"All done! Total clips processed: {sum(results)}")

    return


if __name__ == "__main__":
    main()
