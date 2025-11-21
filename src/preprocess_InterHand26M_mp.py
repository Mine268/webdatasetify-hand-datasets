import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
import tqdm
from collections import defaultdict
from copy import deepcopy
import cv2
import torch
from pathlib import Path
import json
import webdataset as wds
import multiprocessing
from functools import partial
import math
import pickle  # 新增引用

# 假设 utils 中包含了 world2cam, cam2pixel, process_bbox, get_bbox, sanitize_bbox, reorder_joints 以及相关常量
# 请确保 utils.py 在同一目录下
from utils import *

# ================= 配置区域 =================
IH26M_ROOT = r"/data_1/datasets_temp/InterHand2.6M_5fps_batch1/"
SPLIT = "train"  # train val test
# 注意：输出路径增加了 {worker_id} 占位符，防止多进程文件名冲突
OUTPUT_PATTERN = f"ih26m_wds_output/ih26m_{SPLIT}-worker{{worker_id}}-%06d.tar"
MAX_COUNT = 100000  # 已修改：大幅增加数量限制，让切割主要由 MAX_SIZE 决定
MAX_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
NUM_WORKERS = 30 # 建议设置为 CPU 核心数 - 2

os.makedirs(os.path.dirname(OUTPUT_PATTERN), exist_ok=True)

# ================= 全局数据加载 (主进程执行，Linux下子进程通过COW共享) =================
# 为了避免在 spawn 模式下重复加载大文件，建议在 Linux 环境运行
print(f"Loading Annotations for {SPLIT}...")
dataset = COCO(osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_data.json"))

with open(osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_camera.json"), "r") as f:
    cameras = json.load(f)

with open(osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_joint_3d.json"), "r") as f:
    joints = json.load(f)

with open(osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_MANO_NeuralAnnot.json"), "r") as f:
    mano_params = json.load(f)

# Joint Set Definition
joint_set = {
    "joint_num": 42,
    "joints_name": (
        "R_Thumb_4", "R_Thumb_3", "R_Thumb_2", "R_Thumb_1",
        "R_Index_4", "R_Index_3", "R_Index_2", "R_Index_1",
        "R_Middle_4", "R_Middle_3", "R_Middle_2", "R_Middle_1",
        "R_Ring_4", "R_Ring_3", "R_Ring_2", "R_Ring_1",
        "R_Pinky_4", "R_Pinky_3", "R_Pinky_2", "R_Pinky_1",
        "R_Wrist",
        "L_Thumb_4", "L_Thumb_3", "L_Thumb_2", "L_Thumb_1",
        "L_Index_4", "L_Index_3", "L_Index_2", "L_Index_1",
        "L_Middle_4", "L_Middle_3", "L_Middle_2", "L_Middle_1",
        "L_Ring_4", "L_Ring_3", "L_Ring_2", "L_Ring_1",
        "L_Pinky_4", "L_Pinky_3", "L_Pinky_2", "L_Pinky_1",
        "L_Wrist",
    ),
    "flip_pairs": [(i, i + 21) for i in range(21)],
}
joint_set["joint_type"] = {
    "right": np.arange(0, joint_set["joint_num"] // 2),
    "left": np.arange(joint_set["joint_num"] // 2, joint_set["joint_num"]),
}
joint_set["root_joint_idx"] = {
    "right": joint_set["joints_name"].index("R_Wrist"),
    "left": joint_set["joints_name"].index("L_Wrist"),
}

# 定义需要堆叠成 Numpy 数组的字段
NUMPY_KEYS = [
    "hand_bbox", "joint_img", "joint_hand_bbox",
    "joint_cam", "joint_rel", "joint_valid",
    "mano_pose", "mano_shape", "timestamp",
    "focal", "princpt"
]

# ================= 处理函数定义 =================

def process_single_annot(sample, h):
    """单帧处理函数，将在 Worker 进程中被调用"""
    intrinsics = np.array([
        [sample["cam_param"]["focal"][0], 0, sample["cam_param"]["princpt"][0]],
        [0, sample["cam_param"]["focal"][1], sample["cam_param"]["princpt"][1]],
        [0, 0, 1],
    ])

    img_path_parts = Path(sample["img_path"]).parts
    # 兼容路径，取最后4部分或者根据实际情况调整
    img_sub_path = os.path.join(*img_path_parts[-4:])

    handedness = "right" if h == "r" else "left"
    bbox_tight = sample[f"{h}hand_bbox"]  # [xyxy]
    joint_cam = sample["joint_cam"][:21] if h == "r" else sample["joint_cam"][21:]
    joint_valid = (
        sample["joint_valid"][:21] * sample["joint_trunc"][:21]
        if h == "r"
        else sample["joint_valid"][21:] * sample["joint_trunc"][21:]
    )[:, 0]

    # vertices
    r = np.array(sample["cam_param"]["R"]).astype(np.float32)
    # t = np.array(sample["cam_param"]["t"]).astype(np.float32) # unused in snippet

    pose =  np.array(sample["mano_param"][handedness]["pose"]).astype(np.float32)
    shape = np.array(sample["mano_param"][handedness]["shape"]).astype(np.float32)
    # trans = np.array(sample["mano_param"][handedness]["trans"]).astype(np.float32) # unused in snippet

    # MANO param trans
    root_pose = pose[:3]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(r, root_pose))
    pose[:3] = root_pose[:, 0]

    pose =  torch.from_numpy(pose)[None, ...]
    shape = torch.from_numpy(shape)[None, ...]
    # trans = torch.from_numpy(trans)[None, ...]

    joint_rel = joint_cam - joint_cam[-1:]

    # project joint img
    joint_img = joint_cam @ intrinsics.T
    joint_img = joint_img[:, :-1] / joint_img[:, -1:]
    new_intrinsics = intrinsics.copy()
    new_intrinsics[0, 2] -= bbox_tight[0]
    new_intrinsics[1, 2] -= bbox_tight[1]
    joint_bbox_img = joint_cam @ new_intrinsics.T
    joint_bbox_img = joint_bbox_img[:, :-1] / joint_bbox_img[:, -1:]

    focal = [intrinsics[0, 0].item(), intrinsics[1, 1].item()]
    princpt = [intrinsics[0, 2].item(), intrinsics[1, 2].item()]

    # Image IO
    full_img_path = osp.join(IH26M_ROOT, "images", SPLIT, img_sub_path)
    img = cv2.imread(full_img_path)
    if img is None:
         # 尝试备用路径逻辑，根据实际数据集结构调整
         full_img_path = osp.join(IH26M_ROOT, "images", SPLIT, sample["img_path"])
         img = cv2.imread(full_img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {full_img_path}")

    # Encode to WebP
    success, encoded_img_arr = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise Exception(f"Failed to encode the image: {full_img_path}")

    # === 修改：转换为 bytes 可以在 pickle 时更通用，也兼容测试脚本的 np.frombuffer ===
    img_bytes = encoded_img_arr.tobytes()

    # Reorder & Organize
    # 注意: IH26M_RJOINTS_ORDER 和 TARGET_JOINTS_ORDER 需来自 utils
    joint_img = reorder_joints(joint_img, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_hand_bbox = reorder_joints(joint_bbox_img, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_cam = reorder_joints(joint_cam, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_rel = reorder_joints(joint_rel, IH26M_RJOINTS_ORDER, TARGET_JOINTS_ORDER)

    mano_pose = pose[0].numpy()
    mano_shape = shape[0].numpy()
    timestamp = sample["frame_idx"] * 200.0
    focal = np.array(focal)
    princpt = np.array(princpt)

    annot_item = {
        # "img_path": img_sub_path, # 不需要存入最终 Tensor
        "img_bytes": img_bytes,
        "handedness": handedness,
        "hand_bbox": bbox_tight,
        "joint_img": joint_img,
        "joint_hand_bbox": joint_hand_bbox,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_valid": joint_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
        "aid": sample['aid'] # 临时用于 debug/desc，不存入 numpy
    }

    return annot_item

def process_sequence_batch(batch_seqs, worker_id):
    """
    Worker 进程的主入口。
    batch_seqs: List of ((capture_id, seq_name, cam_id), [frames...])
    """
    # 优化：防止 OpenCV 多线程与 Multiprocessing 争抢资源
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # 构造属于当前 Worker 的文件名模式
    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)

    processed_count = 0

    # 开启 ShardWriter
    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        # 遍历分配给该 Worker 的所有序列
        for (capture_id, seq_name, cam_id), annots in batch_seqs:

            # 确保按帧顺序排列
            annots.sort(key=lambda x: x["frame_idx"])

            for h in ['r', 'l']:
                start = 0
                while start < len(annots):
                    # 1. 寻找连续有效帧的起始点
                    while start < len(annots) and annots[start][f"{h}hand_bbox"] is None:
                        start += 1

                    # 2. 寻找连续有效帧的结束点
                    end = start
                    while end < len(annots) and annots[end][f"{h}hand_bbox"] is not None:
                        end += 1

                    # 如果没有找到有效片段，跳出
                    if start >= len(annots):
                        break

                    # ================= 处理单个 Clip (start -> end) =================
                    clip_frames = []
                    clip_descs = []

                    current_start_frame_idx = annots[start]['frame_idx']
                    valid_clip = True

                    # 3. 逐帧处理
                    for i in range(start, end):
                        sample = annots[i]
                        try:
                            processed_frame = process_single_annot(sample, h)

                            desc = {
                                "capture_id": capture_id,
                                "seq_name": seq_name,
                                "cam_id": cam_id,
                                "aid": processed_frame["aid"]
                            }
                            # aid 不写入 tensor
                            del processed_frame["aid"]

                            clip_frames.append(processed_frame)
                            clip_descs.append(desc)

                        except Exception as ex:
                            print(f"[Worker {worker_id}] Error Cap={capture_id} Seq={seq_name} Frame={sample['frame_idx']}: {ex}")
                            valid_clip = False
                            break # 只要有一帧坏了，整个 clip 丢弃或者截断，这里选择丢弃

                    # 更新 start
                    start = end

                    if not valid_clip or len(clip_frames) == 0:
                        continue

                    # ================= 聚合数据并写入 WDS =================
                    key_str = f"{capture_id}_{seq_name}_{cam_id}_{h}_{current_start_frame_idx}"

                    # === 关键修复：显式序列化 Pickle 和 JSON ===
                    # ShardWriter 不会自动处理复杂对象（如 list/dict）到 bytes 的转换

                    # 1. Pickle: List[bytes] -> bytes
                    img_bytes_pickle = pickle.dumps([f["img_bytes"] for f in clip_frames])

                    # 2. JSON: dict/list/str -> str (Writer 会自动 encode 为 utf-8)
                    handedness_json = json.dumps(clip_frames[0]["handedness"])
                    desc_json = json.dumps(clip_descs)

                    wds_sample = {
                        "__key__": key_str,
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

# ================= 主程序 =================

def main():
    print("Building data list from annotations...")
    aid_list = list(dataset.anns.keys())
    datalist = []

    # ---------------- Step 1: 预处理所有 Annotation (CPU 轻量级操作) ----------------
    # 这部分还是单线程跑，因为它很快，主要是字典查找
    for aid in tqdm.tqdm(aid_list, ncols=100, desc="Parsing Annotations"):
        ann = dataset.anns[aid]
        image_id = ann["image_id"]
        img = dataset.loadImgs(image_id)[0]
        img_width, img_height = img["width"], img["height"]
        img_path = img["file_name"]

        capture_id = img["capture"]
        seq_name = img["seq_name"]
        cam = img["camera"]
        frame_idx = img["frame_idx"]
        hand_type = ann["hand_type"]

        # camera parameters
        # 注意：这里只做轻量级的 numpy 转换，不做复杂计算
        t, R = np.array(cameras[str(capture_id)]["campos"][str(cam)], dtype=np.float32).reshape(3), \
               np.array(cameras[str(capture_id)]["camrot"][str(cam)], dtype=np.float32).reshape(3, 3)
        t = -np.dot(R, t.reshape(3, 1)).reshape(3)
        focal, princpt = np.array(cameras[str(capture_id)]["focal"][str(cam)], dtype=np.float32).reshape(2), \
                         np.array(cameras[str(capture_id)]["princpt"][str(cam)], dtype=np.float32).reshape(2)
        cam_param = {"R": R, "t": t, "focal": focal, "princpt": princpt}

        # Validity check
        joint_trunc = np.array(ann["joint_valid"], dtype=np.float32).reshape(-1, 1)
        joint_trunc[joint_set["joint_type"]["right"]] *= joint_trunc[joint_set["root_joint_idx"]["right"]]
        joint_trunc[joint_set["joint_type"]["left"]] *= joint_trunc[joint_set["root_joint_idx"]["left"]]
        if np.sum(joint_trunc) == 0:
            continue

        joint_valid = np.array(joints[str(capture_id)][str(frame_idx)]["joint_valid"], dtype=np.float32).reshape(-1, 1)
        joint_valid[joint_set["joint_type"]["right"]] *= joint_valid[joint_set["root_joint_idx"]["right"]]
        joint_valid[joint_set["joint_type"]["left"]] *= joint_valid[joint_set["root_joint_idx"]["left"]]
        if np.sum(joint_valid) == 0:
            continue

        # joint coordinates
        joint_world = np.array(joints[str(capture_id)][str(frame_idx)]["world_coord"], dtype=np.float32).reshape(-1, 3)
        joint_cam = world2cam(joint_world, R, t)
        joint_cam[np.tile(joint_valid == 0, (1, 3))] = 1.0
        joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

        # body bbox
        body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
        body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
        if body_bbox is None:
            continue

        # left hand bbox
        if np.sum(joint_trunc[joint_set["joint_type"]["left"]]) == 0:
            lhand_bbox = None
        else:
            lhand_bbox = get_bbox(
                joint_img[joint_set["joint_type"]["left"], :],
                joint_trunc[joint_set["joint_type"]["left"], 0],
                extend_ratio=1.2,
            )
            lhand_bbox = sanitize_bbox(lhand_bbox, img_width, img_height)
        if lhand_bbox is None:
            joint_valid[joint_set["joint_type"]["left"]] = 0
            joint_trunc[joint_set["joint_type"]["left"]] = 0
        else:
            lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy

        # right hand bbox
        if np.sum(joint_trunc[joint_set["joint_type"]["right"]]) == 0:
            rhand_bbox = None
        else:
            rhand_bbox = get_bbox(
                joint_img[joint_set["joint_type"]["right"], :],
                joint_trunc[joint_set["joint_type"]["right"], 0],
                extend_ratio=1.2,
            )
            rhand_bbox = sanitize_bbox(rhand_bbox, img_width, img_height)
        if rhand_bbox is None:
            joint_valid[joint_set["joint_type"]["right"]] = 0
            joint_trunc[joint_set["joint_type"]["right"]] = 0
        else:
            rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy

        if lhand_bbox is None and rhand_bbox is None:
            continue

        # mano parameters
        try:
            mano_param = mano_params[str(capture_id)][str(frame_idx)].copy()
            if lhand_bbox is None:
                mano_param["left"] = None
            if rhand_bbox is None:
                mano_param["right"] = None
        except KeyError:
            mano_param = {"right": None, "left": None}

        datalist.append({
            "aid": aid,
            "capture_id": capture_id,
            "seq_name": seq_name,
            "cam_id": cam,
            "frame_idx": frame_idx,
            "img_path": img_path,
            # "img_shape": (img_height, img_width),
            # "body_bbox": body_bbox,
            "lhand_bbox": lhand_bbox,
            "rhand_bbox": rhand_bbox,
            # "joint_img": joint_img,
            "joint_cam": joint_cam,
            "joint_valid": joint_valid,
            "joint_trunc": joint_trunc,
            "cam_param": cam_param,
            "mano_param": mano_param,
            "hand_type": hand_type,
        })

    # ---------------- Step 2: 将帧聚合为序列 (Dict) ----------------
    print("Grouping frames into sequences...")
    seq_list = defaultdict(list)
    for item in tqdm.tqdm(datalist, ncols=100, desc="Grouping"):
        # 浅拷贝，避免修改原始引用（虽然 datalist 之后不再用了）
        item = item.copy()
        capture_id = item["capture_id"]
        seq_name = item["seq_name"]
        cam_id = item["cam_id"]

        # 移除多余 Key 以节省子进程序列化开销
        del item["capture_id"]
        del item["seq_name"]
        del item["cam_id"]

        seq_list[(capture_id, seq_name, cam_id)].append(item)

    # ---------------- Step 3: 切分任务并启动多进程 ----------------
    all_sequences = list(seq_list.items())
    total_seqs = len(all_sequences)
    chunk_size = math.ceil(total_seqs / NUM_WORKERS)

    # 将 List 切分为多个 Chunks
    chunks = [all_sequences[i:i + chunk_size] for i in range(0, total_seqs, chunk_size)]

    print(f"Total Sequences: {total_seqs}")
    print(f"Starting {NUM_WORKERS} workers processing ~{chunk_size} sequences each...")

    # 构建参数：(batch_seqs, worker_id)
    process_args = []
    for i in range(len(chunks)):
        process_args.append((chunks[i], i))

    # 启动进程池
    # 注意：在 Linux 上默认使用 fork，子进程可以直接访问全局的 dataset, cameras 等大对象
    # 无需 pickle 传递，非常高效。
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.starmap(process_sequence_batch, process_args)

    print(f"All done! Total clips processed: {sum(results)}")

if __name__ == "__main__":
    main()