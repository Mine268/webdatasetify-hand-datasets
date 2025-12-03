import os
import os.path as osp
import pickle
import yaml
import json
from tqdm import tqdm
import math
import multiprocessing
import numpy as np
import cv2
import webdataset as wds

from utils import *

HO3D_ROOT = "/mnt/qnap/data/datasets/ho3d_v3/ho3d_v3"
SPLIT = "evaluation"
OUTPUT_PATTERN = f"ho3d_{SPLIT}_wds_output/ho3d_{SPLIT}-worker{{worker_id}}-%06d.tar"
MAX_COUNT = 100000  # 已修改：大幅增加数量限制，让切割主要由 MAX_SIZE 决定
MAX_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
NUM_WORKERS = 1 # 建议设置为 CPU 核心数 - 2

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

def prepare_data():
    # === 读取完整的三维关键点标注 ===
    with open(osp.join(HO3D_ROOT, "evaluation.txt"), "r") as f:
        annot_idx_list = f.read().strip().split("\n")
    annot_idx_map = {v: i for i, v in enumerate(annot_idx_list)}
    with open(osp.join(HO3D_ROOT, "evaluation_xyz.json"), "r") as f:
        joints_gt = json.load(f)

    # === 读取原始文件形成序列组 ===
    annot_seqs = []
    for seq in os.listdir(osp.join(HO3D_ROOT, SPLIT)):
        seq_root = osp.join(HO3D_ROOT, SPLIT, seq, "meta")
        jpeg_files = []
        for jpeg_file in os.listdir(seq_root):
            if jpeg_file.endswith(".pkl"):
                jpeg_files.append(jpeg_file[:-4])  # remove .jpg suffix
        jpeg_files.sort()

        if jpeg_files:
            current_group = []
            prev_num = -1
            for file in jpeg_files:
                current_num = int(file)
                # check the annotation exists
                with open(osp.join(HO3D_ROOT, SPLIT, seq, "meta", file + ".pkl"), "rb") as f:
                    annot = pickle.load(f)
                if not (
                    "handJoints3D" in annot and annot["handJoints3D"] is not None and
                    "camMat" in annot and annot["camMat"] is not None
                ):
                    continue
                # if not (
                #     annot["handJoints3D"] is not None
                #     and annot["camMat"] is not None
                #     and annot["handPose"] is not None
                #     and annot["handBeta"] is not None
                # ):
                #     continue  # skip the invalid annot
                if current_group == [] or prev_num + 1 == current_num:
                    current_group.append(
                        (
                            osp.join(SPLIT, seq, "rgb", file + ".jpg"),
                            osp.join(SPLIT, seq, "meta", file + ".pkl")
                        )
                    )
                    prev_num = current_num
                else:
                    annot_seqs.append(current_group)
                    current_group = []
                    prev_num = current_num
            annot_seqs.append(current_group)

    # === 构造原始标注 ===
    R_x_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    annot_all = []
    for seq in tqdm(annot_seqs, ncols=70):
        annot_seq_all = []
        for img_path, annot_path in seq:
            # 读取图像
            img = cv2.imread(osp.join(HO3D_ROOT, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 读取标注
            with open(osp.join(HO3D_ROOT, annot_path), "rb") as f:
                annot = pickle.load(f)

            # joint_cam, joint_rel
            subject_name = osp.dirname(img_path).split("/")[1]
            img_idx = osp.basename(img_path).split(".")[0]
            annot_ix = annot_idx_map[f"{subject_name}/{img_idx}"]
            joint_cam = np.array(joints_gt[annot_ix])

            joint_cam = joint_cam * np.array([[1, -1, -1]]) * 1e3
            joint_rel = joint_cam - joint_cam[:1]
            joint_img = joint_cam @ annot["camMat"].T
            joint_img = joint_img[..., :2] / joint_img[..., 2:]

            # focal, princpt
            focal = np.array([annot["camMat"][0, 0].item(), annot["camMat"][1, 1].item()])
            princpt = np.array([annot["camMat"][0, 2].item(), annot["camMat"][1, 2].item()])

            # manually compute the bounding box
            x1 = joint_img[:, 0].min().item()
            x2 = joint_img[:, 0].max().item()
            y1 = joint_img[:, 1].min().item()
            y2 = joint_img[:, 1].max().item()
            # expand by 1.1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wx, wy = (x2 - x1) / 2, (y2 - y1) / 2
            x1, x2 = cx - wx * 1.2, cx + wx * 1.2
            y1, y2 = cy - wy * 1.2, cy + wy * 1.2
            # fill the tight bbox
            bbox_tight = np.array([x1, y1, x2, y2], dtype=np.float32)
            joint_bbox_img = joint_img - bbox_tight[None, :2]

            # no MANO provided
            mano_pose = None
            mano_shape = None

            annot_seq_all.append({
                "img_path": img_path,
                "flip": False,
                "bbox_tight": bbox_tight,
                "joint_img": joint_img,
                "joint_bbox_img": joint_bbox_img,
                "joint_cam": joint_cam,
                "joint_rel": joint_rel,
                "mano_pose": mano_pose,  # substracted by mean
                "mano_shape": mano_shape,
                "focal": focal,
                "princpt": princpt,
            })
        annot_all.append(annot_seq_all)

    return annot_all

def process_single_annot(sample, idx: int):
    # img_path
    img_path = sample["img_path"]

    # img_bytes
    img = cv2.imread(osp.join(HO3D_ROOT, img_path))
    success, img_bytes = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise Exception(f"Failed to encode the image: {osp.join(HO3D_ROOT, img_path)}")

    # handedness
    handedness = "right"

    # joint_img, hand_bbox, joint_hand_bbox
    joint_img = sample["joint_img"]
    hand_bbox = sample["bbox_tight"]
    joint_hand_bbox = joint_img - hand_bbox[None, :2]

    # joint_cam, joint_rel, joint_valid
    joint_cam = sample["joint_cam"]
    joint_rel = sample["joint_rel"]
    joint_valid = np.ones_like(joint_cam[:, 0])

    # reorder joints order to target
    joint_img = reorder_joints(joint_img, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_hand_bbox = reorder_joints(joint_hand_bbox, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_cam = reorder_joints(joint_cam, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)
    joint_rel = reorder_joints(joint_rel, HO3D_JOINTS_ORDER, TARGET_JOINTS_ORDER)

    # mano_pose, mano_shape, mano_valid
    if sample["mano_pose"] is not None:
        mano_pose = sample["mano_pose"]
        mano_shape = sample["mano_shape"]
        mano_valid = True
    else:
        mano_pose = np.zeros(shape=(48,))
        mano_shape = np.zeros(shape=(10,))
        mano_valid = False

    # focal, princpt
    focal = sample["focal"]
    princpt = sample["princpt"]

    # timestamp
    timestamp = idx * 33.33333

    return {
        "img_path": img_path,
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
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # 构造属于当前 Worker 的文件名模式
    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)

    processed_count = 0

    # 开启ShardWriter
    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for cx, annots in enumerate(batch_seqs):
            # 之前已经进行了提前处理，全都是有效帧
            start, end = 0, len(annots)

            # 处理单个 clip
            clip_frames = []
            clip_descs = []
            valid_clip = True

            for i in range(start, end):
                sample = annots[i]
                try:
                    processed_frame = process_single_annot(sample, i)
                    clip_descs.append({})
                    clip_frames.append(processed_frame)
                except Exception as ex:
                    print(f"[Worker {worker_id}] Error: {ex}")
                    valid_clip = False
                    break

            if not valid_clip and len(clip_frames) == 0:
                continue

            # === 数据写入wds ===
            key_str = f"{worker_id}_{cx}_ho3d"

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

# === 主程序 ===
def main():
    clips = prepare_data()

    total_seqs = len(clips)
    chunk_size = math.ceil(total_seqs / NUM_WORKERS)
    chunks = [clips[i:i + chunk_size] for i in range(0, total_seqs, chunk_size)]

    print(f"Total Sequences: {total_seqs}")
    print(f"Starting {NUM_WORKERS} workers processing ~{chunk_size} sequences each...")

    process_args = []
    for i in range(len(chunks)):
        process_args.append((chunks[i], i))

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.starmap(process_sequence_batch, process_args)

    print(f"All done! Total clips processed: {sum(results)}")

if __name__ == "__main__":
    main()