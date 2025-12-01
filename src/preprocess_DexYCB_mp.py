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

# === 利用环境变量进行配置 ===
DEX_ROOT = "/data_1/datasets_temp/dexycb"
SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]
SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]
SETUP = os.environ.get("SETUP", "s1")  # s0 s1 s2 s3
SPLIT = os.environ.get("SPLIT", "val")  # train val test
OUTPUT_PATTERN = f"dexycb_{SETUP}_{SPLIT}_wds_output/dexycb_{SETUP}_{SPLIT}-worker{{worker_id}}-%06d.tar"
MAX_COUNT = 100000  # 已修改：大幅增加数量限制，让切割主要由 MAX_SIZE 决定
MAX_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
NUM_WORKERS = 30 # 建议设置为 CPU 核心数 - 2

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

# Seen subjects, camera views, grasped objects.
if SETUP == "s0":
    if SPLIT == "train":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 != 4]
    if SPLIT == "val":
        subject_ind = [0, 1]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]
    if SPLIT == "test":
        subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]

# Unseen subjects.
if SETUP == "s1":
    if SPLIT == "train":
        subject_ind = [0, 1, 2, 3, 4, 5, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
    if SPLIT == "val":
        subject_ind = [6]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
    if SPLIT == "test":
        subject_ind = [7, 8]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))

# Unseen camera views.
if SETUP == "s2":
    if SPLIT == "train":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5]
        sequence_ind = list(range(100))
    if SPLIT == "val":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [6]
        sequence_ind = list(range(100))
    if SPLIT == "test":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [7]
        sequence_ind = list(range(100))

# Unseen grasped objects.
if SETUP == "s3":
    if SPLIT == "train":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)]
    if SPLIT == "val":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
    if SPLIT == "test":
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

# === 数据加载与预处理 ===
# mano姿态的pca参数，dexycb的mano是pca形式的
_mano_tmp = np.load("models/mano/mano_lr_pca.npz")
mano_pca_comps = {k: _mano_tmp[k] for k in _mano_tmp.files}
_mano_tmp.close()
clips = []

for si in tqdm(subject_ind, ncols=50):
    subject = SUBJECTS[si]
    for cap in os.listdir(osp.join(DEX_ROOT, subject)):
        # load shape from meta.yml
        with open(osp.join(DEX_ROOT, subject, cap, "meta.yml"), "r") as f:
            meta = yaml.safe_load(f)
        num_frames = meta["num_frames"]
        extr_path = meta["extrinsics"]
        handedness = meta["mano_sides"][0]
        mano_path = meta["mano_calib"][0]

        # load the mano shape
        with open(
            osp.join(DEX_ROOT, "calibration", f"mano_{mano_path}", "mano.yml"), "r"
        ) as f:
            mano_shape = yaml.safe_load(f)
        # mano_shape = np.array(mano_shape["betas"])  # [10]

        # load the extrinsics
        with open(
            osp.join(
                DEX_ROOT, "calibration", f"extrinsics_{extr_path}", "extrinsics.yml"
            ),
            "r",
        ) as f:
            extrinsics = yaml.load(f, Loader=yaml.FullLoader)

        for sri in serial_ind:
            prev, clip = -1, []
            serial = meta["serials"][sri]
            # load the intrinsics
            with open(
                osp.join(DEX_ROOT, "calibration/intrinsics", f"{serial}_640x480.yml"),
                "r",
            ) as f:
                intrinsics = yaml.load(f, Loader=yaml.FullLoader)
            for i in [x for x in sequence_ind if x < num_frames]:
                annot_npz = np.load(
                    osp.join(DEX_ROOT, subject, cap, serial, f"labels_{i:06}.npz")
                )
                if bool(np.all(annot_npz["joint_2d"] == -1)):  # invalid annotations
                    continue
                else:  # valid annotations
                    item = (
                        serial,  # camera
                        osp.join(subject, cap, serial, f"color_{i:06}.jpg"),  # image
                        handedness,
                        tuple(mano_shape["betas"]),  # mano_shape
                        intrinsics["color"],  # intr
                        extrinsics["extrinsics"][serial],  # extr
                        osp.join(subject, cap, serial, f"labels_{i:06}.npz"),  # annot
                    )
                    if prev == -1 or prev + 1 == i:
                        clip.append(item)
                        prev = i
                    else:
                        clips.append(clip)
                        clip = [item]
                        prev = i
            clips.append(clip)

# === 将单个标注转换为wds的格式 ===
def process_single_annot(sample, label, idx: int):
    """
    Args:
        idx: 该样本在序列中的位置，用于时间戳的计算
    """
    # img_path
    img_path = sample[1]  # str

    # img_bytes
    img = cv2.imread(osp.join(DEX_ROOT, img_path))
    success, img_bytes = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    if not success:
        raise Exception(f"Failed to encode the image: {osp.join(DEX_ROOT, img_path)}")

    # handedness
    handedness = sample[2]  # right, left

    # joint_img, hand_bbox, joint_hand_bbox
    joint_img = label["joint_2d"][0]  # [J,2]
    xm, ym = np.min(joint_img, axis=0)
    xM, yM = np.max(joint_img, axis=0)
    hand_bbox = np.stack([xm, ym, xM, yM], axis=-1)
    joint_hand_bbox = joint_img - hand_bbox[None, :2]

    # joint_cam, joint_rel, joint_valid
    joint_cam = label["joint_3d"][0] * 1e3  # [J,3] 毫米单位
    joint_rel = joint_cam - joint_cam[:1]
    joint_valid = np.ones_like(joint_cam[:, 0])

    # mano_pose, mano_shape, mano_valid
    mano_pose_pca = label["pose_m"][:, :48]  # 最后三个元素是位移，用不着
    mano_pose_pca[:, 3:] = mano_pose_pca[:, 3:] @ mano_pca_comps[handedness]
    mano_pose = mano_pose_pca[0]  # [48]
    mano_shape = np.array(sample[3])
    mano_valid = True

    # timestamp
    # 30 fps, time per frame = 1/30 s = 1000/30 ms = 100/3 = 33.33333
    timestamp = idx * 33.3333333

    # focal, princpt
    focal = np.array([sample[4]["fx"], sample[4]["fy"]])
    princpt = np.array([sample[4]["ppx"], sample[4]["ppy"]])

    return {
        "img_path": img_path, # 不需要存入最终 Tensor
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
            # 读取标注信息
            labels = []
            for i in range(len(annots)):
                label = np.load(osp.join(DEX_ROOT, annots[i][-1]))
                labels.append(label)
            # 划分有效帧
            start = 0
            while start < len(annots):
                # 寻找连续有效帧区间
                while start < len(annots) and np.allclose(labels[start]["pose_m"], 0):
                    start += 1
                end = start
                while end < len(annots) and not np.allclose(labels[end]["pose_m"], 0):
                    end += 1
                if start >= len(annots):
                    break

                # 处理单个 clip
                clip_frames = []
                clip_descs = []
                valid_clip = True

                for i in range(start, end):
                    sample = annots[i]
                    try:
                        processed_frame = process_single_annot(sample, labels[i], i)
                        clip_descs.append({})
                        clip_frames.append(processed_frame)
                    except Exception as ex:
                        print(f"[Worker {worker_id}] Error: {ex}")
                        valid_clip = False
                        break

                # 更新start
                start = end

                if not valid_clip or len(clip_frames) == 0:
                    continue

                # === 数据写入wds ===
                key_str = f"{worker_id}_{cx}_dexycb"

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

# process_sequence_batch(clips[:10], 0)
# === 主程序 ===
def main():
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
