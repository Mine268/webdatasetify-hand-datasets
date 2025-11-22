"""
Webdataset Dataloader
"""
import glob
import copy
from functools import partial
import numpy as np
import webdataset as wds
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision

NPY_KEYS = [
    "hand_bbox.npy", "joint_img.npy", "joint_hand_bbox.npy",
    "joint_cam.npy", "joint_rel.npy", "joint_valid.npy",
    "mano_pose.npy", "mano_shape.npy", "timestamp.npy",
    "focal.npy", "princpt.npy"
]

COLLATE_LIST_KEYS = [
    "imgs", "handedness", "__key__"
]

def clip_to_t_frames(num_frames, stride, source):
    """
    将序列样本拆分为小片小片的连续样本
    """
    for sample in source:
        img_list = sample["img_bytes.pickle"]
        handedness = sample["handedness.json"] # "l" or "r"
        total_frames = len(img_list)
        if total_frames < num_frames:
            continue

        total_samples = (total_frames - num_frames) // stride + 1

        for i in range(total_samples):
            start = i * stride
            end = start + num_frames

            # 构建输出样本 (包含 T 帧数据)
            sub_sample = {
                # 构造唯一的 Key: 原Key_切片序号
                "__key__": f"{sample['__key__']}_{i:04d}",
                "handedness": handedness,
                # 图片字节流：直接切片 List
                "imgs_bytes": img_list[start:end],
            }
            for key in NPY_KEYS:
                if key in sample:
                    out_key = key.replace(".npy", "") # 去后缀
                    # 这里执行的是 Numpy 的第一维切片操作
                    sub_sample[out_key] = sample[key][start:end]

            yield sub_sample

def preprocess_frame(sample):
    """将图像二进制流转换为图片"""
    # 1. 图片解码: Bytes (WebP) -> PIL -> Tensor
    # Writer 中使用的是 cv2.imencode(".webp")，这里用 PIL 打开兼容性很好
    imgs_tensor = []
    for img_bytes in sample["imgs_bytes"]:
        buffer_np = np.frombuffer(img_bytes, dtype=np.uint8).copy()
        buffer = torch.from_numpy(buffer_np)
        img = torchvision.io.decode_webp(buffer)
        imgs_tensor.append(img)
    imgs_tensor = torch.stack(imgs_tensor)

    # 2. 处理其他 Numpy 字段
    result = {
        "imgs": imgs_tensor,
        "handedness": sample["handedness"], # 此时还是 str, collate 时可能需要特殊处理或 drop
    }

    # 自动将所有 numpy 字段转为 Tensor
    for key in sample:
        if key not in ["__key__", "imgs_bytes", "handedness"]:
            # 确保是 float32 (根据你的 writer 逻辑，大部分已经是 float32)
            val = sample[key]
            if isinstance(val, np.ndarray):
                result[key] = torch.from_numpy(val).float()
            else:
                result[key] = torch.tensor(val)

    return result

def collate_fn(batch_wds):
    """对batch数据进行重整，由于图像大小不一，特判使用List"""
    batch_filter = [b for b in batch_wds if b is not None]
    if len(batch_filter) == 0:
        return {}

    collated = {}
    for key in batch_filter[0].keys():
        if key in COLLATE_LIST_KEYS:
            collated[key] = [sample[key] for sample in batch_filter]
        else:
            values = [sample[key] for sample in batch_filter]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values

    return collated

def get_dataloader(
    url, num_frames: int, stride: int, batch_size: int, num_workers: int
) -> wds.WebDataset:
    """获得wds数据加载器"""
    dataset = (
        wds.WebDataset(url, shardshuffle=0)
        .decode()
        .compose(partial(clip_to_t_frames, num_frames, stride))
        .shuffle(5000)
        .map(preprocess_frame)
        .batched(
            batch_size,
            partial=False,
            collation_fn=collate_fn
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )

    return dataloader

if __name__ == "__main__":
    loader = get_dataloader(
        glob.glob(
            "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*.tar"
        ),
        num_frames=1,
        stride=1,
        batch_size=32,
        num_workers=4,
    )

    batch = None
    for batch_ in tqdm(loader, ncols=70):
        batch = copy.deepcopy(batch_)
        break

    # 直接验证数据满足一致性
    import cv2
    import kornia.geometry.transform as T
    import smplx

    bx, tx = 10, 0

    # handedness
    img = cv2.cvtColor(batch["imgs"][bx][tx].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    img1 = img.copy()
    cv2.putText(
        img1,
        batch["handedness"][bx],
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite("temp/origin.png", img1)

    # hand_bbox, hand_img
    img2 = img.copy()
    hand_bbox = batch["hand_bbox"][bx, tx]
    joint_img = batch["joint_img"][bx, tx]
    cv2.rectangle(
        img2,
        (int(hand_bbox[0]), int(hand_bbox[1])),
        (int(hand_bbox[2]), int(hand_bbox[3])),
        (0, 0, 255),
        3
    )
    for i, jnt in enumerate(joint_img):
        cv2.circle(
            img2,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img2,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite("temp/hand_bbox-joint_img.png", img2)

    # joint_hand_bbox
    hand_bbox = torch.round(batch["hand_bbox"][bx, tx:tx + 1])
    joint_hand_bbox = batch["joint_hand_bbox"][bx, tx]
    xm, ym, xM, yM = torch.split(hand_bbox, 1, dim=-1)
    hand_bbox_k = torch.stack(
        [
            torch.cat([xm, ym], dim=-1),
            torch.cat([xM, ym], dim=-1),
            torch.cat([xM, yM], dim=-1),
            torch.cat([xm, yM], dim=-1),
        ],
        dim=1
    )
    bbox_size = hand_bbox[0, 2:] - hand_bbox[0, :2]
    img_hand_bbox = T.crop_and_resize(
        batch["imgs"][bx][tx:tx + 1].float() / 255,
        hand_bbox_k,
        (int(bbox_size[1]), int(bbox_size[0])),
    )
    img_hand_bbox = cv2.cvtColor(
        (img_hand_bbox[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    for i, jnt in enumerate(joint_hand_bbox):
        cv2.circle(
            img_hand_bbox,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img_hand_bbox,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite("temp/joint_hand_bbox.png", img_hand_bbox)

    # joint_cam
    joint_cam = batch["joint_cam"][bx, tx]
    focal = batch["focal"][bx, tx]
    princpt = batch["princpt"][bx, tx]
    joint_reproj_u = focal[0] * joint_cam[:, 0] / joint_cam[:, 2] + princpt[0]
    joint_reproj_v = focal[1] * joint_cam[:, 1] / joint_cam[:, 2] + princpt[1]
    joint_reproj = torch.stack([joint_reproj_u, joint_reproj_v], dim=-1)
    img3 = img.copy()
    for i, jnt in enumerate(joint_reproj):
        cv2.circle(
            img3,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img3,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite("temp/joint_cam.png", img3)

    # joint rel
    joint_cam = batch["joint_cam"][bx, tx]
    joint_rel = batch["joint_rel"][bx, tx]
    assert torch.allclose(
        joint_rel, joint_cam - joint_cam[:1]
    ), "joint_rel not consistent with joint_cam"
    print("joint_rel is consistent with joint_cam")

    # joint valid
    joint_valid = batch["joint_valid"][bx, tx]
    print(f"joint_valid shape={joint_valid.shape}")
    print(f"joint_valid={joint_valid}")

    # mano_shape, mano_pose
    mano_layer = {
        "right": smplx.create("models", "mano", use_pca=False, is_rhand=True),
        "left": smplx.create("models", "mano", use_pca=False, is_rhand=False),
    }
    mano_pose = batch["mano_pose"][bx, tx][None, ...]  # [B,48]
    mano_shape = batch["mano_shape"][bx, tx][None, ...]  # [B,10]
    handedness = batch["handedness"][bx]
    focal = batch["focal"][bx, tx]
    princpt = batch["princpt"][bx, tx]
    img4 = img.copy()
    with torch.inference_mode():
        mano_output = mano_layer[handedness](
            betas=mano_shape,
            global_orient=mano_pose[:, :3],
            hand_pose=mano_pose[:, 3:],
            transl=torch.zeros((1, 3))
        )
    mano_verts = mano_output.vertices[0] * 1e3
    joint_mano_cam = mano_output.joints[0] * 1e3
    joint_cam = batch["joint_cam"][bx, tx]
    mano_verts = mano_verts - joint_mano_cam[:1] + joint_cam[:1]  # 使用gt根关节对齐mano
    mano_verts_reproj_u = focal[0] * mano_verts[:, 0] / mano_verts[:, 2] + princpt[0]
    mano_verts_reproj_v = focal[1] * mano_verts[:, 1] / mano_verts[:, 2] + princpt[1]
    mano_verts_reproj = torch.stack([mano_verts_reproj_u, mano_verts_reproj_v], dim=-1)
    for vt in mano_verts_reproj:
        cv2.circle(
            img4,
            (int(vt[0]), int(vt[1])),
            1, (255, 255, 0), -1
        )
    cv2.imwrite("temp/mano.png", img4)
