"""
Webdataset Dataloader
"""
from typing import *
import glob
import copy
from functools import partial
import numpy as np
import webdataset as wds
from tqdm import tqdm
from einops import rearrange

import torch
from torch.utils.data import DataLoader
import torchvision

import kornia.geometry.transform as T

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
        imgs_path = sample["imgs_path.json"]
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
                # 记录图片地址
                "imgs_path": imgs_path[start:end],
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
        "imgs_path": sample["imgs_path"],
        "imgs": imgs_tensor,
        "handedness": sample["handedness"], # 此时还是 str, collate 时可能需要特殊处理或 drop
    }

    # 自动将所有 numpy 字段转为 Tensor
    for key in sample:
        if key not in ["__key__", "imgs_path", "imgs_bytes", "handedness"]:
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

def preprocess_batch(
    batch_origin,
    patch_size: Tuple[int, int],
    patch_expanstion: float,
    augmentation_flag: bool,
    device: torch.device
):
    """
    将wds的原始数据进行预处理和数据增强，最后送给模型

    Args:
        patch_size: 输出给模型的图像patch的大小
        patch_expansion: patch相较于正方形包围盒扩大的范围
    """
    # TODO: 数据增强

    # 数据规整
    # imgs_path, flip
    imgs_path: List[List[str]] = batch_origin["imgs_path"]
    flip: List[bool] = [v == "left" for v in batch_origin["handedness"]]

    # 计算patches对应的范围，以及patch图像分割
    # hand_bbox, patches_bbox, patches
    hand_bbox: torch.Tensor = batch_origin["hand_bbox"].to(device)  # [B,T,4]
    hand_bbox_center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) / 2  # [B,T,2]
    width, height = torch.split(hand_bbox[..., 2:] - hand_bbox[..., :2], 1, dim=-1)  # [B,T]
    half_edge_len = torch.max(width, height) * patch_expanstion * 0.5  # [B,T]
    patch_bbox_k = torch.stack(  # [B,T,4,2]
        [
            hand_bbox_center - half_edge_len,
            torch.stack([
                hand_bbox_center[..., 0] + half_edge_len[..., 0],
                hand_bbox_center[..., 1] - half_edge_len[..., 0]
            ], dim=-1),
            hand_bbox_center + half_edge_len,
            torch.stack([
                hand_bbox_center[..., 0] - half_edge_len[..., 0],
                hand_bbox_center[..., 1] + half_edge_len[..., 0]
            ], dim=-1),
        ], dim=2
    )
    patch_bbox: torch.Tensor = torch.cat(
        [patch_bbox_k[:, :, 0], patch_bbox_k[:, :, 2]], dim=-1
    )  # [B,T,4]
    patches = []
    for bx, img_orig_tensor in enumerate(batch["imgs"]):
        patch = T.crop_and_resize(
            img_orig_tensor.to(device).float() / 255,
            patch_bbox_k[bx],
            patch_size,
            mode="bilinear"
        )
        patches.append(patch)
    patches: torch.Tensor = torch.stack(patches)

    # 处理关节点二维位置
    joint_img: torch.Tensor = batch["joint_img"].to(device)  # [B,T,J,2]
    joint_patch_bbox: torch.Tensor = joint_img - patch_bbox_k[:, :, :1]  # [B,T,J,2]
    joint_hand_bbox: torch.Tensor = joint_img - hand_bbox[:, :, None, :2]  # [B,T,J,2]

    # 三维位置
    joint_cam: torch.Tensor = batch["joint_cam"].to(device)  # [B,T,J,3]
    joint_rel: torch.Tensor = batch["joint_rel"].to(device)  # [B,T,J,3]
    joint_valid: torch.Tensor = batch["joint_valid"].to(device)  # [B,T,J]

    # MANO标注
    mano_pose: torch.Tensor = batch["mano_pose"].to(device)  # [B,T,48]
    mano_shape: torch.Tensor = batch["mano_shape"].to(device)  # [B,T,10]
    # TODO: mano_valid

    # timestamp
    timestamp: torch.Tensor = batch["timestamp"].to(device)  # [B,T]

    # focal, princpt
    focal: torch.Tensor = batch["focal"].to(device)  # [B,T,2]
    princpt: torch.Tensor = batch["princpt"].to(device)  # [B,T,2]

    # 进行左右翻转
    for bx in range(len(imgs_path)):
        if flip[bx]:
            T_, C, H, W = batch["imgs"][bx].shape
            patch_bbox_w = patch_bbox[bx, :, 2] - patch_bbox[bx, :, 0]  # [T]
            hand_bbox_w =  hand_bbox[bx, :, 2] - hand_bbox[bx, :, 0]  # [T]

            patches[bx] = torch.flip(patches[bx], dims=[-1,])
            patch_bbox[bx, :, 0], patch_bbox[bx, :, 2] = (
                W - patch_bbox[bx, :, 2],
                W - patch_bbox[bx, :, 0],
            )
            hand_bbox[bx, :, 0], hand_bbox[bx, :, 2] = (
                W - hand_bbox[bx, :, 2],
                W - hand_bbox[bx, :, 0]
            )
            joint_img[bx, :, :, 0] = W - joint_img[bx, :, :, 0]
            joint_patch_bbox[bx, :, :, 0] = patch_bbox_w[:, None] - joint_patch_bbox[bx, :, :, 0]
            joint_hand_bbox[bx, :, :, 0] = hand_bbox_w[:, None] - joint_hand_bbox[bx, :, :, 0]
            joint_cam[bx, :, :, 0] *= -1
            joint_rel[bx, :, :, 0] *= -1

            mano_pose_bx = rearrange(mano_pose[bx], "t (j d) -> t j d", d=3)
            mano_pose_bx[:, :, 1:] *= -1
            mano_pose[bx] = rearrange(mano_pose_bx, "t j d -> t (j d)")

            princpt[bx, :, 0] = W - princpt[bx, :, 0]

    return {
        "imgs_path": imgs_path,
        "flip": flip,
        "patches": patches,
        "patch_bbox": patch_bbox,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_patch_bbox": joint_patch_bbox,
        "joint_hand_bbox": joint_hand_bbox,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_valid": joint_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        # "mano_valid": mano_valid,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt
    }

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

def verify_origin_data(batch, output_dir: str, bx: int = 0, tx: int = 0):
    import cv2
    import smplx

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
    cv2.imwrite(f"{output_dir}/origin.png", img1)

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
    cv2.imwrite(f"{output_dir}/hand_bbox-joint_img.png", img2)

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
    cv2.imwrite(f"{output_dir}/joint_hand_bbox.png", img_hand_bbox)

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
    cv2.imwrite(f"{output_dir}/joint_cam.png", img3)

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
    cv2.imwrite(f"{output_dir}/mano.png", img4)

def verify_batch(batch, output_dir: str, source_prefix: str, bx: int = 0, tx: int = 0):
    import cv2
    import smplx

    # origin image
    img = cv2.imread(f"{source_prefix}/" + batch["imgs_path"][bx][tx])
    if batch["flip"][bx]:
        img = img[:, ::-1].copy()
    cv2.imwrite(f"{output_dir}/origin.png", img)

    # patches
    patch = batch["patches"][bx, tx].cpu().permute(1, 2, 0).numpy() * 255
    patch = patch.astype(np.uint8)
    cv2.imwrite(f"{output_dir}/patch.png", patch)

    # hand/patch_bbox, joint_img
    img2 = img.copy()
    patch_bbox = batch["patch_bbox"][bx, tx].cpu().numpy()
    hand_bbox = batch["hand_bbox"][bx, tx].cpu().numpy()
    joint_img = batch["joint_img"][bx, tx].cpu().numpy()
    cv2.rectangle(
        img2,
        (int(patch_bbox[0]), int(patch_bbox[1])),
        (int(patch_bbox[2]), int(patch_bbox[3])),
        (0, 0, 255), 2
    )
    cv2.rectangle(
        img2,
        (int(hand_bbox[0]), int(hand_bbox[1])),
        (int(hand_bbox[2]), int(hand_bbox[3])),
        (0, 255, 0), 2
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
    cv2.imwrite(f"{output_dir}/bbox-joint_img.png", img2)

    # joint_patch_bbox, joint_hand_bbox
    img3 = img.copy()
    joint_patch_bbox = batch["joint_patch_bbox"][bx, tx].cpu().numpy()
    joint_hand_bbox = batch["joint_hand_bbox"][bx, tx].cpu().numpy()
    patch_bbox = batch["patch_bbox"][bx, tx].cpu().numpy()
    hand_bbox = batch["hand_bbox"][bx, tx].cpu().numpy()
    img_patch_bbox = img3[
        int(patch_bbox[1]):int(patch_bbox[3]),
        int(patch_bbox[0]):int(patch_bbox[2])
    ].copy()
    img_hand_bbox = img3[
        int(hand_bbox[1]):int(hand_bbox[3]),
        int(hand_bbox[0]):int(hand_bbox[2])
    ].copy()
    for i, _ in enumerate(joint_patch_bbox):
        cv2.circle(
            img_patch_bbox,
            (int(joint_patch_bbox[i, 0]), int(joint_patch_bbox[i, 1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img_patch_bbox,
            str(i),
            (int(joint_patch_bbox[i, 0]), int(joint_patch_bbox[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            img_hand_bbox,
            (int(joint_hand_bbox[i, 0]), int(joint_hand_bbox[i, 1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img_hand_bbox,
            str(i),
            (int(joint_hand_bbox[i, 0]), int(joint_hand_bbox[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/joint_patch_bbox.png", img_patch_bbox)
    cv2.imwrite(f"{output_dir}/joint_hand_bbox.png", img_hand_bbox)

    # joint_cam
    img4 = img.copy()
    joint_cam = batch["joint_cam"][bx, tx].cpu()
    focal = batch["focal"][bx, tx].cpu()
    princpt = batch["princpt"][bx, tx].cpu()
    joint_reproj_u = focal[0] * joint_cam[:, 0] / joint_cam[:, 2] + princpt[0]
    joint_reproj_v = focal[1] * joint_cam[:, 1] / joint_cam[:, 2] + princpt[1]
    joint_reproj = torch.stack([joint_reproj_u, joint_reproj_v], dim=-1)
    for i, jnt in enumerate(joint_reproj):
        cv2.circle(
            img4,
            (int(jnt[0]), int(jnt[1])),
            3, (255, 50, 50), -1
        )
        cv2.putText(
            img4,
            str(i),
            (int(jnt[0]), int(jnt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(f"{output_dir}/joint_cam.png", img4)

    # joint_rel
    joint_cam = batch["joint_cam"][bx, tx].cpu()
    joint_rel = batch["joint_rel"][bx, tx].cpu()
    assert torch.allclose(
        joint_rel, joint_cam - joint_cam[:1]
    ), "joint_rel not consistent with joint_cam"
    print("joint_rel is consistent with joint_cam")

    # joint valid
    joint_valid = batch["joint_valid"][bx, tx].cpu()
    print(f"joint_valid shape={joint_valid.shape}")
    print(f"joint_valid={joint_valid}")

    # mano_shape, mano_pose
    mano_layer = {
        "right": smplx.create("models", "mano", use_pca=False, is_rhand=True),
        "left": smplx.create("models", "mano", use_pca=False, is_rhand=False),
    }
    mano_pose = batch["mano_pose"][bx, tx][None, ...].cpu()  # [B,48]
    mano_shape = batch["mano_shape"][bx, tx][None, ...].cpu()  # [B,10]
    flip = batch["flip"][bx]
    focal = batch["focal"][bx, tx].cpu()
    princpt = batch["princpt"][bx, tx].cpu()
    img5 = img.copy()
    with torch.inference_mode():
        mano_output = mano_layer["right" if not flip else "left"](
            betas=mano_shape,
            global_orient=mano_pose[:, :3],
            hand_pose=mano_pose[:, 3:],
            transl=torch.zeros((1, 3))
        )
    mano_verts = mano_output.vertices[0] * 1e3
    joint_mano_cam = mano_output.joints[0] * 1e3
    joint_cam = batch["joint_cam"][bx, tx].cpu()
    mano_verts = mano_verts - joint_mano_cam[:1] + joint_cam[:1]  # 使用gt根关节对齐mano
    mano_verts_reproj_u = focal[0] * mano_verts[:, 0] / mano_verts[:, 2] + princpt[0]
    mano_verts_reproj_v = focal[1] * mano_verts[:, 1] / mano_verts[:, 2] + princpt[1]
    mano_verts_reproj = torch.stack([mano_verts_reproj_u, mano_verts_reproj_v], dim=-1)
    for vt in mano_verts_reproj:
        cv2.circle(
            img5,
            (int(vt[0]), int(vt[1])),
            1, (255, 255, 0), -1
        )
    cv2.imwrite(f"{output_dir}/mano.png", img5)

if __name__ == "__main__":
    loader = get_dataloader(
        glob.glob(
            "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/*.tar"
        ),
        num_frames=7,
        stride=1,
        batch_size=32,
        num_workers=4,
    )

    batch = None
    x = 0
    for batch_ in tqdm(loader, ncols=70):
        batch = copy.deepcopy(batch_)
        x += 1
        if x > 1:
            break

    # 直接验证数据满足一致性
    bx, tx = 10, 0
    verify_origin_data(batch, "temp_origin", 10, 0)

    # 验证数据规整的正确性
    batch2 = preprocess_batch(
        batch,
        [256, 256],
        1.1,
        False,
        torch.device("cuda:0")
    )
    verify_batch(
        batch2,
        "temp_processed",
        "/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1/images/train/",
        10,
        0,
    )
