from typing import *
import torch
from einops import rearrange
import kornia.geometry.transform as T

def preprocess_batch(
    batch_origin,
    patch_size: Tuple[int, int],
    patch_expanstion: float,
    scale_z_range: Tuple[float, float],
    scale_f_range: Tuple[float, float],
    augmentation_flag: bool,
    device: torch.device
):
    """
    将wds的原始数据进行预处理和数据增强，最后送给模型

    Args:
        patch_size: 输出给模型的图像patch的大小
        patch_expansion: patch相较于正方形包围盒扩大的范围
        scale_z_range: 进行缩放/平移增强变换的系数范围
        scale_f_range: 进行内参增强变换的焦距乘数的范围
    """
    B, T_ = batch_origin["joint_cam"].shape[:2]
    if not augmentation_flag:
        # 增强参数不动
        rotation_rad = torch.zeros(B, device=device)
        scale_z = torch.ones(B, device=device)
        scale_f = torch.ones(B, device=device)
        old_princpt = new_princpt = batch_origin["princpt"].to(device)
        # 数据规整
        # imgs_path, flip
        imgs_path: List[List[str]] = batch_origin["imgs_path"]
        flip: List[bool] = [v == "left" for v in batch_origin["handedness"]]

        # 计算patches对应的范围，以及patch图像分割
        # hand_bbox, patches_bbox, patches
        hand_bbox: torch.Tensor = batch_origin["hand_bbox"].to(device)  # [B,T,4]
        hand_bbox_center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) * 0.5  # [B,T,2]
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
        for bx, img_orig_tensor in enumerate(batch_origin["imgs"]):
            patch = T.crop_and_resize(
                img_orig_tensor.to(device).float() / 255,
                patch_bbox_k[bx],
                patch_size,
                mode="bilinear"
            )
            patches.append(patch)
        patches: torch.Tensor = torch.stack(patches)

        # 处理关节点二维位置
        joint_img: torch.Tensor = batch_origin["joint_img"].to(device)  # [B,T,J,2]
        joint_patch_bbox: torch.Tensor = joint_img - patch_bbox_k[:, :, :1]  # [B,T,J,2]
        joint_hand_bbox: torch.Tensor = joint_img - hand_bbox[:, :, None, :2]  # [B,T,J,2]

        # 三维位置
        joint_cam: torch.Tensor = batch_origin["joint_cam"].to(device)  # [B,T,J,3]
        joint_rel: torch.Tensor = batch_origin["joint_rel"].to(device)  # [B,T,J,3]
        joint_valid: torch.Tensor = batch_origin["joint_valid"].to(device)  # [B,T,J]

        # MANO标注
        mano_pose: torch.Tensor = batch_origin["mano_pose"].to(device)  # [B,T,48]
        mano_shape: torch.Tensor = batch_origin["mano_shape"].to(device)  # [B,T,10]
        mano_valid: torch.Tensor = batch_origin["mano_valid"].to(device)  # [B,T,10]

        # timestamp
        timestamp: torch.Tensor = batch_origin["timestamp"].to(device)  # [B,T]

        # focal, princpt
        focal: torch.Tensor = batch_origin["focal"].to(device)  # [B,T,2]
        princpt: torch.Tensor = batch_origin["princpt"].to(device)  # [B,T,2]
    else:  # 数据增强分支
        rotation_rad: torch.Tensor = torch.rand(B, device=device) * torch.pi * 2  # [B]
        scale_z: torch.Tensor = (
            torch.rand(B, device=device)
            * (scale_z_range[1] - scale_z_range[0])
            * scale_z_range[0]
        )  # [B]
        scale_f: torch.Tensor = (
            torch.rand(B, device=device)
            * (scale_f_range[1] - scale_f_range[0])
            * scale_f_range[0]
        )  # [B]
        old_princpt = batch_origin["princpt"].to(device)
        new_princpt = (
            torch.randn(B, 1, 2) * torch.norm(old_princpt, dim=-1, keepdim=True) * 0.111111
            + old_princpt
        )

        # 获得新的内参焦距
        focal: torch.Tensor = batch_origin["focal"].to(device)  # [B,T,2]
        new_focal = scale_f[:, None, None] * focal

        # 对二维数据的增强变换
        # 构造图像的旋转变换（以主点为中心）
        rot_2d_mat = torch.zeros((B, T, 2, 2), device=device)
        cos_rot = torch.cos(rotation_rad)
        sin_rot = torch.sin(rotation_rad)
        rot_2d_mat[:, :, 0, 0] = cos_rot
        rot_2d_mat[:, :, 1, 0] = -sin_rot
        rot_2d_mat[:, :, 0, 1] = sin_rot
        rot_2d_mat[:, :, 1, 1] = cos_rot
        # 构造图像的缩放变换（以主点为中心）
        # 构造图像的透视变换
        old_intrinsics = torch.zeros((B, T, 3, 3), device=device)
        old_intrinsics[:, :, 0, 0] = focal[:, :, 0]
        old_intrinsics[:, :, 1, 1] = focal[:, :, 1]
        old_intrinsics[:, :, 0, 2] = princpt[:, :, 0]
        old_intrinsics[:, :, 1, 2] = princpt[:, :, 1]
        old_intrinsics[:, :, 2, 2] = 1
        new_intrinsics = torch.zeros((B, T, 3, 3), device=device)
        new_intrinsics[:, :, 0, 0] = new_focal[:, :, 0]
        new_intrinsics[:, :, 1, 1] = new_focal[:, :, 1]
        new_intrinsics[:, :, 0, 2] = new_princpt[:, :, 0]
        new_intrinsics[:, :, 1, 2] = new_princpt[:, :, 1]
        new_intrinsics[:, :, 2, 2] = 1
        homo_mat = new_intrinsics @ old_intrinsics.inverse()

        # 对三维标注数据的增强
        # 三维变换矩阵
        rot_3d_mat = torch.zeros((B, T, 3, 3), device=device)
        rot_3d_mat[:, :, 0, 0] = cos_rot
        rot_3d_mat[:, :, 1, 0] = -sin_rot
        rot_3d_mat[:, :, 0, 1] = sin_rot
        rot_3d_mat[:, :, 1, 1] = cos_rot
        rot_3d_mat[:, :, 2, 2] = 1
        # z平移矩阵
        scale_z_mat = torch.zeros((B, T, 3, 3), device=device)
        scale_z_mat[:, :, 0, 0] = 1
        scale_z_mat[:, :, 1, 1] = 1
        scale_z_mat[:, :, 2, 2] = scale_z
        # 3D变换矩阵
        aug_3d_mat = scale_z_mat @ rot_3d_mat

        # 对三维的数据进行增强变换
        # joint_img, joint_rel, joint_valid
        joint_cam: torch.Tensor = batch_origin["joint_cam"].to(device)  # [B,T,J,3]
        joint_cam = joint_cam @ aug_3d_mat.T
        joint_rel = joint_cam - joint_cam[:, :, :1]  # 重新计算rel
        joint_valid: torch.Tensor = batch_origin["joint_valid"].to(device)  # [B,T,J]

        # 重新计算二维关节点
        # joint_img
        joint_reproj_u = (
            new_focal[..., None, 0] * joint_cam[..., 0] / joint_cam[..., 2]
            + new_princpt[..., None, 0]
        )
        joint_reproj_v = (
            new_focal[..., None, 1] * joint_cam[..., 1] / joint_cam[..., 2]
            + new_princpt[..., None, 1]
        )
        joint_img: torch.Tensor = torch.stack([joint_reproj_u, joint_reproj_v], dim=-1)  # [B,T,J,2]

        # 基于旋转的结果重新计算手部检测框
        # joint_hand_bbox, joint_patch_bbox
        xm, ym = torch.split(torch.min(joint_img, dim=2), 1, dim=-1)  # [B,T], [B,T]
        xM, yM = torch.split(torch.max(joint_img, dim=2), 1, dim=-1)  # [B,T], [B,T]

    # 进行左右翻转
    for bx in range(len(imgs_path)):
        if flip[bx]:
            T_, C, H, W = batch_origin["imgs"][bx].shape
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
        "mano_valid": mano_valid,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt
    }
