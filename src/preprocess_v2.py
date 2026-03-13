"""
Standalone preprocessing utilities for the V2 WebDataset format.
"""

from typing import Dict, List, Tuple

from einops import rearrange
import torch
import kornia.augmentation as KA
import kornia.geometry.conversions as KC
import kornia.geometry.transform as KT


class PixelLevelAugmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            KA.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
            ),
            KA.RandomGrayscale(p=0.2),
            KA.RandomPosterize(bits=3, p=0.2),
            KA.RandomSharpness(sharpness=0.5, p=0.3),
            KA.RandomEqualize(p=0.1),
            KA.RandomGaussianNoise(mean=0.0, std=0.05, p=0.2),
            KA.RandomMotionBlur(kernel_size=(3, 5), angle=35.0, direction=0.5, p=0.2),
            KA.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
            KA.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.0, p=0.3),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = input_tensor.shape
        output = self.transforms(
            input_tensor.reshape(batch_size * num_frames, channels, height, width)
        ).reshape(batch_size, num_frames, channels, height, width)
        return torch.clamp(output, 0.0, 1.0)


pixel_level_augmentation = PixelLevelAugmentation()


def get_trans_3d_mat(
    rad: torch.Tensor,
    scale: torch.Tensor,
    axis_angle: torch.Tensor,
) -> torch.Tensor:
    device = rad.device
    dtype = rad.dtype
    cos_rad = torch.cos(rad)
    sin_rad = torch.sin(rad)
    prefix_shape = rad.shape

    mat = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    mat[..., 0, 0] = cos_rad
    mat[..., 1, 0] = sin_rad
    mat[..., 0, 1] = -sin_rad
    mat[..., 1, 1] = cos_rad
    mat[..., 2, 2] = scale

    if axis_angle is not None:
        axis_angle_mat = KC.axis_angle_to_rotation_matrix(axis_angle.reshape(-1, 3))
        axis_angle_mat = axis_angle_mat.reshape(*axis_angle.shape[:-1], 3, 3)
        mat = axis_angle_mat @ mat

    return mat


def get_trans_2d_mat(
    rad: torch.Tensor,
    scale_inv: torch.Tensor,
    focal_old: torch.Tensor,
    princpt_old: torch.Tensor,
    focal_new: torch.Tensor,
    princpt_new: torch.Tensor,
    axis_angle: torch.Tensor,
) -> torch.Tensor:
    device = rad.device
    dtype = rad.dtype
    prefix_shape = rad.shape

    old_intr_inv = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    old_intr_inv[..., 0, 0] = 1.0 / focal_old[..., 0]
    old_intr_inv[..., 1, 1] = 1.0 / focal_old[..., 1]
    old_intr_inv[..., 0, 2] = -princpt_old[..., 0] / focal_old[..., 0]
    old_intr_inv[..., 1, 2] = -princpt_old[..., 1] / focal_old[..., 1]
    old_intr_inv[..., 2, 2] = 1.0

    new_intr = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    new_intr[..., 0, 0] = focal_new[..., 0]
    new_intr[..., 1, 1] = focal_new[..., 1]
    new_intr[..., 0, 2] = princpt_new[..., 0]
    new_intr[..., 1, 2] = princpt_new[..., 1]
    new_intr[..., 2, 2] = 1.0

    cos_rad = torch.cos(rad)
    sin_rad = torch.sin(rad)
    mat = torch.zeros(*prefix_shape, 3, 3, device=device, dtype=dtype)
    mat[..., 0, 0] = cos_rad
    mat[..., 1, 0] = sin_rad
    mat[..., 0, 1] = -sin_rad
    mat[..., 1, 1] = cos_rad
    mat = mat * scale_inv[..., None, None]
    mat[..., 2, 2] = 1.0

    if axis_angle is not None:
        axis_angle_mat = KC.axis_angle_to_rotation_matrix(axis_angle.reshape(-1, 3))
        axis_angle_mat = axis_angle_mat.reshape(*axis_angle.shape[:-1], 3, 3)
        mat = axis_angle_mat @ mat

    return new_intr @ mat @ old_intr_inv


def apply_perspective_to_points(
    trans_matrix: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    point_dim = points.shape[-1]
    if trans_matrix.shape[-1] != point_dim + 1 or trans_matrix.shape[-2] != point_dim + 1:
        raise ValueError(
            f"Matrix shape {trans_matrix.shape} does not match point dim {point_dim}"
        )

    points_h = KC.convert_points_to_homogeneous(points)
    points_h_transformed = points_h @ trans_matrix.transpose(-1, -2)
    return KC.convert_points_from_homogeneous(points_h_transformed)


def _bbox_to_corners(bbox: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = torch.split(bbox, 1, dim=-1)
    return torch.stack(
        [
            torch.cat([x1, y1], dim=-1),
            torch.cat([x2, y1], dim=-1),
            torch.cat([x2, y2], dim=-1),
            torch.cat([x1, y2], dim=-1),
        ],
        dim=2,
    )


def _compute_square_patch_bbox(
    hand_bbox: torch.Tensor, patch_expansion: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = (hand_bbox[..., :2] + hand_bbox[..., 2:]) * 0.5
    width = hand_bbox[..., 2] - hand_bbox[..., 0]
    height = hand_bbox[..., 3] - hand_bbox[..., 1]
    half_edge = torch.max(width, height) * patch_expansion * 0.5

    patch_bbox = torch.stack(
        [
            center[..., 0] - half_edge,
            center[..., 1] - half_edge,
            center[..., 0] + half_edge,
            center[..., 1] + half_edge,
        ],
        dim=-1,
    )
    return patch_bbox, _bbox_to_corners(patch_bbox)


def _crop_patches_from_batch(
    batch_imgs: List[torch.Tensor],
    patch_bbox_corners: torch.Tensor,
    patch_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    patches = []
    for batch_idx, imgs in enumerate(batch_imgs):
        patch = KT.crop_and_resize(
            imgs.to(device).float() / 255.0,
            patch_bbox_corners[batch_idx],
            patch_size,
            mode="bilinear",
        )
        patches.append(patch)
    return torch.stack(patches)


def _compute_resized_patch_joints(
    joint_patch_origin: torch.Tensor,
    patch_bbox: torch.Tensor,
    patch_size: Tuple[int, int],
) -> torch.Tensor:
    patch_w = patch_bbox[..., 2] - patch_bbox[..., 0]
    patch_h = patch_bbox[..., 3] - patch_bbox[..., 1]
    scale_x = patch_size[1] / torch.clamp(patch_w, min=1e-6)
    scale_y = patch_size[0] / torch.clamp(patch_h, min=1e-6)

    joint_patch_resized = joint_patch_origin.clone()
    joint_patch_resized[..., 0] *= scale_x[..., None]
    joint_patch_resized[..., 1] *= scale_y[..., None]
    return joint_patch_resized


@torch.no_grad()
def preprocess_batch(
    batch_origin: Dict[str, torch.Tensor],
    patch_size: Tuple[int, int],
    patch_expanstion: float,
    scale_z_range: Tuple[float, float],
    scale_f_range: Tuple[float, float],
    persp_rot_max: float,
    augmentation_flag: bool,
    device: torch.device,
):
    batch_size, num_frames = batch_origin["joint_img"].shape[:2]
    trans_2d_mat = (
        torch.eye(3, device=device, dtype=torch.float32)[None, None]
        .repeat(batch_size, num_frames, 1, 1)
    )

    imgs_path = batch_origin["imgs_path"]
    handedness = batch_origin["handedness"]
    data_source = batch_origin["data_source"]
    source_split = batch_origin["source_split"]
    source_index = batch_origin["source_index"]
    intr_type = batch_origin["intr_type"]
    additional_desc = batch_origin["additional_desc"]
    flip = [value == "left" for value in handedness]

    hand_bbox = batch_origin["hand_bbox"].to(device).clone()
    joint_img = batch_origin["joint_img"].to(device).clone()
    joint_cam = batch_origin["joint_cam"].to(device).clone()
    joint_rel = batch_origin["joint_rel"].to(device).clone()
    joint_2d_valid = batch_origin["joint_2d_valid"].to(device).clone()
    joint_3d_valid = batch_origin["joint_3d_valid"].to(device).clone()
    mano_pose = batch_origin["mano_pose"].to(device).clone()
    mano_shape = batch_origin["mano_shape"].to(device).clone()
    has_mano = batch_origin["has_mano"].to(device).clone()
    has_intr = batch_origin["has_intr"].to(device).clone()
    timestamp = batch_origin["timestamp"].to(device).clone()
    focal = batch_origin["focal"].to(device).clone()
    princpt = batch_origin["princpt"].to(device).clone()

    if not augmentation_flag:
        patch_bbox, patch_bbox_corners = _compute_square_patch_bbox(
            hand_bbox, patch_expanstion
        )
        patches = _crop_patches_from_batch(
            batch_origin["imgs"], patch_bbox_corners, patch_size, device
        )
    else:
        rad = torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
        rad = rad * 2.0 * torch.pi

        scale_z = (
            torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
            * (scale_z_range[1] - scale_z_range[0])
            + scale_z_range[0]
        )
        scale_f = (
            torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
            * (scale_f_range[1] - scale_f_range[0])
            + scale_f_range[0]
        )
        focal_new = focal * scale_f[:, :, None]
        princpt_new = princpt.clone()

        persp_dir_rad = (
            torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
            * 2.0
            * torch.pi
        )
        persp_rot_rad = (
            torch.rand(batch_size, 1, device=device).expand(-1, num_frames)
            * persp_rot_max
        )
        persp_axis_angle = (
            torch.stack(
                [
                    torch.cos(persp_dir_rad),
                    torch.sin(persp_dir_rad),
                    torch.zeros(batch_size, num_frames, device=device),
                ],
                dim=-1,
            )
            * persp_rot_rad[..., None]
        )

        full_trans_3d_mat = get_trans_3d_mat(rad, scale_z, persp_axis_angle)
        has_intr_expand = (has_intr > 0.5)[..., None]
        safe_focal = torch.where(has_intr_expand, focal, torch.ones_like(focal))
        safe_princpt = torch.where(has_intr_expand, princpt, torch.zeros_like(princpt))
        safe_focal_new = torch.where(
            has_intr_expand, focal_new, torch.ones_like(focal_new)
        )
        safe_princpt_new = torch.where(
            has_intr_expand, princpt_new, torch.zeros_like(princpt_new)
        )
        full_trans_2d_mat = get_trans_2d_mat(
            rad,
            1.0 / scale_z,
            safe_focal,
            safe_princpt,
            safe_focal_new,
            safe_princpt_new,
            persp_axis_angle,
        )

        has_intr_mask = has_intr > 0.5
        trans_2d_mat = torch.where(
            has_intr_mask[..., None, None], full_trans_2d_mat, trans_2d_mat
        )

        frame_has_3d = torch.any(joint_3d_valid > 0.5, dim=-1)
        joint_cam_aug = torch.einsum("...jd,...nd->...jn", joint_cam, full_trans_3d_mat)
        joint_rel_aug = joint_cam_aug - joint_cam_aug[:, :, :1]
        joint_cam = torch.where(frame_has_3d[..., None, None], joint_cam_aug, joint_cam)
        joint_rel = torch.where(frame_has_3d[..., None, None], joint_rel_aug, joint_rel)

        frame_has_mano = (has_mano > 0.5).reshape(-1)
        if torch.any(frame_has_mano):
            mano_pose_flat = mano_pose.reshape(-1, mano_pose.shape[-1]).clone()
            root_axis_angle = torch.zeros(
                (batch_size * num_frames, 3),
                device=device,
                dtype=mano_pose.dtype,
            )
            root_axis_angle[:, 2] = rad.reshape(-1)
            root_rot_mat = KC.axis_angle_to_rotation_matrix(root_axis_angle)
            mano_root_rot = KC.axis_angle_to_rotation_matrix(mano_pose_flat[:, :3])
            mano_root_rot = KC.rotation_matrix_to_axis_angle(root_rot_mat @ mano_root_rot)
            mano_pose_flat[frame_has_mano, :3] = mano_root_rot[frame_has_mano]
            mano_pose = mano_pose_flat.reshape(batch_size, num_frames, -1)

        joint_mask = (joint_2d_valid < 0.5)[..., None].expand(-1, -1, -1, 2)
        joint_img = apply_perspective_to_points(trans_2d_mat, joint_img)

        min_xy = torch.min(
            joint_img.masked_fill(joint_mask, float("inf")), dim=-2
        ).values
        max_xy = torch.max(
            joint_img.masked_fill(joint_mask, -float("inf")), dim=-2
        ).values
        hand_bbox_new = torch.cat([min_xy, max_xy], dim=-1)

        valid_joint_count = torch.sum(joint_2d_valid > 0.5, dim=-1, keepdim=True)
        no_valid_joint = valid_joint_count == 0
        hand_bbox = torch.where(
            no_valid_joint.expand_as(hand_bbox), hand_bbox, hand_bbox_new
        )

        patch_bbox, patch_bbox_corners = _compute_square_patch_bbox(
            hand_bbox, patch_expanstion
        )

        patch_bbox_corners_orig = apply_perspective_to_points(
            torch.linalg.inv(trans_2d_mat), patch_bbox_corners
        )
        patches = _crop_patches_from_batch(
            batch_origin["imgs"], patch_bbox_corners_orig, patch_size, device
        )
        patches = pixel_level_augmentation(patches)

        focal = torch.where(has_intr_mask[..., None], focal_new, focal)
        princpt = torch.where(has_intr_mask[..., None], princpt_new, princpt)

    for batch_idx, do_flip in enumerate(flip):
        if not do_flip:
            continue

        _, _, height, width = batch_origin["imgs"][batch_idx].shape
        _ = height

        patches[batch_idx] = torch.flip(patches[batch_idx], dims=[-1])
        patch_bbox[batch_idx, :, 0], patch_bbox[batch_idx, :, 2] = (
            width - patch_bbox[batch_idx, :, 2],
            width - patch_bbox[batch_idx, :, 0],
        )
        hand_bbox[batch_idx, :, 0], hand_bbox[batch_idx, :, 2] = (
            width - hand_bbox[batch_idx, :, 2],
            width - hand_bbox[batch_idx, :, 0],
        )
        joint_img[batch_idx, :, :, 0] = width - joint_img[batch_idx, :, :, 0]
        joint_cam[batch_idx, :, :, 0] *= -1.0
        joint_rel[batch_idx, :, :, 0] *= -1.0

        mano_pose_batch = rearrange(mano_pose[batch_idx], "t (j d) -> t j d", d=3)
        mano_pose_batch[:, :, 1:] *= -1.0
        mano_pose[batch_idx] = rearrange(mano_pose_batch, "t j d -> t (j d)")

        intr_mask = has_intr[batch_idx] > 0.5
        princpt[batch_idx, intr_mask, 0] = width - princpt[batch_idx, intr_mask, 0]

    joint_hand_origin = joint_img - hand_bbox[:, :, None, :2]
    joint_patch_origin = joint_img - patch_bbox[:, :, None, :2]
    joint_patch_resized = _compute_resized_patch_joints(
        joint_patch_origin, patch_bbox, patch_size
    )

    batch_out = {
        "__key__": batch_origin["__key__"],
        "imgs_path": imgs_path,
        "handedness": handedness,
        "data_source": data_source,
        "source_split": source_split,
        "source_index": source_index,
        "intr_type": intr_type,
        "additional_desc": additional_desc,
        "flip": flip,
        "patches": patches,
        "patch_bbox": patch_bbox,
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_patch_origin": joint_patch_origin,
        "joint_patch_resized": joint_patch_resized,
        "joint_hand_origin": joint_hand_origin,
        "joint_patch_bbox": joint_patch_origin,
        "joint_hand_bbox": joint_hand_origin,
        "joint_cam": joint_cam,
        "joint_rel": joint_rel,
        "joint_2d_valid": joint_2d_valid,
        "joint_3d_valid": joint_3d_valid,
        "joint_valid": joint_2d_valid,
        "mano_pose": mano_pose,
        "mano_shape": mano_shape,
        "has_mano": has_mano,
        "mano_valid": has_mano,
        "has_intr": has_intr,
        "timestamp": timestamp,
        "focal": focal,
        "princpt": princpt,
    }
    return batch_out, trans_2d_mat
