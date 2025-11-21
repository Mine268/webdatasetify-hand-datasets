import numpy as np
import os
import os.path as osp
import json
from pycocotools.coco import COCO
import tqdm
from collections import defaultdict
from copy import deepcopy
import pickle as pkl
import cv2
import smplx
import torch
from pathlib import Path

from utils import *


IH26M_ROOT = r"/data_1/datasets_temp/InterHand2.6M_5fps_batch1/"
SPLIT = "val"  # train val test

# IH26M joint set
joint_set = {
    "joint_num": 42,
    "joints_name": (
        "R_Thumb_4",
        "R_Thumb_3",
        "R_Thumb_2",
        "R_Thumb_1",
        "R_Index_4",
        "R_Index_3",
        "R_Index_2",
        "R_Index_1",
        "R_Middle_4",
        "R_Middle_3",
        "R_Middle_2",
        "R_Middle_1",
        "R_Ring_4",
        "R_Ring_3",
        "R_Ring_2",
        "R_Ring_1",
        "R_Pinky_4",
        "R_Pinky_3",
        "R_Pinky_2",
        "R_Pinky_1",
        "R_Wrist",
        "L_Thumb_4",
        "L_Thumb_3",
        "L_Thumb_2",
        "L_Thumb_1",
        "L_Index_4",
        "L_Index_3",
        "L_Index_2",
        "L_Index_1",
        "L_Middle_4",
        "L_Middle_3",
        "L_Middle_2",
        "L_Middle_1",
        "L_Ring_4",
        "L_Ring_3",
        "L_Ring_2",
        "L_Ring_1",
        "L_Pinky_4",
        "L_Pinky_3",
        "L_Pinky_2",
        "L_Pinky_1",
        "L_Wrist",
    ),
    "flip_pairs": [(i, i + 21) for i in range(21)],
}
joint_set["joint_type"] = {
    "right": np.arange(0, joint_set["joint_num"] // 2),
    "left": np.arange(
        joint_set["joint_num"] // 2, joint_set["joint_num"]
    ),
}
joint_set["root_joint_idx"] = {
    "right": joint_set["joints_name"].index("R_Wrist"),
    "left": joint_set["joints_name"].index("L_Wrist"),
}

dataset = COCO(
    osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_data.json")
)
with open(
    osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_camera.json"), "r"
) as f:
    cameras = json.load(f)
with open(
    osp.join(IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_joint_3d.json"),
    "r",
) as f:
    joints = json.load(f)
with open(
    osp.join(
        IH26M_ROOT, f"annotations/{SPLIT}/InterHand2.6M_{SPLIT}_MANO_NeuralAnnot.json"
    ),
    "r",
) as f:
    mano_params = json.load(f)


aid_list = list(dataset.anns.keys())
datalist = []

for aid in tqdm.tqdm(aid_list, ncols=100):
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

    capture_id

    # camera parameters
    t, R = np.array(
        cameras[str(capture_id)]["campos"][str(cam)], dtype=np.float32
    ).reshape(3), np.array(
        cameras[str(capture_id)]["camrot"][str(cam)], dtype=np.float32
    ).reshape(
        3, 3
    )
    t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
    focal, princpt = np.array(
        cameras[str(capture_id)]["focal"][str(cam)], dtype=np.float32
    ).reshape(2), np.array(
        cameras[str(capture_id)]["princpt"][str(cam)], dtype=np.float32
    ).reshape(
        2
    )
    cam_param = {"R": R, "t": t, "focal": focal, "princpt": princpt}

    # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
    joint_trunc = np.array(ann["joint_valid"], dtype=np.float32).reshape(-1, 1)
    joint_trunc[joint_set["joint_type"]["right"]] *= joint_trunc[
        joint_set["root_joint_idx"]["right"]
    ]
    joint_trunc[joint_set["joint_type"]["left"]] *= joint_trunc[
        joint_set["root_joint_idx"]["left"]
    ]
    if np.sum(joint_trunc) == 0:
        continue

    joint_valid = np.array(
        joints[str(capture_id)][str(frame_idx)]["joint_valid"], dtype=np.float32
    ).reshape(-1, 1)
    joint_valid[joint_set["joint_type"]["right"]] *= joint_valid[
        joint_set["root_joint_idx"]["right"]
    ]
    joint_valid[joint_set["joint_type"]["left"]] *= joint_valid[
        joint_set["root_joint_idx"]["left"]
    ]
    if np.sum(joint_valid) == 0:
        continue

    # joint coordinates
    joint_world = np.array(
        joints[str(capture_id)][str(frame_idx)]["world_coord"], dtype=np.float32
    ).reshape(-1, 3)
    joint_cam = world2cam(joint_world, R, t)
    joint_cam[np.tile(joint_valid == 0, (1, 3))] = (
        1.0  # prevent zero division error
    )
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

    datalist.append(
        {
            "aid": aid,
            "capture_id": capture_id,
            "seq_name": seq_name,
            "cam_id": cam,
            "frame_idx": frame_idx,
            "img_path": img_path,
            "img_shape": (img_height, img_width),
            "body_bbox": body_bbox,
            "lhand_bbox": lhand_bbox,
            "rhand_bbox": rhand_bbox,
            "joint_img": joint_img,
            "joint_cam": joint_cam,
            "joint_valid": joint_valid,
            "joint_trunc": joint_trunc,
            "cam_param": cam_param,
            "mano_param": mano_param,
            "hand_type": hand_type,
        }
    )

seq_list = defaultdict(list)
for item in tqdm.tqdm(datalist, ncols=100):
    item = deepcopy(item)
    capture_id = item["capture_id"]
    seq_name = item["seq_name"]
    cam_id = item["cam_id"]
    frame_idx = item["frame_idx"]

    if capture_id is not None and seq_name is not None and cam_id is not None:
        del item["capture_id"]
        del item["seq_name"]
        del item["cam_id"]

        seq_list[(capture_id, seq_name, cam_id)].append(item)

smplx_layer = {
    "right": smplx.create("models", "mano", is_rhand=True, use_pca=False),
    "left": smplx.create("models", "mano", is_rhand=False, use_pca=False),
}

seq_annot = seq_list
for k, v in seq_annot.items():
    v.sort(key=lambda x: x["frame_idx"])

def process_single_annot(sample, h):
    intrinsics = np.array([
        [sample["cam_param"]["focal"][0], 0, sample["cam_param"]["princpt"][0]],
        [0, sample["cam_param"]["focal"][1], sample["cam_param"]["princpt"][1]],
        [0, 0, 1],
    ])

    img_path = os.path.join(*Path(sample["img_path"]).parts[-4:])
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
    t = np.array(sample["cam_param"]["t"]).astype(np.float32)

    pose =  np.array(sample["mano_param"][handedness]["pose"]).astype(np.float32)
    shape = np.array(sample["mano_param"][handedness]["shape"]).astype(np.float32)
    trans = np.array(sample["mano_param"][handedness]["trans"]).astype(np.float32)

    # MANO param trans
    root_pose = pose[:3]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(r, root_pose))
    pose[:3] = root_pose[:, 0]

    pose =  torch.from_numpy(pose)[None, ...]
    shape = torch.from_numpy(shape)[None, ...]
    trans = torch.from_numpy(trans)[None, ...]

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

    annot_item = {
        "img_path": img_path,
        "frame_idx": sample["frame_idx"],
        "handedness": handedness,
        "bbox_tight": bbox_tight,
        "joint_img": joint_img,
        "joint_bbox_img": joint_bbox_img,
        "joint_cam": joint_cam,
        "joint_valid": joint_valid,
        "joint_rel": joint_rel,
        "mano_pose": pose[0].numpy(),
        "mano_shape": shape[0].numpy(),
        "focal": np.array(focal),
        "princpt": np.array(princpt),
    }

    return annot_item

for (capture_id, seq_name, cam_id), annots in tqdm(seq_annot.items(), ncols=100):
    for h in ['r', 'l']:
        # find valid range
        start, end = 0, 0
        while start < len(annots):
            while start < len(annots) and annots[start][f"{h}hand_bbox"] is None:
                start += 1
            end = start
            while end < len(annots) and annots[end][f"{h}hand_bbox"] is not None:
                end += 1

            # guard invalid range
            if not start < len(annots):
                continue

            # process
            single_sequence = {
                "capture_id": capture_id,
                "seq_name": seq_name,
                "cam_id": cam_id,
                "annots": []
            }
            for i in range(start, end):
                sample = annots[i]
                try:
                    sample_processed = process_single_annot(sample, h)
                except Exception as ex:
                    print(f"Error occured at capture={capture_id}, seq={seq_name}, cam={cam_id}, frame={sample['frame_idx']}, ex={str(ex)}")
                    break
                single_sequence["annots"].append(sample_processed)
