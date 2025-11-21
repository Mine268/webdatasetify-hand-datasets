import webdataset as wds
import numpy as np
import cv2
import os
import glob
# import matplotlib.pyplot as plt
import torch
from pprint import pprint
import tqdm

# ================= 配置 =================
# 请根据实际生成的路径调整 pattern
# 注意：Windows 下 glob pattern 分隔符可能需要调整，Linux 下通常没问题
DATA_PATH_PATTERN = "ih26m_wds_output/ih26m_train-worker*.tar"

def test_loading():
    # 1. 查找文件
    tar_files = glob.glob(DATA_PATH_PATTERN)
    if not tar_files:
        print(f"Error: No files found matching {DATA_PATH_PATTERN}")
        return

    print(f"Found {len(tar_files)} tar files. Loading first few samples...")
    print(f"Example file: {tar_files[0]}")

    # 2. 构建 WebDataset Pipeline
    # decode() 会自动根据文件后缀处理：
    # .pickle -> pickle.loads (这里是 img_bytes list)
    # .npy -> np.load
    # .json -> json.loads
    # 显式设置 shardshuffle=False 消除警告
    dataset = (
        wds.WebDataset(tar_files, shardshuffle=True)
        .decode()
    )

    # 3. 遍历并检查
    num_samples_to_check = 5

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

    for i, sample in enumerate(tqdm.tqdm(dataloader)):
        # continue
        if i >= num_samples_to_check:
            break

        print(f"\n=== Sample {i} ===")
        print(f"Key: {sample['__key__']}")

        # 检查基础字段是否存在
        required_keys = [
            "img_bytes.pickle", "handedness.json", "additional_desc.json",
            "hand_bbox.npy", "joint_img.npy"
        ]
        for k in required_keys:
            if k not in sample:
                print(f"[WARNING] Missing key: {k}")

        # A. 检查图像数据
        img_bytes_list = sample["img_bytes.pickle"]
        num_frames = len(img_bytes_list)
        print(f"Sequence Length (Frames): {num_frames}")

        if num_frames > 0:
            # 尝试解码第一帧验证完整性
            first_frame_bytes = img_bytes_list[0]
            # np.frombuffer 是必须的，因为 pickle 出来是 pure bytes
            nparr = np.frombuffer(first_frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                print(f"First Frame Decode: Success (Shape: {img.shape})")
            else:
                print("First Frame Decode: FAILED")

        # B. 检查 Numpy 数据形状
        # 期望形状通常是 (T, ...)
        if "hand_bbox.npy" in sample:
            bbox = sample["hand_bbox.npy"]
            print(f"hand_bbox.npy shape: {bbox.shape} (Expected T={num_frames})")
            # 简单的逻辑检查：bbox 应该是 (T, 4)
            assert bbox.shape[0] == num_frames
            assert bbox.shape[1] == 4

        if "joint_img.npy" in sample:
            joints = sample["joint_img.npy"]
            print(f"joint_img.npy shape: {joints.shape} (Expected T={num_frames}, J=21, D=2)")
            # 简单的逻辑检查
            assert joints.shape[0] == num_frames
            assert joints.shape[1] == 21

        # C. 检查 JSON 元数据
        if "handedness.json" in sample:
            print(f"Handedness: {sample['handedness.json']}")

        if "additional_desc.json" in sample:
            desc = sample["additional_desc.json"]
            print(f"Metadata count: {len(desc)}")
            if len(desc) > 0:
                print(f"First frame aid: {desc[0].get('aid', 'N/A')}")

    print("\nTest Finished.")

if __name__ == "__main__":
    test_loading()