import webdataset as wds
import torch
from torch.utils.data import DataLoader, default_collate
import numpy as np
import io
from PIL import Image
import glob
import tqdm

# ================= 配置区域 =================
# 替换为你生成的数据路径，支持 shell 风格的通配符
URLS = glob.glob("/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train/ih26m_train-worker*.tar")

# 定义需要从 Sequence 拆分出来的 Numpy 字段 (对应 Writer 中的 NUMPY_KEYS)
# 注意：Writer 中 keys 是不带后缀的，但 wds decode 后 key 会带上后缀 (如 .npy)
NPY_KEYS = [
    "hand_bbox.npy", "joint_img.npy", "joint_hand_bbox.npy",
    "joint_cam.npy", "joint_rel.npy", "joint_valid.npy",
    "mano_pose.npy", "mano_shape.npy", "timestamp.npy",
    "focal.npy", "princpt.npy"
]

# ================= 核心处理逻辑 =================

def flatten_sequences(source):
    """
    生成器函数：将读取到的 '序列样本' 拆解为 '单帧样本'。
    WebDataset 的样本在 decode() 后是一个字典。
    """
    for sample in source:
        # 1. 获取序列长度 (基于图片列表)
        # 对应 Writer: img_bytes_pickle = pickle.dumps([bytes, bytes, ...])
        # decode() 后变为: [bytes, bytes, ...]
        img_list = sample["img_bytes.pickle"]
        n_frames = len(img_list)

        # 2. 获取序列级共享属性
        # 对应 Writer: handedness_json
        handedness = sample["handedness.json"] # "right" or "left"

        # 3. 遍历每一帧并 yield
        for i in range(n_frames):
            frame_sample = {
                "__key__": f"{sample['__key__']}_{i:04d}", # 构造唯一的帧 ID
                "img_bytes": img_list[i],                  # 单帧图片 bytes
                "handedness": handedness,                  # 字符串
            }

            # 拆解 Numpy 数组：从 (T, ...) 取第 i 个 -> (...)
            for key in NPY_KEYS:
                # 移除 .npy 后缀作为输出 key，保持代码整洁
                out_key = key.replace(".npy", "")
                frame_sample[out_key] = sample[key][i]

            yield frame_sample

def preprocess_frame(sample):
    """
    单帧预处理：解码图片 Bytes -> Tensor，处理数据类型
    """
    # 1. 图片解码: Bytes (WebP) -> PIL -> Tensor
    # Writer 中使用的是 cv2.imencode(".webp")，这里用 PIL 打开兼容性很好
    img_bytes = sample["img_bytes"]
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # debug
    img = img.resize((256, 256))

    # 转换为 Tensor (C, H, W) 并归一化 [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # 2. 处理其他 Numpy 字段
    result = {
        "image": img_tensor,
        "handedness": sample["handedness"], # 此时还是 str, collate 时可能需要特殊处理或 drop
    }

    # 自动将所有 numpy 字段转为 Tensor
    for key in sample:
        if key not in ["__key__", "img_bytes", "handedness"]:
            # 确保是 float32 (根据你的 writer 逻辑，大部分已经是 float32)
            val = sample[key]
            if isinstance(val, np.ndarray):
                result[key] = torch.from_numpy(val).float()
            else:
                result[key] = torch.tensor(val)

    return result

# 自定义 Collate (可选): 如果需要把 handedness 这种字符串转为 token 或忽略
def custom_collate(batch):
    # 过滤掉不需要 stack 的字段 (如 handedness string) 或者将其单独处理
    # 这里简单示范：使用 default_collate，它会自动忽略无法 stack 的字段或者报错
    # 建议：如果不需要 handedness 参与训练，可以在 preprocess_frame 里删掉它

    # 为了演示，我们假设不需要 handedness 字符串进模型，或者你稍后自己处理
    clean_batch = []
    for sample in batch:
        s = sample.copy()
        if "handedness" in s: del s["handedness"] # 删掉字符串避免报错
        clean_batch.append(s)

    return default_collate(clean_batch)

# ================= Pipeline 构建 =================

def get_dataloader(url, batch_size=64, num_workers=4):
    dataset = (
        wds.WebDataset(url, shardshuffle=True)
        .decode()                    # 关键：自动处理 .pickle (img list), .npy (arrays), .json
        .compose(flatten_sequences)  # 关键：Sequence (T) -> Frames (1) 的拆解器
        .shuffle(5000)               # 关键：在内存中打乱“帧”，打破时序相关性
        .map(preprocess_frame)       # 解码图片 bytes -> Tensor
        .batched(                    # Pipeline Batching (推荐方案)
            batch_size,
            partial=False,
            collation_fn=custom_collate # 自动 stack 为 (B, C, H, W)
        )
    )

    # 使用 batch_size=None 模式
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )

    return loader

# ================= 测试代码 =================
if __name__ == "__main__":
    # 这里的 URL 应该匹配你 Writer 输出的实际路径
    # 记得把 {worker_id} 这种格式化字符串换成真实的通配符
    # 比如: "ih26m_wds_output/ih26m_train-worker*-*.tar"

    loader = get_dataloader(URLS, batch_size=32, num_workers=4)

    print("Starting DataLoader test...")
    for i, batch in enumerate(tqdm.tqdm(loader)):
        imgs = batch["image"]
        joints = batch["joint_cam"]

        # if i == 0:
        #     print(f"Batch Size: {imgs.shape[0]}")
        #     print(f"Image Shape: {imgs.shape}")       # 应为 [32, 3, H, W]
        #     print(f"Joint Cam Shape: {joints.shape}") # 应为 [32, 21, 3]
        #     print(f"Hand Bbox Shape: {batch['hand_bbox'].shape}")
        #     print("Test Passed!")
        #     break