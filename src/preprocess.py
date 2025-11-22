import torch
import kornia.geometry.transform as T

def preprocess_batch(batch, device: torch.device):
    """
    将wds的数据转换为给到模型的数据，同时进行数据增强
    """
    pass