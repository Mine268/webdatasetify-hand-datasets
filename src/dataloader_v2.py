"""
Standalone V2 WebDataset dataloader.
"""

from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import webdataset as wds

try:
    from .schema_v2 import normalize_decoded_clip_sample, slice_normalized_clip_sample
except ImportError:
    from schema_v2 import normalize_decoded_clip_sample, slice_normalized_clip_sample


COLLATE_LIST_KEYS = {"imgs"}


def clip_to_t_frames(
    num_frames: int,
    stride: int,
    source: Iterable[Dict[str, Any]],
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
):
    """
    Normalize decoded clip samples and slice them into fixed-length sub-clips.
    """
    for decoded_sample in source:
        clip_sample = normalize_decoded_clip_sample(
            decoded_sample,
            default_data_source=default_data_source,
            default_source_split=default_source_split,
        )
        total_frames = clip_sample["num_frames"]
        if total_frames < num_frames:
            continue

        total_samples = (total_frames - num_frames) // stride + 1
        for slice_index in range(total_samples):
            start = slice_index * stride
            end = start + num_frames
            yield slice_normalized_clip_sample(
                clip_sample, start, end, slice_index=slice_index
            )


def _decode_webp_bytes(img_bytes: bytes) -> torch.Tensor:
    buffer_np = np.frombuffer(img_bytes, dtype=np.uint8).copy()
    buffer = torch.from_numpy(buffer_np)
    try:
        return torchvision.io.decode_webp(buffer)
    except RuntimeError:
        return torchvision.io.decode_image(buffer, mode=torchvision.io.ImageReadMode.RGB)


def preprocess_frame(sample: Dict[str, Any]) -> Dict[str, Any]:
    imgs_tensor = []
    for img_bytes in sample["imgs_bytes"]:
        imgs_tensor.append(_decode_webp_bytes(img_bytes))
    imgs_tensor = torch.stack(imgs_tensor)

    result: Dict[str, Any] = {
        "__key__": sample["__key__"],
        "imgs_path": sample["imgs_path"],
        "imgs": imgs_tensor,
        "handedness": sample["handedness"],
        "data_source": sample["data_source"],
        "source_split": sample["source_split"],
        "source_index": sample["source_index"],
        "intr_type": sample["intr_type"],
        "additional_desc": sample["additional_desc"],
    }

    for key, value in sample.items():
        if key in result or key in {"num_frames", "imgs_bytes"}:
            continue
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value).float()
        else:
            result[key] = value

    return result


def collate_fn(batch_wds: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_filter = [sample for sample in batch_wds if sample is not None]
    if len(batch_filter) == 0:
        return {}

    collated: Dict[str, Any] = {}
    for key in batch_filter[0].keys():
        if key in COLLATE_LIST_KEYS:
            collated[key] = [sample[key] for sample in batch_filter]
            continue

        values = [sample[key] for sample in batch_filter]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated


def get_dataloader(
    url: Sequence[str],
    num_frames: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    shardshuffle: int = 0,
    shuffle_buffer: int = 5000,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
):
    """
    Build a standalone V2 WebDataset dataloader.
    """
    dataset = (
        wds.WebDataset(url, shardshuffle=shardshuffle)
        .decode()
        .compose(
            partial(
                clip_to_t_frames,
                num_frames,
                stride,
                default_data_source=default_data_source,
                default_source_split=default_source_split,
            )
        )
        .shuffle(shuffle_buffer)
        .map(preprocess_frame)
        .batched(batch_size, partial=False, collation_fn=collate_fn)
    )

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )
