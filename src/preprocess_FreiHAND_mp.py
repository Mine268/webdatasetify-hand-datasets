import json
import math
import multiprocessing
import os
import os.path as osp
import pickle
from typing import List

import cv2
import numpy as np
import webdataset as wds

try:
    from .freihand_common import (
        build_sample_record,
        encode_image_to_webp,
        get_image_count,
        load_image,
        load_split_annotations,
    )
except ImportError:
    from freihand_common import (
        build_sample_record,
        encode_image_to_webp,
        get_image_count,
        load_image,
        load_split_annotations,
    )


SPLIT = os.environ.get("SPLIT", "train")
assert SPLIT in {"train", "evaluation"}, f"Unsupported SPLIT={SPLIT}"
OUTPUT_PATTERN = (
    f"freihand_{SPLIT}_wds_output/freihand_{SPLIT}-worker{{worker_id}}-%06d.tar"
)
MAX_COUNT = 100000
MAX_SIZE = 3 * 1024 * 1024 * 1024
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEBUG_MAX_SAMPLES = int(os.environ.get("DEBUG_MAX_SAMPLES", "0"))

NUMPY_KEYS = [
    "hand_bbox",
    "joint_img",
    "joint_hand_bbox",
    "joint_cam",
    "joint_rel",
    "joint_2d_valid",
    "joint_3d_valid",
    "joint_valid",
    "mano_pose",
    "mano_shape",
    "has_mano",
    "mano_valid",
    "has_intr",
    "timestamp",
    "focal",
    "princpt",
]

os.makedirs(osp.dirname(OUTPUT_PATTERN), exist_ok=True)


def process_single_sample(image_idx: int):
    record = build_sample_record(SPLIT, image_idx)
    img = load_image(SPLIT, image_idx)
    img_bytes = encode_image_to_webp(img)
    record["img_bytes"] = img_bytes
    return record


def process_batch(image_indices: List[int], worker_id: int) -> int:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    worker_pattern = OUTPUT_PATTERN.format(worker_id=worker_id)
    processed_count = 0

    with wds.ShardWriter(worker_pattern, maxcount=MAX_COUNT, maxsize=MAX_SIZE) as sink:
        for image_idx in image_indices:
            try:
                frame = process_single_sample(image_idx)
            except Exception as ex:
                print(f"[Worker {worker_id}] image_idx={image_idx} error: {ex}")
                continue

            source_index = frame["source_index"]
            key_str = (
                f"{source_index['sample_idx']:08d}_"
                f"v{source_index['version_idx']}_"
                f"{SPLIT}_freihand"
            )
            wds_sample = {
                "__key__": key_str,
                "imgs_path.json": json.dumps([frame["img_path"]]),
                "img_bytes.pickle": pickle.dumps([frame["img_bytes"]]),
                "handedness.json": json.dumps(frame["handedness"]),
                "additional_desc.json": json.dumps([frame["additional_desc"]]),
                "data_source.json": json.dumps("freihand"),
                "source_split.json": json.dumps(SPLIT),
                "source_index.json": json.dumps([frame["source_index"]]),
                "intr_type.json": json.dumps("real"),
            }
            for key in NUMPY_KEYS:
                wds_sample[f"{key}.npy"] = np.stack([frame[key]])
            sink.write(wds_sample)
            processed_count += 1

    return processed_count


def build_tasks() -> List[int]:
    total_samples = get_image_count(SPLIT)
    tasks = list(range(total_samples))
    if DEBUG_MAX_SAMPLES > 0:
        tasks = tasks[:DEBUG_MAX_SAMPLES]
    return tasks


def main() -> None:
    print(f"Loading FreiHAND annotations for split={SPLIT} ...")
    load_split_annotations(SPLIT)

    tasks = build_tasks()
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No FreiHAND samples found.")
        return

    worker_count = max(NUM_WORKERS, 1)
    chunk_size = math.ceil(total_tasks / worker_count)
    chunks = [tasks[i : i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    print(f"Total FreiHAND images: {total_tasks}")
    print(f"Starting {len(chunks)} workers processing ~{chunk_size} samples each ...")

    process_args = [(chunk, worker_id) for worker_id, chunk in enumerate(chunks)]
    if len(chunks) == 1:
        results = [process_batch(chunks[0], 0)]
    else:
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            results = pool.starmap(process_batch, process_args)

    print(f"All done! Total FreiHAND clips processed: {sum(results)}")


if __name__ == "__main__":
    main()
