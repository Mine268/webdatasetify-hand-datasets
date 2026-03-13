# Dataset Schema Redesign And COCO-WholeBody Integration

## 1. Document Purpose

This document defines the full dataset schema redesign for this repository after introducing finer-grained supervision flags and COCO-WholeBody integration.

The document covers:

- the current legacy WebDataset sample schema;
- the redesigned canonical schema;
- the exact definition, shape, dtype, and semantics of every old and new field;
- the mapping from each supported dataset into the redesigned schema;
- the fixed virtual intrinsics design for COCO-WholeBody;
- the migration plan from the legacy schema to the redesigned schema.

This document is the source of truth for future writer, dataloader, preprocess, and training-side changes.


## 2. Scope And Terminology

### 2.1 Sample Granularity

In this repository, a single tar sample is a hand clip sample, not a single frame sample.

- For temporal datasets such as InterHand2.6M, DexYCB, HO3D, and HOT3D:
  one tar sample contains `T` consecutive frames from the same hand in the same sequence.
- For COCO-WholeBody:
  one tar sample contains exactly one hand instance from one image, so `T = 1`.

The first dimension of all per-frame numeric arrays is the temporal dimension `T`.

### 2.2 Coordinate Conventions

- `hand_bbox` and `joint_img` are defined in the original image coordinate system.
- `joint_hand_bbox` is defined relative to the top-left corner of `hand_bbox`.
- `joint_cam` and `joint_rel` are defined in camera space, in millimeters.
- `focal` and `princpt` are defined in the original image coordinate system, not in crop coordinates.

### 2.3 Storage Conventions

- `*.json` fields are stored as JSON-serializable objects and decoded by `wds.decode()`.
- `*.pickle` fields are stored as pickled Python objects and decoded by `wds.decode()`.
- `*.npy` fields are stored as NumPy arrays and decoded by `wds.decode()`.


## 3. Legacy Schema (Current Mainline Before Redesign)

The current mainline writers (`*_mp.py` and `preprocess_HOT3D.py`) write a clip sample with the following keys.

### 3.1 Legacy Sample Structure

| Key | Decoded Type | Canonical Shape | Canonical Dtype | Status | Definition |
| --- | --- | --- | --- | --- | --- |
| `__key__` | `str` | scalar | string | active | Unique sample identifier inside WebDataset. One key corresponds to one hand clip sample. |
| `imgs_path.json` | `list[str]` | length `T` | string | active | Relative image path for each frame. |
| `img_bytes.pickle` | `list[bytes-like]` | length `T` | encoded image payload | active | Encoded hand-frame images, typically WebP. Legacy data may contain `bytes` or NumPy buffers originating from `cv2.imencode`. Canonical redesigned storage targets `bytes`. |
| `handedness.json` | `str` | scalar | string | active | Hand side, currently `"right"` or `"left"`. |
| `additional_desc.json` | `list[dict]` | length `T` | JSON object | active | Per-frame auxiliary metadata for debugging or provenance. Contents are dataset-specific. |
| `hand_bbox.npy` | `np.ndarray` | `[T, 4]` | `float32` | active | Tight hand bounding box in original image coordinates, format `[x1, y1, x2, y2]`. |
| `joint_img.npy` | `np.ndarray` | `[T, 21, 2]` | `float32` | active | 2D joint coordinates in original image coordinates, ordered as `TARGET_JOINTS_ORDER`. |
| `joint_hand_bbox.npy` | `np.ndarray` | `[T, 21, 2]` | `float32` | active | 2D joint coordinates relative to `hand_bbox[..., :2]`. |
| `joint_cam.npy` | `np.ndarray` | `[T, 21, 3]` | `float32` | active | 3D joint coordinates in camera space, in millimeters. |
| `joint_rel.npy` | `np.ndarray` | `[T, 21, 3]` | `float32` | active | 3D joint coordinates relative to the root joint in camera space, in millimeters. |
| `joint_valid.npy` | `np.ndarray` | `[T, 21]` | `float32` | legacy | Mixed-semantics joint validity mask. In legacy code this field is used for both 2D-related masking and 3D-related supervision, but its exact meaning varies by dataset. |
| `mano_pose.npy` | `np.ndarray` | `[T, 48]` | `float32` | active | MANO pose parameters. When unavailable, legacy writers often store zeros. |
| `mano_shape.npy` | `np.ndarray` | `[T, 10]` | `float32` | active | MANO shape parameters. When unavailable, legacy writers often store zeros. |
| `mano_valid.npy` | `np.ndarray` | `[T]` | `float32` | legacy | Legacy MANO availability flag. Often stored as bool or float-like values and cast to float by the dataloader. |
| `timestamp.npy` | `np.ndarray` | `[T]` | `float32` | active | Per-frame timestamp in milliseconds. |
| `focal.npy` | `np.ndarray` | `[T, 2]` | `float32` | active | Focal length `[fx, fy]` in original image coordinates. |
| `princpt.npy` | `np.ndarray` | `[T, 2]` | `float32` | active | Principal point `[cx, cy]` in original image coordinates. |

### 3.2 Problems In The Legacy Schema

The legacy schema is sufficient for existing fully- or mostly-3D datasets, but it is not sufficient for mixing in datasets such as COCO-WholeBody.

The main issues are:

- `joint_valid.npy` does not distinguish 2D supervision availability from 3D supervision availability.
- `mano_valid.npy` only describes MANO availability and does not generalize to other supervision types.
- there is no explicit distinction between real intrinsics, fixed virtual intrinsics, and pseudo/fitted intrinsics;
- there is no explicit sample-level data source identifier;
- there is no stable raw-dataset index field that can trace a decoded sample back to the original dataset item;
- `joint_cam.npy`, `joint_rel.npy`, `mano_pose.npy`, and `mano_shape.npy` are always present physically, but their supervision semantics are unclear when the source dataset does not provide those annotations.


## 4. Redesign Goals

The redesigned schema must satisfy the following requirements.

1. Separate 2D supervision availability from 3D supervision availability.
2. Separate MANO supervision availability from geometric keypoint supervision.
3. Separate the existence of intrinsics from the type of intrinsics.
4. Preserve a fixed clip-based sample structure for all datasets.
5. Allow `2D-only` datasets to use the same container format without pretending to provide real 3D or real camera calibration.
6. Make every sample traceable back to its dataset source and raw sample index.
7. Keep enough backward compatibility to migrate the repository incrementally.


## 5. Canonical Redesigned Schema (V2)

The redesigned schema keeps the legacy clip layout and adds finer-grained supervision and provenance fields.

### 5.1 V2 Sample Structure

| Key | Decoded Type | Canonical Shape | Canonical Dtype | Required | Definition |
| --- | --- | --- | --- | --- | --- |
| `__key__` | `str` | scalar | string | yes | Unique WebDataset key. |
| `imgs_path.json` | `list[str]` | length `T` | string | yes | Relative image path for each frame. |
| `img_bytes.pickle` | `list[bytes]` | length `T` | encoded bytes | yes | Encoded image bytes for each frame. In V2 this should be standardized to `bytes`. |
| `handedness.json` | `str` | scalar | string | yes | `"right"` or `"left"`. |
| `additional_desc.json` | `list[dict]` | length `T` | JSON object | yes | Free-form per-frame metadata. This field is for debugging, bookkeeping, or future extensions and is not used as the primary provenance channel. |
| `data_source.json` | `str` | scalar | string | yes | Dataset identifier. Allowed values in the current design are `ih26m`, `dexycb`, `ho3d`, `hot3d`, and `coco_wholebody`. |
| `source_split.json` | `str` | scalar | string | yes | Source split identifier such as `train`, `val`, `test`, or `evaluation`. |
| `source_index.json` | `list[dict]` | length `T` | JSON object | yes | Raw-dataset provenance for each frame. Each element must uniquely map the frame back to the original dataset item. |
| `intr_type.json` | `str` | scalar | string | yes | Intrinsics type. Allowed values are `real`, `fixed_virtual`, `pseudo`, and `none`. |
| `hand_bbox.npy` | `np.ndarray` | `[T, 4]` | `float32` | yes | Hand bounding box in original image coordinates, format `[x1, y1, x2, y2]`. |
| `joint_img.npy` | `np.ndarray` | `[T, 21, 2]` | `float32` | yes | 2D joints in original image coordinates, ordered as `TARGET_JOINTS_ORDER`. |
| `joint_hand_bbox.npy` | `np.ndarray` | `[T, 21, 2]` | `float32` | yes | 2D joints relative to `hand_bbox[..., :2]`. |
| `joint_cam.npy` | `np.ndarray` | `[T, 21, 3]` | `float32` | yes | 3D joints in camera space, in millimeters. For datasets without 3D labels this field remains physically present and uses a placeholder value, typically all zeros. |
| `joint_rel.npy` | `np.ndarray` | `[T, 21, 3]` | `float32` | yes | Root-relative 3D joints in camera space, in millimeters. For datasets without 3D labels this field remains physically present and uses a placeholder value, typically all zeros. |
| `joint_2d_valid.npy` | `np.ndarray` | `[T, 21]` | `float32` | yes | Binary mask for 2D supervision availability. `1` means the 2D joint can be used for 2D losses, 2D masking, and bbox recomputation. `0` means it must be ignored by 2D logic. |
| `joint_3d_valid.npy` | `np.ndarray` | `[T, 21]` | `float32` | yes | Binary mask for 3D supervision availability. `1` means the 3D joint can be used as ground truth. `0` means it must be ignored by all 3D losses and all logic that interprets `joint_cam/joint_rel` as GT. |
| `joint_valid.npy` | `np.ndarray` | `[T, 21]` | `float32` | transition-only | Legacy compatibility mask. New code must not treat this as the primary validity mask. See Section 6.1 for the compatibility rule. |
| `mano_pose.npy` | `np.ndarray` | `[T, 48]` | `float32` | yes | MANO pose parameters. When the source dataset does not provide MANO supervision, this field remains physically present and uses a placeholder value, typically all zeros. |
| `mano_shape.npy` | `np.ndarray` | `[T, 10]` | `float32` | yes | MANO shape parameters. When the source dataset does not provide MANO supervision, this field remains physically present and uses a placeholder value, typically all zeros. |
| `has_mano.npy` | `np.ndarray` | `[T]` | `float32` | yes | Binary mask indicating whether MANO supervision is available for the frame. `1` means MANO targets are valid GT. `0` means `mano_pose/mano_shape` are placeholders only. |
| `mano_valid.npy` | `np.ndarray` | `[T]` | `float32` | transition-only | Legacy compatibility alias for MANO availability. New code must prefer `has_mano.npy`. |
| `has_intr.npy` | `np.ndarray` | `[T]` | `float32` | yes | Binary mask indicating whether intrinsics are available and can be used by projection-related logic. `1` means `focal/princpt` are defined. `0` means they must be ignored. |
| `timestamp.npy` | `np.ndarray` | `[T]` | `float32` | yes | Timestamp in milliseconds. For single-image datasets or datasets without meaningful time, a deterministic placeholder such as `0` may be used. |
| `focal.npy` | `np.ndarray` | `[T, 2]` | `float32` | yes | Focal length `[fx, fy]` in original image coordinates. May represent real intrinsics, fixed virtual intrinsics, pseudo intrinsics, or placeholders depending on `intr_type` and `has_intr`. |
| `princpt.npy` | `np.ndarray` | `[T, 2]` | `float32` | yes | Principal point `[cx, cy]` in original image coordinates. May represent real intrinsics, fixed virtual intrinsics, pseudo intrinsics, or placeholders depending on `intr_type` and `has_intr`. |

### 5.2 Placeholder Policy

The V2 schema keeps a physically fixed tensor layout across all datasets. Therefore, some datasets must use placeholders for unavailable targets.

The placeholder rules are:

- if a dataset does not provide 3D keypoint supervision:
  `joint_cam.npy` and `joint_rel.npy` must still be written with shape `[T, 21, 3]`, typically filled with zeros, and `joint_3d_valid.npy` must be all zeros;
- if a dataset does not provide MANO supervision:
  `mano_pose.npy` and `mano_shape.npy` must still be written with shapes `[T, 48]` and `[T, 10]`, typically filled with zeros, and `has_mano.npy` must be all zeros;
- if a dataset does not provide real intrinsics but the pipeline chooses to synthesize fixed virtual intrinsics:
  `has_intr.npy` must be all ones and `intr_type.json` must be `fixed_virtual`;
- if a dataset truly has no usable intrinsics and no synthesized intrinsics:
  `focal.npy` and `princpt.npy` must still be written with shapes `[T, 2]`, typically filled with zeros, `has_intr.npy` must be all zeros, and `intr_type.json` must be `none`.


## 6. Exact Definition Of Old And New Fields

This section is normative. Every field below must follow the specified shape and meaning.

### 6.1 Legacy Compatibility Fields

| Field | Shape | Dtype | Definition | V2 Rule |
| --- | --- | --- | --- | --- |
| `joint_valid.npy` | `[T, 21]` | `float32` | Legacy mixed validity mask used by older code paths. | Retained only for migration. For datasets where `joint_2d_valid == joint_3d_valid`, write that shared mask. For `2D-only` datasets such as COCO-WholeBody, write `joint_valid = joint_2d_valid` for compatibility only. New code must never use this field as a 3D supervision mask. |
| `mano_valid.npy` | `[T]` | `float32` | Legacy MANO availability flag. | Retained only for migration and written as an alias of `has_mano.npy`. New code must prefer `has_mano.npy`. |

### 6.2 Core Geometry Fields

| Field | Shape | Dtype | Definition |
| --- | --- | --- | --- |
| `hand_bbox.npy` | `[T, 4]` | `float32` | Hand box in original image coordinates, format `[x1, y1, x2, y2]`. |
| `joint_img.npy` | `[T, 21, 2]` | `float32` | 2D joints in original image coordinates, ordered as `TARGET_JOINTS_ORDER`. |
| `joint_hand_bbox.npy` | `[T, 21, 2]` | `float32` | `joint_img - hand_bbox[..., None, :2]`. |
| `joint_cam.npy` | `[T, 21, 3]` | `float32` | Camera-space 3D joints in millimeters. Placeholder zeros are allowed only when `joint_3d_valid` is zero. |
| `joint_rel.npy` | `[T, 21, 3]` | `float32` | Root-relative 3D joints in millimeters. Placeholder zeros are allowed only when `joint_3d_valid` is zero. |
| `timestamp.npy` | `[T]` | `float32` | Timestamp in milliseconds. |
| `focal.npy` | `[T, 2]` | `float32` | `[fx, fy]` in original image coordinates. |
| `princpt.npy` | `[T, 2]` | `float32` | `[cx, cy]` in original image coordinates. |

### 6.3 Supervision Availability Fields

| Field | Shape | Dtype | Definition |
| --- | --- | --- | --- |
| `joint_2d_valid.npy` | `[T, 21]` | `float32` | Binary mask for 2D supervision availability. Used for 2D losses, 2D masking, and recomputing 2D-derived boxes. |
| `joint_3d_valid.npy` | `[T, 21]` | `float32` | Binary mask for 3D supervision availability. Used for all losses that consume `joint_cam` or `joint_rel` as GT. |
| `has_mano.npy` | `[T]` | `float32` | Binary mask for MANO supervision availability. |
| `has_intr.npy` | `[T]` | `float32` | Binary mask for intrinsics availability. If `has_intr = 0`, `focal/princpt` must not be used for supervision or geometry transforms. |

### 6.4 Provenance And Intrinsics-Type Fields

| Field | Shape | Dtype | Definition |
| --- | --- | --- | --- |
| `data_source.json` | scalar | string | Dataset source identifier. Allowed values: `ih26m`, `dexycb`, `ho3d`, `hot3d`, `coco_wholebody`. |
| `source_split.json` | scalar | string | Dataset split identifier. |
| `source_index.json` | length `T` | JSON list of dicts | Per-frame raw-dataset index records. Each dict must allow exact reverse lookup to the raw dataset frame or instance. |
| `intr_type.json` | scalar | string | Intrinsics type. Allowed values: `real`, `fixed_virtual`, `pseudo`, `none`. |

### 6.5 Image And Metadata Fields

| Field | Shape | Dtype | Definition |
| --- | --- | --- | --- |
| `imgs_path.json` | length `T` | list of string | Relative image path per frame. |
| `img_bytes.pickle` | length `T` | list of bytes | Encoded image bytes per frame. The canonical V2 representation is `bytes` for each frame. |
| `handedness.json` | scalar | string | Hand side, either `"right"` or `"left"`. |
| `additional_desc.json` | length `T` | list of dict | Free-form per-frame metadata, not a substitute for `source_index.json`. |


## 7. Field Usage Contract For Loader, Preprocess, And Training

The redesigned schema changes the meaning of validity and availability flags. The following rules are mandatory.

### 7.1 2D Logic

The following logic must use `joint_2d_valid.npy` and must not use `joint_3d_valid.npy`:

- 2D keypoint losses;
- 2D visibility masking;
- recomputation of hand bbox or patch bbox from projected joints;
- any augmentation logic that treats `joint_img` as GT.

### 7.2 3D Logic

The following logic must use `joint_3d_valid.npy` and must not use `joint_valid.npy`:

- `joint_cam` loss;
- `joint_rel` loss;
- any logic that interprets `joint_cam/joint_rel` as GT rather than placeholders.

### 7.3 MANO Logic

The following logic must use `has_mano.npy`:

- MANO pose loss;
- MANO shape loss;
- any use of `mano_pose.npy` or `mano_shape.npy` as GT.

### 7.4 Intrinsics Logic

The following logic must use `has_intr.npy` and inspect `intr_type.json`:

- projection of predicted 3D joints to 2D;
- intrinsics-aware augmentation;
- any camera-parameter supervision.

The semantics of `intr_type.json` are:

- `real`: real calibration from the source dataset;
- `fixed_virtual`: deterministic virtual calibration constructed by the pipeline;
- `pseudo`: fitted or estimated calibration, not real calibration;
- `none`: no usable intrinsics.

If `intr_type != real`, training code must not treat `focal/princpt` as real calibration GT.


## 8. Fixed Virtual Intrinsics Design For COCO-WholeBody

COCO-WholeBody does not provide real camera intrinsics. The repository will integrate it using fixed virtual intrinsics.

### 8.1 Source Data Location

- dataset root: `/mnt/qnap/data/datasets/coco2017`
- train annotation: `/mnt/qnap/data/datasets/coco2017/annotations/coco_wholebody_train_v1.0.json`
- val annotation: `/mnt/qnap/data/datasets/coco2017/annotations/coco_wholebody_val_v1.0.json`

### 8.2 COCO Sample Granularity

Each COCO image may contribute up to two hand samples:

- one right-hand sample;
- one left-hand sample.

Each COCO hand sample is a single-frame clip, so `T = 1`.

### 8.3 Fixed Virtual Intrinsics

For an image of size `(W, H)`, the fixed virtual intrinsics are defined in original image coordinates:

- `cx = (W - 1) / 2`
- `cy = (H - 1) / 2`
- `fx = gamma * max(W, H)`
- `fy = gamma * max(W, H)`

The current fixed design uses:

- `gamma = 1.0`

Therefore the canonical stored values are:

```python
focal = np.array([max(W, H), max(W, H)], dtype=np.float32)
princpt = np.array([(W - 1) * 0.5, (H - 1) * 0.5], dtype=np.float32)
```

The intrinsics-related flags for COCO-WholeBody are:

- `has_intr.npy = [1]`
- `intr_type.json = "fixed_virtual"`

### 8.4 COCO Supervision Semantics

For COCO-WholeBody:

- `joint_2d_valid.npy` is derived from hand keypoint visibility;
- `joint_3d_valid.npy` is all zeros;
- `joint_cam.npy` and `joint_rel.npy` are zero placeholders;
- `has_mano.npy` is all zeros;
- `mano_pose.npy` and `mano_shape.npy` are zero placeholders;
- `focal.npy` and `princpt.npy` are fixed virtual intrinsics, not real intrinsics.

COCO-WholeBody samples are therefore `2D-only` samples with usable virtual intrinsics for reprojection-based training, but without real 3D or real camera calibration GT.


## 9. Per-Dataset Mapping Into V2

This section defines how each currently supported dataset maps into the redesigned schema.

### 9.1 InterHand2.6M

| Field Group | Mapping Rule |
| --- | --- |
| `data_source.json` | `"ih26m"` |
| `source_split.json` | environment variable `SPLIT` |
| `source_index.json` | per-frame dict with `capture_id`, `seq_name`, `cam_id`, `frame_idx`, `aid`, and `handedness` |
| `intr_type.json` | `"real"` |
| `has_intr.npy` | all ones |
| `joint_2d_valid.npy` | hand-specific reordered validity derived from `joint_valid * joint_trunc` |
| `joint_3d_valid.npy` | same as `joint_2d_valid.npy` under the current annotation contract |
| `joint_valid.npy` | same as `joint_2d_valid.npy` during migration |
| `has_mano.npy` | `1` if `sample["mano_param"][handedness]` exists, else `0` |
| `mano_valid.npy` | alias of `has_mano.npy` during migration |

### 9.2 DexYCB

| Field Group | Mapping Rule |
| --- | --- |
| `data_source.json` | `"dexycb"` |
| `source_split.json` | `SPLIT` |
| `source_index.json` | per-frame dict with `subject`, `sequence`, `serial`, `frame_idx`, and `handedness` |
| `intr_type.json` | `"real"` |
| `has_intr.npy` | all ones |
| `joint_2d_valid.npy` | all ones |
| `joint_3d_valid.npy` | all ones |
| `joint_valid.npy` | all ones during migration |
| `has_mano.npy` | all ones |
| `mano_valid.npy` | alias of `has_mano.npy` during migration |

### 9.3 HO3D Train And Evaluation

| Field Group | Mapping Rule |
| --- | --- |
| `data_source.json` | `"ho3d"` |
| `source_split.json` | `train` or `evaluation` |
| `source_index.json` | per-frame dict with `seq_name`, `frame_name`, and `handedness` |
| `intr_type.json` | `"real"` |
| `has_intr.npy` | all ones |
| `joint_2d_valid.npy` | all ones |
| `joint_3d_valid.npy` | all ones |
| `joint_valid.npy` | all ones during migration |
| `has_mano.npy` | `1` when MANO labels exist, else `0` |
| `mano_valid.npy` | alias of `has_mano.npy` during migration |

### 9.4 HOT3D

| Field Group | Mapping Rule |
| --- | --- |
| `data_source.json` | `"hot3d"` |
| `source_split.json` | `train` or `test` |
| `source_index.json` | per-frame dict with `sequence_name`, `timestamp_ns`, and `handedness` |
| `intr_type.json` | `"real"` |
| `has_intr.npy` | all ones |
| `joint_2d_valid.npy` | all ones except interpolated `Thumb_1`, which is zero |
| `joint_3d_valid.npy` | all ones except interpolated `Thumb_1`, which is zero |
| `joint_valid.npy` | same as `joint_2d_valid.npy` during migration |
| `has_mano.npy` | copied from source MANO validity |
| `mano_valid.npy` | alias of `has_mano.npy` during migration |

### 9.5 COCO-WholeBody

| Field Group | Mapping Rule |
| --- | --- |
| `data_source.json` | `"coco_wholebody"` |
| `source_split.json` | `train` or `val` |
| `source_index.json` | per-frame dict with `image_id`, `ann_id`, and `handedness` |
| `intr_type.json` | `"fixed_virtual"` |
| `has_intr.npy` | all ones |
| `joint_2d_valid.npy` | derived from COCO hand keypoint visibility |
| `joint_3d_valid.npy` | all zeros |
| `joint_valid.npy` | equal to `joint_2d_valid.npy` for migration only |
| `joint_cam.npy` | zero placeholders |
| `joint_rel.npy` | zero placeholders |
| `has_mano.npy` | all zeros |
| `mano_valid.npy` | alias of `has_mano.npy` during migration |
| `mano_pose.npy` | zero placeholders |
| `mano_shape.npy` | zero placeholders |
| `timestamp.npy` | zero placeholder |


## 10. Requirements For `source_index.json`

`source_index.json` is required in V2 and must be length `T`.

Each element in `source_index.json` must:

- identify the original dataset item unambiguously;
- contain only JSON-serializable scalar values or small strings;
- remain stable across tar regeneration as long as the raw dataset does not change;
- be sufficient for reverse lookup and debugging.

Recommended examples:

```json
[
  {
    "capture_id": 12,
    "seq_name": "0002_dinosaur",
    "cam_id": "400262",
    "frame_idx": 128,
    "aid": 391829,
    "handedness": "right"
  }
]
```

```json
[
  {
    "image_id": 391895,
    "ann_id": 848372,
    "handedness": "left"
  }
]
```


## 11. Recommended Canonical Dtypes

Although some legacy writers currently emit bool-like arrays for certain flags, the redesigned schema standardizes the following dtypes:

| Field Category | Canonical Dtype |
| --- | --- |
| geometric coordinates and boxes | `float32` |
| timestamps | `float32` |
| validity masks and availability flags | `float32` with values in `{0, 1}` |
| JSON metadata | JSON-native scalar, string, list, or dict |
| encoded images | `bytes` inside `img_bytes.pickle` |

This dtype convention matches the current dataloader behavior, which converts NumPy arrays to float tensors.


## 12. Migration Plan

The migration from the legacy schema to V2 must proceed in three phases.

### Phase 1: Writer Compatibility

All writers emit:

- all legacy fields required by current code paths;
- all V2 fields introduced in this document.

At this phase:

- `joint_valid.npy` remains present as a compatibility alias;
- `mano_valid.npy` remains present as a compatibility alias.

### Phase 2: Loader And Preprocess Migration

The dataloader and preprocessing code are updated to:

- prefer `joint_2d_valid.npy` over `joint_valid.npy` for 2D logic;
- prefer `joint_3d_valid.npy` for 3D logic;
- prefer `has_mano.npy` over `mano_valid.npy`;
- inspect `has_intr.npy` and `intr_type.json` before consuming `focal/princpt`.

Fallback to legacy fields is allowed only to support already-generated old tar files.

### Phase 3: Training-Side Semantic Cleanup

Training and loss code are updated so that:

- `joint_cam.npy` and `joint_rel.npy` are interpreted as GT only where `joint_3d_valid.npy == 1`;
- `mano_pose.npy` and `mano_shape.npy` are interpreted as GT only where `has_mano.npy == 1`;
- `focal.npy` and `princpt.npy` are interpreted according to `intr_type.json`;
- provenance fields are available for debugging, filtering, and dataset-specific logic.

After all code paths are migrated and legacy tar files are regenerated, the repository may remove direct dependence on `joint_valid.npy` and `mano_valid.npy`.


## 13. Summary Of The Redesign

The redesign makes three structural changes:

1. supervision is split into explicit channels:
   `joint_2d_valid`, `joint_3d_valid`, `has_mano`, and `has_intr`;
2. intrinsics semantics become explicit via `intr_type`;
3. provenance becomes explicit via `data_source`, `source_split`, and `source_index`.

Under this design:

- existing 3D datasets keep their current physical tensor structure;
- COCO-WholeBody can be integrated without pretending to provide real 3D or real calibration;
- legacy fields remain available during migration but are clearly defined as compatibility-only aliases.
