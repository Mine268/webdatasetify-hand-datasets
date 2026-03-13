# AGENTS.md - WebDatasetify Hand Datasets

This document provides essential information for AI coding agents working on this project.

## Project Overview

This project is a data preprocessing pipeline that converts popular hand pose estimation datasets into [WebDataset](https://github.com/webdataset/webdataset) format for efficient training of hand pose estimation models.

**Supported Datasets:**
- **InterHand2.6M**: Large-scale 3D interacting hand dataset
- **DexYCB**: Dexterous grasping dataset with YCB objects
- **HO3D_v3**: Hand-Object interaction dataset
- **HOT3D**: Hand and object tracking in 3D (requires projectaria_tools)

**Output Format:**
The pipeline converts datasets into WebDataset shards (.tar files) containing:
- Image sequences (WebP encoded, stored as pickle bytes)
- 3D joint coordinates (camera space, relative to root)
- 2D joint projections (image space)
- MANO hand model parameters (pose and shape)
- Camera intrinsics (focal, principal point)
- Bounding boxes and metadata

## Technology Stack

| Component | Purpose |
|-----------|---------|
| Python 3.x | Core language |
| PyTorch 2.x | Tensor operations, GPU acceleration |
| WebDataset 1.0.2 | Efficient data storage/loading |
| OpenCV (cv2) | Image I/O, encoding/decoding |
| NumPy | Array operations |
| Kornia 0.8.2 | Geometric transformations, augmentations |
| smplx 0.1.28 | MANO hand model loading |
| projectaria-tools 2.1.1 | HOT3D dataset processing |
| pycocotools | COCO format parsing (InterHand2.6M) |
| einops | Tensor reshaping operations |

## Project Structure

```
webdatasetify-hand-datasets/
├── src/                          # Source code
│   ├── preprocess.py             # Data augmentation and preprocessing
│   ├── dataloader.py             # WebDataset loader with frame clipping
│   ├── utils.py                  # Joint reordering, coordinate transforms
│   ├── data_reorganizer.py       # Merge shard files utility
│   ├── preprocess_InterHand26M_mp.py   # InterHand2.6M converter (multi-process)
│   ├── preprocess_InterHand26M.py      # InterHand2.6M converter (single-process)
│   ├── preprocess_DexYCB_mp.py         # DexYCB converter
│   ├── preprocess_HO3D_mp.py           # HO3D converter
│   ├── preprocess_HO3D_eval.py         # HO3D evaluation set converter
│   └── preprocess_HOT3D.py             # HOT3D converter
├── test/                         # Test scripts
│   ├── test_wds.py              # Basic WebDataset loading test
│   └── test_wds_dl.py           # Dataloader with frame flattening test
├── models/                       # MANO model files
│   └── mano/                    # MANO hand model files (not in git)
│       ├── MANO_LEFT.pkl        # Left hand model
│       ├── MANO_RIGHT.pkl       # Right hand model
│       ├── mano_lr_pca.npz      # PCA components for pose
│       └── mano_right_mean.npy  # Mean pose for right hand
├── requirements.txt              # Python dependencies
├── run.sh                        # Example workflow script
└── .vscode/launch.json          # VS Code debug configurations
```

## Build and Run Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preprocessing

All preprocessing scripts use environment variables for configuration:

**InterHand2.6M:**
```bash
SPLIT=train python src/preprocess_InterHand26M_mp.py
SPLIT=val python src/preprocess_InterHand26M_mp.py
SPLIT=test python src/preprocess_InterHand26M_mp.py
```

**DexYCB:**
```bash
# Setup variants: s0 (seen subjects), s1 (unseen subjects), s2 (unseen cameras), s3 (unseen objects)
SETUP=s1 SPLIT=train python src/preprocess_DexYCB_mp.py
SETUP=s1 SPLIT=val python src/preprocess_DexYCB_mp.py
SETUP=s1 SPLIT=test python src/preprocess_DexYCB_mp.py
```

**HO3D:**
```bash
SPLIT=train python src/preprocess_HO3D_mp.py
SPLIT=evaluation python src/preprocess_HO3D_mp.py
```

**HOT3D:**
```bash
SPLIT=train python src/preprocess_HOT3D.py
SPLIT=test python src/preprocess_HOT3D.py
```

### Data Reorganization

Merge scattered shard files into organized tar archives:

```bash
# Example from run.sh
SRC=ih26m_train_wds_output/ih26m_train-worker*.tar \
DST=/path/to/output/train/%06d.tar \
    python src/data_reorganizer.py
```

### Testing

```bash
# Test WebDataset loading
python test/test_wds.py

# Test dataloader with preprocessing
python test/test_wds_dl.py
```

### Running Dataloader (Debug/Verification)

```bash
# Direct run of dataloader module for verification
python src/dataloader.py
```

## Code Style Guidelines

### Naming Conventions
- **Functions**: `snake_case` (e.g., `process_single_annot`, `get_bbox`)
- **Classes**: `PascalCase` (e.g., `PixelLevelAugmentation`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `IH26M_ROOT`, `NUMPY_KEYS`)
- **Private**: Single underscore prefix (e.g., `_JOINT_REORDER_CACHE`)

### Code Organization
- Use Chinese comments for internal logic explanations
- Use English for docstrings and function descriptions
- Type hints are encouraged (e.g., `def func(x: torch.Tensor) -> np.ndarray`)

### Tensor/Array Conventions
- Image format: `(B, T, C, H, W)` for batched videos, `(C, H, W)` for single images
- Joint coordinates: `(B, T, J, 2)` for 2D, `(B, T, J, 3)` for 3D
- Joints are reordered to `TARGET_JOINTS_ORDER` (21 joints, see `src/utils.py`)
- Camera coordinates in millimeters (mm)
- Timestamps in milliseconds (ms)

### Key Data Structures

**WebDataset Sample Format (stored in .tar):**
```python
{
    "__key__": "unique_sample_id",
    "imgs_path.json": json.dumps(["path/to/frame1.jpg", ...]),  # List of paths
    "img_bytes.pickle": pickle.dumps([bytes, bytes, ...]),      # List of WebP bytes
    "handedness.json": json.dumps("right" or "left"),
    "additional_desc.json": json.dumps([{}, ...]),              # Per-frame metadata
    "hand_bbox.npy": np.array([T, 4]),          # [x1, y1, x2, y2]
    "joint_img.npy": np.array([T, 21, 2]),      # 2D joint positions
    "joint_hand_bbox.npy": np.array([T, 21, 2]), # Joints relative to bbox
    "joint_cam.npy": np.array([T, 21, 3]),      # 3D camera coordinates
    "joint_rel.npy": np.array([T, 21, 3]),      # Relative to wrist (root)
    "joint_valid.npy": np.array([T, 21]),       # Validity mask
    "mano_pose.npy": np.array([T, 48]),         # MANO pose params
    "mano_shape.npy": np.array([T, 10]),        # MANO shape params
    "mano_valid.npy": np.array([T]),            # MANO validity flag
    "timestamp.npy": np.array([T]),             # Timestamps in ms
    "focal.npy": np.array([T, 2]),              # Focal lengths [fx, fy]
    "princpt.npy": np.array([T, 2]),            # Principal point [cx, cy]
}
```

## Testing Instructions

### Test Files Overview

1. **`test/test_wds.py`**: Basic WebDataset loading verification
   - Checks if tar files can be decoded correctly
   - Validates shape consistency of tensors
   - Verifies image decoding from WebP bytes

2. **`test/test_wds_dl.py`**: Advanced dataloader testing
   - Tests sequence-to-frame flattening
   - Verifies custom collation
   - Tests batch loading with preprocessing

### Running Tests

```bash
# Set up data path in test file first, then:
python test/test_wds.py

# For dataloader test, update URLS variable in test_wds_dl.py, then:
python test/test_wds_dl.py
```

### Manual Verification

The `dataloader.py` module includes verification functions:
- `verify_origin_data()`: Visualizes original data before preprocessing
- `verify_batch()`: Visualizes data after preprocessing/augmentation

These functions save images to disk for manual inspection.

## Configuration Reference

### Environment Variables

| Variable | Script | Description |
|----------|--------|-------------|
| `SPLIT` | All preprocessors | Dataset split: `train`/`val`/`test`/`evaluation` |
| `SETUP` | DexYCB | Setup variant: `s0`/`s1`/`s2`/`s3` |
| `SRC` | data_reorganizer | Source pattern for shards |
| `DST` | data_reorganizer | Destination pattern for merged shards |

### Dataset Paths (Hardcoded)

Update these in the respective preprocessing scripts:
- InterHand2.6M: `/data_1/datasets_temp/InterHand2.6M_5fps_batch1/`
- DexYCB: `/data_1/datasets_temp/dexycb`
- HO3D: `/data_1/datasets_temp/HO3D_v3`
- HOT3D: `/mnt/qnap/data/datasets/hot3d/`

### Shard Writer Configuration

All preprocessors use consistent shard settings:
- `MAX_COUNT = 100000`: Maximum samples per shard
- `MAX_SIZE = 3GB`: Maximum shard file size
- `NUM_WORKERS = 8-30`: Parallel processing workers

## Important Notes for Developers

### MANO Model Setup

The MANO model files require preprocessing before use (see `models/mano/conduct.md`):

1. Convert all members in `MANO_LEFT/RIGHT.pkl` to `np.array` type
2. Extract `hand_components` (45×45) into `mano_lr_pca.npz` with keys `left` and `right`
3. Extract `hands_mean` from `MANO_RIGHT.pkl` to `mano_right_mean.npy`

These files are excluded from git (see `.gitignore`).

### Multi-Processing Considerations

- All `*_mp.py` scripts use multiprocessing.Pool for parallel processing
- OpenCV threading is disabled (`cv2.setNumThreads(0)`) to prevent resource contention
- Global data is loaded in main process and shared via copy-on-write (Linux)
- Each worker writes to separate shard files to avoid conflicts

### Joint Coordinate Systems

The project standardizes on a unified joint ordering (`TARGET_JOINTS_ORDER`):
```
Wrist -> Thumb_1-4 -> Index_1-4 -> Middle_1-4 -> Ring_1-4 -> Pinky_1-4
(21 joints total)
```

Source datasets have different orderings (see `src/utils.py` for mappings):
- `IH26M_RJOINTS_ORDER`: InterHand2.6M order
- `HO3D_JOINTS_ORDER`: HO3D order
- `HOT3D_JOINTS_ORDER`: HOT3D order (lacks Thumb_1, interpolated)

Use `reorder_joints()` function to convert between orderings.

### HOT3D Dataset Special Handling

HOT3D uses VRS format and requires:
- projectaria_tools for data loading
- Special handling for Thumb_1 (interpolated from wrist and Thumb_2)
- Different coordinate system conventions
- See `src/preprocess_HOT3D.py` for implementation details

## Security Considerations

- The project reads/writes files from hardcoded absolute paths
- Ensure these paths exist and have proper permissions
- Output directories are created automatically with `exist_ok=True`
- Be cautious with `run.sh` as it contains example paths that may not exist on your system
- MANO model files are binary pickles - only load trusted files

## Common Issues

1. **CUDA out of memory**: Reduce batch size in `dataloader.py`
2. **OpenCV errors**: Ensure image paths are correct and images exist
3. **MANO loading errors**: Check that model files are properly preprocessed
4. **Shard file conflicts**: Ensure output directories are clean before re-running
5. **Memory issues during preprocessing**: Reduce `NUM_WORKERS` if system RAM is limited

# 重要提示
1. 每一次执行修改之前必须经过与我先讨论修改方案，不要擅自执行
2. 每次回答完成之后加一句GG
