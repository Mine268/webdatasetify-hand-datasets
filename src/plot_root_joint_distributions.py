"""
Plot root joint XYZ distributions from precomputed statistic outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", osp.abspath("temp/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MANIFEST_PATH = "manifest.json"
DEFAULT_OUTPUT_DIR = "statistic/plots"
AXIS_KEYS = ("x_mm", "y_mm", "z_mm")
AXIS_LABELS = ("X (mm)", "Y (mm)", "Z (mm)")
AXIS_COLORS = ("#d1495b", "#2e86ab", "#2a9d8f")
HIST_BINS = 256
TRIM_PERCENTILES = (0.5, 99.5)
TRAIN_SAMPLING_WEIGHTS = {
    "AssemblyHands": 0.10,
    "COCO-WholeBody": 0.10,
    "DexYCB": 0.10,
    "FreiHAND": 0.15,
    "HO3D_v3": 0.10,
    "HOT3D": 0.10,
    "InterHand2.6M": 0.20,
    "MTC": 0.15,
    "RHD": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot root joint XYZ distributions.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _slugify(rel_split_dir: str) -> str:
    return rel_split_dir.replace("/", "__")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_root_positions(split_summary: Dict[str, Any]) -> np.ndarray:
    path = split_summary["root_positions_path"]
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32)


def _trim_range(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, TRIM_PERCENTILES)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(values.min())
        hi = float(values.max())
    if lo == hi:
        pad = 1.0 if lo == 0.0 else abs(lo) * 0.05
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def _full_range(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        pad = 1.0 if lo == 0.0 else abs(lo) * 0.05
        lo -= pad
        hi += pad
    return lo, hi


def _plot_hist(
    ax: plt.Axes,
    values: np.ndarray,
    color: str,
    title: str,
    xlim: Tuple[float, float],
    density: bool = True,
    annotate: bool = True,
) -> None:
    if values.size == 0:
        ax.text(0.5, 0.5, "No valid 3D root", ha="center", va="center", fontsize=10)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)
        return

    ax.hist(
        values,
        bins=HIST_BINS,
        range=xlim,
        density=density,
        color=color,
        alpha=0.8,
        edgecolor="none",
    )
    p1, p50, p99 = np.percentile(values, [1, 50, 99])
    ax.axvline(float(p50), color="black", linestyle="-", linewidth=1.2, alpha=0.9)
    ax.axvline(float(p1), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(float(p99), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    if annotate:
        ax.text(
            0.02,
            0.96,
            f"n={values.size:,}\np50={p50:.1f}\np1={p1:.1f}\np99={p99:.1f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )


def _save_split_plot(split_summary: Dict[str, Any], output_dir: str) -> str:
    rel_split_dir = split_summary["rel_split_dir"]
    root_positions = _load_root_positions(split_summary)
    out_dir = osp.join(output_dir, "splits")
    _ensure_dir(out_dir)
    out_path = osp.join(out_dir, f"{_slugify(rel_split_dir)}__xyz_distribution.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for col, (axis_key, axis_label, color) in enumerate(
        zip(AXIS_KEYS, AXIS_LABELS, AXIS_COLORS)
    ):
        values = root_positions[:, col] if root_positions.size > 0 else np.array([], dtype=np.float32)
        full_xlim = _full_range(values)
        trim_xlim = _trim_range(values)
        _plot_hist(
            axes[0, col],
            values,
            color=color,
            title=f"{axis_label} full range",
            xlim=full_xlim,
        )
        _plot_hist(
            axes[1, col],
            values,
            color=color,
            title=f"{axis_label} trimmed {TRIM_PERCENTILES[0]}-{TRIM_PERCENTILES[1]}%",
            xlim=trim_xlim,
        )
        axes[1, col].set_xlabel(axis_label)
        axes[0, col].set_ylabel("Density")
        axes[1, col].set_ylabel("Density")

    fig.suptitle(
        f"{rel_split_dir} | valid_root_frames={split_summary['num_valid_root_frames']:,}",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_dataset_overlay_plot(dataset_summary: Dict[str, Any], output_dir: str) -> str:
    dataset_name = dataset_summary["dataset_name"]
    splits = dataset_summary["splits"]
    out_dir = osp.join(output_dir, "datasets")
    _ensure_dir(out_dir)
    out_path = osp.join(out_dir, f"{dataset_name}__split_overlay.png")

    split_arrays: List[Tuple[Dict[str, Any], np.ndarray]] = [
        (split_summary, _load_root_positions(split_summary)) for split_summary in splits
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for col, (axis_label, color) in enumerate(zip(AXIS_LABELS, AXIS_COLORS)):
        axis_values = [
            arr[:, col]
            for _split, arr in split_arrays
            if arr.size > 0
        ]
        if axis_values:
            combined = np.concatenate(axis_values, axis=0)
            xlim = _trim_range(combined)
        else:
            xlim = (-1.0, 1.0)

        ax = axes[col]
        has_data = False
        for idx, (split_summary, arr) in enumerate(split_arrays):
            if arr.size == 0:
                continue
            has_data = True
            values = arr[:, col]
            hist, bin_edges = np.histogram(values, bins=HIST_BINS, range=xlim, density=True)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(
                centers,
                hist,
                linewidth=1.6,
                label=f"{split_summary['split_name']} (n={values.size:,})",
            )
        if not has_data:
            ax.text(0.5, 0.5, "No valid 3D root", ha="center", va="center", fontsize=10)
        ax.set_title(f"{axis_label} trimmed overlay")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)
        if has_data:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{dataset_name} split comparison | valid_root_frames={dataset_summary['num_valid_root_frames']:,}",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_all_datasets_overview(dataset_summaries: Sequence[Dict[str, Any]], output_dir: str) -> str:
    out_path = osp.join(output_dir, "all_datasets_overview.png")
    rows = len(dataset_summaries)
    fig, axes = plt.subplots(rows, 3, figsize=(15, max(3.0 * rows, 8.0)), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])

    for row, dataset_summary in enumerate(dataset_summaries):
        dataset_name = dataset_summary["dataset_name"]
        split_arrays = [
            _load_root_positions(split_summary) for split_summary in dataset_summary["splits"]
        ]
        non_empty_arrays = [arr for arr in split_arrays if arr.size > 0]
        if non_empty_arrays:
            dataset_arr = np.concatenate(non_empty_arrays, axis=0)
        else:
            dataset_arr = np.zeros((0, 3), dtype=np.float32)

        for col, (axis_label, color) in enumerate(zip(AXIS_LABELS, AXIS_COLORS)):
            ax = axes[row, col]
            values = dataset_arr[:, col] if dataset_arr.size > 0 else np.array([], dtype=np.float32)
            xlim = _trim_range(values)
            _plot_hist(
                ax,
                values,
                color=color,
                title=f"{dataset_name} | {axis_label}",
                xlim=xlim,
                annotate=False,
            )
            if row == rows - 1:
                ax.set_xlabel(axis_label)
            ax.set_ylabel("Density")
            if values.size > 0:
                p50 = np.percentile(values, 50)
                ax.text(
                    0.98,
                    0.96,
                    f"n={values.size:,}\np50={p50:.1f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
                )

    fig.suptitle("Root joint XYZ distributions by dataset (trimmed view)", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_all_datasets_overlay(dataset_summaries: Sequence[Dict[str, Any]], output_dir: str) -> str:
    out_path = osp.join(output_dir, "all_datasets_overlay.png")
    dataset_arrays: List[Tuple[str, np.ndarray]] = []
    for dataset_summary in dataset_summaries:
        split_arrays = [
            _load_root_positions(split_summary) for split_summary in dataset_summary["splits"]
        ]
        non_empty_arrays = [arr for arr in split_arrays if arr.size > 0]
        if not non_empty_arrays:
            continue
        dataset_arrays.append(
            (dataset_summary["dataset_name"], np.concatenate(non_empty_arrays, axis=0))
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for col, axis_label in enumerate(AXIS_LABELS):
        axis_values = [arr[:, col] for _name, arr in dataset_arrays]
        xlim = _trim_range(np.concatenate(axis_values, axis=0)) if axis_values else (-1.0, 1.0)
        ax = axes[col]
        for dataset_name, arr in dataset_arrays:
            values = arr[:, col]
            hist, bin_edges = np.histogram(values, bins=HIST_BINS, range=xlim, density=True)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(centers, hist, linewidth=1.6, label=f"{dataset_name} (n={values.shape[0]:,})")
        ax.set_title(f"{axis_label} overlay")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("All datasets root joint XYZ overlay (trimmed view)", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _collect_dataset_arrays(
    dataset_summaries: Sequence[Dict[str, Any]]
) -> List[Tuple[str, np.ndarray]]:
    dataset_arrays: List[Tuple[str, np.ndarray]] = []
    for dataset_summary in dataset_summaries:
        split_arrays = [
            _load_root_positions(split_summary) for split_summary in dataset_summary["splits"]
        ]
        non_empty_arrays = [arr for arr in split_arrays if arr.size > 0]
        if not non_empty_arrays:
            continue
        dataset_arrays.append(
            (dataset_summary["dataset_name"], np.concatenate(non_empty_arrays, axis=0))
        )
    return dataset_arrays


def _save_all_datasets_overlay_full_range(
    dataset_summaries: Sequence[Dict[str, Any]], output_dir: str
) -> str:
    out_path = osp.join(output_dir, "all_datasets_overlay_full_range.png")
    dataset_arrays = _collect_dataset_arrays(dataset_summaries)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for col, axis_label in enumerate(AXIS_LABELS):
        axis_values = [arr[:, col] for _name, arr in dataset_arrays]
        xlim = _full_range(np.concatenate(axis_values, axis=0)) if axis_values else (-1.0, 1.0)
        ax = axes[col]
        for dataset_name, arr in dataset_arrays:
            values = arr[:, col]
            hist, bin_edges = np.histogram(values, bins=HIST_BINS, range=xlim, density=True)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(centers, hist, linewidth=1.6, label=f"{dataset_name} (n={values.shape[0]:,})")
        ax.set_title(f"{axis_label} full-range overlay")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("All datasets root joint XYZ overlay (full range)", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_all_datasets_combined_full_range(
    dataset_summaries: Sequence[Dict[str, Any]], output_dir: str
) -> str:
    out_path = osp.join(output_dir, "all_datasets_combined_full_range.png")
    dataset_arrays = _collect_dataset_arrays(dataset_summaries)
    combined = (
        np.concatenate([arr for _name, arr in dataset_arrays], axis=0)
        if dataset_arrays
        else np.zeros((0, 3), dtype=np.float32)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for col, (axis_label, color) in enumerate(zip(AXIS_LABELS, AXIS_COLORS)):
        ax = axes[col]
        values = combined[:, col] if combined.size > 0 else np.array([], dtype=np.float32)
        xlim = _full_range(values)
        _plot_hist(
            ax,
            values,
            color=color,
            title=f"{axis_label} combined full range",
            xlim=xlim,
            annotate=True,
        )
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Density")

    fig.suptitle(
        f"All datasets combined root joint XYZ distributions | n={combined.shape[0]:,}",
        fontsize=16,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _weighted_quantile_from_hist(
    hist_mass: np.ndarray, bin_edges: np.ndarray, q: float
) -> float:
    if hist_mass.sum() <= 0:
        return 0.0
    target = q * hist_mass.sum()
    cdf = np.cumsum(hist_mass)
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, hist_mass.shape[0] - 1))
    return float(0.5 * (bin_edges[idx] + bin_edges[idx + 1]))


def _build_train_split_map(
    dataset_summaries: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for dataset_summary in dataset_summaries:
        for split_summary in dataset_summary["splits"]:
            if str(split_summary["rel_split_dir"]).endswith("/train"):
                result[dataset_summary["dataset_name"]] = split_summary
                break
    return result


def _save_train_weighted_combined_full_range(
    dataset_summaries: Sequence[Dict[str, Any]], output_dir: str
) -> Tuple[str, str]:
    out_path = osp.join(output_dir, "train_weighted_combined_full_range.png")
    weights_path = osp.join(output_dir, "train_weighted_sampling_weights.json")
    train_split_map = _build_train_split_map(dataset_summaries)

    effective_weights: Dict[str, float] = {}
    skipped_weights: Dict[str, Dict[str, Any]] = {}
    train_arrays: List[Tuple[str, float, np.ndarray]] = []
    for dataset_name, raw_weight in TRAIN_SAMPLING_WEIGHTS.items():
        split_summary = train_split_map.get(dataset_name)
        if split_summary is None:
            skipped_weights[dataset_name] = {
                "weight": raw_weight,
                "reason": "missing_train_split",
            }
            continue
        arr = _load_root_positions(split_summary)
        if arr.size == 0:
            skipped_weights[dataset_name] = {
                "weight": raw_weight,
                "reason": "no_valid_3d_root",
                "rel_split_dir": split_summary["rel_split_dir"],
            }
            continue
        effective_weights[dataset_name] = raw_weight
        train_arrays.append((dataset_name, raw_weight, arr))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for col, (axis_label, color) in enumerate(zip(AXIS_LABELS, AXIS_COLORS)):
        ax = axes[col]
        axis_values = [arr[:, col] for _name, _weight, arr in train_arrays]
        xlim = _full_range(np.concatenate(axis_values, axis=0)) if axis_values else (-1.0, 1.0)
        bin_edges = np.linspace(xlim[0], xlim[1], HIST_BINS + 1, dtype=np.float64)
        hist_mass = np.zeros((HIST_BINS,), dtype=np.float64)
        total_mass = 0.0

        for dataset_name, dataset_weight, arr in train_arrays:
            values = arr[:, col]
            per_sample_weight = dataset_weight / max(values.shape[0], 1)
            weighted_counts, _ = np.histogram(
                values,
                bins=bin_edges,
                weights=np.full(values.shape[0], per_sample_weight, dtype=np.float64),
            )
            hist_mass += weighted_counts
            total_mass += float(weighted_counts.sum())

        bin_widths = np.diff(bin_edges)
        density = hist_mass / np.maximum(bin_widths, 1e-12)
        if total_mass > 0:
            density = density / total_mass
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.fill_between(centers, density, step="mid", color=color, alpha=0.75)
        ax.plot(centers, density, color=color, linewidth=1.2)

        if total_mass > 0:
            p1 = _weighted_quantile_from_hist(hist_mass, bin_edges, 0.01)
            p50 = _weighted_quantile_from_hist(hist_mass, bin_edges, 0.50)
            p99 = _weighted_quantile_from_hist(hist_mass, bin_edges, 0.99)
            ax.axvline(p50, color="black", linestyle="-", linewidth=1.2, alpha=0.9)
            ax.axvline(p1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(p99, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(
                0.02,
                0.96,
                f"weighted mass={total_mass:.2f}\np50={p50:.1f}\np1={p1:.1f}\np99={p99:.1f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
            )

        ax.set_title(f"{axis_label} weighted combined full range")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)

    used_weight_lines = "\n".join(
        f"{name}: {weight:.2f}" for name, weight in sorted(effective_weights.items())
    )
    if skipped_weights:
        used_weight_lines += "\nSkipped:\n" + "\n".join(
            f"{name}: {info['reason']}" for name, info in sorted(skipped_weights.items())
        )
    fig.suptitle(
        "Train-only weighted combined root joint XYZ distributions (full range)",
        fontsize=16,
    )
    fig.text(
        0.995,
        0.5,
        used_weight_lines,
        va="center",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    weight_payload = {
        "raw_train_sampling_weights": TRAIN_SAMPLING_WEIGHTS,
        "effective_3d_weights": effective_weights,
        "skipped_weights": skipped_weights,
    }
    with open(weights_path, "w", encoding="utf-8") as fp:
        json.dump(weight_payload, fp, ensure_ascii=False, indent=2)
    return out_path, weights_path


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.manifest_path)
    output_dir = osp.abspath(args.output_dir)
    _ensure_dir(output_dir)

    split_plot_paths: List[str] = []
    dataset_overlay_paths: List[str] = []
    for dataset_summary in manifest["datasets"]:
        for split_summary in dataset_summary["splits"]:
            split_plot_paths.append(_save_split_plot(split_summary, output_dir))
        dataset_overlay_paths.append(_save_dataset_overlay_plot(dataset_summary, output_dir))

    overview_path = _save_all_datasets_overview(manifest["datasets"], output_dir)
    overlay_path = _save_all_datasets_overlay(manifest["datasets"], output_dir)
    overlay_full_range_path = _save_all_datasets_overlay_full_range(
        manifest["datasets"], output_dir
    )
    combined_full_range_path = _save_all_datasets_combined_full_range(
        manifest["datasets"], output_dir
    )
    train_weighted_combined_full_range_path, train_weighted_weights_path = (
        _save_train_weighted_combined_full_range(manifest["datasets"], output_dir)
    )

    report = {
        "manifest_path": osp.abspath(args.manifest_path),
        "output_dir": output_dir,
        "overview_path": overview_path,
        "overlay_path": overlay_path,
        "overlay_full_range_path": overlay_full_range_path,
        "combined_full_range_path": combined_full_range_path,
        "train_weighted_combined_full_range_path": train_weighted_combined_full_range_path,
        "train_weighted_weights_path": train_weighted_weights_path,
        "dataset_overlay_paths": dataset_overlay_paths,
        "split_plot_paths": split_plot_paths,
    }
    report_path = osp.join(output_dir, "plot_manifest.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print(f"Saved plots to {output_dir}")
    print(f"Saved plot manifest to {report_path}")


if __name__ == "__main__":
    main()
