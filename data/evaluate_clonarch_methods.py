#!/usr/bin/env python3
"""Evaluate ClonArch interpolation and boundary-extraction variants.

Expected prevalence layout is N x C, where N = rows * cols and C is the
number of clones. Optional mask arrays may be N x C, rows x cols (one clone),
rows x cols x C, or C x rows x cols.

Boundary distance is the average symmetric surface distance (ASSD). This is
different from prevalence MAE; the formula originally supplied for boundary
distance was a duplicate of prevalence MAE.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import ndimage


def read_numeric_matrix(path: str | Path, drop_first_column: bool = False) -> np.ndarray:
    """Read a headerless numeric CSV as a 2-D float array."""
    frame = pd.read_csv(path, header=None)
    if drop_first_column:
        frame = frame.iloc[:, 1:]
    matrix = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix in {path}, found {matrix.shape}")
    return matrix


def load_array(path: str | Path, drop_first_column: bool = False) -> np.ndarray:
    """Load .npy, single-array .npz, or headerless numeric .csv/.txt."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(f"{path} must contain exactly one array")
        return archive[archive.files[0]]
    return read_numeric_matrix(path, drop_first_column=drop_first_column)


def as_prevalence_matrix(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Normalize prevalence data to shape (rows * cols, clones)."""
    values = np.asarray(values, dtype=float)
    n = rows * cols
    if values.ndim == 1 and values.size == n:
        return values.reshape(n, 1)
    if values.ndim == 2 and values.shape[0] == n:
        return values
    if values.ndim == 2 and values.shape[1] == n:
        return values.T
    if values.ndim == 3 and values.shape[:2] == (rows, cols):
        return values.reshape(n, values.shape[2])
    if values.ndim == 3 and values.shape[1:] == (rows, cols):
        return np.moveaxis(values, 0, -1).reshape(n, values.shape[0])
    raise ValueError(
        f"Cannot convert shape {values.shape} to ({n}, clones) for a "
        f"{rows} x {cols} grid"
    )


def as_mask_stack(values: np.ndarray, rows: int, cols: int, clones: int) -> np.ndarray:
    """Normalize binary masks to shape (rows, cols, clones)."""
    values = np.asarray(values)
    n = rows * cols
    if values.ndim == 2 and values.shape == (n, clones):
        values = values.reshape(rows, cols, clones)
    elif values.ndim == 2 and values.shape == (rows, cols) and clones == 1:
        values = values[:, :, None]
    elif values.ndim == 3 and values.shape == (clones, rows, cols):
        values = np.moveaxis(values, 0, -1)
    elif values.ndim != 3 or values.shape != (rows, cols, clones):
        raise ValueError(
            f"Mask shape {values.shape} is incompatible with "
            f"({rows}, {cols}, {clones})"
        )
    return values.astype(bool)


def make_masks(prevalence: np.ndarray, rows: int, cols: int, threshold: float) -> np.ndarray:
    """Threshold an N x C prevalence matrix into rows x cols x C masks."""
    return (prevalence >= threshold).reshape(rows, cols, prevalence.shape[1])


def binary_boundary(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Return the inner one-pixel boundary of a binary region."""
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def boundary_distances(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing: tuple[float, float],
    connectivity: int,
) -> tuple[float, float]:
    """Return (ASSD, Hausdorff) between two binary boundaries.

    If exactly one mask is empty, both distances receive the physical diagonal
    of the grid as a finite worst-case penalty. If both are empty, both are 0.
    """
    truth_any = bool(truth.any())
    prediction_any = bool(prediction.any())
    if not truth_any and not prediction_any:
        return 0.0, 0.0
    if truth_any != prediction_any:
        diagonal = math.hypot(
            (truth.shape[0] - 1) * spacing[0],
            (truth.shape[1] - 1) * spacing[1],
        )
        return diagonal, diagonal

    truth_boundary = binary_boundary(truth, connectivity)
    prediction_boundary = binary_boundary(prediction, connectivity)

    distance_to_truth = ndimage.distance_transform_edt(
        ~truth_boundary, sampling=spacing
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        ~prediction_boundary, sampling=spacing
    )

    truth_to_prediction = distance_to_prediction[truth_boundary]
    prediction_to_truth = distance_to_truth[prediction_boundary]
    assd = 0.5 * (
        float(truth_to_prediction.mean())
        + float(prediction_to_truth.mean())
    )
    hausdorff = max(
        float(truth_to_prediction.max()),
        float(prediction_to_truth.max()),
    )
    return assd, hausdorff


def component_count(mask: np.ndarray, connectivity: int) -> int:
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    _, count = ndimage.label(mask, structure=structure)
    return int(count)


def evaluate_clone(
    truth_u: np.ndarray,
    prediction_u: np.ndarray,
    truth_mask: np.ndarray,
    prediction_mask: np.ndarray,
    valid_points: np.ndarray,
    spacing: tuple[float, float],
    connectivity: int,
) -> dict[str, float | int]:
    truth_mask = truth_mask & valid_points
    prediction_mask = prediction_mask & valid_points

    intersection = int(np.count_nonzero(truth_mask & prediction_mask))
    union = int(np.count_nonzero(truth_mask | prediction_mask))
    truth_area = int(np.count_nonzero(truth_mask))
    prediction_area = int(np.count_nonzero(prediction_mask))

    iou = 1.0 if union == 0 else intersection / union
    denominator = truth_area + prediction_area
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator

    prevalence_valid = valid_points.reshape(-1)
    prevalence_mae = float(
        np.mean(
            np.abs(
                truth_u[prevalence_valid]
                - prediction_u[prevalence_valid]
            )
        )
    )

    assd, hausdorff = boundary_distances(
        truth_mask, prediction_mask, spacing, connectivity
    )

    if truth_area == 0:
        area_error = 0.0 if prediction_area == 0 else np.nan
    else:
        area_error = abs(prediction_area - truth_area) / truth_area

    truth_components = component_count(truth_mask, connectivity)
    prediction_components = component_count(prediction_mask, connectivity)

    return {
        "iou": iou,
        "dice": dice,
        "boundary_assd": assd,
        "prevalence_mae": prevalence_mae,
        "hausdorff_distance": hausdorff,
        "clone_area_error": area_error,
        "truth_area": truth_area,
        "predicted_area": prediction_area,
        "truth_components": truth_components,
        "predicted_components": prediction_components,
        "component_error": abs(prediction_components - truth_components),
    }


def evaluate_method(
    truth_prevalence: np.ndarray,
    predicted_prevalence: np.ndarray,
    rows: int,
    cols: int,
    method: str,
    prevalence_threshold: float = 0.1,
    truth_masks: np.ndarray | None = None,
    predicted_masks: np.ndarray | None = None,
    tumor_mask: np.ndarray | None = None,
    spacing: tuple[float, float] = (1.0, 1.0),
    connectivity: int = 8,
    clone_names: Iterable[str] | None = None,
    runtime_seconds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one interpolation + boundary-extraction method."""
    truth = as_prevalence_matrix(truth_prevalence, rows, cols)
    prediction = as_prevalence_matrix(predicted_prevalence, rows, cols)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Truth shape {truth.shape} does not match prediction {prediction.shape}"
        )
    clones = truth.shape[1]

    if truth_masks is None:
        truth_mask_stack = make_masks(truth, rows, cols, prevalence_threshold)
    else:
        truth_mask_stack = as_mask_stack(truth_masks, rows, cols, clones)

    if predicted_masks is None:
        prediction_mask_stack = make_masks(
            prediction, rows, cols, prevalence_threshold
        )
    else:
        prediction_mask_stack = as_mask_stack(predicted_masks, rows, cols, clones)

    if tumor_mask is None:
        valid_points = np.ones((rows, cols), dtype=bool)
    else:
        tumor_mask = np.asarray(tumor_mask)
        if tumor_mask.size != rows * cols:
            raise ValueError(
                f"Tumor mask has {tumor_mask.size} values; expected {rows * cols}"
            )
        valid_points = tumor_mask.reshape(rows, cols).astype(bool)

    if clone_names is None:
        names = [f"clone_{index}" for index in range(clones)]
    else:
        names = list(clone_names)
        if len(names) != clones:
            raise ValueError(f"Received {len(names)} clone names for {clones} clones")

    records = []
    for clone_index, clone_name in enumerate(names):
        record = evaluate_clone(
            truth[:, clone_index],
            prediction[:, clone_index],
            truth_mask_stack[:, :, clone_index],
            prediction_mask_stack[:, :, clone_index],
            valid_points,
            spacing,
            connectivity,
        )
        record.update(
            {
                "method": method,
                "clone_index": clone_index,
                "clone_name": clone_name,
                "prevalence_threshold": prevalence_threshold,
                "runtime_seconds": runtime_seconds,
            }
        )
        records.append(record)

    per_clone = pd.DataFrame.from_records(records)
    metric_columns = [
        "iou",
        "dice",
        "boundary_assd",
        "prevalence_mae",
        "hausdorff_distance",
        "clone_area_error",
        "component_error",
    ]
    summary_record: dict[str, float | str | int | None] = {
        "method": method,
        "clone_count": clones,
        "prevalence_threshold": prevalence_threshold,
        "runtime_seconds": runtime_seconds,
    }
    for metric in metric_columns:
        values = per_clone[metric].to_numpy(dtype=float)
        finite = np.isfinite(values)
        summary_record[f"mean_{metric}"] = (
            float(values[finite].mean()) if finite.any() else np.nan
        )
        summary_record[f"median_{metric}"] = (
            float(np.median(values[finite])) if finite.any() else np.nan
        )
        summary_record[f"valid_clones_{metric}"] = int(finite.sum())

    return per_clone, pd.DataFrame([summary_record])


def time_method(function: Callable, *args, **kwargs):
    """Run an interpolation/boundary function and return (result, seconds)."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def parse_clone_names(path: str | None) -> list[str] | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        names = json.loads(text)
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(names, list):
        raise ValueError("Clone-name file must contain a list")
    return [str(name) for name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Ground-truth prevalence CSV")
    parser.add_argument("--prediction", required=True, help="Predicted prevalence CSV")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--method", required=True, help="Method/version label")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--truth-masks", help="Optional ground-truth .npy/.npz/CSV masks")
    parser.add_argument("--predicted-masks", help="Optional extracted .npy/.npz/CSV masks")
    parser.add_argument("--tumor-mask", help="Optional .npy/.npz/CSV valid-region mask")
    parser.add_argument("--row-spacing", type=float, default=1.0)
    parser.add_argument("--col-spacing", type=float, default=1.0)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--clone-names", help="Optional JSON list or one-name-per-line file")
    parser.add_argument("--runtime-seconds", type=float)
    parser.add_argument("--truth-drop-first-column", action="store_true")
    parser.add_argument("--prediction-drop-first-column", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    truth = load_array(args.truth, args.truth_drop_first_column)
    prediction = load_array(args.prediction, args.prediction_drop_first_column)
    truth_masks = load_array(args.truth_masks) if args.truth_masks else None
    predicted_masks = load_array(args.predicted_masks) if args.predicted_masks else None
    tumor_mask = load_array(args.tumor_mask) if args.tumor_mask else None

    per_clone, summary = evaluate_method(
        truth,
        prediction,
        rows=args.rows,
        cols=args.cols,
        method=args.method,
        prevalence_threshold=args.threshold,
        truth_masks=truth_masks,
        predicted_masks=predicted_masks,
        tumor_mask=tumor_mask,
        spacing=(args.row_spacing, args.col_spacing),
        connectivity=args.connectivity,
        clone_names=parse_clone_names(args.clone_names),
        runtime_seconds=args.runtime_seconds,
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    per_clone_path = prefix.with_name(prefix.name + "_per_clone.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    per_clone.to_csv(per_clone_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {per_clone_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
