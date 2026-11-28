"""
metrics.py

Compare interpolated clone boundaries against the ground truth using

  * IoU  (Jaccard index) of the regions  {field >= cut}
  * BDE  (boundary displacement error) between the extracted contours

BDE (Freixenet et al., 2002 style, symmetric): contours are densified to points
spaced `spacing` grid cells apart; for every point on one boundary we take the
distance to the nearest point on the other boundary, average that, and average
the two directions:

    BDE = 0.5 * ( mean_{a in A} d(a, B) + mean_{b in B} d(b, A) )

Units are grid cells (one cell = 1.0). Also reported: both directions
separately and the Hausdorff distance (worst-case displacement).

Inputs come from interpolate.py, which saves `maps/results_<prop>.npz`
(fields + the marching-squares contours it computed). No node/sklearn needed here.

Usage
-----
    python metrics.py maps/results_0.5.npz
    python metrics.py maps/results_0.5.npz --unobserved-only --csv metrics.csv

From Python
-----------
    import metrics
    df = metrics.evaluate_results("maps/results_0.5.npz")
    row = metrics.evaluate_clone(gt, est, 0.1, 0.1, gt_contours, est_contours)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- region metric

def iou(gt_region, est_region, eval_mask=None):
    """Jaccard index of two boolean regions, optionally restricted to eval_mask.
    Both regions empty -> 1.0 (they agree that nothing is there)."""
    a = np.asarray(gt_region, dtype=bool)
    b = np.asarray(est_region, dtype=bool)
    if eval_mask is not None:
        keep = np.asarray(eval_mask, dtype=bool)
        a, b = a[keep], b[keep]
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


# -------------------------------------------------------------- boundary metric

def _is_outer_frame(path, rows, cols, tol=1e-6):
    """True for the rectangle that hugs the (zero-margined) grid edge."""
    pts = np.asarray(path, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        return False
    xs, ys = pts[:, 0], pts[:, 1]
    spans_x = (xs.min() <= tol) and (xs.max() >= (cols - 1) - tol)
    spans_y = (ys.min() <= tol) and (ys.max() >= (rows - 1) - tol)
    on_border = np.all(
        (np.abs(xs) <= tol) | (np.abs(xs - (cols - 1)) <= tol)
        | (np.abs(ys) <= tol) | (np.abs(ys - (rows - 1)) <= tol)
    )
    return bool(spans_x and spans_y and on_border)


def _densify(path, spacing):
    """Resample a polyline so consecutive points are <= spacing apart."""
    pts = np.asarray(path, dtype=float)
    chunks = [pts[:1]]
    for p, q in zip(pts[:-1], pts[1:]):
        k = max(int(np.ceil(np.linalg.norm(q - p) / spacing)), 1)
        t = np.linspace(0.0, 1.0, k + 1)[1:, None]
        chunks.append(p + (q - p) * t)
    return np.vstack(chunks)


def boundary_points(contours, padded_shape, spacing=0.25):
    """Flatten a list of marching-squares paths into one (n, 2) point array."""
    rows, cols = padded_shape
    chunks = []
    for path in contours:
        pts = np.asarray(path, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        if _is_outer_frame(pts, rows, cols):
            continue
        chunks.append(_densify(pts, spacing))
    return np.vstack(chunks) if chunks else np.empty((0, 2))


def boundary_displacement_error(points_a, points_b):
    """Symmetric BDE (+ directional means and Hausdorff) between two point sets.
    Both empty -> 0. Exactly one empty -> NaN (distance is undefined)."""
    na, nb = len(points_a), len(points_b)
    if na == 0 and nb == 0:
        return dict(bde=0.0, bde_gt_to_est=0.0, bde_est_to_gt=0.0, hausdorff=0.0)
    if na == 0 or nb == 0:
        nan = float("nan")
        return dict(bde=nan, bde_gt_to_est=nan, bde_est_to_gt=nan, hausdorff=nan)
    d_ab = cKDTree(points_b).query(points_a)[0]
    d_ba = cKDTree(points_a).query(points_b)[0]
    return dict(
        bde=float(0.5 * (d_ab.mean() + d_ba.mean())),
        bde_gt_to_est=float(d_ab.mean()),
        bde_est_to_gt=float(d_ba.mean()),
        hausdorff=float(max(d_ab.max(), d_ba.max())),
    )


# ----------------------------------------------------------------- combined API

def evaluate_clone(gt, est, gt_cut, est_cut, gt_contours, est_contours,
                   eval_mask=None, spacing=0.25):
    """IoU + BDE for one clone layer.

    gt, est       : (N, N) fields; regions are {gt >= gt_cut}, {est >= est_cut}
                    (est_cut is the prevalence threshold for mean fields, or the
                    confidence level for a probability field)
    *_contours    : marching-squares paths computed on the zero-margined fields
    eval_mask     : optional (N, N) bool; IoU only counts these cells
    """
    gt = np.asarray(gt, dtype=float)
    padded = (gt.shape[0] + 2, gt.shape[1] + 2)
    out = {"iou": iou(gt >= gt_cut, np.asarray(est) >= est_cut, eval_mask)}
    out.update(boundary_displacement_error(
        boundary_points(gt_contours, padded, spacing),
        boundary_points(est_contours, padded, spacing),
    ))
    return out


def evaluate_results(path, unobserved_only=False, spacing=0.25):
    """Evaluate every clone and method stored in a results .npz from interpolate.py.
    Returns a DataFrame with one row per (clone, method)."""
    data = np.load(path, allow_pickle=False)
    T = float(data["threshold"])
    levels = [float(l) for l in data["levels"]]
    mask = data["mask"].astype(bool)
    eval_mask = ~mask if unobserved_only else None

    rows = []
    for k in range(1, int(data["n_clones"]) + 1):
        gt = data[f"gt_{k}"]
        contours = json.loads(str(data[f"contours_{k}"]))
        methods = [("bilinear", data[f"bilinear_{k}"], T, "bilinear"),
                   ("gp_mean", data[f"gp_mean_{k}"], T, "gp_mean")]
        methods += [(f"gp_prob>={l:g}", data[f"prob_{k}"], l, f"prob_{l:g}") for l in levels]
        for name, field, cut, ckey in methods:
            r = evaluate_clone(gt, field, T, cut, contours["gt"], contours[ckey],
                               eval_mask=eval_mask, spacing=spacing)
            rows.append({"clone": k, "method": name, **r})
    return pd.DataFrame(rows)


def summarize(df):
    """Print per-clone and per-method (mean over clones, NaNs skipped) tables."""
    cols = ["iou", "bde", "hausdorff"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print("\nPer clone:")
        print(df[["clone", "method"] + cols].to_string(index=False, na_rep="n/a"))
        print("\nMean over clones (BDE/Hausdorff skip clones where a boundary is missing):")
        print(df.groupby("method", sort=False)[cols].mean().to_string(na_rep="n/a"))
    print("\nBDE / Hausdorff are in grid cells. n/a = one of the two boundaries is empty.")


def main():
    ap = argparse.ArgumentParser(description="IoU + BDE for ClonArch interpolation results")
    ap.add_argument("results", help="results_<prop>.npz written by interpolate.py")
    ap.add_argument("--unobserved-only", action="store_true",
                    help="compute IoU only on cells that were NOT biopsied")
    ap.add_argument("--spacing", type=float, default=0.25,
                    help="contour resampling step in grid cells (default 0.25)")
    ap.add_argument("--csv", help="also write the full table to this CSV")
    args = ap.parse_args()

    df = evaluate_results(args.results, args.unobserved_only, args.spacing)
    summarize(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()