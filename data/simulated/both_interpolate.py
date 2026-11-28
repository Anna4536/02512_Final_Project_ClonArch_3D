import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulate_dataset import run_simulation
from scipy.interpolate import griddata
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import json
import subprocess
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JS_FILE = BASE_DIR / "../../dist/marchingsquares-isolines.js"
DEFAULT_JS_SCRIPT = BASE_DIR / "get_contours.js"
OUT_DIR = BASE_DIR / "maps"   # = data/simulated/maps when this file lives in data/simulated

'''
Interpolate.py

Bilinear / Gaussian-process interpolation + Marching Squares on simulated dataset.
Uses the same algorithms as data_prep.ipynb for interpolation and marching squares.

Three ways to get a boundary for one clone:
  1. bilinear mean field            -> contour at prevalence threshold t
  2. Gaussian-process mean field    -> contour at prevalence threshold t
  3. GP probability field P(U >= t) -> contour at a confidence level (e.g. 0.9)


'''






def random_cell_mask(n_rows, n_cols, sample_fraction=0.25, seed=43):
    rng = np.random.default_rng(seed)
    total = n_rows * n_cols
    k = int(round(total * sample_fraction))
    flat = np.zeros(total, dtype=bool)
    if k > 0:
        flat[rng.choice(total, size=k, replace=False)] = True
    return flat.reshape((n_rows, n_cols))


def get_clone_field(matrix_u, N, clone_idx=0, order="F"):
    # matrix_u has shape (N*N, num_clones); f_row = r + N * c 
    arr = np.asarray(matrix_u[:, clone_idx], dtype=float)
    return arr.reshape((N, N), order=order).T


def mean_square_err(N,ground_truth,new,mask):
    # Mean square error of non-masked values

    invert = ~mask

    mean = (ground_truth[invert]-new[invert])**2

    mse = mean.sum()/len(ground_truth[invert])
    print(mse)
    return mse

def marching_squares(
    mutant_matrix: np.ndarray,
    threshold: float = 0.01,
    js_file: str | Path = DEFAULT_JS_FILE,
    js_script: str | Path = DEFAULT_JS_SCRIPT,
    options: dict | None = None,
) -> list:
    arr = np.asarray(mutant_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mutant matrix, got shape {arr.shape}.")

    payload = {
        "matrix": arr.tolist(),
        "threshold": float(threshold),
        "options": options or {},
        "js_path": str(Path(js_file).resolve()),
    }

    result = subprocess.run(
        ["node", str(Path(js_script).resolve())],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)

def is_outer_frame(path, rows, cols, tol=1e-6):
    pts = np.asarray(path, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        return False
    xs, ys = pts[:, 0], pts[:, 1]
    spans_x = (xs.min() <= tol) and (xs.max() >= (cols - 1) - tol)
    spans_y = (ys.min() <= tol) and (ys.max() >= (rows - 1) - tol)
    all_on_border = np.all(
        (np.abs(xs) <= tol)
        | (np.abs(xs - (cols - 1)) <= tol)
        | (np.abs(ys) <= tol)
        | (np.abs(ys - (rows - 1)) <= tol)
    )
    return spans_x and spans_y and all_on_border

def plot_contours(contours, matrix, name, id, prop, title=None, label="Mutant proportion", out_dir=OUT_DIR):
    rows, cols = np.asarray(matrix).shape
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(matrix, origin="upper", vmin=0,vmax=1)
    fig.colorbar(im, ax=ax,label=label)

    for path in contours:
        if is_outer_frame(path, rows, cols):
            continue
        pts = np.asarray(path, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.8)

    plt.tight_layout()
    if title is None:
        title = f"Simulated Dataset:\n Mutant {id+1} Linear Interpolation, {int(prop*100)}% known"
    plt.title(title)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{id+1}_{prop}_{name}.png")
    plt.close()


def margins(one,N):

    one = np.insert(one, 0, values=np.zeros(N), axis=0)
    one = np.insert(one, len(one), values = np.zeros(N), axis=0)
    one = np.insert(one, 0, values = np.zeros(N+2), axis=1)
    one = np.insert(one, len(one[0]), values = np.zeros(N+2), axis=1)

    return one

def bilin_interp(gt,mask):
    # Perform bilinear interpolation on a clone layer.

    ONE = pd.DataFrame(gt) # convert to dataframe

    ONE[~mask] = np.nan
    
    row = ONE.interpolate(method='linear', limit_direction='both', axis=1)
    col = ONE.interpolate(method='linear', limit_direction='both', axis=0)

    avg = (row + col) / 2
    
    interpolated = np.array(avg)

    return interpolated





# ---------------------------------------------------------------------------
# Gaussian process interpolation + probability contours
# ---------------------------------------------------------------------------

def gp_interpolate(gt, mask, random_state=42, n_restarts=2):
    """
    Fit a Gaussian process to the observed cells of one clone layer and predict
    the posterior mean and standard deviation on the whole grid.

    gt   : (N, N) ground-truth clone prevalence; only cells where mask is True are used
    mask : (N, N) bool, True = biopsied / observed

    Returns (mean, std), both (N, N). Observed cells keep their measured value
    with std = 0 (same convention as data_prep.ipynb).
    """
    gt = np.asarray(gt, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("gp_interpolate needs at least one observed cell.")

    rows, cols = np.where(mask)
    X = np.column_stack([cols, rows]).astype(float)
    y = gt[rows, cols]

    scaler = StandardScaler().fit(X)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0))
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 0.25))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # sklearn ConvergenceWarning on bound hits
        model.fit(scaler.transform(X), y)

    grid_rows, grid_cols = np.indices(gt.shape)
    query = np.column_stack([grid_cols.ravel(), grid_rows.ravel()]).astype(float)
    mu, sd = model.predict(scaler.transform(query), return_std=True)

    mean = np.clip(mu.reshape(gt.shape), 0.0, 1.0)
    std = sd.reshape(gt.shape)

    mean[mask] = gt[mask]
    std[mask] = 0.0
    return mean, std


def exceedance_probability(mean, std, threshold):
    """P(U >= threshold) for U ~ N(mean, std^2), elementwise."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    prob = np.where(mean >= threshold, 1.0, 0.0)          # std == 0 case
    nonzero = std > 1e-9
    prob[nonzero] = norm.sf((threshold - mean[nonzero]) / std[nonzero])
    return prob


def probability_contours(
    mean,
    std,
    prevalence_threshold=0.1,
    levels=(0.5, 0.9),
    **ms_kwargs,
):
    """
    Boundary extraction on the probability field P(U >= prevalence_threshold).

    Marching squares is run on the (zero-margined) probability field once per
    confidence level. A high level (0.9) is a conservative boundary: we are 90%
    sure the clone exceeds the threshold inside it. A low level (0.1) is a
    generous boundary: outside it the clone is very unlikely to exceed it.
    Level 0.5 is close to contouring the mean.

    Returns (contours_by_level: dict, prob_field_with_margin: ndarray)
    """
    N = np.asarray(mean).shape[0]
    prob = margins(exceedance_probability(mean, std, prevalence_threshold), N)
    ms_kwargs.setdefault("options", {"requireLineFrame": False})
    contours = {
        float(level): marching_squares(prob, threshold=level, **ms_kwargs)
        for level in levels
    }
    return contours, prob


def plot_comparison(panels, mask, path, suptitle=""):
    """
    panels: list of dicts {title, field (margined), contours: {color: [paths]}, label}
    mask  : (N, N) bool of observed cells (drawn as white dots)
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    obs_r, obs_c = np.where(mask)
    for ax, p in zip(axes, panels):
        field = np.asarray(p["field"])
        rows, cols = field.shape
        im = ax.imshow(field, origin="upper", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, label=p.get("label", ""))
        ax.scatter(obs_c + 1, obs_r + 1, s=4, c="white", alpha=0.7)   # +1: margin
        for color, paths in p["contours"].items():
            for pts in paths:
                if is_outer_frame(pts, rows, cols):
                    continue
                pts = np.asarray(pts, dtype=float)
                if pts.ndim == 2 and pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.6)
        ax.set_title(p["title"], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(suptitle)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.close()


def save_results(path, arrays, contours_by_clone, mask, threshold, levels, prop):
    """Write fields + contours so metrics.py can score them without re-running anything."""
    payload = dict(arrays)
    for k, c in contours_by_clone.items():
        payload[f"contours_{k}"] = np.array(json.dumps(c))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, mask=np.asarray(mask), threshold=threshold, levels=np.array(levels, dtype=float),
        prop=prop, n_clones=len(contours_by_clone), **payload,
    )
    print(f"Saved {path}")


def main():

    # Number of rows/cols
    N = 20
    NORMAL_PURITY_THRESHOLD = 0.20

    # Phylogeny Tree (contains 5 clone species, including 1)
    phylogeny = {   
        "1": [2, 3],
        "2": [],
        "3": [4, 5],
        "4": [],
        "5": []
    }

    # Proportion of biopsies to include during interpolation
    prop = 0.5



    # First, we run the simulation to get our ground truth dataset.
    # This shows proportion of each biopsy that is each clone species.
    _,matrix_u,_,_ = run_simulation(N,NORMAL_PURITY_THRESHOLD,phylogeny,seed=43,show_plot=False)
    matrix_u = np.array(matrix_u)

    # Second, we randomly select which biopsies we interpolate from. This mask is conserved across all interpolations in the dataset.
    mask = random_cell_mask(N,N,prop)

    # Prevalence threshold used for the boundary, and confidence levels for the
    # probability contours.
    T = 0.1
    LEVELS = (0.5, 0.9)   # add 0.1 for a generous outer boundary (noisy far from biopsies)
    LEVEL_COLORS = {0.1: "cyan", 0.5: "orange", 0.9: "red"}

    saved_arrays, saved_contours = {}, {}

    for id in range(5): # There are 5 clone species in the phylogeny tree
        
        # Then, we isolate the clone species in the simulated ground truth dataset.
        gt = get_clone_field(matrix_u,N,id) 

        # (a) bilinear interpolation (original ClonArch behaviour)
        bilinear = bilin_interp(gt,mask)

        # (b) Gaussian process: posterior mean and standard deviation
        gp_mean, gp_std = gp_interpolate(gt, mask)

        # (c) probability that the clone prevalence exceeds T
        # The marching squares doesn't like pixels touching the edge, so we add a margin of 0s.
        gt_m = margins(gt,N)
        bilinear_m = margins(bilinear,N)
        gp_mean_m = margins(gp_mean,N)

        ms_opts = dict(js_file=DEFAULT_JS_FILE, options={"requireLineFrame": False})
        contours_gt = marching_squares(gt_m, threshold=T, **ms_opts)
        contours_bilinear = marching_squares(bilinear_m, threshold=T, **ms_opts)
        contours_gp_mean = marching_squares(gp_mean_m, threshold=T, **ms_opts)
        contours_prob, prob_m = probability_contours(
            gp_mean, gp_std, prevalence_threshold=T, levels=LEVELS, **ms_opts
        )

        # Plot contours on the ground truth, bilinear, and GP.
        plot_contours(contours_gt, gt_m, "gt", id, prop)
        plot_contours(contours_bilinear, bilinear_m, "int", id, prop)
        plot_contours(contours_gp_mean, gp_mean_m, "gp_mean", id, prop,
                      title=f"Simulated Dataset:\n Mutant {id+1} GP mean, {int(prop*100)}% known")
        for level in LEVELS:
            plot_contours(contours_prob[float(level)], prob_m, f"gp_prob{int(level*100)}", id, prop,
                          title=f"Mutant {id+1}: P(U >= {T}) contour at {level:.0%}, {int(prop*100)}% known",
                          label=f"P(U >= {T})")

        plot_comparison(
            [
                {"title": "Ground truth", "field": gt_m, "label": "U",
                 "contours": {"red": contours_gt}},
                {"title": "Bilinear, contour at t", "field": bilinear_m, "label": "U",
                 "contours": {"red": contours_bilinear}},
                {"title": "GP mean, contour at t", "field": gp_mean_m, "label": "U",
                 "contours": {"red": contours_gp_mean}},
                {"title": f"GP P(U>={T}); contours " + "/".join(f"{l:.0%}" for l in LEVELS),
                 "field": prob_m, "label": f"P(U >= {T})",
                 "contours": {LEVEL_COLORS.get(l, "white"): contours_prob[float(l)] for l in LEVELS}},
            ],
            mask,
            OUT_DIR / f"{id+1}_{prop}_comparison.png",
            suptitle=f"Mutant {id+1}, {int(prop*100)}% of cells observed (white dots)",
        )

        # Keep everything needed for metrics.py
        k = id + 1
        saved_arrays.update({
            f"gt_{k}": gt, f"bilinear_{k}": bilinear, f"gp_mean_{k}": gp_mean,
            f"gp_std_{k}": gp_std, f"prob_{k}": exceedance_probability(gp_mean, gp_std, T),
        })
        saved_contours[k] = {
            "gt": contours_gt, "bilinear": contours_bilinear, "gp_mean": contours_gp_mean,
            **{f"prob_{l:g}": contours_prob[float(l)] for l in LEVELS},
        }

    results_path = OUT_DIR / f"results_{prop}.npz"
    save_results(results_path, saved_arrays, saved_contours, mask, T, LEVELS, prop)

    import metrics
    metrics.summarize(metrics.evaluate_results(results_path))


if __name__ == "__main__":

    main()