import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulate_dataset import run_simulation
from scipy.interpolate import griddata
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JS_FILE = BASE_DIR / "../../dist/marchingsquares-isolines.js"
DEFAULT_JS_SCRIPT = BASE_DIR / "get_contours.js"

'''
Interpolate.py

Bilinear interpolation + Marching Squares on simulated dataset.
Uses the same algorithms as data_prep.ipynb for interpolation and marching squares


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

    print(mean.sum()/len(ground_truth[invert]))

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

def plot_contours(contours, matrix, name, id, prop):
    rows, cols = np.asarray(matrix).shape
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(matrix, origin="upper", vmin=0,vmax=1)
    fig.colorbar(im, ax=ax,label="Mutant proportion")

    for path in contours:
        if is_outer_frame(path, rows, cols):
            continue
        pts = np.asarray(path, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.8)

    plt.tight_layout()
    plt.title(f"Simulated Dataset:\n Mutant {id+1} Linear Interpolation, {int(prop*100)}% known")

    plt.savefig(f"data/simulated/maps/{id+1}_{prop}_{name}.png")
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

    for id in range(5): # There are 5 clone species in the phylogeny tree
        
        # Then, we isolate the clone species in the simulated ground truth dataset.
        gt = get_clone_field(matrix_u,N,id) 

        # We perform bilinear interpolation.
        interpolated = bilin_interp(gt,mask)

        # The marching squares doesn't like pixels touching the edge, so we add a margin of 0s.
        interpolated = margins(interpolated,N)
        gt = margins(gt,N)

        # Then, we run marching squares to get our contours. This uses ClonArch's marching squares algorithm.
        contours_int = marching_squares(
            interpolated,
            threshold=0.1,
            js_file=BASE_DIR / "../../dist/marchingsquares-isolines.js",
            options={"requireLineFrame": False},
        )
        contours_gt = marching_squares(
            gt,
            threshold=0.1,
            js_file=BASE_DIR / "../../dist/marchingsquares-isolines.js",
            options={"requireLineFrame": False},
        )

        # Plot contours on the interpolation and ground truth.
        plot_contours(contours_gt,gt,f"gt",id,prop)
        plot_contours(contours_int,interpolated,f"int",id,prop)




        





if __name__ == "__main__":

    main()




