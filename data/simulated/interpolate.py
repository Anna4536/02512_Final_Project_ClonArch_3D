import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulate_dataset import run_simulation
from scipy.interpolate import griddata

def random_cell_mask(n_rows, n_cols, sample_fraction=0.25, seed=43):
    rng = np.random.default_rng(seed)
    total = n_rows * n_cols
    k = int(round(total * sample_fraction))
    flat = np.zeros(total, dtype=bool)
    if k > 0:
        flat[rng.choice(total, size=k, replace=False)] = True
    return flat.reshape((n_rows, n_cols))

def reconstruct_from_mask(
    field: np.ndarray,
    mask: np.ndarray,
    method: str = "linear",
    clip_range: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Reconstruct a full (N, N) grid from sampled cells using scipy.interpolate.griddata."""
    n_rows, n_cols = field.shape
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    known_pts = np.column_stack([rows[mask], cols[mask]]).astype(float)
    known_vals = field[mask].astype(float)

    all_pts = np.column_stack([rows.ravel(), cols.ravel()]).astype(float)

    if len(known_vals) == 0:
        recon = np.zeros_like(field, dtype=float)
    else:
        recon = griddata(known_pts, known_vals, all_pts, method=method).reshape(
            (n_rows, n_cols)
        )
        # Fill any cells outside the convex hull of sampled points
        if np.isnan(recon).any():
            nearest = griddata(
                known_pts, known_vals, all_pts, method="nearest"
            ).reshape((n_rows, n_cols))
            recon = np.where(np.isnan(recon), nearest, recon)

    # Preserve exact values at sampled cells
    recon[mask] = known_vals

    if clip_range is not None:
        recon = np.clip(recon, clip_range[0], clip_range[1])

    return recon


def get_clone_field(matrix_u, N, clone_idx=0, order="F"):
    # matrix_u has shape (N*N, num_clones); f_row = r + N * c 
    arr = np.asarray(matrix_u[:, clone_idx], dtype=float)
    return arr.reshape((N, N), order=order).T


def mean_square_err(N,ground_truth,new,mask):
    # Mean square error of non-masked values

    invert = ~mask

    mean = (ground_truth[invert]-new[invert])**2

    print(mean.sum()/len(ground_truth[invert]))



def main():

    # Number of rows/cols
    N = 20
    NORMAL_PURITY_THRESHOLD = 0.20

    # Phylogeny Tree
    phylogeny = {   
        "1": [2, 3],
        "2": [],
        "3": [4, 5],
        "4": [],
        "5": []
    }

    # Make simulated grid
    grid,matrix_u,matrix_f,grid_plot = run_simulation(N,NORMAL_PURITY_THRESHOLD,phylogeny,seed=43,show_plot=False)
    matrix_u = np.array(matrix_u)

    one = get_clone_field(matrix_u,N,1)
    mask = random_cell_mask(N,N,0.75)

    new = reconstruct_from_mask(one,mask,method="linear",clip_range=(0,1))

    #print(mean_square_err(N,one,new,mask))


    ONE = pd.DataFrame(one)


    ONE[~mask] = np.nan

    
    row = ONE.interpolate(method='linear', limit_direction='both', axis=1)
    col = ONE.interpolate(method='linear', limit_direction='both', axis=0)

    avg = (row + col) / 2
    
    interpolated = np.array(avg)

    plt.imshow(interpolated)
    plt.show()



        





if __name__ == "__main__":

    main()




