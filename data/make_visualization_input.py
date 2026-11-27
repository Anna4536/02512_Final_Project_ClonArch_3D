import json
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("f_gaussian_lp.csv")
STD_CSV = Path("gp_uncertainty.csv")
# Row labels of the interpolated matrix: sample name for measured spots, "Interpolated" otherwise
COORDS_CSV = Path("new_fff_ppp_gaussian_process.csv")
OUTPUT_JS = Path("gp_interpolated.js")


# Trees copied from LP.ipynb. They are zero-indexed there.
trees_zero_indexed = [
    {
        18: [19],
        19: [],
        7: [18, 17, 16],
        17: [],
        16: [],
        4: [7],
        3: [15, 14],
        15: [],
        14: [],
        2: [13],
        13: [],
        1: [9],
        9: [],
        0: [12, 11, 10, 8, 6, 5, 4, 3, 2, 1],
        12: [],
        11: [],
        10: [],
        8: [],
        6: [],
        5: [],
    },
    {
        18: [19],
        19: [],
        7: [18, 17, 16],
        17: [],
        16: [],
        4: [7],
        3: [15, 14],
        15: [],
        14: [],
        2: [13],
        13: [],
        1: [9, 8],
        9: [],
        8: [],
        0: [12, 11, 10, 6, 5, 4, 3, 2, 1],
        12: [],
        11: [],
        10: [],
        6: [],
        5: [],
    },
]


def convert_tree_to_one_indexed(tree):
    """ClonArch's JavaScript visualization uses one-indexed nodes."""

    return {
        str(parent + 1): [child + 1 for child in children]
        for parent, children in tree.items()
    }


phylogeny = [
    convert_tree_to_one_indexed(tree)
    for tree in trees_zero_indexed
]


mutation_labels = [
    ["trunk"],
    ["MLL", "CHUK"],
    ["RIMS2"],
    ["C15orf42", "JAK2"],
    ["MUC16"],
    ["FLNB"],
    ["LRP2"],
    ["ZNF498", "LRRTM4"],
    ["PPP1R3B"],
    ["ITGB2"],
    ["GCK"],
    ["KIAA0427", "GABRA1"],
    ["A2M"],
    ["CACNA1A", "MAP4"],
    ["NACAD", "VIPR2"],
    ["DBF4B"],
    ["BAZ1B"],
    ["CNOT10", "PRKAR1B"],
    ["THRA"],
    ["PHF19"],
]


# LP.ipynb writes no header and no index.
matrix = pd.read_csv(INPUT_CSV, header=None)
gp_std_matrix = pd.read_csv(STD_CSV, header=None)

expected_shape = (42 * 44, 20)

if matrix.shape != expected_shape:
    raise ValueError(
        f"Expected matrix shape {expected_shape}, "
        f"but found {matrix.shape}."
    )

if gp_std_matrix.shape != expected_shape:
    raise ValueError(
        f"Expected GP uncertainty shape {expected_shape}, "
        f"but found {gp_std_matrix.shape}."
    )

if matrix.isna().any().any():
    raise ValueError("LP matrix contains missing values.")

if gp_std_matrix.isna().any().any():
    raise ValueError(
        "GP uncertainty matrix contains missing values."
    )

if (gp_std_matrix < 0).any().any():
    raise ValueError(
        "GP standard deviations cannot be negative."
    )

coordinates = pd.read_csv(COORDS_CSV, index_col=0).index.astype(str).tolist()
if len(coordinates) != len(matrix):
    raise ValueError("coordinates and LP matrix have different lengths")

data = {
    "cols": 44,
    "rows": 42,

    # main.js expects one threshold per tree node.
    "u_thresholds": [0.1] * 20,
    "f_thresholds": [0.1] * 20,
    "default_threshold": 0.1,

    "coordinates": coordinates,
    "mut_labels": mutation_labels,

    # Start with all mutations selected.
    "selected_mutations": list(range(1, 21)),

    # LP-adjusted GP mean frequencies.
    "input": matrix.values.tolist(),

    # GP posterior standard deviations.
    "gp_std": gp_std_matrix.values.tolist(),

    # Use the probability field for boundary extraction.
    "contour_mode": "probability",

    # Extract the boundary where
    # P(prevalence >= slider threshold) = 0.95.
    "probability_threshold": 0.001,
}


with OUTPUT_JS.open("w", encoding="utf-8") as output:
    output.write("phylogeny = ")
    json.dump(phylogeny, output, indent=2)
    output.write(";\n\n")

    output.write("data = ")
    json.dump(data, output, indent=2)
    output.write(";\n")


print(f"Wrote {OUTPUT_JS}")
print(f"Matrix shape: {matrix.shape}")
print(f"Number of trees: {len(phylogeny)}")