import json
import matplotlib.patches as mpatches
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


'''
Simulate_dataset.py
This is a callable version of simulation.ipynb, used for making a simulated dataset.
It's pretty much an exact copy of simulation.ipynb, but slightly more organized.


'''




# ## Utility Functions


def get_root(tree):
    nodes = set(tree.keys())
    for node in tree:
        for child in tree[node]:
            child = str(child)
            try:
                nodes.remove(child)
            except Exception as e:
                print("Exception: " + str(e))
                # print("Already removed" + str(child))

    if len(nodes) > 1:
        print("Error: more than one node remaining in parent list")
        return -1
    return nodes.pop()


def get_parent(clone, tree):
    for key in tree:
        if int(clone) in tree[key]:
            return key
    return -1


# ## Begin
#
# Define the inputs here;
# * `N` = size of grid (NxN)
# * `phylogeny` = phylogenetic tree
#
# These are global variables.
#
# Assumptions:
# * phylogeny starts at 1; 0 is reserved for normal


def get_starting_pos(clone, root, N, phylogeny, positions):
    '''
    Input: 
        - clone 
        - positions (dictionary of clones and their current positions on the matrix)
    Output: 
        a pair that represents a spot on the matrix 
        - If root, we will start at the center (N/2, N/2)
        - Otherwise, we will randomly pick a place that the clone's parent is at
    '''
    if clone == root:
        return (int(N / 2), int(N / 2))
    else:
        parent = get_parent(clone, phylogeny)
        arr = np.arange(len(positions[parent]))
        rand_idx = np.random.choice(arr)
        return positions[parent][rand_idx]


def within_bounds(N,coordinates):
    '''
    Input:
        - coordinates (x,y)
    Output:
        - boolean value indicating whether coordinates are within grid bounds
    '''
    x = coordinates[0]
    y = coordinates[1]
    if x >= 0 and x < N and y >= 0 and y < N:
        return True
    return False


def get_position_probability(N,coordinate, clone, grid):
    '''
    Input:
        - coordinate to move to 
        - clone 
    Output:
        - probability of clone moving to specified position 
        Rules:
            base probability starts at 1
            +1 if space is empty
            +1 for each neighbor that has clone 
    '''

    prob = 1.0
    x = coordinate[0]
    y = coordinate[1]

    if not within_bounds(N,(x, y)):
        return 0.0

    if len(grid[x][y]) == 1:
        prob += 1
    # for each direction, add one IF there is a tumor

    # up
    if within_bounds(N,(x, y + 1)) and len(grid[x][y + 1]) > 1:
        prob += 1
    # down
    elif within_bounds(N,(x, y - 1)) and len(grid[x][y - 1]) > 1:
        prob += 1
    # left
    elif within_bounds(N,(x - 1, y)) and len(grid[x - 1][y]) > 1:
        prob += 1
    # right
    elif within_bounds(N,(x + 1, y)) and len(grid[x + 1][y]) > 1:
        prob += 1

    return prob ** 2


# UP
# DOWN
# LEFT
# RIGHT


def move_clone(N,coordinates, clone, grid):
    '''
    Input:
        - coordinate (x,y) of current position
        - clone 
        - current grid status
    Output:
        - coordinate (x,y) of position to move to
    '''
    x = coordinates[0]
    y = coordinates[1]
    # moves_idx=['up','down','left','right']
    moves_idx = [0, 1, 2, 3]
    moves = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
    probs = [0.25, 0.25, 0.25, 0.25]
    moved = False

    for idx, i in enumerate(moves):
        probs[idx] = get_position_probability(N,i, clone, grid)
    probs = probs / (np.sum(probs))
    draw = np.random.choice(moves_idx, 1, p=probs)[0]
    return moves[draw]
    #     while not moved:
    #         move=np.random.choice(moves)
    #         #print(move)
    #         if move == 'up' and within_bounds((x, y+1)):
    #             return (x, y+1)
    #         elif move == 'down' and within_bounds((x, y-1)):
    #             return (x, y-1)
    #         elif move == 'left' and within_bounds((x-1, y)):
    #             return (x-1, y)
    #         elif move == 'right' and within_bounds((x+1, y)):
    #             return (x+1, y)


def get_grid(phylogeny, N, purity, root):
    # alphas[0] will be the normal; purity should be < NORMAL_PURITY_THRESHOLD (0.2)
    alphas = [1, 8, 7, 6, 9, 6, 8]
    grid = [[{"0": 0.0} for x in range(N)] for y in range(N)]
    positions = {}
    for clone in phylogeny.keys():

        positions[clone] = []
        # initialize to start in middle
        cur = get_starting_pos(clone, root, N, phylogeny, positions)
        # probability of moving
        while np.random.normal(0, 1) > -2:
            x = cur[0]
            y = cur[1]

            # add position to clone position dictionary
            positions[clone].append(cur)

            # generate frequency from dirichlet
            grid[x][y][clone] = 0.0
            freqs = np.random.dirichlet(alphas[: len(grid[x][y].keys())])
            # freqs=np.around(np.array(freqs),4)
            while freqs[0] > purity:
                freqs = np.random.dirichlet(alphas[: len(grid[x][y].keys())])
            for idx, key in enumerate(grid[x][y]):
                grid[x][y][key] = freqs[idx]

            # move
            cur = move_clone(N,cur, clone, grid)
    return grid


# ## Visualize matrix


def stringify(arr):
    num = ""
    for i in arr:
        num += i
    return int(num)


def plot_grid(N,matrix):
    image_grid = [[stringify(matrix[x][y].keys()) for x in range(N)] for y in range(N)]
    r = lambda: np.random.randint(0, 255)
    brewer = [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
        "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
        "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
        "#000000", "#ffed6f", "#aaaaaa", "#bbbbbb",
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
        "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
        "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
    ]
    c = []
    for i in range(len(np.unique(image_grid))):
        #         r = lambda: np.random.randint(0,255)/255
        #         c.append((r(),r(),r()))
        c.append(brewer[i])

    # cmap = colors.ListedColormap(['black','blue','red','green','yellow','white'])
    cmap = colors.ListedColormap(c)
    bounds = np.unique(image_grid) - 0.01
    norm = colors.BoundaryNorm(bounds, cmap.N)
    values = np.unique(image_grid)
    patches = [
        mpatches.Patch(color=c[i], label="Clones {l}".format(l=values[i]))
        for i in range(len(values))
    ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("Simulation Grid")
    plt.imshow(image_grid, cmap=cmap, norm=norm)

    return image_grid


def pretty_print(matrix):
    for i in matrix:
        print(i)


# ## Generate matrix U and F
# * `u` is the clonal prevalence matrix
# * `f` is the frequency matrix


def construct_u(N, phylogeny,grid):
    U = [[0 for x in range(len(phylogeny))] for y in range(N * N)]
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            for key in grid[r][c]:
                # ignore normal for matrix F
                if key != "0":
                    # f_col stands for the clone
                    f_col = int(key) - 1
                    # number of columns * x + y
                    f_row = r + N * c
                    U[f_row][f_col] = grid[r][c][key]
    return U


def get_f(tree, u):
    F = [row[:] for row in u]
    root = get_root(tree)

    def recurse(node, parent, u, temp_F):
        for child in tree[node]:
            recurse(str(child), node, u, temp_F)
        if node != root:
            parent_idx = int(parent) - 1
            node_idx = int(node) - 1
            for row in range(len(temp_F)):
                temp_F[row][parent_idx] += temp_F[row][node_idx]

        return temp_F

    F = recurse(root, root, u, F)
    return F


def test(matrix):
    for row in matrix:
        for cell in row:
            if cell > 1 or cell < -1:
                print("Error: " + str(cell))


# ## The full pipeline


def run_simulation(n_grid, purity, tree, seed=None, show_plot=False):
    N = n_grid
    phylogeny = tree
    if seed is not None:
        np.random.seed(seed=seed)
    root = get_root(phylogeny)
    num_clones = len(phylogeny.keys())
    grid = get_grid(phylogeny, N, purity, root)

    img_grid = [[stringify(grid[x][y].keys()) for x in range(N)] for y in range(N)]

    #pretty_print(img_grid)
    grid_plot = plot_grid(N,grid)
    if show_plot:
        plt.show()
    else:
        plt.close()

    matrix_u = construct_u(N,phylogeny,grid)
    matrix_f = get_f(phylogeny, matrix_u)
    test(matrix_f)

    return grid, matrix_u, matrix_f, grid_plot


if __name__ == "__main__":
    # necessary inputs
    NORMAL_PURITY_THRESHOLD = 0.20
    N = 20
    phylogeny = {
        "1": [2, 3],
        "2": [],
        "3": [4, 5],
        "4": [],
        "5": [],
    }
    grid, matrix_u, matrix_f, grid_plot = run_simulation(N, NORMAL_PURITY_THRESHOLD, phylogeny, seed=43, show_plot=True)


    