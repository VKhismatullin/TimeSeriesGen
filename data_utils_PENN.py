import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict, Counter

def plot_figure(x, y, ax, color):
    """
    Plot a human figure using arrows to represent body parts.
    This function connects keypoints into limbs and joints.
    """
    left = np.arange(2, 13, 2) - 1
    right = left + 1

    head = np.array([x[0], y[0]])
    x_r, y_r = x[right], y[right]
    x_l, y_l = x[left], y[left]
    x_mid = (x_l + x_r) / 2
    y_mid = (y_l + y_r) / 2

    d = 0.01  # arrow width

    # Head to shoulders
    arr = plt.arrow(
        x_mid[0], y_mid[0],
        head[0] - x_mid[0], head[1] - y_mid[0],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Shoulders to left/right
    arr = plt.arrow(
        x_mid[0], y_mid[0],
        x_l[0] - x_mid[0], y_l[0] - y_mid[0],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)
    arr = plt.arrow(
        x_mid[0], y_mid[0],
        x_r[0] - x_mid[0], y_r[0] - y_mid[0],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Shoulders to elbows
    arr = plt.arrow(
        x_l[0], y_l[0],
        x_l[1] - x_l[0], y_l[1] - y_l[0],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)
    arr = plt.arrow(
        x_r[0], y_r[0],
        x_r[1] - x_r[0], y_r[1] - y_r[0],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Elbows to hands
    arr = plt.arrow(
        x_l[1], y_l[1],
        x_l[2] - x_l[1], y_l[2] - y_l[1],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)
    arr = plt.arrow(
        x_r[1], y_r[1],
        x_r[2] - x_r[1], y_r[2] - y_r[1],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Shoulders to pelvis
    arr = plt.arrow(
        x_mid[0], y_mid[0],
        x_mid[3] - x_mid[0], y_mid[3] - y_mid[0],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Pelvis to left/right pelvis
    arr = plt.arrow(
        x_mid[3], y_mid[3],
        x_l[3] - x_mid[3], y_l[3] - y_mid[3],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)
    arr = plt.arrow(
        x_mid[3], y_mid[3],
        x_r[3] - x_mid[3], y_r[3] - y_mid[3],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Pelvis to knees
    arr = plt.arrow(
        x_l[3], y_l[3],
        x_l[4] - x_l[3], y_l[4] - y_l[3],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)
    arr = plt.arrow(
        x_r[3], y_r[3],
        x_r[4] - x_r[3], y_r[4] - y_r[3],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)

    # Knees to ankles
    arr = plt.arrow(
        x_l[4], y_l[4],
        x_l[5] - x_l[4], y_l[5] - y_l[4],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)
    arr = plt.arrow(
        x_r[4], y_r[4],
        x_r[5] - x_r[4], y_r[5] - y_r[4],
        head_width=d, head_length=d, color=color
    )
    ax.add_patch(arr)


def visualize(y, X=None, title=None, file=None):
    """
    Visualize human poses of one action (one per subplot) using plot_figure.
    Optionally overlays a second set of poses (X).
    """
    fig, ax = plt.subplots(3, 5, sharex='all', sharey='all', figsize=(14, 6))
    
    title = 'Human action in silhouettes over time' if title is None else title
    plt.suptitle(title, fontsize=14)

    for j in range(15):
        ax = plt.subplot(3, 5, j + 1)
        plot_figure(
            y[j * 2, :13].astype(float),
            -y[j * 2, 13:].astype(float),
            ax, 'r'
        )
        if X is not None:
            plot_figure(
                X[j * 2, :13].astype(float),
                -X[j * 2, 13:].astype(float),
                ax, 'k'
            )
    if file is None:
        plt.show()
    else:
        plt.savefig(file)

        
def load_PENN_data(path=None, shuffle=False, uniform_shuffle=False):
    # --- 1. LOAD DATASET ---
    with open('datasets/UPENN_no_anomaly_STD.pkl' if path is None else path, 'rb') as f:
        loaded_dict = pickle.load(f)

    # Map each unique action and pose to an integer label
    names = {x: i for i, x in enumerate(OrderedDict(Counter([s['action'][0] for s in loaded_dict])).keys())}
    poses = {x: i for i, x in enumerate(OrderedDict(Counter([s['pose'][0] for s in loaded_dict])).keys())}

    # Preallocate data arrays
    autoenc_X = np.zeros((len(loaded_dict), 30, 13 * 2))  # 30 timesteps, 13 joints (x + y)
    autoenc_Y = np.zeros((len(loaded_dict), 3))           # [action_id, pose_id, index]

    # Fill arrays with pose sequences and label encodings
    for i, sample in enumerate(loaded_dict):
        autoenc_X[i, :, :13] = sample['x']  # x-coordinates
        autoenc_X[i, :, 13:] = sample['y']  # y-coordinates
        autoenc_Y[i, 0] = names[sample['action'][0]]  # action label
        autoenc_Y[i, 1] = poses[sample['pose'][0]]    # pose label
        autoenc_Y[i, 2] = sample['ind']               # sequence index
    autoenc_Y = autoenc_Y.astype(int)

    # --- 2. SHUFFLE AND SORT ---
    if shuffle:
        # Shuffle the dataset completely (remove ordering bias)
        data_size = autoenc_X.shape[0]
        choice = np.random.choice(data_size, data_size, replace=False)
        autoenc_X = autoenc_X[choice]
        autoenc_Y = autoenc_Y[choice]

        # Sort samples by pose_id (column 1 of autoenc_Y)
        sort_indices = autoenc_Y[:, 1].argsort()
        autoenc_X = autoenc_X[sort_indices]
        autoenc_Y = autoenc_Y[sort_indices]

    # --- 3. STRUCTURED PERMUTATION FOR BALANCED DATA ORDERING ---
    if uniform_shuffle:
        # Define number of blocks (e.g., number of pose groups)
        block_width = 32
        data_size = autoenc_X.shape[0]

        block_height = data_size // block_width

        # Step 1: Sort by action within each block
        for i in range(block_width):
            start = i * block_height
            end = (i + 1) * block_height
            sort = autoenc_Y[start:end, 0].argsort()
            autoenc_X[start:end] = autoenc_X[start:end][sort]
            autoenc_Y[start:end] = autoenc_Y[start:end][sort]

        # Step 2: Shuffle entire blocks
        choice = np.random.choice(block_width, block_width, replace=False)
        new_autoenc_X = np.zeros_like(autoenc_X)
        new_autoenc_Y = np.zeros_like(autoenc_Y)

        for i in range(block_width):
            src = choice[i]
            new_autoenc_X[i*block_height:(i+1)*block_height] = autoenc_X[src*block_height:(src+1)*block_height]
            new_autoenc_Y[i*block_height:(i+1)*block_height] = autoenc_Y[src*block_height:(src+1)*block_height]

        autoenc_X = new_autoenc_X
        autoenc_Y = new_autoenc_Y

        # Step 3: Interleave samples across blocks at each row index
        for j in range(block_height):
            choice1 = np.random.choice(block_width, block_width, replace=False)
            for i in range(block_width):
                new_autoenc_X[i * block_height + j] = autoenc_X[choice1[i] * block_height + j]
                new_autoenc_Y[i * block_height + j] = autoenc_Y[choice1[i] * block_height + j]

        # Replace with the final shuffled result
        autoenc_X = new_autoenc_X
        autoenc_Y = new_autoenc_Y
    return autoenc_X, autoenc_Y