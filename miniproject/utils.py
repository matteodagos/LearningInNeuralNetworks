from typing import List, Tuple, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import trange

import network


def reconstruct_image_from_patches(
        patch_size : int,
        n_horizontal_patches: int,
        n_vertical_patches: int,
        patches: List[np.ndarray],
        patch_subsampling_factor=1
) -> np.ndarray:
    """
    Reconstructs an image from patches.
    :param patch_size:
    :param n_horizontal_patches:
    :param n_vertical_patches:
    :param patches: list of patches of the image
    :param patch_subsampling_factor: if patches are overlapping, allow to discard every other patch_subsampling_factor
    :return:
    """

    im_patches_np = np.array(patches)
    patches_reshaped = im_patches_np.reshape(
        n_vertical_patches,
        n_horizontal_patches,
        patch_size,
        patch_size
    )[::patch_subsampling_factor, ::patch_subsampling_factor]

    image = patches_reshaped.transpose(0, 2, 1, 3).reshape(
        int(n_vertical_patches * patch_size / patch_subsampling_factor),
        int(n_horizontal_patches * patch_size / patch_subsampling_factor),
    )

    return image

def show_RF(W, patch_size=16, title="Receptive Field"):
    RF = W.reshape(patch_size, patch_size)
    max_value = np.max(np.abs(W))
    plt.imshow(RF, cmap='PiYG', vmin=-max_value, vmax=max_value)
    plt.colorbar()
    plt.grid(False)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_RFs(Ws, n_col=5, title='', patch_size=16):
    n = len(Ws)
    n_row = int(np.ceil(n / n_col))
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, sharex=True, sharey=True, figsize=(2*n_col,2*n_row))

    max_value = np.max(np.abs(Ws))

    for i, ax in enumerate(axs.flat):
        display = i < len(Ws)
        W = Ws[i].reshape(patch_size, patch_size) if display else np.zeros((patch_size, patch_size))
        display_img = ax.imshow(W, cmap='PiYG', vmin=-max_value, vmax=max_value)
        ax.set_visible(display)
        ax.axis('off')

    fig.colorbar(display_img, ax=axs, orientation='horizontal')
    fig.suptitle(title)
    plt.show()

def extract_patches(
        img : np.ndarray,
        patch_size : int = 16,
        n_patches : int = 5000,
        verbose : bool = False,
) -> List[np.ndarray]:
    """
    extract specified number of patches of given size from the image
    :param verbose:
    :param img: numpy array of the image of shape (H, W) (grayscale)
    :param patch_size: side of a square patch to extract (default is 16)
    :param n_patches: number of patches to extract (default is 5000)
    :return: list of flattened patches. may return less than n_patches if patch sizes are not multiple of image sizes
    """
    img_height, img_width = img.shape
    a = img_height / img_width

    if verbose: print(f"Image size (W, H): {img_width} x {img_height}, aspect ratio: {a:.2f}")

    n_horizontal_displacement = int(np.floor(np.sqrt(n_patches / a)))
    n_vertical_displacement = int(np.floor(n_patches / n_horizontal_displacement))

    horizontal_stride = int(np.floor((img_width - patch_size) / (n_horizontal_displacement - 1)))
    vertical_stride = int(np.floor((img_height - patch_size) / (n_vertical_displacement - 1)))

    if verbose: print(
        f"Returning {n_horizontal_displacement} x {n_vertical_displacement}"
        f" = {n_horizontal_displacement*n_vertical_displacement} patches."
        f" Using horizontal stride of {horizontal_stride} and vertical stride of {vertical_stride}."
    )

    patches = []
    for ver_displacement_idx in range(n_vertical_displacement):
        for hor_displacement_idx in range(n_horizontal_displacement):
            x = hor_displacement_idx * horizontal_stride
            y = ver_displacement_idx * vertical_stride
            x_end = x + patch_size
            y_end = y + patch_size
            patch = img[y:y_end, x:x_end]
            patches.append(patch.flatten())


    return patches

def normalize(img : Image.Image) -> np.ndarray:
    """
    Normalize the image to have zero mean and unit variance.
    """
    np_img = np.array(img)
    mean = np.mean(np_img)
    std = np.std(np_img)
    return (np_img - mean) / std


def load_dataset(path: str) -> List[np.ndarray]:
    """
    Returns list of patches from images at the given path. Assume images are in .bmp format and grayscale.
    images are normalized to have zero mean and unit variance.
    """
    patch_list = []
    for filename in os.listdir(path):

        if filename.endswith('.bmp'):
            img = Image.open(os.path.join(path, filename))
            normalized_img = normalize(img)
            patches = extract_patches(normalized_img)
            patch_list.extend(patches)

    return patch_list


def train_network(
        net : network.Visual_network,
        patches: List[np.ndarray],
        n_iter: int,
        theta: np.ndarray,
        gamma: float=0.01,
        seed: int=0,
        progress_bar: bool=False,
        **kwargs,
) -> None:
    """
    Train the network using the given patches and parameters.
    :param net:
    :param patches:
    :param theta:
    :param n_iter:
    :param gamma:
    :param seed:
    :param progress_bar: if True, show a progress bar
    :return:
    """
    # use seed to generate n_iter indices between 0 and len(patches)
    rng = np.random.RandomState(seed)

    def step():
            idx = rng.randint(0, len(patches))
            x = patches[idx]
            net.update(x, gamma, theta, **kwargs)

    if progress_bar:
        for _ in trange(n_iter, desc="Training network", leave=False):
            step()
    else:
        for _ in range(n_iter):
            step()



def plot_v_list(v_list, subsampling_factor):
    v_list_np = np.array(v_list[::subsampling_factor])
    v_list_np = v_list_np.reshape(len(v_list_np), -1)
    diffs = np.zeros((len(v_list_np), len(v_list_np)))
    for row_one in range(len(v_list_np)):
        for row_two in range(len(v_list_np)):
            diffs[row_one, row_two] = np.linalg.norm(v_list_np[row_one] - v_list_np[row_two])
    plt.figure(figsize=(17, 8))
    plt.imshow(diffs)
    plt.title("Distance between the lateral connections weights at different iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Iteration")
    plt.xticks(np.arange(0, len(v_list_np), subsampling_factor), np.arange(0, len(v_list_np), subsampling_factor)*subsampling_factor, rotation=90)
    plt.yticks(np.arange(0, len(v_list_np), subsampling_factor), np.arange(0, len(v_list_np), subsampling_factor)*subsampling_factor)
    plt.colorbar()

def pixel_to_xy(i: int, j: int, patch_size: int = 16) -> Tuple[int, int]:
    """
    Convert image-index (row i, col j) to (x, y) coordinates with
    origin at the *bottom-left* of the patch.
    :param i: row index of the pixel
    :param j: column index of the pixel
    :param patch_size: height of the patch (assumed square, so width is also patch_size)
    """
    if not (0 <= i < patch_size and 0 <= j < patch_size):
        raise ValueError("indices (i, j) must be within the patch bounds")

    x = j
    y = (patch_size - 1) - i
    return x, y

def xy_to_pixel(x: int, y: int, patch_size: int = 16) -> Tuple[int, int]:
    """
    Convert (x, y) coordinates with origin at the *bottom-left* of the patch
    to image-index (row i, col j).
    :param x: x coordinate
    :param y: y coordinate
    :param patch_size: height of the patch (assumed square, so width is also patch_size)
    """
    i = (patch_size - 1) - y
    j = x
    return i, j

def threshold_weights(w: np.ndarray, lambda_: float, patch_size: int = 16) -> np.ndarray:
    """
    Threshold the weights using a given lambda value.
    :param w: weights to threshold
    :param lambda_: threshold value
    :param patch_size: size of the patch (default is 16)
    :return: thresholded weights
    """
    W = w.reshape((patch_size, patch_size))
    return np.where(np.abs(W) < lambda_, 0, W)

def is_boundary(W_lambda: np.ndarray, x: int, y: int) -> bool:
    """
    Return True iff the 3 × 3 patch centred on (x, y) contains both
    at least one strictly positive and one strictly negative entry.
    :param W_lambda : thresholded receptive-field matrix.
    :param x : x coordinate of the centre pixel.
    :param y : y coordinate of the centre pixel.
    :return: True if the patch contains both positive and negative values, False otherwise.
    """
    i,j = xy_to_pixel(x, y)
    i_start, i_end = max(i - 1, 0), min(i + 2, W_lambda.shape[0])
    j_start, j_end = max(j - 1, 0), min(j + 2, W_lambda.shape[1])
    patch = W_lambda[i_start:i_end, j_start:j_end]

    has_positive = np.any(patch > 0).astype(bool)
    has_negative = np.any(patch < 0).astype(bool)

    return has_positive and has_negative

def find_boundary(W_lambda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the boundary pixels in the thresholded receptive-field matrix.
    :param W_lambda: thresholded receptive-field matrix.
    :return: a binary matrix of the same shape as W_lambda, where 1 indicates a boundary pixel and 0 indicates a non-boundary pixel.
    """
    coordinates = []
    W_boundary = np.zeros_like(W_lambda)
    for x in range(W_lambda.shape[1]):
        for y in range(W_lambda.shape[0]):
            if is_boundary(W_lambda, x, y):
                i,j = xy_to_pixel(x, y)
                W_boundary[i, j] = 1
                coordinates.append(np.array([x, y]))
    return W_boundary, np.array(coordinates)

def linear_regression(X_b: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares line y = α·x + β through a set of boundary pixels.
    :param X_b: Iterable of (x, y) tuples representing pixel coordinates.
    :return: Tuple containing the slope (α̂) and intercept (β̂) of the best-fit line.
    :raises ValueError: If X_b is not an iterable of (x, y) pairs.
    """
    pts = np.asarray(X_b, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("X_b must be an iterable of (x, y) pairs")

    x = pts[:, 0]                      
    y = pts[:, 1]                    

    A = np.column_stack((x, np.ones_like(x)))

    alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]

    return alpha, beta

def find_parameters(X_b: np.ndarray, alpha: float, beta: float, patch_size: int = 16) -> dict:
    """
    Find length, orientation and endpoints with PCA (robust for any tilt).
    X_b : array (N,2) of boundary (x,y) in bottom-left coords.
    """
    X = X_b - X_b.mean(axis=0, keepdims=True)      

    _, _, vt = np.linalg.svd(X, full_matrices=False)
    u = vt[0]                                   

    t = X @ u
    P0 = X_b.mean(0) + t.min()*u
    P1 = X_b.mean(0) + t.max()*u

    dx, dy = P1 - P0
    length      = float(np.hypot(dx, dy))
    orientation = float(np.degrees(np.arctan2(dy, dx)) % 180)
    loc_px      = xy_to_pixel(P0[0], P0[1], patch_size)

    return {
        "length":       length,
        "orientation":  orientation,
        "location_xy":  tuple(P0),
        "end_xy":       tuple(P1),
        "location_px":  loc_px
    }


def draw_segment(params: dict, H: int, ax=None, line_kwargs=None, title=None):
    if line_kwargs is None:
        line_kwargs = dict(color="crimson", linewidth=3, zorder=5)

    x0, y0 = params["location_xy"]
    x1, y1 = params["end_xy"]          

    i0, j0 = xy_to_pixel(x0, y0, H)
    i1, j1 = xy_to_pixel(x1, y1, H)

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlim(-.5, H-.5)
        ax.set_ylim(H-.5, -.5)
        ax.set_aspect("equal")

    ax.scatter(j0, i0, s=60, c="black", zorder=6)
    ax.plot([j0, j1], [i0, i1], **line_kwargs)  
    ax.set_xticks([]); ax.set_yticks([])
    if title is None:
        title = "Parametrised stripe"
    ax.set_title(title)
