import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.feature import peak_local_max
from tqdm import tqdm


def load_images(data_path, frame_nrs, type='tif', lead_0=5, timing=True):
    """
    Load selected .tif images from a directory into a 3D numpy array.

    Args:
        data_path (str): Path to the directory containing .tif images.
        frame_nrs (list of int): List of frame numbers to load.
        type (str): File extension to load.
        lead_0 (int): Number of leading zeros in the file names.
        timing (bool): If True, show a progress bar while loading images.

    Returns:
        imgs (np.ndarray): 3D array of images (image_index, y, x).
    """

    # List all files in the directory
    files = natsorted(
            [f for f in os.listdir(data_path) if f.endswith('.' + type)])

    # Filter files to include only those that match the specified frame numbers
    files = [f for f in files if any(f.endswith(f"{nr:0{lead_0}d}.tif") for nr
                                     in frame_nrs) and not f.startswith('.')]
    if not files:
        raise FileNotFoundError(f"No files found in {data_path} matching "
                                f"frame numbers {frame_nrs} with type '"
                                f"{type}'.")

    # Read images into a 3D numpy array
    imgs = np.array([cv.imread(os.path.join(data_path, f), cv.IMREAD_GRAYSCALE)
                     for f in tqdm(files, desc='Reading images', disable=not
        timing)], dtype=np.uint64)
    return imgs


def downsample(imgs, factor):
    """Downsample a 2D image by summing non-overlapping blocks
     of size (block_size, block_size).

     Args:
        imgs (np.ndarray): 3D array of images (image_index, y, x).
        factor (int): Size of the blocks to sum over.

    Returns:
            np.ndarray: 3D array of downsampled images (image_index, y, x).
         """

    # Get image stack dimensions, check divisibility
    n_img, h, w = imgs.shape
    assert h % factor == 0 and w % factor == 0, \
        "Image dimensions must be divisible by block_size"

    # Reshape the image into blocks and sum over the blocks
    return imgs.reshape(n_img, h // factor, factor,
                        w // factor, factor).sum(axis=(2, 4))


def split_n_shift(img, n_windows, overlap=0, shift=(0, 0),
                  shift_mode='before', plot=False):
    """
    Split a 2D image array (y, x) into (overlapping) windows,
    with optional edge cut-off for shifted images.

    Args:
        img (np.ndarray): 2D array of image values (y, x).
        n_windows (tuple): Number of windows in (y, x) direction.
        overlap (float): Fractional overlap between windows (0 = no overlap).
        shift (tuple): (dy, dx) shift in pixels - can be (0, 0).
        shift_mode (str): 'before' or 'after' shift: which frame is considered?
        plot (bool): If True, plot the windows on the image.

    Returns:
        windows (np.ndarray): 4D array of image windows
            (window_y_idx, window_x_idx, y, x).
        centres (np.ndarray): 3D array of window centres
            (window_y_idx, window_x_idx, 2).
    """
    # Get dimensions
    h, w = img.shape
    n_y, n_x = n_windows
    dy, dx = shift.astype(int)

    # Calculate window size including overlap
    size_y = min(int(h // n_y * (1 + overlap)), h)
    size_x = min(int(w // n_x * (1 + overlap)), w)

    # Get the top-left corner of each window to create grid of window coords
    y_indices = np.linspace(0, h - size_y, num=n_y, dtype=int)
    x_indices = np.linspace(0, w - size_x, num=n_x, dtype=int)
    grid = np.stack(np.meshgrid(y_indices, x_indices, indexing="ij"), axis=-1)

    # Compute centres (window_y_idx, window_x_idx, 2)
    centres = np.stack((grid[:, :, 0] + size_y / 2,
                        grid[:, :, 1] + size_x / 2), axis=-1)

    # Determine cut-off direction: +1 for 'before', -1 for 'after'
    mode_sign = 1 if shift_mode == 'before' else -1

    # Show windows and centres on the image if requested
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img.astype(float) / img.max() * 255, cmap='gray')

    # Pre-allocate and fill the windows
    windows = np.empty((n_y, n_x,
                        size_y - abs(dy), size_x - abs(dx)),
                        dtype=img.dtype)
    # Calculate cut-off for each direction
    cut_y0 = max(0, mode_sign * dy)
    cut_y1 = max(0, -mode_sign * dy)
    cut_x0 = max(0, mode_sign * dx)
    cut_x1 = max(0, -mode_sign * dx)

    for i, y in enumerate(y_indices):
        for j, x in enumerate(x_indices):
            y0 = y + cut_y0
            y1 = y + size_y - cut_y1
            x0 = x + cut_x0
            x1 = x + size_x - cut_x1
            windows[i, j] = img[y0:y1, x0:x1]

            if plot:
                color = ['orange', 'blue'][(i + j) % 2]
                rect = plt.Rectangle((x + cut_x0, y + cut_y0),
                                     x + size_x - cut_x1 - (x + cut_x0),
                                     y + size_y - cut_y1 - (y + cut_y0),
                                     edgecolor=color, facecolor='none',
                                     linewidth=1.5)
                ax.add_patch(rect)
                ax.scatter(centres[i, j, 1], centres[i, j, 0], c=color,
                           marker='x', s=40)

    # Finish plot
    if plot:
        plt.xlim(-20, w + 20)
        plt.ylim(-20, h + 20)
        plt.show()

    return windows, centres


def find_peaks(corr_map, num_peaks=1, min_distance=5):
    """
    Find peaks in a correlation map.

    Args:
        corr_map (np.ndarray): 2D array of correlation values.
        num_peaks (int): Number of peaks to find.
        min_distance (int): Minimum distance between peaks in pixels.

    Returns:
        peaks (np.ndarray): Array of peak coordinates shaped (num_peaks, 2)
        intensities (np.ndarray): Intensities of the found peaks.
    """

    if num_peaks == 1:
        # Find the single peak
        peaks = np.argwhere(np.amax(corr_map) == corr_map).astype(np.float64)
    else:
        # Find multiple peaks using peak_local_max
        peaks = peak_local_max(corr_map, min_distance=min_distance,
                               num_peaks=num_peaks, exclude_border=True).astype(np.float64)

    # If a smaller number of peaks is found, pad with NaNs
    if peaks.shape[0] < num_peaks:
        peaks = np.pad(peaks, ((0, num_peaks - peaks.shape[0]), (0, 0)),
                       mode='constant', constant_values=np.nan)

    # Calculate the intensities of the peaks
    if np.all(np.isnan(peaks)):
        intensities = np.array([np.nan] * num_peaks)
    else:
        intensities = corr_map[peaks[:, 0].astype(int), peaks[:, 1].astype(int)]

    return peaks, intensities


def remove_outliers(coords, y_max, x_max, strip=True):
    """
    Remove outliers from a list of coordinates based on specified limits.
    - For x < 0: keep only points inside a semicircle of radius y_max
    centered at (0,0)
    - For x >= 0: keep only points inside a rectangle [-0.5, x_max] x [
    -y_max, y_max]

    Args:
        coords (np.ndarray): 2D or 3D array of coordinates (y, x).
        y_max (float): Maximum y-coordinate for the semicircle.
        x_max (float): Maximum x-coordinate for the rectangle.
        strip (bool): If True, reduce the array to 2D by taking only the first
                      non-NaN coordinate.
    """

    # Coords might be an 3D array. Reshape it to 2D for processing
    orig_shape = coords.shape
    coords = coords.reshape(-1, coords.shape[-1])

    # Set all non-valid coordinates to NaN
    mask = (((coords[:, 1] < 0) &
             (coords[:, 1] ** 2 + coords[:, 0] ** 2 <= y_max ** 2))
            | ((coords[:, 1] >= 0) & (coords[:, 1] <= x_max)
               & (np.abs(coords[:, 0]) <= y_max)))
    coords[~mask] = np.array([np.nan, np.nan])

    # Reshape back to original shape
    coords = coords.reshape(orig_shape)

    # If needed, reduce the array to 2D by taking only the first non-NaN
    # coordinate
    if strip and coords.ndim > 2:
        coords_stripped = np.full([coords.shape[0], 2], np.nan,
                                  dtype=np.float64)
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                if ~np.any(coords[i, j, :] == np.nan):
                    # If there are non NaNs, save these coordinates
                    coords_stripped[i, :] = coords[i, j, :]
                    break
                elif j == coords.shape[1] - 1:
                    # If all coordinates are NaN, set to NaN
                    coords_stripped[i, :] = np.array([np.nan, np.nan])
        coords = coords_stripped

    return coords


def three_point_gauss(array):
    """
    Fit a Gaussian to three points.

    Args:
        array (np.ndarray): 1D array of three points, peak in the middle.
    Returns:
        float: Subpixel correction value.
    """

    # Check if the input is a 1D array
    if array.ndim != 1 or array.shape[0] != 3:
        raise ValueError("Input must be a 1D array with exactly three elements.")

    # Calculate the subpixel correction using the Gaussian fit formula
    return (0.5 * (np.log(array[0]) - np.log(array[2])) /
            ((np.log(array[0])) + np.log(array[2]) - 2 * np.log(array[1])))


def subpixel(corr_map, peak):
    """
    Use a Gaussian fit to refine the peak coordinates.

    Args:
        corr_map (np.ndarray): 2D array of correlation values.
        peak (np.ndarray): Coordinates of the peak (y, x).
    Returns:
        np.ndarray: Refined peak coordinates with subpixel correction.
    """

    # Apply three-point Gaussian fit to peak coordinates in two directions
    y_corr = three_point_gauss(corr_map[peak[0] - 1:peak[0] + 2, peak[1]])
    x_corr = three_point_gauss(corr_map[peak[0], peak[1] - 1:peak[1] + 2])

    # Add subpixel correction to the peak coordinates
    return peak.astype(np.float64) + np.array([y_corr, x_corr])