import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from scipy.interpolate import make_smoothing_spline
from skimage.feature import peak_local_max
from tqdm import trange, tqdm


def backup(mode: str, proc_path: str, filename: str, var_names=None, test_mode=False, **kwargs) -> tuple[bool, dict]:
    """
    Load or save a backup file from/to the specified path.

    Args:
        mode (str): 'load' or 'save' to specify the operation.
        proc_path (str): Path to the directory containing the backup file.
        filename (str): Name of the backup file to load/save.
        test_mode (bool): If True, do not load/save the file.
        var_names (list): List of variable names to load (for load mode).
        **kwargs: Variables to save (for save mode). Use as: backup("save", path, file, var1=value1, var2=value2, ...)

    Returns:
        For load mode: loaded_vars (dict)
        For save mode: success (bool)
    """
    # If in test mode, return appropriate values
    if test_mode:
        return False, {}

    # Load mode
    elif mode == 'load':
        # Check if the file exists
        filepath = os.path.join(proc_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: backup file {filename} not found in {proc_path}.")
            return False, {}

        # Load the data from the .npz file
        else:
            loaded_vars = {}
            with np.load(filepath) as data:
                if var_names is None:
                    # Load all variables in the file
                    for k in data.files:
                        loaded_vars[k] = data[k]
                else:
                    # Load only requested variables
                    for k in var_names:
                        if k in data:
                            loaded_vars[k] = data[k]
                        else:
                            print(f"Warning: {k} not found in {filepath}")
            print(f"Loaded data from {filepath}")
            return True, loaded_vars

    # Save mode
    elif mode == 'save':
        if not kwargs:
            print("Warning: No variables provided for saving.")
            return False, {}
        
        # Save the variables to a .npz file
        filepath = os.path.join(proc_path, filename)
        np.savez(filepath, **kwargs)
        print(f"Saved data to {filepath}")
        return True, {}

    # If mode is not recognized, return False
    else:
        print(f"Error: Unrecognized mode '{mode}'. Use 'load' or 'save'.")
        return False


def load_images(data_path, frame_nrs, format='tif', lead_0=5, timing=True):
    """
    Load selected .tif images from a directory into a 3D numpy array.

    Args:
        data_path (str): Path to the directory containing .tif images.
        frame_nrs (list of int): List of frame numbers to load.
        format (str): File extension to load.
        lead_0 (int): Number of leading zeros in the file names.
        timing (bool): If True, show a progress bar while loading images.

    Returns:
        imgs (np.ndarray): 3D array of images (image_index, y, x).
    """

    # Check if the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # List all files in the directory
    files = natsorted(
            [f for f in os.listdir(data_path) if f.endswith('.' + format)])

    # Filter files to include only those that match the specified frame numbers
    files = [f for f in files if any(f.endswith(f"{nr:0{lead_0}d}.tif") for nr
                                     in frame_nrs) and not f.startswith('.')]
    if not files:
        raise FileNotFoundError(f"No files found in {data_path} matching "
                                f"frame numbers {frame_nrs} with type '"
                                f"{format}'.")

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
    mode_sign = 1 if shift_mode == 'after' else -1

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
    intensities = np.full(num_peaks, np.nan)
    
    # Only calculate intensities for valid (non-NaN) peaks
    valid_mask = ~np.isnan(peaks).any(axis=1)
    if np.any(valid_mask):
        valid_peaks = peaks[valid_mask]
        intensities[valid_mask] = corr_map[valid_peaks[:, 0].astype(int), valid_peaks[:, 1].astype(int)]

    return peaks, intensities


def filter_outliers(mode, coords, a=None, b=None):
    """
    Remove outliers from coordinates based on spatial and intensity criteria.
    
    Args:
        mode (str): Filtering mode:
            - 'semicircle_rect': Filter a semicircle and rectangle
            - 'circle': Filter a circle at the origin
            - 'intensity': Filter based on intensity values of the provided peaks
        coords (np.ndarray): ND coordinate array of shape (..., 2)
        a (float or np.ndarray): Radius for filtering (or intensity values in 'intensity' mode)
        b (float): Width for rectangle filtering or relative threshold in 'intensity' mode       
    
    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN
    """
    
    # Store original shape and flatten for processing
    orig_shape = coords.shape
    coords = coords.reshape(-1, coords.shape[-1])
    
    # OPTION 1: filter a semicircle of radius a (x < 0)
    # and a rectangle of height 2a and width b (x >= 0).
    if mode == 'semicircle_rect':
        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' parameters must be provided for "
                             "'semicircle_rect' mode.")
        mask = (((coords[..., 1] < 0) &
                 (coords[..., 1] ** 2 + coords[:, 0] ** 2 <= a ** 2))
                | ((coords[:, 1] >= 0) & (coords[:, 1] <= b)
                   & (np.abs(coords[:, 0]) <= a)))
    
    # OPTION 2: filter a circle at the origin with radius a
    elif mode == 'circle':
        # TODO: Test circular filtering
        mask = (coords[:, 0] ** 2 + coords[:, 1] ** 2 <= a ** 2)

        if b is not None:
            print("Warning: 'b' parameter is ignored in 'circle' mode.")

    # OPTION 3: filter based on intensity values of the provided peaks
    elif mode == 'intensity':
        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' parameters must be provided for "
                             "'intensity' mode.")
        
        # Check that a is an ND numpy array of intensity values
        if not isinstance(a, np.ndarray) or a.shape != orig_shape[:-1]:
            raise ValueError("Parameter 'a' must be an ND numpy array of intensity values matching the shape of the coordinates.")
        if not isinstance(b, (int, float)):
            raise ValueError("Parameter 'b' must be an integer or float representing the relative threshold.")

        # Reshape a to match the flattened coords
        ints = a.reshape(-1)

        # Calculate the intensity threshold
        int_min = b * np.nanmax(ints)

        # Create a mask based on the intensity threshold
        mask = (ints >= int_min)

    else:
        raise ValueError(f"Unknown filtering mode: {mode}")

    # Apply the mask to the coordinates*
    coords[~mask] = np.array([np.nan, np.nan])   
    
    # Reshape back to original shape
    coords = coords.reshape(orig_shape)
    
    # In intensity mode, also return the filtered intensities
    if mode == 'intensity':
        ints[~mask] = np.nan
        return coords, ints
    else:
        return coords


def cart2polar(coords):
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        coords (np.ndarray): ND array of shape (..., 2) with (y, x) coordinates.

    Returns:
        np.ndarray: ND array of shape (n_corrs, 2) with (r, phi) coordinates.
    """
    # Calculate the magnitude and angle
    r = np.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2)
    phi = np.arctan2(coords[..., 1], coords[..., 0])

    # Stack the results to form a new array
    polar_coords = np.stack((r, phi), axis=-1)
    return polar_coords


def filter_neighbours(coords, thr=1, n_nbs=2):
    """
    Filter out coordinates that are too different from their neighbours.

    Args:
        coords (np.ndarray): 4D coordinate array of shape (n_corrs, n_wins_y, n_wins_x, 2).
        thr (float): Threshold; how many standard deviations can a point be away from its neighbours.
        nr_neighbours (int): Number of neighbours in each dimension to consider for filtering.

    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN.
    """

    # Number of neighbours can be specified in each dimension and must be even
    if isinstance(n_nbs, int):
        n_nbs = (n_nbs, n_nbs, n_nbs)
    elif isinstance(n_nbs, tuple) and len(n_nbs) == 3:
        n_nbs = n_nbs
    else:
        raise ValueError("n_nbs must be integer or a tuple of three integers.")
    if any(n % 2 != 0 for n in n_nbs):
        raise ValueError("n_nbs must be even in each dimension.")

    # Create a copy for output
    coords_output = coords.copy()
    
    # Get a sliding window view of the input coordinates
    n_corrs, n_wins_y, n_wins_x, _ = coords.shape
    nbs = np.lib.stride_tricks.sliding_window_view(coords,
    (n_nbs[0] + 1, n_nbs[1] + 1, n_nbs[2] + 1, 1))[..., 0]

    # Iterate over each coordinate, first spatially, then temporally
    for i in range(n_corrs):
        for j in range(n_wins_y):
            for k in range(n_wins_x):
                # Edge handling: clamp to valid sliding window range
                i_nbs = np.clip(i, n_nbs[0], n_corrs - n_nbs[0] - 1) - n_nbs[0]
                j_nbs = np.clip(j, n_nbs[1], n_wins_y - n_nbs[1] - 1) - n_nbs[1]
                k_nbs = np.clip(k, n_nbs[2], n_wins_x - n_nbs[2] - 1) - n_nbs[2]

                # Skip if the coordinate is already NaN in the input
                if np.any(np.isnan(coords[i, j, k, :])):
                    # print(f"Skipping coordinate ({i}, j}, {k}) as it is NaN.")
                    continue
                
                # Need minimum number of neighbors for reliable statistics
                # min_neighbors = 3
                # if len(valid_neighbors) < min_neighbors:
                #     # print(f"Skipping coordinate ({i}, {j}, {k}): only {len(valid_neighbors)} valid neighbors (need >= {min_neighbors})")
                #     continue
                
                # Calculate the median and standard deviation
                med = np.median(nbs[i_nbs, j_nbs, k_nbs], axis=(1, 2, 3))
                std = np.nanstd(nbs[i_nbs, j_nbs, k_nbs], axis=(1, 2, 3))
                # print(f"Processing coordinate ({i}, {j}, {k}): median={med}, std={std}, n_neighbors={len(valid_neighbors)}")

                # Check if the current coordinate is within the threshold
                if not np.all(np.abs(coords[i, j, k, :] - med) <= thr * std):
                    # print(f"(Filtered out: {coords_input[i, j, k, 0]}, {coords_input[i, j, k, 1]} not within {thr} std from median {med})")
                    coords_output[i, j, k, :] = (np.nan, np.nan)
  

    return coords_output


def first_valid(arr):
    """
    Function to find the first non-NaN value in an array
    """    

    # Check if the input is a 1D array
    if arr.ndim == 1:
        for c in arr:
            if not np.isnan(c):
                return c
        # If no valid value found, return NaN
        return np.nan
    
    # Throw an error if the input is not 1D
    else:
        raise ValueError("Input must be a 1D array.")


def strip_peaks(coords, axis=-2):
    """
    Reduce array dimensionality by selecting the first valid peak along an axis containing options.
    
    Args:
        coords (np.ndarray): N-D array where one axis represents different peaks
        axis (int): Axis along which to reduce the array
            (default: second-to-last axis)
    
    Returns:
        np.ndarray: Array with one axis reduced
    """

    if coords.ndim < 3:
        return coords  # Nothing to strip

    # Apply the first_valid function along the specified axis  
    coords = np.apply_along_axis(first_valid, axis, coords)
    return coords


def replace_nans():
    """Function to replace NaN values with weighted average of neighbours (merge with strip_peaks?)"""
    return

def smooth(time, disps, col='both', lam=5e-7, type=int):
    """
    Smooth displacement data along a specified axis using a smoothing spline.
    
    Args:
        time (np.ndarray): 1D array of time values.
        disps (np.ndarray): 2D array of displacement values.
        col (str or int): Column to smooth:
            - 'both': Smooth both columns (y and x displacements).
            - int: Index of the column to smooth (0 for y, 1 for x).
        lam (float): Smoothing parameter.
        type (type): Type to convert the smoothed displacements to.

    Returns:
        tuple: Tuple containing:
            - time (np.ndarray): 1D array of time values.
            - disps (np.ndarray): 2D array of smoothed displacements.
    """

    # Work on copy
    disps_spl = disps.copy()
    orig_shape = disps_spl.shape

    # Try to squeeze displacements array, then check if 2D
    disps_spl = disps_spl.squeeze() if disps_spl.ndim > 2 else disps_spl
    if disps_spl.ndim != 2:
        raise ValueError("disps must be a 2D array with shape (n_time, 2).")

    # Mask any NaN values in the displacements
    mask = ~np.isnan(disps_spl).any(axis=1)

    # If cols is 'both', apply smoothing to both columns
    if col == 'both':
        for i in range(disps_spl.shape[1]):
            disps_spl[:, i] = make_smoothing_spline(time[mask], disps_spl[mask, i], lam=lam)(time).astype(type)

    # Otherwise, apply smoothing to the specified column
    elif isinstance(col, int):
        disps_spl[:, col] = make_smoothing_spline(time[mask], disps_spl[mask, col], lam=lam)(time).astype(type)
    else:
        raise ValueError("cols must be 'both' or an integer index.")
    
    return disps_spl.reshape(orig_shape)


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


def save_cfig(directory, filename, format='pdf', test_mode=False, verbose=True):
    """
    Save the current matplotlib figure to a file.

    Args:
        directory (str): Directory to save the figure.
        filename (str): Name of the file to save the figure as.
        format (str): File format to save the figure in (e.g., 'pdf', 'png').
        test_mode (bool): If True, do not save the figure.
        verbose (bool): If True, print a message when saving the figure.
    """

    # Only run when not in test mode
    if test_mode:
        return
    
    # Otherwise, save figure
    else:
        # Set directory and file format
        filename = f"{filename}.{format}"
        filepath = os.path.join(directory, filename)

        # Save the figure
        plt.savefig(filepath, transparent=True, bbox_inches='tight',
                    format=format)
        if verbose:
            print(f"Figure saved to {filepath}")

    # # Show the figure
    # plt.show()

    return
