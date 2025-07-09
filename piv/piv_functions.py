import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.feature import peak_local_max
from tqdm import tqdm


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


def filter_outliers(coords, mode, a=None, b=None, intensities=None, int_thr=0):
    """
    Filter outliers from coordinates based on spatial and intensity criteria.
    
    Args:
        coords (np.ndarray): N-D array where last dimension is (y, x) coordinates.
        mode (str): Filtering mode. Options:
            - 'semicircle_rect': For x < 0: semicircle of radius a; For x >= 0: rectangle [-0.5, b] x [-a, a]
            - 'circle': Circle of radius a centered at origin
        a (float): First parameter (y-direction limit or radius)
        b (float): Second parameter (x-direction limit)
        intensities (np.ndarray): Intensity values corresponding to coordinates
        int_thr (float): Minimum intensity threshold for valid coordinates
    
    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN
    """
    
    # Store original shape and flatten for processing
    orig_shape = coords.shape
    coords = coords.reshape(-1, coords.shape[-1])
    
    if mode == 'semicircle_rect':
        # Current implementation: semicircle for x < 0, rectangle for x >= 0
        mask = (((coords[..., 1] < 0) &
                 (coords[..., 1] ** 2 + coords[:, 0] ** 2 <= a ** 2))
                | ((coords[:, 1] >= 0) & (coords[:, 1] <= b)
                   & (np.abs(coords[:, 0]) <= a)))
    
    elif mode == 'circle':
        # TODO: Test circular filtering
        mask = (coords[:, 0] ** 2 + coords[:, 1] ** 2 <= a ** 2)
    
    else:
        raise ValueError(f"Unknown filtering mode: {mode}")
    
    # Apply spatial filtering
    coords[~mask] = np.array([np.nan, np.nan])
    
    # Apply intensity filtering if provided
    if intensities is not None:
        # TODO: Implement intensity filtering
        # Reshape intensities to match flattened coords
        intensities_flat = intensities.reshape(-1)
        low_intensity_mask = intensities_flat < int_thr
        coords[low_intensity_mask] = np.array([np.nan, np.nan])
    
    # Reshape back to original shape
    coords = coords.reshape(orig_shape)
    
    return coords


def filter_outliers_local(coords, mode='temporal', window_size=3, threshold=2.0):
    """
    Filter outliers based on local consistency (temporal or spatial).
    
    Args:
        coords (np.ndarray): Coordinate array
        mode (str): Local filtering mode:
            - 'temporal': Compare with previous/next timesteps
            - 'spatial': Compare with neighboring windows
            - 'peer_peaks': Use other candidate peaks for validation
        window_size (int): Size of the local window for comparison
        threshold (float): Threshold for outlier detection (in standard deviations)
    
    Returns:
        np.ndarray: Filtered coordinates with outliers set to NaN
    """
    # TODO: Implement local filtering modes
    
    if mode == 'temporal':
        # TODO: Implement temporal consistency filtering
        # Could use rolling window statistics or interpolation-based detection
        raise NotImplementedError("Temporal filtering not yet implemented")
    
    elif mode == 'spatial':
        # TODO: Implement spatial consistency filtering
        # Compare with neighboring windows in the same frame
        raise NotImplementedError("Spatial filtering not yet implemented")
    
    elif mode == 'peer_peaks':
        # TODO: Implement peer peak validation
        # Use other candidate peaks to validate the primary peak
        raise NotImplementedError("Peer peak filtering not yet implemented")
    
    else:
        raise ValueError(f"Unknown local filtering mode: {mode}")
    
    return coords


def strip_peaks(coords, mode='first_valid'):
    """
    Reduce array dimensionality by selecting peaks along the second-to-last axis.
    
    Args:
        coords (np.ndarray): N-D array where second-to-last axis represents different peaks
        mode (str): Peak selection mode:
            - 'first_valid': Take first non-NaN peak
            - 'best_intensity': Take peak with highest intensity (requires intensities)
            - 'median': Take median of valid peaks
            - 'mean': Take mean of valid peaks
    
    Returns:
        np.ndarray: Array with second-to-last axis reduced
    """
    if coords.ndim < 3:
        return coords  # Nothing to strip
    
    if mode == 'first_valid':
        # Current implementation
        def first_valid(arr):
            for c in arr:
                if not np.any(np.isnan(c)):
                    return c
            return np.full(arr.shape[-1], np.nan)
        
        return np.apply_along_axis(first_valid, -2, coords)
    
    elif mode == 'best_intensity':
        # TODO: Implement intensity-based peak selection
        # Would need intensities as additional parameter
        raise NotImplementedError("Intensity-based peak selection not yet implemented")
    
    elif mode == 'median':
        # TODO: Implement median peak selection
        raise NotImplementedError("Median peak selection not yet implemented")
    
    elif mode == 'mean':
        # TODO: Implement mean peak selection
        raise NotImplementedError("Mean peak selection not yet implemented")
    
    else:
        raise ValueError(f"Unknown peak selection mode: {mode}")


def smooth_temporal(coords, method='spline', **kwargs):
    """
    Apply temporal smoothing to coordinate time series.
    
    Args:
        coords (np.ndarray): 2D array of coordinates (time, coordinate)
        method (str): Smoothing method:
            - 'spline': Smoothing spline (current implementation)
            - 'gaussian': Gaussian filter
            - 'median': Median filter
            - 'savgol': Savitzky-Golay filter
        **kwargs: Method-specific parameters
    
    Returns:
        np.ndarray: Smoothed coordinates
    """
    # TODO: Implement different smoothing methods
    
    if method == 'spline':
        # TODO: Move spline smoothing from piv.py here
        # Would use scipy.interpolate.make_smoothing_spline
        raise NotImplementedError("Spline smoothing not yet implemented")
    
    elif method == 'gaussian':
        # TODO: Implement Gaussian smoothing
        raise NotImplementedError("Gaussian smoothing not yet implemented")
    
    elif method == 'median':
        # TODO: Implement median filtering
        raise NotImplementedError("Median filtering not yet implemented")
    
    elif method == 'savgol':
        # TODO: Implement Savitzky-Golay filtering
        raise NotImplementedError("Savitzky-Golay filtering not yet implemented")
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
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
    
    # Othrwise, save figure
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
