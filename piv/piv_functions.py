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
        coords (np.ndarray): ND array of shape (..., 2) with (y, x) coordinates

    Returns:
        np.ndarray: ND array of shape (..., 2) with (r, phi) coordinates
    """
    # Calculate the magnitude and angle
    r = np.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2)
    phi = np.arctan2(coords[..., 1], coords[..., 0])

    # Stack the results to form a new array
    polar_coords = np.stack((r, phi), axis=-1)
    return polar_coords


def validate_n_nbs(n_nbs, max_shape=None):
    """
    Validate and process n_nbs parameter for filter_neighbours function.
    
    Args:
        n_nbs (int, str, or tuple): Number of neighbours specification
            - int: Number of neighbours in each dimension (must be even).
            - str: "all" to use the full dimension length.
            - tuple: Three values specifying neighbours in each dimension.
        max (tuple): Shape of the dimensions to use if n_nbs is "all"

    Returns:
        tuple: Processed n_nbs values
    """

    # Convert to list
    if isinstance(n_nbs, (int, str)):
        n_nbs = [n_nbs, n_nbs, n_nbs]
    elif isinstance(n_nbs, tuple):
        n_nbs = list(n_nbs)
    else:
        raise ValueError("n_nbs must be integer, 'all', or a tuple of three values (int or 'all').")
    
    # Process each dimension
    for i, n in enumerate(n_nbs):
        if n == "all":
            # Use dimension length (make it even if necessary)
            n_nbs[i] = max_shape[i] - 2 if max_shape[i] % 2 == 0 else max_shape[i] - 1
        elif isinstance(n, int):
            if n % 2 != 0:
                raise ValueError(f"n_nbs must be even in each dimension. Got {n} for dimension {i}.")
        else:
            raise ValueError(f"Each element of n_nbs must be an integer or 'all'. Got {n} for dimension {i}.")
    
    return tuple(n_nbs)


def filter_neighbours(coords, thr=1, n_nbs=2, mode="xy", replace=False, verbose=False):
    """
    Filter out coordinates that are too different from their neighbours.

    Args:
        coords (np.ndarray): 4D coordinate array of shape (n_corrs, n_wins_y, n_wins_x, 2).
        thr (float): Threshold; how many standard deviations can a point be away from its neighbours.
        n_nbs (int, str, or tuple): Number of neighbours in each dimension to consider for filtering. Can be an integer, "all", or a tuple of three values (int or "all").
        mode (str): Which coordinates should be within std*thr from the median:
            - "x": Compare x coordinates only
            - "y": Compare y coordinates only
            - "xy": Compare both x and y coordinates
            - "r": Compare vector lengths only
        replace (bool): Replace outliers and pre-existing NaN values with the median of neighbours.
        verbose (bool): If True, print additional information during processing.

    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN or replaced with median.
    """

    # Get dimensions and validate n_nbs
    n_corrs, n_wins_y, n_wins_x, _ = coords.shape
    n_nbs = validate_n_nbs(n_nbs, (n_corrs, n_wins_y, n_wins_x))

    # Create a copy for output
    coords_output = coords.copy()
    
    # Get a set of sliding windows around each coordinate
    # Note this function is slow
    nbs = np.lib.stride_tricks.sliding_window_view(coords,
    (n_nbs[0] + 1, n_nbs[1] + 1, n_nbs[2] + 1, 1))[..., 0]

    # Iterate over each coordinate
    for i in range(n_corrs):
        for j in range(n_wins_y):
            for k in range(n_wins_x):
                
                # Edge handling: clamp to valid sliding window range
                i_nbs = np.clip(i, n_nbs[0]//2, n_corrs - n_nbs[0]//2 - 1) - n_nbs[0]//2
                j_nbs = np.clip(j, n_nbs[1]//2, n_wins_y - n_nbs[1]//2 - 1) - n_nbs[1]//2
                k_nbs = np.clip(k, n_nbs[2]//2, n_wins_x - n_nbs[2]//2 - 1) - n_nbs[2]//2

                # Calculate the median and standard deviation first
                med = np.nanmedian(nbs[i_nbs, j_nbs, k_nbs], axis=(1, 2, 3))
                std = np.nanstd(nbs[i_nbs, j_nbs, k_nbs], axis=(1, 2, 3))

                # TODO: make all below more compact
                # Check if the coordinate is already NaN in the input
                is_nan = np.any(np.isnan(coords[i, j, k, :]))
                
                # Else, check if the current coordinate is within the threshold
                if not is_nan:
                    if mode == "x":
                        is_outlier = not np.abs(coords[i, j, k, 1] - med[1]) <= thr * std[1]
                    elif mode == "y":
                        is_outlier = not np.abs(coords[i, j, k, 0] - med[0]) <= thr * std[1]
                    elif mode == "xy":
                        is_outlier = not np.all(np.abs(coords[i, j, k, :] - med) <= thr * std)
                    elif mode == "r":
                        # Calculate the vector length and compare
                        vec_length = np.sqrt(coords[i, j, k, 0] ** 2 + coords[i, j, k, 1] ** 2)
                        med_length = np.sqrt(med[0] ** 2 + med[1] ** 2)
                        is_outlier = not np.abs(vec_length - med_length) <= thr * std.mean()
                    else:
                        raise ValueError(f"Unknown mode: {mode}. Use 'x', 'y', 'xy', or 'r'.")
                else:
                    is_outlier = False
                
                # If verbose, print the status of the coordinate
                if verbose:
                    if is_nan:
                        print(f"Coordinate ({i}, {j}, {k}) is NaN, skipping.")
                    elif is_outlier:
                        print(f"Coordinate ({i}, {j}, {k}) is an outlier: "
                              f"{coords[i, j, k, :]} (med: {med}, std: {std}).")
                    else:
                        print(f"Coordinate ({i}, {j}, {k}) is valid: "
                              f"{coords[i, j, k, :]}.")

                # Apply replacement or filtering logic
                if replace and (is_nan or is_outlier):
                    coords_output[i, j, k, :] = med
                elif not replace and is_outlier:
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


# %% GPT GENERATED PLOTTING CODE: TO BE ADJUSTED

def plot_first_pass_vx(time, vel1_unf, vel1, vel1x_spl, n_peaks1, proc_path, test_mode=False):
    """
    Plot vx velocity over time for first pass analysis.
    
    Args:
        time (np.ndarray): Time array
        vel1_unf (np.ndarray): Unfiltered velocities 
        vel1 (np.ndarray): Filtered velocities
        vel1x_spl (np.ndarray): Smoothed vx velocities
        n_peaks1 (int): Number of peaks
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    fig0, ax0 = plt.subplots()
    ax0.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 1],
                c='gray', s=2, label='Other peaks')
    ax0.scatter(1000 * time, vel1_unf[:, 0, 0, 0, 1], c='blue', s=10,
                label='Most prominent peak')
    ax0.scatter(1000 * time, vel1[:, 0, 0, 1], c='orange', s=4,
                label='After outlier removal')
    ax0.plot(1000 * time, vel1x_spl, color='red',
                label='Displacement to be used\n in 2nd pass (smoothed)')
    ax0.set_ylim([-5, 45])
    ax0.set_xlabel('Time (ms)')
    ax0.set_ylabel('vx (m/s)')
    ax0.legend(loc='upper right', fontsize='small', framealpha=1)

    save_cfig(proc_path, 'disp1_vx_t', test_mode=test_mode)


def plot_first_pass_vy(time, vel1_unf, vel1, n_peaks1, proc_path, test_mode=False):
    """
    Plot vy velocity over time for first pass analysis.
    
    Args:
        time (np.ndarray): Time array
        vel1_unf (np.ndarray): Unfiltered velocities 
        vel1 (np.ndarray): Filtered velocities
        n_peaks1 (int): Number of peaks
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    fig0b, ax0b = plt.subplots()
    ax0b.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 0],
                 c='gray', s=2, label='Other peaks')
    ax0b.scatter(1000 * time, vel1_unf[:, 0, 0, 0, 0], c='blue', s=10,
                 label='Most prominent peak')
    ax0b.scatter(1000 * time, vel1[:, 0, 0, 0], c='orange', s=4,
                 label='After outlier removal')
    ax0b.set_ylim([-5, 45])
    ax0b.set_xlabel('Time (ms)')
    ax0b.set_ylabel('vy (m/s)')
    ax0b.legend(loc='upper right', fontsize='small', framealpha=1)

    save_cfig(proc_path, 'disp1_vy_t', test_mode=test_mode)


def plot_first_pass_vy_vx(vel1, proc_path, test_mode=False):
    """
    Plot vy vs vx scatter plot for first pass analysis.
    
    Args:
        vel1 (np.ndarray): Filtered velocities
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    fig1, ax1 = plt.subplots()
    ax1.scatter(vel1[:, 0, 0, 1], vel1[:, 0, 0, 0], c='blue', s=4)
    ax1.set_xlabel('vx (m/s)')
    ax1.set_ylabel('vy (m/s)')
    save_cfig(proc_path, 'disp1_vy_vx', test_mode=test_mode)


def plot_velocity_field(vel2_unf, vel2, centres, time, sample_frame, n_peaks2, res_avg, proc_path, test_mode=False):
    """
    Plot velocity field for a sample frame from second pass analysis.
    
    Args:
        vel2_unf (np.ndarray): Unfiltered velocities from second pass
        vel2 (np.ndarray): Filtered velocities from second pass
        centres (np.ndarray): Window centres
        time (np.ndarray): Time array
        sample_frame (int): Frame index to plot
        n_peaks2 (int): Number of peaks
        res_avg (float): Average resolution
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Plot velocity vectors at window centres for the sample frame
    if centres is not None:
        # Plot all window centres in gray
        for j in range(n_peaks2):
            valid_mask = ~np.isnan(vel2_unf[sample_frame, :, :, j, :]).any(axis=-1)
            if np.any(valid_mask):
                y_pos, x_pos = np.where(valid_mask)
                ax2.scatter(centres[y_pos, x_pos, 1] * res_avg * 1000, 
                           centres[y_pos, x_pos, 0] * res_avg * 1000, 
                           c='lightgray', s=10, alpha=0.5)
        
        # Plot filtered velocities
        valid_mask = ~np.isnan(vel2[sample_frame, :, :, :]).any(axis=-1)
        if np.any(valid_mask):
            y_pos, x_pos = np.where(valid_mask)
            
            # Create velocity vectors
            u = vel2[sample_frame, y_pos, x_pos, 1]  # vx
            v = vel2[sample_frame, y_pos, x_pos, 0]  # vy
            x_centers = centres[y_pos, x_pos, 1] * res_avg * 1000  # mm
            y_centers = centres[y_pos, x_pos, 0] * res_avg * 1000  # mm
            
            # Plot velocity vectors
            ax2.quiver(x_centers, y_centers, u, v,
                      scale=200, scale_units='xy', angles='xy', 
                      color='blue', alpha=0.8, width=0.003)

    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title(f'Velocity field at t = {time[sample_frame]*1000:.2f} ms')
    ax2.grid(True, alpha=0.3)

    save_cfig(proc_path, 'disp2_velocity_field', test_mode=test_mode)


def plot_velocity_profiles(vel2, centres, time, sample_frame, res_avg, proc_path, test_mode=False):
    """
    Plot velocity profiles along the centerline for second pass analysis.
    
    Args:
        vel2 (np.ndarray): Filtered velocities from second pass
        centres (np.ndarray): Window centres
        time (np.ndarray): Time array
        sample_frame (int): Frame index to plot
        res_avg (float): Average resolution
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot vx vs y
    y_positions = centres[:, 0, 0] * res_avg * 1000  # mm
    vx_profile = vel2[sample_frame, :, 0, 1]  # vx at centerline
    vy_profile = vel2[sample_frame, :, 0, 0]  # vy at centerline
    
    ax3a.plot(vx_profile, y_positions, 'b-o', markersize=4, label='vx')
    ax3a.set_xlabel('vx (m/s)')
    ax3a.set_ylabel('y (mm)')
    fig3.suptitle(f'Velocity profiles at t = {time[sample_frame]*1000:.2f} ms')
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim([-5, 40])  # Set x-limits for vx profile
    ax3a.set_ylim([0, 21])  # Set y-limits for consistency
    
    ax3b.plot(vy_profile, y_positions, 'r-o', markersize=4, label='vy')
    ax3b.set_xlabel('vy (m/s)')
    ax3b.set_ylabel('y (mm)')

    # Use same scaling as ax3a for consistency
    ax3b.set_xlim(ax3a.get_xlim())
    ax3b.set_ylim(ax3a.get_ylim())

    ax3b.grid(True, alpha=0.3)
    
    save_cfig(proc_path, 'disp2_velocity_profiles', test_mode=test_mode)


def plot_second_pass_vy_vx(vel2, proc_path, test_mode=False):
    """
    Plot vy vs vx scatter plot for second pass analysis.
    
    Args:
        vel2 (np.ndarray): Filtered velocities from second pass
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    fig4, ax4 = plt.subplots()
    ax4.scatter(vel2[:, 0, 0, 1], vel2[:, 0, 0, 0], c='blue', s=4)
    ax4.set_xlabel('vx (m/s)')
    ax4.set_ylabel('vy (m/s)')
    save_cfig(proc_path, 'disp2_vy_vx', test_mode=test_mode)


def create_velocity_profiles_video(vel2, centres, time, n_corrs, res_avg, proc_path, test_mode=False):
    """
    Create a video of velocity profiles over time.
    
    Args:
        vel2 (np.ndarray): Filtered velocities from second pass
        centres (np.ndarray): Window centres
        time (np.ndarray): Time array
        n_corrs (int): Number of correlation frames
        res_avg (float): Average resolution
        proc_path (str): Path to save processed data
        test_mode (bool): Test mode flag
    """
    from matplotlib import animation as ani
    
    fig_video, (ax_vx, ax_vy) = plt.subplots(1, 2, figsize=(12, 5))
    writer = ani.FFMpegWriter(fps=10)

    video_path = os.path.join(proc_path, 'disp2.mp4')
    with writer.saving(fig_video, video_path, dpi=150):
        for i in range(n_corrs):
            # Clear both axes
            ax_vx.clear()
            ax_vy.clear()
            
            # Get y positions and velocity profiles for current frame
            y_positions = centres[:, 0, 0] * res_avg * 1000  # mm
            vx_profile = vel2[i, :, 0, 1]  # vx at centerline
            vy_profile = vel2[i, :, 0, 0]  # vy at centerline
            
            # Plot vx profile
            ax_vx.plot(vx_profile, y_positions, 'b-o', markersize=4, label='vx')
            ax_vx.set_xlabel('vx (m/s)')
            ax_vx.set_ylabel('y (mm)')
            ax_vx.grid(True, alpha=0.3)
            ax_vx.set_xlim([-5, 40])
            ax_vx.set_ylim([0, 21])
            
            # Plot vy profile  
            ax_vy.plot(vy_profile, y_positions, 'r-o', markersize=4, label='vy')
            ax_vy.set_xlabel('vy (m/s)')
            ax_vy.set_ylabel('y (mm)')
            ax_vy.grid(True, alpha=0.3)
            ax_vy.set_xlim([-5, 40])
            ax_vy.set_ylim([0, 21])
            
            # Set consistent title
            fig_video.suptitle(f'Velocity profiles at t = {time[i]*1000:.2f} ms')
            
            writer.grab_frame()
    plt.close(fig_video)
    print(f"Figure saved to {video_path}")