import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from skimage.feature import peak_local_max
from tqdm import trange, tqdm


# TODO: Set up module.
""""
piv_functions/
├── __init__.py
├── correlation.py
├── filtering.py
├── io.py
├── plotting.py
├── processing.py
└── utils.py

E.g.:
# Import all functions from submodules
from .io import backup, read_img, read_imgs
from .processing import downsample, split_n_shift
from .correlation import calc_corr, calc_corrs, sum_corr, sum_corrs
from .utils import find_peaks, three_point_gauss, subpixel, find_disp, find_disps
from .filtering import filter_outliers, filter_neighbours, cart2polar, validate_n_nbs, first_valid, strip_peaks
from .plotting import save_cfig
from .processing import smooth

# Optional: define __all__ for explicit exports
__all__ = [
    'backup', 'read_img', 'read_imgs',
    'downsample', 'split_n_shift', 
    'calc_corr', 'calc_corrs', 'sum_corr', 'sum_corrs',
    'find_peaks', 'three_point_gauss', 'subpixel', 'find_disp', 'find_disps',
    'filter_outliers', 'filter_neighbours', 'cart2polar', 'validate_n_nbs', 
    'first_valid', 'strip_peaks', 'smooth', 'save_cfig'
]

"""

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



def read_img(file_path: str) -> np.ndarray | None:
    """
    Read a single image file using OpenCV.

    Args:
        filepath (str): Full path to the image file to load.

    Returns:
        np.ndarray | None: Loaded image as grayscale array, or None if loading failed.
    """
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load {file_path}")
    return img


def read_imgs(data_path: str, frame_nrs: list[int], format: str = 'tif', lead_0: int = 5, timing: bool = True) -> np.ndarray:

    """
    Load selected images from a directory into a 3D numpy array.

    Args:
        data_path (str): Path to the directory containing images.
        frame_nrs (list[int]): List of frame numbers to load.
        format (str): File extension to load.
        lead_0 (int): Number of leading zeros in the file names.
        timing (bool): If True, show a progress bar while loading images.

    Returns:
        np.ndarray: 3D array of images (image_index, y, x).
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

    # Read images into a 3D numpy array in parallel
    file_paths = [os.path.join(data_path, f) for f in files]
    
    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        imgs = list(tqdm(executor.map(read_img, file_paths), 
                        total=len(file_paths), 
                        desc='Reading images'))

    # Convert list of images to a numpy array
    imgs = np.array(imgs, dtype=np.uint64)
    return imgs


def downsample(imgs: np.ndarray, factor: int) -> np.ndarray:
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


def split_n_shift(img: np.ndarray, n_wins: tuple[int, int], overlap: float = 0, shift=(0, 0), shift_mode: str = 'before', plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 2D image array (y, x) into (overlapping) windows,
    with optional edge cut-off for shifted images.

    Args:
        img (np.ndarray): 2D array of image values (y, x).
        n_wins (tuple[int, int]): Number of windows in (y, x) direction.
        overlap (float): Fractional overlap between windows (0 = no overlap).
        shift (tuple[int, int] | np.ndarray): (dy, dx) shift in pixels - can be (0, 0).
        shift_mode (str): 'before' or 'after' shift: which frame is considered?
        plot (bool): If True, plot the windows on the image.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - windows: 4D array of image windows (window_y_idx, window_x_idx, y, x)
            - centres: 3D array of window centres (window_y_idx, window_x_idx, 2)
    """
    # Get dimensions
    h, w = img.shape
    n_y, n_x = n_wins
    dy, dx = np.asarray(shift, dtype=int)

    # Calculate window size including overlap
    size_y = min(int(h // n_y * (1 + overlap)), h)
    size_x = min(int(w // n_x * (1 + overlap)), w)
    # print(f"Window size: {size_y} x {size_x}, overlap: {overlap:.2f}")

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


def calc_corr(i: int, imgs: np.ndarray, n_wins: tuple[int, int], shifts: np.ndarray, overlap: float, centres: np.ndarray) -> dict:
    """
    Calculate correlation maps for a single set of frames.

    Args:
        i (int): Frame index
        imgs (np.ndarray): 3D array of images (frame, y, x)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray): Array of shifts per frame (frame, y_shift, x_shift)
        overlap (float): Fractional overlap between windows (0 = no overlap)
        centres (np.ndarray): Window centres from first frame
        
    Returns:
        dict: Correlation maps for this frame as {(frame, win_y, win_x): (correlation_map, map_center)}
    """
    
    # Split images into windows with shifts
    wnd0, centres_curr = split_n_shift(imgs[i], n_wins, shift=shifts[i], shift_mode='before', overlap=overlap)
    wnd1, _ = split_n_shift(imgs[i + 1], n_wins, shift=shifts[i], shift_mode='after', overlap=overlap)

    frame_corr_maps = {}
    
    # Calculate correlation maps for all windows
    for j in range(n_wins[0]):
        for k in range(n_wins[1]):
            corr_map = sig.correlate(wnd1[j, k], wnd0[j, k], method='fft', mode='same')
            map_center = centres[j, k] if centres is not None else centres_curr[j, k]
            frame_corr_maps[(i, j, k)] = (corr_map, map_center)
    
    return frame_corr_maps


def calc_corrs(imgs: np.ndarray, n_wins: tuple[int, int], shifts: np.ndarray | None = None, overlap: float = 0, ds_fac: int = 1):
    """
    Calculate correlation maps for all frames and windows.

    Args:
        imgs (np.ndarray): 3D array of images (frame, y, x)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray | None): Optional array of shifts per window (frame, y_shift, x_shift). If None, shift zero is used.
        overlap (float): Fractional overlap between windows (0 = no overlap)
        ds_fac (int): Downsampling factor (1 = no downsampling)

    Returns:
        dict: Correlation maps as {(frame, win_y, win_x): (correlation_map, map_center)}
    """
    n_corrs = len(imgs) - 1

    # Apply downsampling if needed
    if ds_fac > 1:
        imgs = downsample(imgs, ds_fac)

    # Handle shifts - default to zero if not provided
    if shifts is None:
        shifts = np.zeros((n_corrs, 2))

    # Get centres from first frame
    _, centres = split_n_shift(imgs[0], n_wins, shift=shifts[0], shift_mode='before', overlap=overlap)

    # Prepare arguments for multithreading
    calc_corr_partial = partial(calc_corr, imgs=imgs, n_wins=n_wins, shifts=shifts, overlap=overlap, centres=centres)

    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(executor.map(calc_corr_partial, range(n_corrs)), 
                                 total=n_corrs, 
                                 desc='Calculating correlation maps'))
    
    # Combine results from all frames
    corr_maps = {}
    for frame_result in frame_results:
        corr_maps.update(frame_result)

    return corr_maps


def sum_corr(i: int, corrs: dict, shifts: np.ndarray, n_tosum: int, n_wins: tuple[int, int], n_corrs: int) -> dict:
    """
    Sum correlation maps for a single set of frames.

    Args:
        i (int): Frame index
        corrs (dict): Correlation maps from calc_corrs
        shifts (np.ndarray): 2D array of shifts (frame, y_shift, x_shift)
        n_tosum (int): Number of correlation maps to sum
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        n_corrs (int): Total number of correlation frames
        
    Returns:
        dict: Summed correlation maps for this frame as {(frame, win_y, win_x): (summed_map, new_center)}
    """
    
    # Calculate window bounds for summing: odd = symmetric, even = asymmetric
    i0 = max(0, i - (n_tosum - 1) // 2)
    i1 = min(n_corrs, i + n_tosum // 2 + 1)
    ref = shifts[i]  # Reference shift for current frame
    
    frame_corrs_sum = {}
    
    for j in range(n_wins[0]):
        for k in range(n_wins[1]):

            # Collect all correlation maps to sum and their relative shifts
            maps = []
            sfts = []

            for f in range(i0, i1):
                # Calculate shift difference (in pixels) relative to reference
                d = np.round(shifts[f] - ref).astype(int)

                # Extract correlation map from tuple (corr_map, map_center)
                corr_map, _ = corrs[(f, j, k)]
                maps.append(corr_map)
                sfts.append(d)

            if len(maps) == 1:
                # Only one map to sum, no alignment needed
                smap = maps[0]
                ctr = np.array(smap.shape) // 2
            else:
                # Calculate the expanded size needed to fit all shifted maps
                sh0 = maps[0].shape
                mn = np.min(sfts, axis=0)
                mx = np.max(sfts, axis=0)
                nshape = (sh0[0] + mx[0] - mn[0], sh0[1] + mx[1] - mn[1])

                # Calculate new center position in expanded map
                ctr = (sh0[0] // 2 - mn[0], sh0[1] // 2 - mn[1])
                smap = np.zeros(nshape)

                # Add each map at its shifted position in the expanded array
                for m, s in zip(maps, sfts):
                    # Calculate start and end indices for placement
                    sy, sx = s - mn
                    ey, ex = sy + m.shape[0], sx + m.shape[1]
                    smap[sy:ey, sx:ex] += m

            # Store the summed map and its center for this window
            frame_corrs_sum[(i, j, k)] = (smap, ctr)
    
    return frame_corrs_sum


def sum_corrs(corrs: dict, shifts: np.ndarray, n_tosum: int, n_wins: tuple[int, int]) -> dict:
    """
    Sum correlation maps with windowing and alignment.

    Args:
        corrs (dict): Correlation maps from calc_corrs as {(frame, win_y, win_x): (correlation_map, map_center)}
        shifts (np.ndarray): 2D array of shifts (frame, y_shift, x_shift)
        n_tosum (int): Number of correlation maps to sum (1 = no summation, even = asymmetric)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)

    Returns:
        dict: Summed correlation maps as {(frame, win_y, win_x, k): (summed_map, new_center)}
    """
    
    # Verify that n_tosum is a positive integer
    if n_tosum < 1 or not isinstance(n_tosum, int):
        raise ValueError("n_tosum must be a positive integer")

    # Always use dictionary storage
    n_corrs = max(key[0] for key in corrs.keys()) + 1
    
    # Prepare arguments for multithreading
    sum_corr_partial = partial(sum_corr, corrs=corrs, shifts=shifts, n_tosum=n_tosum, n_wins=n_wins, n_corrs=n_corrs)
    
    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(executor.map(sum_corr_partial, range(n_corrs)), 
                                 total=n_corrs, 
                                 desc='Summing correlation maps'))
    
    # Combine results from all frames
    corrs_sum = {}
    for frame_result in frame_results:
        corrs_sum.update(frame_result)
    
    return corrs_sum
    

def find_peaks(corr: np.ndarray, n_peaks: int = 1, min_dist: int = 5, floor: float | None = None):

    """
    Find peaks in a correlation map.

    Args:
        corr (np.ndarray): 2D array of correlation values.
        n_peaks (int): Number of peaks to find.
        min_dist (int): Minimum distance between peaks in pixels.
        floor (float | None): Optional floor threshold for peak detection.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - peaks: Array of peak coordinates shaped (n_peaks, 2)
            - intensities: Intensities of the found peaks.
    """

    # Based on the median of the correlation map, set a floor
    if floor is not None:
        floor = floor * np.nanmedian(corr, axis=None)

        # # Check whether the floor is below the standard deviation
        # if floor < np.nanstd(corr_map, axis=None):
        #     print(f"Warning: floor {floor} is above the standard deviation.")

    if n_peaks == 1:
        # Find the single peak
        peaks = np.argwhere(np.amax(corr) == corr).astype(np.float64)
    else:
        # Find multiple peaks using peak_local_max
        peaks = peak_local_max(corr, min_distance=min_dist,
                               num_peaks=n_peaks, exclude_border=True, threshold_abs=floor).astype(np.float64)

    # If a smaller number of peaks is found, pad with NaNs
    if peaks.shape[0] < n_peaks:
        peaks = np.pad(peaks, ((0, n_peaks - peaks.shape[0]), (0, 0)),
                       mode='constant', constant_values=np.nan)

    # Calculate the intensities of the peaks
    ints = np.full(n_peaks, np.nan)
    
    # Only calculate intensities for valid (non-NaN) peaks
    valid_mask = ~np.isnan(peaks).any(axis=1)
    if np.any(valid_mask):
        valid_peaks = peaks[valid_mask]
        ints[valid_mask] = corr[valid_peaks[:, 0].astype(int), valid_peaks[:, 1].astype(int)]

    return peaks, ints


def three_point_gauss(array: np.ndarray) -> float:

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

    # Check if middle value is not the peak
    if array[1] < array[0] or array[1] < array[2]:
        raise ValueError("Middle value must be the peak of the three-point array.")
    
    # Shortcut for the symmetric case
    if array[0] == array[2]:
        return 0.0
    
    # Replace any zero values with 1 to avoid log(0) issues
    array1 = np.where(array <= 0, 1, array)
    
    # Calculate the denominator (PIV book §5.4.5)
    den = (np.log(array1[0]) + np.log(array1[2]) - 2 * np.log(array1[1]))
    
    # If the denominator is too small, return 0 to avoid division by zero
    if np.abs(den) < 1e-10:
        return 0.0
    else:
        return (0.5 * (np.log(array1[0]) - np.log(array1[2])) / den)


def subpixel(corr: np.ndarray, peak: np.ndarray) -> np.ndarray:

    """
    Use a Gaussian fit to refine the peak coordinates.

    Args:
        corr (np.ndarray): 2D array of correlation values.
        peak (np.ndarray): Coordinates of the peak (y, x).

    Returns:
        np.ndarray: Refined peak coordinates with subpixel correction.
    """

    # Apply three-point Gaussian fit to peak coordinates in two directions
    y_corr = three_point_gauss(corr[peak[0] - 1:peak[0] + 2, peak[1]])
    x_corr = three_point_gauss(corr[peak[0], peak[1] - 1:peak[1] + 2])

    # Add subpixel correction to the peak coordinates
    return peak.astype(np.float64) + np.array([y_corr, x_corr])


def find_disp(i: int, corrs: dict, shifts: np.ndarray, n_wins: tuple[int, int], n_peaks: int, ds_fac: int, subpx: bool = False, **find_peaks_kwargs) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Find peaks and calculate displacements for a single correlation map.
    
    Args:
        i (int): Correlation map index
        corrs (dict): Correlation maps as {(frame, win_y, win_x): (correlation_map, map_center)}
        shifts (np.ndarray): 2D array of shifts (frame, y_shift, x_shift)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        n_peaks (int): Number of peaks to find
        ds_fac (int): Downsampling factor to account for in displacement calculation
        subpx (bool): If True, apply subpixel accuracy using Gaussian fitting
        **find_peaks_kwargs: Additional arguments for find_peaks function (min_distance, floor, etc.)
        
    Returns:
        tuple: (frame_index, frame_disps, frame_ints) for this frame
    """
    
    # Get reference shift (from current frame) - same for all windows
    ref_shift = shifts[i]
    
    # Initialize output arrays for this frame
    frame_disps = np.full((n_wins[0], n_wins[1], n_peaks, 2), np.nan)
    frame_ints = np.full((n_wins[0], n_wins[1], n_peaks), np.nan)
    
    for j in range(n_wins[0]):
        for k in range(n_wins[1]):
            # Get correlation map and center
            corr_map, map_center = corrs[(i, j, k)]
            
            # Find peaks in the correlation map
            peaks, peak_ints = find_peaks(corr_map, n_peaks=n_peaks, 
                                               **find_peaks_kwargs)
            
            # Apply subpixel correction if requested
            if subpx:
                for p in range(n_peaks):
                    if not np.isnan(peaks[p]).any():  # Only apply to valid peaks
                        # Check if peak is not on the edge (needed for subpixel correction)
                        peak_y, peak_x = peaks[p].astype(int)
                        if (peak_y > 0 and peak_y < corr_map.shape[0] - 1 and 
                            peak_x > 0 and peak_x < corr_map.shape[1] - 1):
                            peaks[p] = subpixel(corr_map, peaks[p].astype(int))
            
            # Store intensities
            frame_ints[j, k, :] = peak_ints
            
            # Calculate displacements for all peaks
            frame_disps[j, k, :, :] = (ref_shift + 
                                          (peaks - map_center) * ds_fac)
    
    return i, frame_disps, frame_ints


def find_disps(corrs: dict, shifts: np.ndarray, n_wins: tuple[int, int], n_peaks: int, ds_fac: int = 1, subpx: bool = False, **find_peaks_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in correlation maps and calculate displacements.

    Args:
        corrs (dict): Correlation maps as {(frame, win_y, win_x): (correlation_map, map_center)}
        shifts (np.ndarray): 2D array of shifts (frame, y_shift, x_shift)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        n_peaks (int): Number of peaks to find
        ds_fac (int): Downsampling factor to account for in displacement calculation
        subpx (bool): If True, apply subpixel accuracy using Gaussian fitting
        **find_peaks_kwargs: Additional arguments for find_peaks function (min_distance, floor, etc.)

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - disps: 4D array (frame, win_y, win_x, peak, 2)
            - ints: 3D array (frame, win_y, win_x, peak)
    """
    
    # Determine number of frames from dictionary keys
    n_corrs = max(key[0] for key in corrs.keys()) + 1
    
    # Initialize output arrays
    disps = np.full((n_corrs, n_wins[0], n_wins[1], n_peaks, 2), np.nan)
    ints = np.full((n_corrs, n_wins[0], n_wins[1], n_peaks), np.nan)
    
    # Prepare arguments for multithreading
    find_disp_partial = partial(find_disp, corrs=corrs, shifts=shifts, n_wins=n_wins, n_peaks=n_peaks, ds_fac=ds_fac, subpx=subpx, **find_peaks_kwargs)

    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(executor.map(find_disp_partial, range(n_corrs)), 
                                 total=n_corrs, 
                                 desc='Finding peaks'))
    
    # Combine results from all frames
    for frame_idx, frame_disps, frame_ints in frame_results:
        disps[frame_idx] = frame_disps
        ints[frame_idx] = frame_ints
    
    return disps, ints
    

def filter_outliers(mode: str, coords: np.ndarray, a: float | np.ndarray | None = None, b: float | None = None, verbose: bool = False):

    """
    Remove outliers from coordinates based on spatial and intensity criteria.

    Args:
        mode (str): Filtering mode:
            - 'semicircle_rect': Filter a semicircle and rectangle
            - 'circle': Filter a circle at the origin
            - 'intensity': Filter based on intensity values of the provided peaks
        coords (np.ndarray): ND coordinate array of shape (..., 2)
        a (float | np.ndarray | None): Radius for filtering (or intensity values in 'intensity' mode)
        b (float | None): Width for rectangle filtering or relative threshold in 'intensity' mode  
        verbose (bool): If True, print summary of filtering.     

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]:
            - Filtered coordinates with invalid points set to NaN
            - (if mode == 'intensity') also returns filtered intensities
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
    
    if verbose:
        # Print summary statistics
        print(f"Post-processing: global filter removed {np.sum(~mask)} out of {coords.shape[0]} coordinates in mode '{mode}'")

    # Reshape back to original shape
    coords = coords.reshape(orig_shape)
    
    # In intensity mode, also return the filtered intensities
    if mode == 'intensity':
        ints[~mask] = np.nan
        return coords, ints
    else:
        return coords


def cart2polar(coords: np.ndarray) -> np.ndarray:

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


def validate_n_nbs(n_nbs: int | str | tuple[int, int, int], max_shape: tuple[int, int, int] | None = None):

    """
    Validate and process n_nbs parameter for filter_neighbours function.

    Args:
        n_nbs (int | str | tuple): Neighbourhood size specification (including center point)
            - int: Neighbourhood size in each dimension (must be odd).
            - str: "all" to use the full dimension length.
            - tuple: Three values specifying neighbourhood size in each dimension.
        max_shape (tuple[int, int, int] | None): Shape of the dimensions to use if n_nbs is "all"

    Returns:
        tuple[int, int, int]: Processed n_nbs values (neighbourhood sizes)
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
            # Use dimension length (make it odd if necessary)
            n_nbs[i] = max_shape[i] - 1 if max_shape[i] % 2 == 0 else max_shape[i]
        elif isinstance(n, int):
            if n % 2 == 0:
                raise ValueError(f"n_nbs must be odd in each dimension (neighbourhood size including center). Got {n} for dimension {i}.")
        else:
            raise ValueError(f"Each element of n_nbs must be an integer or 'all'. Got {n} for dimension {i}.")
    
    return tuple(n_nbs)


def filter_neighbours(coords: np.ndarray, thr: float = 1, n_nbs: int | str | tuple[int, int, int] = 3, mode: str = "xy", replace: bool = False, verbose: bool = False):

    """
    Filter out coordinates that are too different from their neighbours.

    Args:
        coords (np.ndarray): 4D coordinate array of shape (n_corrs, n_wins_y, n_wins_x, 2).
        thr (float): Threshold; how many standard deviations can a point be away from its neighbours.
        n_nbs (int | str | tuple): Size of neighbourhood in each dimension to consider for filtering (including center point). Can be an integer, "all", or a tuple of three values (int or "all").
        mode (str): Which coordinates should be within std*thr from the median:
            - "x": Compare x coordinates only
            - "y": Compare y coordinates only
            - "xy": Compare both x and y coordinates
            - "r": Compare vector lengths only
        replace (bool): Replace outliers and pre-existing NaN values with the median of neighbours.
        verbose (bool): If True, print summary statistics about filtering.

    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN or replaced with median.
    """

    # Get dimensions and validate n_nbs
    n_corrs, n_wins_y, n_wins_x, _ = coords.shape
    n_nbs = validate_n_nbs(n_nbs, (n_corrs, n_wins_y, n_wins_x))

    # Create a copy for output
    coords_output = coords.copy()
    
    # Initialize counters for verbose mode
    if verbose:
        outlier_count = 0
        nan_replaced_count = 0
        outlier_replaced_count = 0
    
    # Get a set of sliding windows around each coordinate
    # Note this function is slow
    nbs = np.lib.stride_tricks.sliding_window_view(coords,
    (n_nbs[0], n_nbs[1], n_nbs[2], 1))[..., 0]

    # Iterate over each coordinate
    for i in range(n_corrs):
        for j in range(n_wins_y):
            for k in range(n_wins_x):

                # First handle the coordinates at the edges, which are not in the centre of a neighbourhood
                i_nbs = (np.clip(i, (n_nbs[0] - 1)//2, n_corrs - (n_nbs[0] - 1)//2 - 1) 
                         - (n_nbs[0] - 1)//2)
                j_nbs = (np.clip(j, (n_nbs[1] - 1)//2, n_wins_y - (n_nbs[1] - 1)//2 - 1) 
                         - (n_nbs[1] - 1)//2)
                k_nbs = (np.clip(k, (n_nbs[2] - 1)//2, n_wins_x - (n_nbs[2] - 1)//2 - 1) 
                         - (n_nbs[2] - 1)//2)
                nb = nbs[i_nbs, j_nbs, k_nbs]

                # Calculate the median and standard deviation
                med = np.nanmedian(nb, axis=(1, 2, 3))
                std = np.nanstd(nb, axis=(1, 2, 3))

                # Check if the coordinate is already NaN in the input
                coord = coords[i, j, k, :]
                is_nan = np.any(np.isnan(coord))
                
                # Check if the current coordinate is an outlier
                is_outlier = False
                if not is_nan:
                    if mode == "x":
                        is_outlier = np.abs(coord[1] - med[1]) > thr * std[1]
                    elif mode == "y":
                        is_outlier = np.abs(coord[0] - med[0]) > thr * std[0]
                    elif mode == "xy":
                        is_outlier = np.any(np.abs(coord - med) > thr * std)
                    elif mode == "r":
                        vec_length = np.linalg.norm(coord)
                        med_length = np.linalg.norm(med)
                        is_outlier = np.abs(vec_length - med_length) > thr * std.mean()
                    else:
                        raise ValueError(f"Unknown mode: {mode}. Use 'x', 'y', 'xy', or 'r'.")
                
                # Update counters for verbose mode
                if verbose:
                    if is_outlier:
                        outlier_count += 1
                        if replace:
                            outlier_replaced_count += 1
                    if is_nan and replace:
                        nan_replaced_count += 1
                
                # Detailed verbose output (commented out for simplicity)
                # if verbose:
                #     status = "NaN" if is_nan else ("outlier" if is_outlier else "valid")
                #     print(f"Coordinate ({i}, {j}, {k}) is {status}: {coord}" +
                #           (f" (med: {med}, std: {std})" if status == "outlier" else ""))

                # Apply replacement or filtering logic
                if (replace and (is_nan or is_outlier)) or (not replace and is_outlier):
                    coords_output[i, j, k, :] = med if replace else np.array([np.nan, np.nan])

    # Print summary for verbose mode
    if verbose:
        if replace:
            print(f"Post-processing: neighbour filter replaced {outlier_replaced_count}/{len(coords_output)} outliers and {nan_replaced_count} other NaNs")
        else:
            print(f"Post-processing: neighbour filter removed {outlier_count}/{len(coords_output)} outliers")

    return coords_output


def first_valid(arr: np.ndarray) -> float | int | np.generic:

    """
    Function to find the first non-NaN value in a 1D array.

    Args:
        arr (np.ndarray): 1D array

    Returns:
        float | int | np.generic: First non-NaN value, or np.nan if none found
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


def strip_peaks(coords: np.ndarray, axis: int = -2, verbose: bool = False) -> np.ndarray:

    """
    Reduce array dimensionality by selecting the first valid peak along an axis containing options.

    Args:
        coords (np.ndarray): N-D array where one axis represents different peaks
        axis (int): Axis along which to reduce the array (default: second-to-last axis)

    Returns:
        np.ndarray: (N-1)-D array with one axis reduced
    """

    if coords.ndim < 3:
        return coords  # Nothing to strip
    
    # Apply the first_valid function along the specified axis  
    coords_str = np.apply_along_axis(first_valid, axis, coords.copy())

    # Report on the number of NaNs
    if verbose:
        n_nans_i = np.sum(np.any(np.isnan(coords[:, :, :, 0, :]), axis=-1))
        n_nans_f = np.sum(np.any(np.isnan(coords_str), axis=-1))

        print(f"Post-processing: {n_nans_i}/{np.prod(coords.shape[0:3])} most likely peak candidates invalid; left with {n_nans_f} after taking next-best peak")
    return coords_str


def smooth(time: np.ndarray, disps: np.ndarray, col: str | int = 'both', lam: float = 5e-7, type: type = int) -> np.ndarray:

    """
    Smooth displacement data along a specified axis using a smoothing spline.

    Args:
        time (np.ndarray): 1D array of time values.
        disps (np.ndarray): 2D array of displacement values.
        col (str | int): Column to smooth:
            - 'both': Smooth both columns (y and x displacements).
            - int: Index of the column to smooth (0 for y, 1 for x).
        lam (float): Smoothing parameter. Larger = more smoothing.
        type (type): Type to convert the smoothed displacements to.

    Returns:
        np.ndarray: 2D array of smoothed displacements (same shape as input)
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


def save_cfig(directory: str, filename: str, format: str = 'pdf', test_mode: bool = False, verbose: bool = True):
    """
    Save the current matplotlib figure to a file.

    Args:
        directory (str): Directory to save the figure.
        filename (str): Name of the file to save the figure as.
        format (str): File format to save the figure in (e.g., 'pdf', 'png').
        test_mode (bool): If True, do not save the figure.
        verbose (bool): If True, print a message when saving the figure.

    Returns:
        None
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