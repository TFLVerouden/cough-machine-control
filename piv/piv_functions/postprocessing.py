"""
Post-processing functions for PIV analysis.

This module contains functions for filtering displacement data,
removing outliers, smoothing data, and other post-processing operations.
"""

import numpy as np
from scipy.interpolate import make_smoothing_spline


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
        print(f"Post-processing: global filter removed {np.sum(~mask)}/{coords.shape[0]} coordinates in mode '{mode}'")

    # Reshape back to original shape
    coords = coords.reshape(orig_shape)
    
    # In intensity mode, also return the filtered intensities
    if mode == 'intensity':
        ints[~mask] = np.nan
        return coords, ints
    else:
        return coords


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
                i_nbs = (np.clip(i, (n_nbs[0] - 1)//2, 
                                 n_corrs - (n_nbs[0] - 1)//2 - 1) 
                         - (n_nbs[0] - 1)//2)
                j_nbs = (np.clip(j, (n_nbs[1] - 1)//2, 
                                 n_wins_y - (n_nbs[1] - 1)//2 - 1) 
                         - (n_nbs[1] - 1)//2)
                k_nbs = (np.clip(k, (n_nbs[2] - 1)//2, 
                                 n_wins_x - (n_nbs[2] - 1)//2 - 1) 
                         - (n_nbs[2] - 1)//2)
                nb = nbs[i_nbs, j_nbs, k_nbs]

                # If the neighbourhood is empty, skip to the next coordinate
                if np.all(np.isnan(nb)):
                    continue

                # If entire neighbourhood is identical, replace and skip
                if np.all(nb == nb[0, 0, 0, :]):
                    if replace:
                        coords_output[i, j, k, :] = nb[0, 0, 0, :]
                    continue

                # Calculate the median and standard deviation
                med = np.nanmedian(nb, axis=(1, 2, 3))
                std = np.nanstd(nb, axis=(1, 2, 3))
                
                # If std is 0 or NaN, skip outlier detection
                if np.any(np.isnan(std)) or np.any(std == 0):
                    continue

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
