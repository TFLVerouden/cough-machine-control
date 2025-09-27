"""
Post-processing functions for PIV analysis.

This module contains functions for filtering displacement data,
removing outliers, smoothing data, and other post-processing operations.
"""

import numpy as np
from scipy.interpolate import make_smoothing_spline
from tqdm import trange


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


def _validate_n_nbs(n_nbs: int | str | tuple[int, int, int], max_shape: tuple[int, int, int]):
    """Validate and process n_nbs parameter for filter_neighbours function."""
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


def _edge_nbs(i, j, k, n_nbs, nbs, n_corrs, n_wins_y, n_wins_x):
    """Get neighbourhood slice, handling coordinates at edges."""
    # Get neighbourhood indices, handling coordinates at edges
    i_nbs = (np.clip(i, (n_nbs[0] - 1)//2, 
                     n_corrs - (n_nbs[0] - 1)//2 - 1) 
             - (n_nbs[0] - 1)//2)
    j_nbs = (np.clip(j, (n_nbs[1] - 1)//2, 
                     n_wins_y - (n_nbs[1] - 1)//2 - 1) 
             - (n_nbs[1] - 1)//2)
    k_nbs = (np.clip(k, (n_nbs[2] - 1)//2, 
                     n_wins_x - (n_nbs[2] - 1)//2 - 1) 
             - (n_nbs[2] - 1)//2)
    
    return nbs[i_nbs, j_nbs, k_nbs]


def _outlier_dist(coord, med, threshold, mode):
    """Check if coordinate is outlier and calculate distance to median."""
    if mode == "x":
        is_outl = np.abs(coord[1] - med[1]) > threshold[1]
        dist = np.abs(coord[1] - med[1])
    elif mode == "y":
        is_outl = np.abs(coord[0] - med[0]) > threshold[0]
        dist = np.abs(coord[0] - med[0])
    elif mode == "xy":
        is_outl = np.any(np.abs(coord - med) > threshold)
        dist = np.linalg.norm(coord - med)
    elif mode == "r":
        vec_length = np.linalg.norm(coord)
        med_length = np.linalg.norm(med)
        is_outl = np.abs(vec_length - med_length) > threshold.mean()
        dist = np.abs(vec_length - med_length)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'x', 'y', 'xy', or 'r'.")
    
    return is_outl, dist


def filter_neighbours(coords: np.ndarray, n_nbs: int | str | tuple[int, int, int] = 3, thr: float | tuple[float, float] = 1, thr_unit: str = "std", mode: str = "xy", replace: bool | str = False, verbose: bool = False, timing: bool = False) -> np.ndarray:

    """
    Filter out coordinates that are too different from their neighbours.

    Args:
        coords (np.ndarray): 4D or 5D coordinate array of shape
            (n_corrs,   n_wins_y, n_wins_x, [n_peaks,] 2).
        n_nbs (int | str | tuple): Size of neighbourhood in each dimension
            to consider for filtering (including center point). Can be
            an integer, "all", or a tuple of three values (int or "all").
        thr (float | tuple[float, float]): Threshold; how many standard
            deviations or pixels can a point be away from its neighbours.
            Can be a single value (applied to both x and y)
            or a tuple (thr_y, thr_x) for separate thresholds.
        thr_unit (str): Unit of the threshold:
            - "std": Standard deviations (default)
            - "pxs": Pixels (absolute distance)
        mode (str): Which coords should be within threshold from the median:
            - "x": Compare x coordinates only
            - "y": Compare y coordinates only
            - "xy": Compare both x and y coordinates
            - "r": Compare vector lengths only
        replace (bool | str): Replacement strategy for outliers:
            - False: Set outliers to NaN
            - True or "median": Replace outliers with median of neighbours
            - "closest": Replace outliers with closest valid candidate peak,
                but only if there is a non-outlier peak available.
        verbose (bool): If True, print summary statistics about filtering.
        timing (bool): If True, print timing information.

    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN or replaced.
    """

    # INITIALISATION
    # Convert input without extra candidate peaks to 5D
    was_4d = coords.ndim == 4
    if was_4d:
        coords = coords[:, :, :, np.newaxis, :]
    n_corrs, n_wins_y, n_wins_x, _, _ = coords.shape
    
    # Move NaNs to end (no effect on 4D->5D since there's only 1 peak)
    coords = strip_peaks(coords, axis=-2, mode='sort', verbose=False)

    # Validate n_nbs
    n_nbs = _validate_n_nbs(n_nbs, (n_corrs, n_wins_y, n_wins_x))
    
    # Initialize counters for verbose mode
    if verbose:
        outlier_count = 0
        nan_replaced_count = 0
        outlier_replaced_count = 0
    
    # Handle replace parameter
    if replace is True:
        replace = "median"
    elif replace not in [False, "median", "closest"]:
        raise ValueError("replace must be False, True, 'median', or 'closest'")
    
    if replace == "closest" and was_4d:
        raise ValueError("'closest' replacement mode requires 5D input array")

    # Get copy for output, and extract 1st valid peak for neighbour analysis
    coords_out = coords.copy()
    coords_work = strip_peaks(coords, axis=-2, mode='reduce', verbose=False)

    # Get a set of sliding windows around each (bulk) coordinate
    nbs = np.lib.stride_tricks.sliding_window_view(coords_work,
    (n_nbs[0], n_nbs[1], n_nbs[2], 1))[..., 0]

    # TODO: Multi-thread
    # Iterate over each coordinate
    for i in trange(n_corrs, desc="Filtering neighbours", disable=not timing):
        for j in range(n_wins_y):
            for k in range(n_wins_x):

                # NEIGHBOURHOOD ANALYSIS
                # Get neighbourhood, handling coordinates at edges
                nb = _edge_nbs(i, j, k, n_nbs, nbs, n_corrs, n_wins_y, n_wins_x)

                # If the neighbourhood is empty, skip to the next coordinate
                if np.all(np.isnan(nb)):
                    continue

                # Calculate the median
                med = np.nanmedian(nb, axis=(1, 2, 3))
                
                # Check for identical neighbourhood (no outliers possible)
                if np.all(nb == nb[0, 0, 0, :]):
                    if replace:
                        # If entire neighbourhood is identical, replace with that value
                        coords_out[i, j, k, 0, :] = nb[0, 0, 0, :]
                    continue
                
                # COORDINATE VALIDATION
                # Calculate actual threshold based on mode
                if thr_unit == "std":
                    std = np.nanstd(nb, axis=(1, 2, 3))
                    # Skip if std is invalid
                    if np.any(np.isnan(std)) or np.any(std == 0):
                        continue
                    # Handle tuple or scalar threshold
                    if isinstance(thr, tuple):
                        thr_cur = np.array([thr[0] * std[0], thr[1] * std[1]])  # (thr_y, thr_x)
                    else:
                        thr_cur = thr * std
                else:  # thr_unit == "pxs"
                    # Handle tuple or scalar threshold for pixel mode
                    if isinstance(thr, tuple):
                        thr_cur = np.array([thr[0], thr[1]])  # (thr_y, thr_x)
                    else:
                        thr_cur = np.array([thr, thr])

                # Get the coordinate to check
                coord = coords[i, j, k, 0, :]
                        
                # Check if the coordinate is already NaN in the input
                is_nan = np.any(np.isnan(coord))
                
                # Check if the current coordinate is an outlier
                is_outl = False
                if not is_nan:
                    is_outl, _ = _outlier_dist(coord, med, thr_cur, mode)
                
                # Update counters for verbose mode
                if verbose:
                    if is_outl:
                        outlier_count += 1
                        if replace:
                            outlier_replaced_count += 1
                    if is_nan and replace:
                        nan_replaced_count += 1

                # REPLACEMENT OR FILTERING
                # Apply replacement or filtering logic
                if is_nan or is_outl:
                    if replace == "median":
                        coords_out[i, j, k, 0, :] = med
                    elif replace == "closest":
                        # Find the closest valid candidate peak inline
                        min_distance = np.inf
                        closest_peak = None
                        
                        for peak in coords[i, j, k, :, :]:
                            # Skip NaN peaks
                            if np.any(np.isnan(peak)):
                                continue
                                
                            # Check if it is outlier
                            is_outl, dist = _outlier_dist(peak, med,
                                                          thr_cur, mode)
                            
                            if not is_outl and dist < min_distance:
                                min_distance = dist
                                closest_peak = peak
                        
                        # Only replace if we found a valid peak!
                        if closest_peak is not None:
                            coords_out[i, j, k, 0, :] = closest_peak
                    elif not replace:
                        coords_out[i, j, k, 0, :] = np.array([np.nan, np.nan])

    # Print summary for verbose mode
    if verbose:
        if replace:
            total_coords = np.prod(coords.shape[:-2])
            print(f"Post-processing: neighbour filter ({thr_unit}) replaced {outlier_replaced_count}/{total_coords} outliers and {nan_replaced_count} other NaNs")
        else:
            total_coords = np.prod(coords.shape[:-2])
            print(f"Post-processing: neighbour filter ({thr_unit}) removed {outlier_count}/{total_coords} outliers")

    # Restore original shape for 4D arrays
    if was_4d:
        coords_out = coords_out[:, :, :, 0, :]

    return coords_out


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


def strip_peaks(coords: np.ndarray, axis: int = -2, mode: str = 'reduce', verbose: bool = False) -> np.ndarray:

    """
    Process peaks along an axis containing options.

    Args:
        coords (np.ndarray): N-D array where one axis represents different peaks
        axis (int): Axis along which to process the array (default: second-to-last axis)
        mode (str): Processing mode:
            - 'reduce': Reduce array dimensionality by selecting the first valid peak
            - 'sort': Remove NaNs between peaks without reducing dimensionality
        verbose (bool): If True, print summary information

    Returns:
        np.ndarray: Processed array
            - 'reduce' mode: (N-1)-D array with one axis reduced
            - 'sort' mode: Same dimensionality as input with NaNs moved to end
    """

    if coords.ndim < 3:
        return coords  # Nothing to strip
    
    if mode == 'reduce':
        # Apply the first_valid function along the specified axis  
        coords_str = np.apply_along_axis(first_valid, axis, coords.copy())

        # Report on the number of NaNs
        if verbose:
            n_nans_i = np.sum(np.any(np.isnan(coords[:, :, :, 0, :]), axis=-1))
            n_nans_f = np.sum(np.any(np.isnan(coords_str), axis=-1))

            print(f"Post-processing: {n_nans_i}/{np.prod(coords.shape[0:3])} most likely peak candidates invalid; left with {n_nans_f} after taking next-best peak")
        return coords_str
    
    elif mode == 'sort':
        # Sort peaks to move NaNs to the end without reducing dimensionality
        coords_sor = coords.copy()
        
        # Get all dimensions except the peak axis and coordinate axis
        pk_ax = axis if axis >= 0 else len(coords.shape) + axis
        
        # Iterate through all positions and sort peaks at each location        
        for idx in np.ndindex(coords.shape[:pk_ax] + coords.shape[pk_ax+1:-1]):
            # Create full index for accessing the peak dimension
            full_idx = idx[:pk_ax] + (slice(None),) + idx[pk_ax:]
            
            # Get peaks for this location
            peaks = coords_sor[full_idx]  # Shape: (n_peaks, 2)
            
            # Find which peaks are valid (not NaN)
            mask = np.any(np.isnan(peaks), axis=-1)

            # Reorder: valid peaks first, then NaN peaks
            coords_sor[full_idx] = peaks[np.concatenate([np.where(~mask)[0],
                                                         np.where(mask)[0]])]
        
        if verbose:
            print(f"Post-processing: peak sorting maintained "
                  f"{np.sum(~np.any(np.isnan(coords_sor), axis=-1))} valid "
                  f"peaks (was {np.sum(~np.any(np.isnan(coords), axis=-1))})")
        return coords_sor
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'reduce' or 'sort'.")


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
