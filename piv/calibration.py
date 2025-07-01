import numpy as np
import cv2 as cv
from scipy import spatial
import pickle
import os

def all_distances(points):
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must be a 2D array with shape (n_points, 2)")
    return spatial.distance.cdist(points, points, 'euclidean')

def calibrate_grid(path, spacing, roi=None, init_grid=(4, 4), binary_thr=100,
                   blur_ker=(3, 3), open_ker=(3, 3), print_prec=8):
    """
    Calculate resolution from a grid.

    Parameters:
        path (str): Path to the image file.
        spacing (float): Real-world spacing between grid points [m].
        roi (list): Region to crop to (y_start, y_end, x_start, x_end).
                    If None, the entire image is used.
        init_grid (tuple): Initial grid size (columns, rows).
        binary_thr (int): Threshold value for binarising the image.
        blur_ker (tuple): Kernel size for Gaussian blur (width, height).
        open_ker (tuple): Kernel size for morphological open (width, height).
        print_prec (int): Number of decimal places for printing resolution.

    Returns:
        res_avg (float): Average (weighted) resolution from all dot pairs.
        res_std (float): Standard deviation (weighted) in the resolution.
    """

    # Load the image and convert it to grayscale
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Crop the image to the specified region of interest (ROI)
    if roi is None:
        roi = [0, img.shape[0], 0, img.shape[1]]
    img = img[roi[0]:roi[1], roi[2]:roi[3]]

    # Apply morphological opening and Gaussian blur to preprocess the image
    img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones(open_ker, np.uint8))
    img = cv.GaussianBlur(img, blur_ker, 0)

    # Binarize the image using the specified threshold
    img = np.where(img > binary_thr, 255, img)

    # Initialize grid size and flag for maximum columns found
    grid_size = list(init_grid)
    max_columns_found = False

    # Loop to find the largest fitting grid size
    while True:
        grid_found, centres = cv.findCirclesGrid(
            img, tuple(grid_size), flags=cv.CALIB_CB_SYMMETRIC_GRID)

        if grid_found and not max_columns_found:
            # Increase the number of columns if the maximum hasn't been reached
            grid_size[0] += 1
        elif not grid_found and not max_columns_found:
            # Fix the number of columns and start increasing rows
            max_columns_found = True
            grid_size[0] -= 1
            grid_size[1] += 1
        elif max_columns_found:
            # Only increase the number of rows
            grid_found, centres = cv.findCirclesGrid(
                img, tuple(grid_size), flags=cv.CALIB_CB_SYMMETRIC_GRID)
            if grid_found:
                grid_size[1] += 1
            else:
                # Revert to the last successful row count and exit the loop
                grid_size[1] -= 1
                break

    print(f"Grid found: {grid_size[0]} cols ᳵ {grid_size[1]} rows")

    # Reshape the detected centers and generate grid points in real-world units
    centres = centres.reshape(-1, 2)
    grid_points = np.array([[x, y] for y in range(grid_size[1])
                            for x in range(grid_size[0])]) * spacing

    # Calculate pairwise distances in real-world and pixel units
    dist_real = all_distances(grid_points)
    dist_pixel = all_distances(centres)

    # Compute resolution and standard deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        all_res = dist_real / dist_pixel

    mask = np.eye(*dist_pixel.shape, dtype=bool).__invert__()
    res_avg = np.average(all_res[mask], weights=dist_pixel[mask])
    res_std = np.sqrt(np.average((all_res[mask]-res_avg)**2, weights=dist_pixel[mask]))

    print(f"Resolution (+- std): {res_avg:.{print_prec}f} +- {res_std:.{print_prec}f} m/px")

    # Save the resolution and standard deviation to a file
    with open(path.replace('.tif', '_res_std.txt'), 'wb') as f:
        np.savetxt(f,[res_avg, res_std])
    print("Resolution saved to disk.")

    return res_avg, res_std


if __name__ == "__main__":
    # Define the path to the image and other parameters
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
   
    path = "cough-machine-control/piv/calibration/250624_calibration_PIV_500micron.tif"
    cal_path = parent_dir + "\\" + path

    # cal_path = ('/Users/tommieverouden/PycharmProjects/cough-machine-control/'
    #             'piv/calibration/250624_calibration_PIV_500micron.tif')
    # cal_path = 'D:\Experiments\PIV\250624_calibration_PIV_500micron.tif'
    cal_spacing = 0.001  # m
    cal_roi = [50, 725, 270, 375]

    # Run the calibration function
    calibrate_grid(cal_path, cal_spacing, roi=cal_roi)

    # Resolution (+- std): 0.0508 +- 0.0001 mm/px