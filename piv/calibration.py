import numpy as np
import cv2 as cv
from scipy import spatial
import pickle


def all_distances(points):
    # Check points are in the right shape
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must be a 2D array with shape (n_points, 2)")

    # Calculate the distance between all points
    return spatial.distance.cdist(points, points, 'euclidean')


# Set variables
path = '/Users/tommieverouden/PycharmProjects/cough-machine-control/piv/calibration/250624_calibration_PIV_500micron.tif'
# path = 'D:\Experiments\PIV\250624_calibration_PIV_500micron.tif'
spacing = 1  # mm
roi = [50, 725, 270, 375]  # [y_start, y_end, x_start, x_end]
binary_thresh = 100

# Import image, convert to grayscale, 8-bit (for blob detector) image
img = cv.imread(path, cv.IMREAD_UNCHANGED)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Crop only the area containing dots
img = img[roi[0]:roi[1], roi[2]:roi[3]]

# Opening up image to get rid of lines
kernel = np.ones((3, 3), np.uint8)
img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# Gaussian blur
img = cv.GaussianBlur(img, (3, 3), 0)

# Set all pixels above threshold to white
img = np.where(img > binary_thresh, 255, img)

# Initialize grid size
grid_size = [4, 4]  # [columns, rows]
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
        grid_size[0] -= 1  # Revert to the last successful column count
        grid_size[1] += 1
    elif max_columns_found:
        # Only increase the number of rows
        grid_found, centres = cv.findCirclesGrid(
            img, tuple(grid_size), flags=cv.CALIB_CB_SYMMETRIC_GRID)
        if grid_found:
            grid_size[1] += 1
        else:
            grid_size[1] -= 1
            break  # Exit the loop if no grid is found with the current size

print(f"Grid size found: {grid_size}")

# Reshape centers to match grid size
centres = centres.reshape(-1, 2)

# Generate all the grid point coordinates in real space units (img xy flipped)
grid_points = np.array([[x, y] for y in range(grid_size[1]) for x in range(grid_size[0])])

# Calculate the distances in pixel space and real space
dist_real = all_distances(grid_points)
dist_pixel = all_distances(centres)

# Divide for resolutions (ignore divide by zero warning)
with np.errstate(divide='ignore', invalid='ignore'):
    all_res = dist_real / dist_pixel

# Calculate avg and std; mask diagonal to avoid division by zero
mask = np.eye(*dist_pixel.shape, dtype=bool).__invert__()
res_avg = np.average(all_res[mask], weights=dist_pixel[mask])
res_std = np.sqrt(
        np.average((all_res[mask]-res_avg)**2, weights=dist_pixel[mask]))

# Print the results
print(f"Resolution (+- std): {res_avg:.4f} +- {res_std:.4f} mm/px")

# Save resolution to disk with the same name as the original image
with open(path.replace('.tif', '_res_std.pkl'), 'wb') as f:
    pickle.dump([res_avg, res_std], f)