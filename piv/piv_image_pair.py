import os

import cv2 as cv
import numpy as np
from natsort import natsorted
from scipy import signal as sig
from skimage.feature import peak_local_max
from tqdm import tqdm
import matplotlib.pyplot as plt


def split_image(imgs, nr_windows, overlap=0):
    """
    Split a 3D image array (n_img, y, x) into (overlapping) windows. Windows should all have the same size.

    Parameters:
        imgs (np.ndarray): 3D array of image values (image_index, y, x).
        nr_windows (tuple): Number of windows in (y, x) direction.
        overlap (float): Fractional overlap between windows (0 = no overlap).
            E.g. a value of 0.5 means that the window extends to the centre of the neighbouring windows.

    Returns:
        windows (np.ndarray): 5D array of image windows
            (image_index, window_y_idx, window_x_idx, y, x).
        centres (np.ndarray): 3D array of window centres
            (window_y_idx, window_x_idx, 2).
    """
    n_img, img_y, img_x = imgs.shape
    n_y, n_x = nr_windows

    # Calculate window size including overlap
    size_y = min(int(img_y // n_y * (1 + overlap)), img_y)
    size_x = min(int(img_x // n_x * (1 + overlap)), img_x)

    # Get the top-left corner of each window
    y_indices = np.linspace(0, img_y - size_y, num=n_y, dtype=int)
    x_indices = np.linspace(0, img_x - size_x, num=n_x, dtype=int)

    # Create grid of window coordinates
    grid = np.stack(np.meshgrid(y_indices, x_indices, indexing="ij"), axis=-1)

    # Compute centres (window_y_idx, window_x_idx, 2)
    centres = np.stack(
        (grid[:, :, 0] + size_y / 2, grid[:, :, 1] + size_x / 2), axis=-1
    )

    # Allocate output array
    windows = np.empty((n_img, n_y, n_x, size_y, size_x), dtype=imgs.dtype)
    for img_idx in range(n_img):
        for i, y in enumerate(y_indices):
            for j, x in enumerate(x_indices):
                windows[img_idx, i, j] = imgs[img_idx, y:y + size_y, x:x + size_x]

    return windows, centres

def find_peaks(corr_map, num_peaks=1, min_distance=5):
        if num_peaks == 1:
            # Find the single peak
            peak_coords = np.unravel_index(np.argmax(corr_map), corr_map.shape)

            # Return the peak coordinates and the intensity
            return np.array([peak_coords]), corr_map[peak_coords]

        else:
            # Find multiple peaks using peak_local_max
            peaks = peak_local_max(corr_map, min_distance=min_distance, num_peaks=num_peaks)

            # Return the peak coordinates and their intensities
            intensities = corr_map[peaks[:, 0], peaks[:, 1]]
            return peaks, intensities



def subpixel(corr_map, peak_coords):
    # Use a Gaussian fit to refine the peak coordinates

    def three_point_gauss(array):
        return (0.5 * (np.log(array[0]) - np.log(array[2])) /
                ((np.log(array[0])) + np.log(array[2]) - 2 * np.log(array[1])))

    # For both axes...
    for axis in range(2):
        # Get the values around the peak
        values = corr_map[peak_coords[0] - 1:peak_coords[0] + 2, peak_coords[1] - 1:peak_coords[1] + 2]

        # Apply the three-point Gaussian fit
        subpixel_offset = three_point_gauss(values[:, axis])

        # Update the peak coordinates with the subpixel offset
        peak_coords[axis] += subpixel_offset

    return peak_coords


if __name__ == "__main__":
    # Set variables
    # path = '/Volumes/Data/Data/250623 PIV/250624_1333_80ms_whand'
    path = '/Users/tommieverouden/PycharmProjects/cough-machine-control/piv/test_pair'
    frame_nrs = [930, 931]  # Frame numbers to compare

    # List all images in folder; filter for .tif files; sort; get specific frames
    files = natsorted([f for f in os.listdir(path) if f.endswith('.tif')])
    files = [f for f in files if any(f.endswith(f"{nr:05d}.tif") for nr in frame_nrs) if not f.startswith('.')]

    # Import images into 3D numpy array (image_index, y, x)
    imgs = np.array([cv.imread(os.path.join(path, f), cv.IMREAD_GRAYSCALE)
                       for f in tqdm(files, desc='Reading images')],
                      dtype=np.uint64)

    # TODO: Pre-process images (background subtraction? thresholding?
    #  binarisation to reduce relative influence of bright particles?
    #  low-pass filter to remove camera noise?
    #  mind increase in measurement uncertainty -> PIV book page 140)

    # Threshold images


    print(f"Image size: {imgs.shape}")

    windows, centres = split_image(imgs, nr_windows=(16, 1), overlap=0.5)

    # # Plot the windows and centres on top of the first image
    # plt.imshow(imgs[0], cmap='gray')
    # for i, window in enumerate(windows):
    #     y, x = centres[i]
    #     rect = plt.Rectangle((x - window.shape[1] / 2, y - window.shape[0] / 2),
    #                          window.shape[1], window.shape[0], linewidth=1,
    #                          edgecolor='r', facecolor='none')
    #     plt.gca().add_patch(rect)
    #     plt.plot(x, y, 'ro')  # Plot the centre
    # plt.show()

    # Cycle through all windows in one specific image and correlate them with the corresponding windows in the other image
    maps = np.array([[sig.correlate(window[1], window[0], method='fft')
             for window in zip(windows[0], windows[1])]])

    # TODO: Any processing of the correlation map happens here (i.e. blacking out all pixels outside of a positive semi-circle)

    # Create an array of indices to plot the correlation map, such that the origin (zero-shift DC value) is in the centre
    # of the correlation map
    y_indices = np.arange(maps.shape[2]) - maps.shape[2] // 2
    x_indices = np.arange(maps.shape[3]) - maps.shape[3] // 2
    y_indices, x_indices = np.meshgrid(y_indices, x_indices, indexing='ij')

    # Find local maxima in the correlation map
    peaks = peak_local_max(maps[0, 7, 0], min_distance=5, threshold_abs=0.1, num_peaks=5)
    intensities = maps[0, 7, 0][peaks[:, 0], peaks[:, 1]]

    # Plot the normalised correlation map with the local maxima
    corr_map = maps[0, 7, 0] / np.max(maps[0, 7, 0])  # Normalize to the maximum value
    plt.imshow(corr_map, cmap='hot')
    plt.colorbar(label='Correlation')
    plt.scatter(peaks[:, 1], peaks[:, 0], color='blue', label='Local maxima')
    plt.show()

    print(find_peaks(maps[0, 7, 0]))
    print(find_peaks(maps[0, 7, 0], num_peaks=5, min_distance=5))

    print()
    # # Find the maximum correlation value and its coordinates
    # max_corr = np.max(corr_map)
    # max_coords = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    #
    # # Plot correlation map and maximum, normalized to the maximum value
    # corr_map = corr_map / max_corr  # Normalize to the maximum value
    # plt.imshow(corr_map, cmap='hot')
    # plt.colorbar(label='Correlation')
    # # plt.scatter(max_coords[1], max_coords[0], color='blue', label='Max Correlation')
    # plt.show()