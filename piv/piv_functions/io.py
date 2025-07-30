"""
Input/output functions for PIV analysis.

This module handles file operations including reading images,
loading/saving backup data, and saving figures.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
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
        return False, {}


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


def read_imgs(data_path: str, frame_nrs: list[int] | str, format: str = 'tif', lead_0: int = 5, timing: bool = True) -> np.ndarray:

    """
    Load selected images from a directory into a 3D numpy array.

    Args:
        data_path (str): Path to the directory containing images.
        frame_nrs (list[int] | str): List of frame numbers to load,
            or "all" to load all images.
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

    # Handle "all" option or specific frame numbers
    if frame_nrs == "all":
        # Load all images - filter by format and exclude hidden files
        files = [f for f in files if f.endswith('.' + format) and not f.startswith('.')]
    else:
        # Filter files to include only those that match the specified frame numbers
        files = [f for f in files if any(f.endswith(f"{nr:0{lead_0}d}.{format}") for nr
                                         in frame_nrs) and not f.startswith('.')]
    
    if not files:
        raise FileNotFoundError(f"No files found in {data_path} with the specified criteria and format '{format}'.")

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


def save_cfig(directory: str, file_name: str, format: str = 'pdf', test_mode: bool = False, verbose: bool = True):
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
        file_name = f"{file_name}.{format}"
        filepath = os.path.join(directory, file_name)

        # Save the figure
        plt.savefig(filepath, transparent=True, bbox_inches='tight',
                    format=format)
        if verbose:
            print(f"Figure saved to {filepath}")

    # # Show the figure
    # plt.show()

    return
