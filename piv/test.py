import getpass
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from tqdm import tqdm

import piv_functions as piv


# Set experimental parameters
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
dt = 1 / 40000  # [s] 

# Data processing settings
v_max = [15, 100]  # [m/s]
ds_fac = 4  # First pass downsampling factor
n_peaks1 = 10  # Number of peaks to find in first pass correlation map
n_peaks2 = 5
n_windows2 = (8, 1)  # Number of windows in second pass (rows, cols)

# File handling
current_dir = os.path.dirname(os.path.abspath(__file__))
cal_path = os.path.join(current_dir, "calibration",
                        "250624_calibration_PIV_500micron_res_std.txt")
user = getpass.getuser()
if user == "tommieverouden":
    data_path = os.path.join("/Volumes/Data/Data/250623 PIV/", meas_name)
elif user == "sikke":
    data_path = os.path.join("D:\\Experiments\\PIV\\", meas_name)

# Data saving settings
disp1_var_names = ['time', 'disp1_unf', 'int1_unf', 'n_corrs']

# In the current directory, create a folder for processed data
# named the same as the final part of the data_path
proc_path = os.path.join(current_dir, 'processed', os.path.basename(data_path))

# Read calibration data
if not os.path.exists(cal_path):
    raise FileNotFoundError(f"Calibration file not found: {cal_path}")
res_avg, _ = np.loadtxt(cal_path)

# Convert max velocities to max displacements in px
d_max = np.array(v_max) * dt / res_avg  # m/s -> px/frame

# Load data
bckp1_loaded, loaded_vars = piv.backup("load", proc_path, "pass1.npz", disp1_var_names)

if bckp1_loaded:
    # Extract loaded variables using the same names as defined in disp1_var_names
    for var_name in disp1_var_names:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing backup data.")

# Filter data
print("Number of NaNs:", np.sum(np.isnan(disp1_unf)),"/", disp1_unf.size)
# disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, a=d_max[0], b=d_max[1])
disp1 = disp1_unf
print("Number of NaNs:", np.sum(np.isnan(disp1)),"/", disp1.size)
disp1, int1 = piv.filter_outliers('intensity', disp1, a=int1_unf, b=0.00005)
print("Number of NaNs:", np.sum(np.isnan(disp1)),"/", disp1.size)

disp1 = np.expand_dims(disp1, axis=2)

# Apply the filter_neighbours function
disp1 = piv.filter_neighbours(disp1, thr=0.1, n_nbs=(4, 2, 0))

# Count the number of NaN values in the filtered coordinates
print("Number of NaNs:", np.sum(np.isnan(disp1)),"/", disp1.size)

# %% Plot a single frame
frame_idx = 500
frame_coords = disp1[frame_idx, :, :, :]

# Create a mask for valid coordinates (not NaN)
valid_mask = ~np.isnan(frame_coords[:, :, 0])

# Get valid and invalid coordinates
valid_coords = frame_coords[valid_mask]
invalid_coords = frame_coords[~valid_mask]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot valid coordinates in blue
if len(valid_coords) > 0:
    ax.scatter(valid_coords[:, 1], valid_coords[:, 0], 
              c='blue', s=30, alpha=0.7, label=f'Valid ({len(valid_coords)})')

# Plot invalid coordinates in red (just the positions where they would be)
invalid_y, invalid_x = np.where(~valid_mask)
if len(invalid_y) > 0:
    ax.scatter(invalid_x, invalid_y, 
              c='red', s=30, alpha=0.7, marker='x', 
              label=f'Invalid ({len(invalid_y)})')

ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title(f'Coordinate filtering results for frame {frame_idx}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Frame {frame_idx}: {len(valid_coords)} valid, {len(invalid_y)} invalid coordinates")

# disp1 = piv.strip_peaks(disp1, axis=-2)  # Strip peaks with NaN values

# %%
