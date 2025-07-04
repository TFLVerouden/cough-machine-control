import getpass
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from tqdm import tqdm

import piv_functions as piv

# Set experimental parameters
test_mode = True
meas_name = '250624_1333_80ms_whand'  # Name of the measurement
frame_nrs = list(range(500, 550)) if test_mode else list(range(1, 6000))
dt = 1 / 40000  # [s]

# Data processing settings
v_max = [15, 150]  # [m/s]
ds_fac = 4  # First pass downsampling factor
n_peaks1 = 10  # Number of peaks to find in first pass correlation map
n_peaks2 = 3
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

# In the current directory, create a folder for processed data
# named the same as the final part of the data_path
proc_path = os.path.join(current_dir, 'processed', os.path.basename(data_path))
if not os.path.exists(proc_path) and not test_mode:
    os.makedirs(proc_path)

# Read calibration data
res_avg, _ = np.loadtxt(cal_path)

# Convert max velocities to max displacements in px
d_max = np.array(v_max) * dt / res_avg  # m/s -> px/frame

# %% FIRST PASS: Full frame correlation ========================================

# Shortcut: if a disp1.npz file already exists, load it
disp1_path = os.path.join(proc_path, 'disp1.npz')
if os.path.exists(disp1_path) and not test_mode:
    with np.load(disp1_path) as data:
        disp1 = data['disp1']
        disp1_unf = data['disp1_unf']
        time = data['time']
    print("Loaded existing disp1.npz file.")

# Otherwise, start from scratch
else:
    # Load images
    imgs = piv.load_images(data_path, frame_nrs,
                           type='tif', lead_0=5, timing=True)

    # TODO: Pre-process images (background subtraction? thresholding?
    #  binarisation to reduce relative influence of bright particles?
    #  low-pass filter to remove camera noise?
    #  mind increase in measurement uncertainty -> PIV book page 140)

    n_corrs = len(imgs) - 1

    # Downsample a copy of the images
    imgs_ds = piv.downsample(imgs.copy(), ds_fac)

    # Pre-allocate array for all peaks: (n_frames, num_peaks, 2) [vy, vx]
    disp1_unf = np.full((n_corrs, n_peaks1, 2), np.nan)

    # Define time arrays beforehand
    time = np.linspace((frame_nrs[0] - 1) * dt,
                       (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

    # Go through all frames and calculate the correlation map
    for i in tqdm(range(n_corrs), desc='First pass'):
        corr_map = sig.correlate(imgs_ds[i + 1], imgs_ds[i], method='fft')

        # TODO: Any processing of the correlation map happens here
        #  (i.e. blacking out pixels or something)

        # Find peaks in the correlation map
        peaks, int_unf = piv.find_peaks(corr_map, num_peaks=n_peaks1,
                                        min_distance=5)

        # Calculate displacements for all peaks
        disp1_unf[i, :, :] = (peaks - np.array(
                corr_map.shape) // 2) * ds_fac  # shape (n_found, 2)

    # Save unfiltered displacements
    disp1 = disp1_unf.copy()

    # Outlier removal
    # TODO: Do something with the intensities of the peaks?
    disp1 = piv.remove_outliers(disp1, y_max=d_max[0], x_max=d_max[1],
                                strip=True)

    # Save the displacements to a file
    if not test_mode:
        np.savez(os.path.join(proc_path, 'disp1'), time=time, disp1=disp1,
                 disp1_unf=disp1_unf, int_unf=int_unf)

# Interpolate data to smooth out the x_displacement in time
disp1_spl = make_smoothing_spline(time[~np.isnan(disp1[:, 1])],
                                  disp1[~np.isnan(disp1[:, 1]), 1], lam=5e-7)
disp1_spl = disp1_spl(time).astype(int)
disp1_spl = np.row_stack([np.zeros(len(disp1_spl)), disp1_spl]).T

# Calculate velocities for plot
vel1_unf = disp1_unf * res_avg / dt
vel1 = disp1 * res_avg / dt
vel1x_spl = disp1_spl[:,0] * res_avg / dt

# Scatter plot vx(t)
plt.figure()
plt.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 1],
            c='gray', s=2, label='Other peaks')
plt.scatter(1000 * time, vel1_unf[:, 0, 1], c='blue', s=10,
            label='Most prominent peak')
plt.scatter(1000 * time, vel1[:, 1], c='orange', s=4,
            label='After outlier removal')
plt.plot(1000 * time, vel1x_spl, color='red',
         label='Displacement to be used\n in 2nd pass (smoothed)')
plt.ylim([-5, 70])
plt.xlabel('Time (ms)')
plt.ylabel('vx (m/s)')
plt.legend(loc='upper right', fontsize='small', framealpha=1)

# Save plot as pdf
if not test_mode:
    plt.savefig(os.path.join(proc_path, 'disp1_vx_t.pdf'),
                bbox_inches='tight')
plt.show()

# %% SECOND PASS: Split image into windows and correlate =======================

# Shortcut: if a disp2.npz file already exists, load it
disp2_path = os.path.join(proc_path, 'disp2.npz')
if os.path.exists(disp2_path) and not test_mode:
    with np.load(disp2_path) as data:
        disp2 = data['disp2']
    print("Loaded existing disp2.npz file.")

# Otherwise, start from scratch
else:
    # Pre-allocate array for all peaks: (frame idx, window idx, peak idx, 2)
    disp2_unf = np.full((n_corrs, n_windows2[0], n_peaks2, 2), np.nan)
    corr_maps2 = []

    for i in tqdm(range(n_corrs), desc='Second pass'):

        # Split the images into horizontal rectangular windows, shifted by
        # the interpolated/smoothed displacements from the first pass
        wnd0, centres = piv.split_n_shift(imgs[i], n_windows2,
                                          shift=disp1_spl[i, :],
                                          shift_mode='before')
        wnd1, _ = piv.split_n_shift(imgs[i + 1], n_windows2,
                                    shift=disp1_spl[i, :], shift_mode='after')

        # Correlate the windows
        corr_maps2.append(np.array([[sig.correlate(window[1], window[0],
                                              method='fft')
                          for window in zip(wnd0, wnd1)]]))

        # Find peaks in the correlation maps
        for j in range(n_windows2[0]):
            peaks, int_unf2 = piv.find_peaks(corr_maps2[i][0, j, 0, ...],
                                             num_peaks=n_peaks2,
                                            min_distance=5)

            # Calculate displacements for all peaks
            disp2_unf[i, j, :, :] = (peaks - np.array(corr_maps2[i].shape[-2:]) // 2
                                 + disp1_spl[i, :])

    # Save unfiltered displacements
    disp2 = disp2_unf.copy()

    # Outlier removal
    disp2 = piv.remove_outliers(disp2, y_max=d_max[0], x_max=d_max[1],
                                strip=True)


# # Todo: outliers (see step 3 in PIV book page 148)
print()