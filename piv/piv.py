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
test_mode = False
from_disk = True  # If True, load data from disk instead of processing it
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'  # Name of the measurement
frame_nrs = list(range(500, 574)) if test_mode else list(range(500, 574))
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
disp1_save = ['time', 'disp1', 'disp1_unf', 'disp1_spl', 'int1_unf',
              'n_corrs']

# In the current directory, create a folder for processed data
# named the same as the final part of the data_path
proc_path = os.path.join(current_dir, 'processed', os.path.basename(data_path))
if not os.path.exists(proc_path) and not test_mode:
    os.makedirs(proc_path)

# Read calibration data
res_avg, _ = np.loadtxt(cal_path)

# Convert max velocities to max displacements in px
d_max = np.array(v_max) * dt / res_avg  # m/s -> px/frame

# # %% FIRST PASS: Full frame correlation ========================================
# Shortcut: if data is saved on disk, load it
if from_disk and not test_mode:
    disp1_path = os.path.join(proc_path, 'pass1.npz')

    with np.load(disp1_path) as data:
        for k in disp1_save:
            if k in data:
                locals()[k] = data[k]
            else:
                print(f"Warning: {k} not found in {disp1_path}")
    print(f"Loaded data from {disp1_path}")

# Otherwise, start from scratch with loading images
else:
    imgs = piv.load_images(data_path, frame_nrs, format='tif', lead_0=5,
                           timing=True)

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
        corr_map = sig.correlate(imgs_ds[i + 1], imgs_ds[i],
                                 method='fft', mode='same')

        # TODO: Any processing of the correlation map happens here
        #  (i.e. blacking out pixels or something)

        # Find peaks in the correlation map
        peaks, int1_unf = piv.find_peaks(corr_map, num_peaks=n_peaks1,
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

    # Interpolate data to smooth out the x_displacement in time
    disp1_spl = make_smoothing_spline(time[~np.isnan(disp1[:, 1])],
                                      disp1[~np.isnan(disp1[:, 1]), 1], lam=5e-7)
    disp1_spl = disp1_spl(time).astype(int)
    disp1_spl = np.row_stack([np.zeros(len(disp1_spl)), disp1_spl]).T

    if not test_mode:
        # Save data to a file
        save_dict = {k: locals()[k] for k in disp1_save if k in locals()}
        np.savez(os.path.join(proc_path, 'pass1'), **save_dict)

    # Calculate velocities for plot
    vel1_unf = disp1_unf * res_avg / dt
    vel1 = disp1 * res_avg / dt
    vel1x_spl = disp1_spl[:, 1] * res_avg / dt

    # Scatter plot vx(t)
    fig0, ax0 = plt.subplots()
    ax0.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 1],
                c='gray', s=2, label='Other peaks')
    ax0.scatter(1000 * time, vel1_unf[:, 0, 1], c='blue', s=10,
                label='Most prominent peak')
    ax0.scatter(1000 * time, vel1[:, 1], c='orange', s=4,
                label='After outlier removal')
    ax0.plot(1000 * time, vel1x_spl, color='red',
             label='Displacement to be used\n in 2nd pass (smoothed)')
    ax0.set_ylim([-5, 70])
    ax0.set_xlabel('Time (ms)')
    ax0.set_ylabel('vx (m/s)')
    ax0.legend(loc='upper right', fontsize='small', framealpha=1)

    piv.save_cfig(proc_path, 'disp1_vx_t', test_mode=test_mode)

    # Plot all velocities vy(vx)
    fig1, ax1 = plt.subplots()
    ax1.scatter(vel1[:, 1], vel1[:, 0], c='blue', s=4)
    ax1.set_xlabel('vx (m/s)')
    ax1.set_ylabel('vy (m/s)')

    piv.save_cfig(proc_path, 'disp1_vy_vx', test_mode=test_mode)

# # %% SECOND PASS: Split image into windows and correlate =======================
#
# # Shortcut: if a disp2.npz file already exists, load it
# disp2_path = os.path.join(proc_path, 'disp2.npz')
# if os.path.exists(disp2_path) and not test_mode:
#     with np.load(disp2_path) as data:
#         disp2 = data['disp1']
#         disp2_unf = data['disp2_unf']
#         time = data['time']
#     print("Loaded existing disp2.npz file.")
#
# # Otherwise, start from scratch
# else:
#
# # Pre-allocate array for all peaks: (frame idx, window idx, peak idx, 2)
# disp2_unf = np.full((n_corrs, n_windows2[0], n_peaks2, 2), np.nan)
#
# for i in tqdm(range(n_corrs), desc='Second pass'):
#
#     # Split the images into horizontal rectangular windows, shifted by
#     # the interpolated/smoothed displacements from the first pass
#     wnd0, centres = piv.split_n_shift(imgs[i], n_windows2,
#                                       shift=disp1_spl[i, :],
#                                       shift_mode='before')
#     wnd1, _ = piv.split_n_shift(imgs[i + 1], n_windows2,
#                                 shift=disp1_spl[i, :],
#                                 shift_mode='after')
#
#     for j in range(n_windows2[0]):
#         # Calculate the correlation map for each window pair
#         # (i.e. the first window of frame i with the first window of frame i+1)
#         corr_map = sig.correlate(wnd1[j, 0], wnd0[j, 0],
#                                  method='fft', mode='same')
#
#         # Find peaks in the correlation maps
#         peaks, int2_unf = piv.find_peaks(corr_map, num_peaks=n_peaks2,
#                                         min_distance=3)
#
#         # Calculate displacements for all peaks
#         disp2_unf[i, j, :, :] = (disp1_spl[i, :] + peaks
#                                  - (np.array(corr_map.shape) // 2))
#
# # Save unfiltered displacements
# disp2 = disp2_unf.copy()
#
# # Outlier removal
# disp2 = piv.remove_outliers(disp2, y_max=d_max[0], x_max=d_max[1],
#                             strip=True)
# disp2_show = piv.remove_outliers(disp2_unf, y_max=d_max[0], x_max=d_max[1],
#                                  strip=False)
#
# # Save the displacements to a file
# if not test_mode:
#     np.savez(os.path.join(proc_path, 'disp2'), disp2=disp2, disp2_unf=disp2_unf,
#              int2_unf=int2_unf, centres=centres)
#
# # For each frame, plot the velocity vectors at the window centres
# vel2_show = disp2_show * res_avg / dt
# vel2 = disp2 * res_avg / dt
# #
# # # Set up video writer
# # if not test_mode:
# #     fig1, ax = plt.subplots()
# #     writer = ani.FFMpegWriter(fps=10)
# #
# #     video_path = os.path.join(proc_path, 'disp2.mp4')
# #     with writer.saving(fig1, video_path, dpi=150):
# #         for i in range(n_corrs):
# #             ax.clear()
# #             for j in range(n_peaks2):
# #                 ax.scatter(vel2_show[i, :, j, 0],
# #                            centres[:, 0, 0] * res_avg * 1000, c='gray', s=2)
# #                 ax.scatter(vel2_show[i, :, j, 1],
# #                            centres[:, 0, 0] * res_avg * 1000, c='gray', s=2)
# #             ax.plot(vel2[i, :, 0], centres[:, 0, 0] * res_avg * 1000,
# #                     label='vy')
# #             ax.plot(vel2[i, :, 1], centres[:, 0, 0] * res_avg * 1000,
# #                     label='vx')
# #             ax.set_title(f't = {((i + 1) * dt * 1000):.2f} ms')
# #             ax.set_xlabel('v (m/s)')
# #             ax.set_ylabel('y (mm)')
# #             ax.set_xlim([-5, 60])
# #             ax.legend(loc='upper right')
# #             writer.grab_frame()
# #     plt.close(fig1)
# #
# # # # Todo: outliers (see step 3 in PIV book page 148)
# # print()