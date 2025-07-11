import getpass
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from tqdm import trange, tqdm

import piv_functions as piv


# Set experimental parameters
test_mode = False
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
frame_nrs = list(range(3000, 3100)) if test_mode else list(range(1, 6000))
dt = 1 / 40000  # [s] 

# Data processing settings
v_max = [15, 100]  # [m/s]
ds_fac = 4  # First pass downsampling factor
n_peaks1 = 10  # Number of peaks to find in first pass correlation map
n_wins1 = (1, 1)
n_peaks2 = 5
n_wins2 = (8, 1)  # Number of windows in second pass (rows, cols)

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
disp1_var_names = ['time', 'disp1', 'disp1_unf', 'disp1_spl', 'int1_unf', 'n_corrs']

# In the current directory, create a folder for processed data
# named the same as the final part of the data_path
proc_path = os.path.join(current_dir, 'processed', os.path.basename(data_path))
if not os.path.exists(proc_path) and not test_mode:
    os.makedirs(proc_path)

# Read calibration data
if not os.path.exists(cal_path):
    raise FileNotFoundError(f"Calibration file not found: {cal_path}")
res_avg, _ = np.loadtxt(cal_path)

# Convert max velocities to max displacements in px
d_max = np.array(v_max) * dt / res_avg  # m/s -> px/frame


# FIRST PASS: Full frame correlation ===========================================
bckp1_loaded, loaded_vars = piv.backup("load", proc_path, "pass1.npz", disp1_var_names, test_mode)

if bckp1_loaded:
    # Extract loaded variables using the same names as defined in disp1_var_names
    for var_name in disp1_var_names:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing backup data.")

if not bckp1_loaded:
    # Load images from disk    
    imgs = piv.load_images(data_path, frame_nrs, format='tif', lead_0=5,
                           timing=True)

    # TODO: Pre-process images (background subtraction? thresholding?
    #  binarisation to reduce relative influence of bright particles?
    #  low-pass filter to remove camera noise?
    #  mind increase in measurement uncertainty -> PIV book page 140)

    # Calculate the number of correlation frames
    n_corrs = len(imgs) - 1

    # Downsample a copy of the images
    imgs_ds = piv.downsample(imgs.copy(), ds_fac)

    # TODO: Adjust so dimensions are always the same, containing space for windows in 2 directions

    # Pre-allocate arrays for all peaks
    disp1_unf = np.full((n_corrs, n_wins1[0], n_wins1[1], n_peaks1, 2), np.nan)
    int1_unf = np.full((n_corrs,  n_wins1[0], n_wins1[1], n_peaks1), np.nan)

    # Define time arrays beforehand
    time = np.linspace((frame_nrs[0] - 1) * dt,
                       (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

    # Go through all frames and calculate the correlation map
    for i in tqdm(range(n_corrs), desc='First pass'):
        corr_map = sig.correlate(imgs_ds[i + 1], imgs_ds[i],
                                 method='fft', mode='same')

        # TODO: Any processing of the correlation map should happen here
        #  (i.e. blacking out pixels or something)

        # Find peaks in the correlation map
        peaks, int1_unf[i, 0, 0, :] = piv.find_peaks(corr_map, num_peaks=n_peaks1,
                                        min_distance=5)

        # Calculate displacements for all peaks
        disp1_unf[i, 0, 0, :, :] = (
            peaks - np.array( corr_map.shape) // 2) * ds_fac

    # Save unfiltered displacements
    disp1 = disp1_unf.copy()

# Outlier removal using the new modular functions
disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, a=d_max[0], b=d_max[1])
disp1 = piv.strip_peaks(disp1, axis=-2)
print(f"Number of NaNs: {np.sum(np.isnan(disp1))}")

# TODO: filter_neighbours could also consider unstripped peaks?
disp1 = piv.filter_neighbours(disp1, thr=1, n_nbs=(10, 0, 0))
print(f"Number of NaNs: {np.sum(np.isnan(disp1))}")


# Interpolate data to smooth out the x_displacement in time
disp1_spl = make_smoothing_spline(time[~np.isnan(disp1[:, 0, 0, 1])],
                                    disp1[~np.isnan(disp1[:, 0, 0, 1]), 0, 0, 1], lam=5e-7)
disp1_spl = disp1_spl(time).astype(int)
disp1_spl = np.row_stack([np.zeros(len(disp1_spl)), disp1_spl]).T

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass1.npz", test_mode=test_mode,
            time=time, disp1=disp1, disp1_unf=disp1_unf, disp1_spl=disp1_spl,
            int1_unf=int1_unf, n_corrs=n_corrs)

# Calculate velocities for plot
vel1_unf = disp1_unf * res_avg / dt
vel1 = disp1 * res_avg / dt
vel1x_spl = disp1_spl[:, 1] * res_avg / dt

# Scatter plot vx(t)
fig0, ax0 = plt.subplots()
ax0.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 1],
            c='gray', s=2, label='Other peaks')
ax0.scatter(1000 * time, vel1_unf[:, 0, 0, 0, 1], c='blue', s=10,
            label='Most prominent peak')
ax0.scatter(1000 * time, vel1[:, 0, 0, 1], c='orange', s=4,
            label='After outlier removal')
ax0.plot(1000 * time, vel1x_spl, color='red',
            label='Displacement to be used\n in 2nd pass (smoothed)')
ax0.set_ylim([-5, 70])
ax0.set_xlabel('Time (ms)')
ax0.set_ylabel('vx (m/s)')
ax0.legend(loc='upper right', fontsize='small', framealpha=1)

piv.save_cfig(proc_path, 'disp1_vx_t', test_mode=test_mode)

# # Scatter plot vy(t)
# fig0b, ax0b = plt.subplots()
# ax0b.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 0],
#              c='gray', s=2, label='Other peaks')
# ax0b.scatter(1000 * time, vel1_unf[:, 0, 0], c='blue', s=10,
#              label='Most prominent peak')
# ax0b.scatter(1000 * time, vel1[:, 0], c='orange', s=4,
#              label='After outlier removal')
# ax0b.set_ylim([-5, 70])
# ax0b.set_xlabel('Time (ms)')
# ax0b.set_ylabel('vy (m/s)')
# ax0b.legend(loc='upper right', fontsize='small', framealpha=1)

# piv.save_cfig(proc_path, 'disp1_vy_t', test_mode=test_mode)

# # Plot all velocities vy(vx)
# fig1, ax1 = plt.subplots()
# ax1.scatter(vel1[:, 1], vel1[:, 0], c='blue', s=4)
# ax1.set_xlabel('vx (m/s)')
# ax1.set_ylabel('vy (m/s)')
# piv.save_cfig(proc_path, 'disp1_vy_vx', test_mode=test_mode)

# # Plot intensity distribution histogram
# fig2, ax2 = plt.subplots()
# ax2.hist(int1_unf[~np.isnan(int1_unf)], bins=100, log=True)
# ax2.set_xlabel('Intensity')
# ax2.set_ylabel('Count')


# SECOND PASS: Split image into windows and correlate ==========================

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
# # Outlier removal using the new modular functions
# disp2 = piv.filter_outliers(disp2, mode='semicircle_rect', a=d_max[0], b=d_max[1])
# disp2 = piv.strip_peaks(disp2, mode='first_valid')
# disp2_show = piv.filter_outliers(disp2_unf, mode='semicircle_rect', a=d_max[0], b=d_max[1])
# # disp2_show keeps all peaks (no stripping) for visualization
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


# Also later: combine autocorrs might not be best, rather fit profile with turbulence model from turbulence book (Burgers equation, max 3 params)



# Finally, show all figures
plt.show()
