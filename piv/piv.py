import os
import cv2 as cv
import numpy as np
from natsort import natsorted
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from tqdm import tqdm
import matplotlib.pyplot as plt
import getpass

import piv_functions as piv

# Set experimental parameters
test_mode = True
meas_name = '250624_1333_80ms_whand'  # Name of the measurement
frame_nrs = [930, 931] if test_mode else list(range(1, 6000))
dt = 1/40000 # [s]

# Data processing settings
v_max = [15, 150] # [m/s]
ds_fac = 4  # First pass downsampling factor
num_peaks = 10  # Number of peaks to find in first pass correlation map

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


#%% FIRST PASS: Full frame correlation =========================================

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

    n_frames = len(imgs) - 1

    # Pre-allocate array for all peaks: (n_frames, num_peaks, 2) [vy, vx]
    disp1 = np.full((n_frames, num_peaks, 2), np.nan)

    # Define time arrays beforehand
    time = np.linspace((frame_nrs[0] - 1) * dt, (frame_nrs[0] - 1 + n_frames - 1) * dt, n_frames)

    for i in tqdm(range(n_frames), desc='First pass'):
        img1 = piv.downsample(imgs[i + 1], ds_fac)
        img0 = piv.downsample(imgs[i], ds_fac)
        corr_map = sig.correlate(img1, img0, method='fft')
        peaks, int_unf = piv.find_peaks(corr_map, num_peaks=num_peaks, min_distance=5)

        # Calculate velocities for all peaks
        disp1[i, :, :] = (peaks - np.array(corr_map.shape) // 2) * ds_fac  # shape (n_found, 2)

    # Save unfiltered displacements
    disp1_unf = disp1.copy()

    # Outlier removal
    # TODO: Do something with the intensities of the peaks?
    disp1 = piv.remove_outliers(disp1, y_max=d_max[0], x_max=d_max[1], strip=True)

    # Save the displacements to a file
    if not test_mode:
        np.savez(os.path.join(proc_path, 'disp1'), time=time, disp1=disp1,
                 disp1_unf=disp1_unf, int_unf=int_unf)

# # Interpolate data to smooth out the x_displacement in time
# disp1_spl = make_smoothing_spline(time[~np.isnan(disp1[:, 1])],
#                                   disp1[~np.isnan(disp1[:, 1]), 1], lam=5e-7)
# disp1_spl = disp1_spl(time).astype(int)
#
# # Calculate velocities for plot
# vel1_unf = disp1_unf * res_avg / dt
# vel1 = disp1 * res_avg / dt
# vel1x_spl = disp1_spl * res_avg / dt
#
# # Scatter plot vx(t)
# plt.figure()
# plt.scatter(np.tile(1000*time[:, None], (1, num_peaks)), vel1_unf[..., 1],
#             c='gray', s=2, label='Other peaks')
# plt.scatter(1000*time, vel1_unf[:, 0, 1], c='blue', s=10,
#             label='Most prominent peak')
# plt.scatter(1000*time, vel1[:, 1], c='orange', s=4,
#             label='After outlier removal')
# plt.plot(1000*time, vel1x_spl, label='Displacement to be used\n in 2nd pass (smoothed)', color='red')
# plt.ylim([-15, 150])
# plt.xlabel('Time (ms)')
# plt.ylabel('vx (m/s)')
# plt.legend(loc='upper right', fontsize='small', framealpha=1)
#
# # Save plot as pdf
# if ~test_mode:
#     plt.savefig(os.path.join(proc_path, 'disp1_vx_t.pdf'), bbox_inches='tight')
# plt.show()

# Split image nr 930 into windows
windows1, centres1 = piv.split_image(imgs, (4,1), overlap=0.2, shift=(0, 20), shift_mode='after',
            plot=True)
windows0, centres0 = piv.split_image(imgs, (4,1), overlap=0.2, shift=(0, 20), shift_mode='before',
            plot=True)
print()

# map1 = sig.correlate(downsample(imgs[1], factor=8), downsample(imgs[0], factor=8), method='fft')
# peaks, _ = find_peaks(map1, num_peaks=5, min_distance=5)
#
# # Todo: check this list of peak with previous and next frame (see step 3 in PIV book page 148)
# # If none match, interpolate between the two frames. For now, just take the first peak.
#
# disp1 = peaks[0] - np.array(map1.shape) // 2
# print(disp1 * 8 * res_avg / dt)

# # Split images into overlapping windows
# windows, centres = split_image(imgs, nr_windows=(16, 1), overlap=0.5)
#
# # # Plot the windows and centres on top of the first image
# # plt.imshow(imgs[0], cmap='gray')
# # for i, window in enumerate(windows):
# #     y, x = centres[i]
# #     rect = plt.Rectangle((x - window.shape[1] / 2, y - window.shape[0] / 2),
# #                          window.shape[1], window.shape[0], linewidth=1,
# #                          edgecolor='r', facecolor='none')
# #     plt.gca().add_patch(rect)
# #     plt.plot(x, y, 'ro')  # Plot the centre
# # plt.show()
#
# # Cycle through all windows in one specific image and correlate them with the corresponding windows in the other image
# maps = np.array([[sig.correlate(window[1], window[0], method='fft')
#          for window in zip(windows[0], windows[1])]])
#
# # TODO: Any processing of the correlation map happens here (i.e. blacking out all pixels outside of a positive semi-circle)
#
# peak, int = find_peaks(maps[0, 7, 0])
# print("Peak coordinates:", peak)
# print("Peak intensity:", int)
# peak = subpixel(maps[0, 7, 0], peak[0])
# print("Subpixel peak coordinates:", peak)
#
# # Get displacement vector from the peak coordinates
# displacement_vector = peak - np.array(maps.shape[3:]) // 2
#
# print(displacement_vector * res_avg / dt)
