import getpass
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
from tqdm import trange, tqdm

import piv_functions as piv

# Add the functions directory to the path and import CVD check
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'functions'))
import cvd_check as cvd

# Set CVD-friendly colors
cvd.set_cvd_friendly_colors()


# Set experimental parameters
test_mode = False
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
frame_nrs = list(range(600, 1000)) if test_mode else list(range(1, 6000))
dt = 1 / 40000  # [s] 

# Data processing settings
v_max = [5, 45]  # [m/s]
ds_fac = 4  # First pass downsampling factor
n_peaks1 = 10  # Number of peaks to find in first pass correlation map
n_wins1 = (1, 1)
n_peaks2 = 10
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
disp2_var_names = ['disp2', 'disp2_unf', 'int2_unf', 'centres']

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

    # Go through all frames and calculate the correlation map
    for i in tqdm(range(n_corrs), desc='First pass'):
        corr_map = sig.correlate(imgs_ds[i + 1], imgs_ds[i],
                                 method='fft', mode='same')

        # TODO: Any processing of the correlation map should happen here
        #  (i.e. blacking out pixels or something)

        # Find peaks in the correlation map
        peaks, int1_unf[i, 0, 0, :] = piv.find_peaks(corr_map, num_peaks=n_peaks1, min_distance=5)

        # Calculate displacements for all peaks
        disp1_unf[i, 0, 0, :, :] = (
            peaks - np.array( corr_map.shape) // 2) * ds_fac

    # Save unfiltered displacements
    disp1 = disp1_unf.copy()

# Outlier removal
disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, a=d_max[0], b=d_max[1], verbose=True)
disp1 = piv.strip_peaks(disp1, axis=-2)
print(f"Keeping only brightest candidate, left with {np.sum(np.isnan(disp1))}/{np.size(disp1)} NaNs.")
disp1_nbs = piv.filter_neighbours(disp1.copy(), thr=1, n_nbs=(40, 0, 0), verbose=True)

# Define time arrays beforehand
time = np.linspace((frame_nrs[0] - 1) * dt,
                    (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

# Smooth the x displacement in time
disp1_spl = piv.smooth(time, disp1_nbs.copy(), lam=5e-7, type=int)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass1.npz", test_mode=test_mode,
            time=time, disp1=disp1, disp1_unf=disp1_unf, disp1_spl=disp1_spl,
            int1_unf=int1_unf, n_corrs=n_corrs)

# Calculate velocities for plot
vel1_unf = disp1_unf * res_avg / dt
vel1 = disp1 * res_avg / dt
vel1_nbs = disp1_nbs * res_avg / dt
vel1x_spl = disp1_spl[:, 0, 0, 1] * res_avg / dt

# Scatter plot vx(t)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.tile(time[:, None] * 1000, (1, n_peaks1)).flatten(),
           vel1_unf[:, 0, 0, :, 1].flatten(), 'x', c='gray', alpha=0.5, ms=4, label='Unfiltered vx')
ax.plot(1000 * time, vel1[:, 0, 0, 1], 'o', ms=4, c=cvd.get_color(1),
           label='Filtered globally')
ax.plot(1000 * time, vel1_nbs[:, 0, 0, 1], 'o', mfc='none', ms=4, c='black', label='Filtered neighbours')
ax.plot(1000 * time, vel1x_spl, c=cvd.get_color(1), label='Displacement to be used\n in 2nd pass (smoothed)')

ax.set_ylim(v_max[0] * -1.1, v_max[1] * 1.1)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('First pass summary')
ax.legend()
ax.grid()

piv.save_cfig(proc_path, "disp1", test_mode=test_mode)


# SECOND PASS: Split image into windows and correlate ==========================

# Try to load existing backup data
bckp2_loaded, loaded_vars2 = piv.backup("load", proc_path, "pass2.npz", disp2_var_names, test_mode)

if bckp2_loaded:
    # Extract loaded variables using the same names as defined in disp2_var_names
    for var_name in disp2_var_names:
        globals()[var_name] = loaded_vars2.get(var_name)
    print("Loaded existing second pass backup data.")

if not bckp2_loaded:
    # Ensure we have the images loaded (in case only second pass backup failed)
    if 'imgs' not in globals():
        # Load images from disk    
        imgs = piv.load_images(data_path, frame_nrs, format='tif', lead_0=5,
                               timing=True)
    
    # Pre-allocate array for all peaks: (frame idx, window idx, peak idx, 2)
    disp2_unf = np.full((n_corrs, n_wins2[0], n_wins2[1], n_peaks2, 2), np.nan)
    int2_unf = np.full((n_corrs, n_wins2[0], n_wins2[1], n_peaks2), np.nan)

    for i in tqdm(range(n_corrs), desc='Second pass'):
        # Split the images into horizontal rectangular windows, shifted by
        # the interpolated/smoothed displacements from the first pass
        wnd0, centres = piv.split_n_shift(imgs[i], n_wins2,
                                          shift=disp1_spl[i, 0, 0, :],
                                          shift_mode='before')
        wnd1, _ = piv.split_n_shift(imgs[i + 1], n_wins2,
                                    shift=disp1_spl[i, 0, 0, :],
                                    shift_mode='after')

        # Loop through all windows
        for j in range(n_wins2[0]):
            for k in range(n_wins2[1]):
                # Calculate the correlation map for each window pair
                corr_map = sig.correlate(wnd1[j, k], wnd0[j, k],
                                         method='fft', mode='same')

                # TODO: Any processing of the correlation map could happen here
                #  (i.e. blacking out pixels or something)

                # Find peaks in the correlation maps
                peaks, int2_unf[i, j, k, :] = piv.find_peaks(corr_map, num_peaks=n_peaks2, floor=10, min_distance=3)

                # Calculate displacements for all peaks
                disp2_unf[i, j, k, :, :] = (disp1_spl[i, 0, 0, :] + peaks
                                           - np.array(corr_map.shape) // 2)

    # Save unfiltered displacements
    disp2 = disp2_unf.copy()

# Basic global outlier removal of unreasonable displacements
disp2 = piv.filter_outliers('semicircle_rect', disp2_unf, a=d_max[0], b=d_max[1], verbose=True)
disp2 = piv.strip_peaks(disp2, axis=-2)
print(f"Keeping only brightest candidate, left with {np.sum(np.isnan(disp2))}/{np.size(disp2)} NaNs.")

# Neighbour filtering
disp2_nbs = piv.filter_neighbours(disp2.copy(), thr=4, n_nbs=(20, 2, 0), verbose=True, mode='r')

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass2.npz", test_mode=test_mode,
           disp2=disp2, disp2_unf=disp2_unf, int2_unf=int2_unf, centres=centres)

# Calculate velocities for plots
vel2_unf = disp2_unf * res_avg / dt
vel2 = disp2 * res_avg / dt
vel2_nbs = disp2_nbs * res_avg / dt

# # Plot the velocity profiles for randomly selected frames
# np.random.seed(42)  # For reproducible results
# sample_frames = np.random.choice(n_corrs, size=min(10, n_corrs), replace=False)
# sample_frames = np.sort(sample_frames)  # Sort for better organization

# for sample_frame in sample_frames:
#     fig0, ax0 = plt.subplots(figsize=(10, 6))

#     y_pos = centres[:, 0, 0] * res_avg * 1000
#     vx2 = vel2[sample_frame, :, 0, 1]
#     vx2_nbs = vel2_nbs[sample_frame, :, 0, 1]
#     vy2 = vel2[sample_frame, :, 0, 0]
#     vy2_nbs = vel2_nbs[sample_frame, :, 0, 0]

#     ax0.plot(vx2, y_pos, '-o', c=cvd.get_color(1), label='vx (filtered)')
#     ax0.plot(vx2_nbs, y_pos, 'o', mfc='none', c='black', label='vx (filtered neighbours)')
#     ax0.plot(vy2, y_pos, '-o', c=cvd.get_color(0), label='vy (filtered)')
#     ax0.plot(vy2_nbs, y_pos, 'o', mfc='none', c='black', label='vy (filtered neighbours)')

#     ax0.set_xlabel('Velocity (m/s)')
#     ax0.set_ylabel('y position (mm)')
#     # ax0.set_yticklabels([])

#     ax0.legend()
#     ax0.grid()

#     ax0.set_xlim(v_max[0] * -1.1, v_max[1] * 1.1)
#     ax0.set_title(f'Velocity profiles at frame {sample_frame + 1} ({time[sample_frame] * 1000:.2f} ms)')


# Plot the median velocity in time, show the min and max as a shaded area - Regular filtering
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Plot vy (vertical velocity)
ax1.plot(time * 1000, np.nanmean(vel2[:, :, :, 0], axis=(1, 2)), label='Mean vy')
ax1.fill_between(time * 1000,
                 np.nanmin(vel2[:, :, :, 0], axis=(1, 2)),
                 np.nanmax(vel2[:, :, :, 0], axis=(1, 2)),
                 alpha=0.3, label='Min/Max vy')

# Plot vx (horizontal velocity)
ax1.plot(time * 1000, np.nanmedian(vel2[:, :, :, 1], axis=(1, 2)), label='Mean vx')
ax1.fill_between(time * 1000,
                 np.nanmin(vel2[:, :, :, 1], axis=(1, 2)),
                 np.nanmax(vel2[:, :, :, 1], axis=(1, 2)),
                 alpha=0.3, label='Min/Max vx')
ax1.set_ylim(v_max[0] * -1.1, v_max[1] * 1.1)

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Second pass (after global filter)')
ax1.legend()
ax1.grid()

piv.save_cfig(proc_path, "disp2_med",  test_mode=test_mode)

# Set up video writer for velocity profiles - based on random samples plotting style
if not test_mode:
    from matplotlib import animation as ani
    
    fig_video, ax_video = plt.subplots(figsize=(10, 6))
    writer = ani.FFMpegWriter(fps=10)

    video_path = os.path.join(proc_path, 'pass2.mp4')
    with writer.saving(fig_video, video_path, dpi=150):
        for i in range(n_corrs):
            # Clear the axis
            ax_video.clear()
            
            # Get data for current frame (same as random samples code)
            y_pos = centres[:, 0, 0] * res_avg * 1000
            vx2 = vel2[i, :, 0, 1]
            vx2_nbs = vel2_nbs[i, :, 0, 1]
            vy2 = vel2[i, :, 0, 0]
            vy2_nbs = vel2_nbs[i, :, 0, 0]

            # Plot using the same style as the random samples
            ax_video.plot(vx2, y_pos, '-o', c=cvd.get_color(1), label='vx (filtered)')
            ax_video.plot(vx2_nbs, y_pos, 'o', mfc='none', c='black', label='vx (filtered neighbours)')
            ax_video.plot(vy2, y_pos, '-o', c=cvd.get_color(0), label='vy (filtered)')
            ax_video.plot(vy2_nbs, y_pos, 'o', mfc='none', c='black', label='vy (filtered neighbours)')

            ax_video.set_xlabel('Velocity (m/s)')
            ax_video.set_ylabel('y position (mm)')
            ax_video.set_xlim(v_max[0] * -1.1, v_max[1] * 1.1)
            ax_video.set_title(f'Velocity profiles at frame {i + 1} ({time[i] * 1000:.2f} ms)')
            ax_video.legend()
            ax_video.grid()
            
            writer.grab_frame()
    
    plt.close(fig_video)
    print(f"Video saved to {video_path}")

# TODO: combine autocorrs might not be best, rather fit profile with turbulence model from turbulence book (Burgers equation, with max 3 params)

# Finally, show all figures
plt.show()
