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
test_mode = True
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
frame_nrs = list(range(4500, 5500)) if test_mode else list(range(1, 6000))
dt = 1 / 40000  # [s]

# Data processing settings
v_max = [5, 45]  # [m/s]
ds_fac = 4  # First pass downsampling factor
sum_corrs1 = 21  # Number of correlation frames to sum in first pass (1 = no summation, even = asymmetric)
sum_corrs2 = 21  # Number of correlation frames to sum in second pass (1 = no summation, even = asymmetric)
n_peaks1 = 10  # Number of peaks to find in first pass correlation map
n_wins1 = (1, 1)
n_peaks2 = 10
n_wins2 = (8, 1)  # Number of windows in second pass (rows, cols)

# Validate that sum_corrs1 and sum_corrs2 are positive integers
if sum_corrs1 < 1 or not isinstance(sum_corrs1, int):
    raise ValueError("sum_corrs1 must be a positive integer (1 = no summation, odd = symmetric, even = asymmetric)")
if sum_corrs2 < 1 or not isinstance(sum_corrs2, int):
    raise ValueError("sum_corrs2 must be a positive integer (1 = no summation, odd = symmetric, even = asymmetric)")

# File handling
current_dir = os.path.dirname(os.path.abspath(__file__))
cal_path = os.path.join(current_dir, "calibration",
                        "250624_calibration_PIV_500micron_res_std.txt")
user = getpass.getuser()
if user == "tommieverouden":
    # data_path = os.path.join("/Volumes/Data/Data/250623 PIV/", meas_name)
    data_path = os.path.join("/Users/tommieverouden/Documents/Current data/250623 PIV/", meas_name)
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
    # LOADING & CORRELATION
    # Load images from disk    
    imgs = piv.read_images(data_path, frame_nrs, format='tif', lead_0=5,
                           timing=True)

    # TODO: Pre-process images (background subtraction? thresholding?
    #  binarisation to reduce relative influence of bright particles?
    #  low-pass filter to remove camera noise?
    #  mind increase in measurement uncertainty -> PIV book page 140)

    # Number of correlation maps is shorter than the number of images
    n_corrs = len(imgs) - 1

    # Downsample a copy of the images
    imgs_ds = piv.downsample(imgs.copy(), ds_fac)

    # Go through all frames and calculate the correlation map
    corr1 = np.zeros((n_corrs, n_wins1[0], n_wins1[1], *imgs_ds.shape[1:]))
    for i in range(n_corrs):
        corr1[i, 0, 0, ...] = sig.correlate(imgs_ds[i + 1], imgs_ds[i],
                                 method='fft', mode='same')

        # TODO: Any processing of the correlation map should happen here
        #  (i.e. blacking out pixels or something)

    # Apply windowing (keeping same number of output frames)
    corr1_sum = np.zeros_like(corr1)
    for i in range(n_corrs):
        # Calculate window bounds: odd = symmetric, even = asymmetric (extra frame after)
        start_idx = max(0, i - (sum_corrs1 - 1) // 2)
        end_idx = min(n_corrs, i + sum_corrs1 // 2 + 1)

        # Sum correlation maps in the window
        corr1_sum[i, 0, 0, ...] = np.sum(corr1[start_idx:end_idx, 0, 0, ...], axis=0)
    corr1 = corr1_sum

    # Go through all frames and find peaks in the correlation maps
    # TODO: unf variable naming is messy    
    disp1_unf = np.full((n_corrs, n_wins1[0], n_wins1[1], n_peaks1, 2), np.nan)
    int1_unf = np.full((n_corrs,  n_wins1[0], n_wins1[1], n_peaks1), np.nan)
    
    for i in range(n_corrs):
        # Find peaks in the correlation map
        peaks, int1_unf[i, 0, 0, :] = piv.find_peaks(corr1[i, 0, 0, ...], num_peaks=n_peaks1, min_distance=5)

        # Calculate displacements for all peaks
        disp1_unf[i, 0, 0, :, :] = (
            peaks - np.array(corr1[i, 0, 0, ...].shape) // 2) * ds_fac

    # Save unfiltered displacements
    disp1 = disp1_unf.copy()

# POST-PROCESSING
# Outlier removal
disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, a=d_max[0], b=d_max[1], verbose=True)
disp1 = piv.strip_peaks(disp1, axis=-2)
print(f"Post-processing: kept only brightest candidate, left with {np.sum(np.isnan(disp1))}/{np.size(disp1)} NaNs.")
# TODO: Make it so neighbour filtering is done over an odd number of neighbours, counting the value itself. This is more consistent
disp1_nbs = piv.filter_neighbours(disp1.copy(), thr=1, n_nbs=(40, 0, 0),  verbose=True)

# Define time arrays beforehand
time = np.linspace((frame_nrs[0] - 1) * dt,
                    (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

# Smooth the x displacement in time
disp1_spl = piv.smooth(time, disp1_nbs.copy(), lam=4e-7, type=int)

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
ax.set_title('First pass')
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
    # LOADING & CORRELATION
    # Ensure we have the images loaded (in case only second pass backup failed)
    if 'imgs' not in globals():
        # Load images from disk    
        imgs = piv.read_images(data_path, frame_nrs, format='tif', lead_0=5,
                               timing=True)
    
    # Step 1: Calculate all correlation maps for all windows and frames
    print("Step 1: Calculating all correlation maps...")
    corr_maps = {}  # Store as dict since shapes may vary: {(frame, window_j, window_k): corr_map}
    centres = None  # Will be set from the first frame
    
    for i in trange(n_corrs, desc='Calculating correlation maps'):
        # Split the images into windows
        wnd0, frame_centres = piv.split_n_shift(imgs[i], n_wins2,
                                               shift=disp1_spl[i, 0, 0, :],
                                               shift_mode='before')
        wnd1, _ = piv.split_n_shift(imgs[i + 1], n_wins2,
                                   shift=disp1_spl[i, 0, 0, :],
                                   shift_mode='after')
        
        # Store centres from the first frame
        if centres is None:
            centres = frame_centres
        
        # Calculate correlation maps for all windows
        for j in range(n_wins2[0]):
            for k in range(n_wins2[1]):
                corr_map = sig.correlate(wnd1[j, k], wnd0[j, k], method='fft', mode='same')
                corr_maps[(i, j, k)] = corr_map
    
    # Step 2: Sum correlation maps with alignment and size expansion
    print("Step 2: Summing correlation maps...")
    summed_corr_maps = {}  # Store summed maps: {(frame, window_j, window_k): (summed_map, new_center)}
    
    for i in trange(n_corrs, desc='Summing correlation maps'):
        # Calculate window bounds: odd = symmetric, even = asymmetric (extra frame after)
        start_idx = max(0, i - (sum_corrs2 - 1) // 2)
        end_idx = min(n_corrs, i + sum_corrs2 // 2 + 1)
        
        # Get reference shift (from current frame) - same for all windows
        ref_shift = disp1_spl[i, 0, 0, :]
        
        for j in range(n_wins2[0]):
            for k in range(n_wins2[1]):
                
                # Collect all correlation maps to sum with their shifts
                maps_to_sum = []
                shifts_to_apply = []
                
                for frame_idx in range(start_idx, end_idx):
                    current_shift = disp1_spl[frame_idx, 0, 0, :]
                    shift_diff = current_shift - ref_shift
                    shift_diff_px = np.round(shift_diff).astype(int)
                    
                    maps_to_sum.append(corr_maps[(frame_idx, j, k)])
                    shifts_to_apply.append(shift_diff_px)
                
                # Sum maps with size expansion
                if len(maps_to_sum) == 1:
                    # No summation needed
                    summed_map = maps_to_sum[0]
                    new_center = np.array(summed_map.shape) // 2
                else:
                    # Calculate the expanded size needed
                    base_shape = maps_to_sum[0].shape
                    min_shift = np.min(shifts_to_apply, axis=0)
                    max_shift = np.max(shifts_to_apply, axis=0)
                    
                    # Calculate new size to accommodate all shifted maps
                    new_shape = (base_shape[0] + max_shift[0] - min_shift[0],
                                base_shape[1] + max_shift[1] - min_shift[1])
                    
                    # Calculate new center position
                    new_center = (base_shape[0] // 2 - min_shift[0],
                                 base_shape[1] // 2 - min_shift[1])
                    
                    # Initialize summed map
                    summed_map = np.zeros(new_shape)
                    
                    # Add each map at its shifted position
                    for map_idx, (corr_map, shift) in enumerate(zip(maps_to_sum, shifts_to_apply)):
                        # Calculate position in the expanded map
                        start_y = shift[0] - min_shift[0]
                        start_x = shift[1] - min_shift[1]
                        end_y = start_y + corr_map.shape[0]
                        end_x = start_x + corr_map.shape[1]
                        
                        # Add to the summed map
                        summed_map[start_y:end_y, start_x:end_x] += corr_map
                
                # Store the summed map and its center
                summed_corr_maps[(i, j, k)] = (summed_map, new_center)
    
    # Step 3: Find peaks in all summed correlation maps
    print("Step 3: Finding peaks...")
    disp2_unf = np.full((n_corrs, n_wins2[0], n_wins2[1], n_peaks2, 2), np.nan)
    int2_unf = np.full((n_corrs, n_wins2[0], n_wins2[1], n_peaks2), np.nan)
    
    for i in tqdm(range(n_corrs), desc='Finding peaks'):
        # Get reference shift (from current frame) - same for all windows  
        ref_shift = disp1_spl[i, 0, 0, :]
        
        for j in range(n_wins2[0]):
            for k in range(n_wins2[1]):
                summed_map, map_center = summed_corr_maps[(i, j, k)]
                
                # Find peaks in the summed correlation map
                peaks, int2_unf[i, j, k, :] = piv.find_peaks(summed_map, 
                                                           num_peaks=n_peaks2, 
                                                           floor=10, 
                                                           min_distance=3)

                # Calculate displacements for all peaks
                disp2_unf[i, j, k, :, :] = (ref_shift + peaks - map_center)

    # Save unfiltered displacements
    disp2 = disp2_unf.copy()

# POST-PROCESSING

# Basic global outlier removal of unreasonable displacements
disp2 = piv.filter_outliers('semicircle_rect', disp2_unf, a=d_max[0], b=d_max[1], verbose=True)
disp2 = piv.strip_peaks(disp2, axis=-2)
print(f"Post-processing: kept only brightest candidate, left with {np.sum(np.isnan(disp2))}/{np.size(disp2)} NaNs.")

# Very light neighbour filtering to remove extremes and replace missing values
disp2 = piv.filter_neighbours(disp2, thr=5, n_nbs=(50, 2, 0), verbose=True, mode='r', replace=True)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass2.npz", test_mode=test_mode,
           disp2=disp2, disp2_unf=disp2_unf, int2_unf=int2_unf, centres=centres)

# Calculate velocities for plots
vel2_unf = disp2_unf * res_avg / dt
vel2 = disp2 * res_avg / dt

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
ax1.plot(time * 1000, np.nanmedian(vel2[:, :, :, 0], axis=(1, 2)), label='Median vy')
ax1.fill_between(time * 1000,
                 np.nanmin(vel2[:, :, :, 0], axis=(1, 2)),
                 np.nanmax(vel2[:, :, :, 0], axis=(1, 2)),
                 alpha=0.3, label='Min/Max vy')

# Plot vx (horizontal velocity)
ax1.plot(time * 1000, np.nanmedian(vel2[:, :, :, 1], axis=(1, 2)), label='Median vx')
ax1.fill_between(time * 1000,
                 np.nanmin(vel2[:, :, :, 1], axis=(1, 2)),
                 np.nanmax(vel2[:, :, :, 1], axis=(1, 2)),
                 alpha=0.3, label='Min/Max vx')
ax1.set_ylim(v_max[0] * -1.1, v_max[1] * 1.1)

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Second pass')
ax1.legend()
ax1.grid()

piv.save_cfig(proc_path, "disp2_med",  test_mode=test_mode)

# Set up video writer for velocity profiles
if not test_mode:
    from matplotlib import animation as ani
    
    fig_video, ax_video = plt.subplots(figsize=(10, 6))
    writer = ani.FFMpegWriter(fps=10)

    video_path = os.path.join(proc_path, 'pass2.mp4')
    with writer.saving(fig_video, video_path, dpi=150):
        for i in trange(n_corrs, desc='Creating video'):
            # Clear the axis
            ax_video.clear()
            
            # Get data for current frame (same as random samples code)
            y_pos = centres[:, 0, 0] * res_avg * 1000
            vx2 = vel2[i, :, 0, 1]
            # vx2_nbs = vel2_nbs[i, :, 0, 1]
            vy2 = vel2[i, :, 0, 0]
            # vy2_nbs = vel2_nbs[i, :, 0, 0]

            # Plot using the same style as the random samples
            ax_video.plot(vx2, y_pos, '-o', c=cvd.get_color(1), label='vx')
            ax_video.plot(vy2, y_pos, '-o', c=cvd.get_color(0), label='vy')
            # ax_video.plot(vy2_nbs, y_pos, 'o', mfc='none', c='black', label='vy (filtered neighbours)')

            ax_video.set_xlabel('Velocity (m/s)')
            ax_video.set_ylabel('y position (mm)')
            ax_video.set_xlim(v_max[0] * -1.1, v_max[1] * 1.1)
            ax_video.set_ylim(0, 21.12)
            ax_video.set_title(f'Velocity profiles at frame {i + 1} ({time[i] * 1000:.2f} ms)')
            ax_video.legend()
            ax_video.grid()
            
            writer.grab_frame()
    
    plt.close(fig_video)
    print(f"Video saved to {video_path}")

# TODO: combine autocorrs might not be best, rather fit profile with turbulence model from turbulence book (Burgers equation, with max 3 params)

# Finally, show all figures
plt.show()
