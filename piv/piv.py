import getpass
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from scipy import signal as sig
from tqdm import trange, tqdm

import piv_functions as piv


# Set experimental parameters
test_mode = True
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
frame_nrs = list(range(3000, 3100)) if test_mode else list(range(1, 6000))
dt = 1 / 40000  # [s] 

# Data processing settings
v_max = [10, 50]  # [m/s]
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

# Outlier removal
disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, a=d_max[0], b=d_max[1])
disp1 = piv.strip_peaks(disp1, axis=-2)
print(f"Number of NaNs: {np.sum(np.isnan(disp1))}/{np.size(disp1)}")

# TODO: filter_neighbours could also consider unstripped peaks?
disp1 = piv.filter_neighbours(disp1, thr=1, n_nbs=(20, 0, 0))
print(f"Number of NaNs: {np.sum(np.isnan(disp1))}/{np.size(disp1)}")

# Define time arrays beforehand
time = np.linspace((frame_nrs[0] - 1) * dt,
                    (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

# Smooth the x displacement in time
disp1_spl = piv.smooth(time, disp1.copy(), lam=5e-7, type=int)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass1.npz", test_mode=test_mode,
            time=time, disp1=disp1, disp1_unf=disp1_unf, disp1_spl=disp1_spl,
            int1_unf=int1_unf, n_corrs=n_corrs)

# Calculate velocities for plot
vel1_unf = disp1_unf * res_avg / dt
vel1 = disp1 * res_avg / dt
vel1x_spl = disp1_spl[:, 0, 0, 1] * res_avg / dt

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
ax0.set_ylim([-5, 45])
ax0.set_xlabel('Time (ms)')
ax0.set_ylabel('vx (m/s)')
ax0.legend(loc='upper right', fontsize='small', framealpha=1)

piv.save_cfig(proc_path, 'disp1_vx_t', test_mode=test_mode)

# Scatter plot vy(t)
fig0b, ax0b = plt.subplots()
ax0b.scatter(np.tile(1000 * time[:, None], (1, n_peaks1)), vel1_unf[..., 0],
             c='gray', s=2, label='Other peaks')
ax0b.scatter(1000 * time, vel1_unf[:, 0, 0, 0, 0], c='blue', s=10,
             label='Most prominent peak')
ax0b.scatter(1000 * time, vel1[:, 0, 0, 0], c='orange', s=4,
             label='After outlier removal')
ax0b.set_ylim([-5, 45])
ax0b.set_xlabel('Time (ms)')
ax0b.set_ylabel('vy (m/s)')
ax0b.legend(loc='upper right', fontsize='small', framealpha=1)

piv.save_cfig(proc_path, 'disp1_vy_t', test_mode=test_mode)

# Plot all velocities vy(vx)
fig1, ax1 = plt.subplots()
ax1.scatter(vel1[:, 0, 0, 1], vel1[:, 0, 0, 0], c='blue', s=4)
ax1.set_xlabel('vx (m/s)')
ax1.set_ylabel('vy (m/s)')
piv.save_cfig(proc_path, 'disp1_vy_vx', test_mode=test_mode)


# SECOND PASS: Split image into windows and correlate ==========================

# Data saving settings for second pass
disp2_var_names = ['disp2', 'disp2_unf', 'int2_unf', 'centres']

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

                # TODO: Any processing of the correlation map should happen here
                #  (i.e. blacking out pixels or something)

                # Find peaks in the correlation maps
                peaks, int2_unf[i, j, k, :] = piv.find_peaks(corr_map, num_peaks=n_peaks2,
                                                min_distance=3)

                # Calculate displacements for all peaks
                disp2_unf[i, j, k, :, :] = (disp1_spl[i, 0, 0, :] + peaks
                                           - np.array(corr_map.shape) // 2)

    # Save unfiltered displacements
    disp2 = disp2_unf.copy()

# Basic global outlier removal of unreasonable displacements
disp2 = piv.filter_outliers('semicircle_rect', disp2_unf, a=d_max[0], b=d_max[1])
disp2 = piv.strip_peaks(disp2, axis=-2)
print(f"Number of NaNs: {np.sum(np.isnan(disp2))}/{np.size(disp2)}")
            
# TODO: filtering neighbours
disp2 = piv.filter_neighbours(disp2, thr=100, n_nbs=(2, 2, 0))
print(f"Number of NaNs: {np.sum(np.isnan(disp2))}/{np.size(disp2)}")

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass2.npz", test_mode=test_mode,
           disp2=disp2, disp2_unf=disp2_unf, int2_unf=int2_unf, centres=centres)

# Calculate velocities for plots
vel2_unf = disp2_unf * res_avg / dt
vel2 = disp2 * res_avg / dt

# Plot velocity field for a sample frame
sample_frame = min(50, n_corrs - 1) if not test_mode else min(10, n_corrs - 1)

# Create a figure for velocity vectors
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot velocity vectors at window centres for the sample frame
if centres is not None:
    # Plot all window centres in gray
    for j in range(n_peaks2):
        valid_mask = ~np.isnan(vel2_unf[sample_frame, :, :, j, :]).any(axis=-1)
        if np.any(valid_mask):
            y_pos, x_pos = np.where(valid_mask)
            ax2.scatter(centres[y_pos, x_pos, 1] * res_avg * 1000, 
                       centres[y_pos, x_pos, 0] * res_avg * 1000, 
                       c='lightgray', s=10, alpha=0.5)
    
    # Plot filtered velocities
    valid_mask = ~np.isnan(vel2[sample_frame, :, :, :]).any(axis=-1)
    if np.any(valid_mask):
        y_pos, x_pos = np.where(valid_mask)
        
        # Create velocity vectors
        u = vel2[sample_frame, y_pos, x_pos, 1]  # vx
        v = vel2[sample_frame, y_pos, x_pos, 0]  # vy
        x_centers = centres[y_pos, x_pos, 1] * res_avg * 1000  # mm
        y_centers = centres[y_pos, x_pos, 0] * res_avg * 1000  # mm
        
        # Plot velocity vectors
        ax2.quiver(x_centers, y_centers, u, v,
                  scale=200, scale_units='xy', angles='xy', 
                  color='blue', alpha=0.8, width=0.003)

ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
ax2.set_title(f'Velocity field at t = {time[sample_frame]*1000:.2f} ms')
# ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

piv.save_cfig(proc_path, 'disp2_velocity_field', test_mode=test_mode)

# Plot velocity profiles along the centerline
if centres is not None and n_wins2[1] == 1:  # Only for 1D window arrays
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot vx vs y
    y_positions = centres[:, 0, 0] * res_avg * 1000  # mm
    vx_profile = vel2[sample_frame, :, 0, 1]  # vx at centerline
    vy_profile = vel2[sample_frame, :, 0, 0]  # vy at centerline
    
    ax3a.plot(vx_profile, y_positions, 'b-o', markersize=4, label='vx')
    ax3a.set_xlabel('vx (m/s)')
    ax3a.set_ylabel('y (mm)')
    fig3.suptitle(f'Velocity profiles at t = {time[sample_frame]*1000:.2f} ms')
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim([-5, 40])  # Set x-limits for vx profile
    
    ax3b.plot(vy_profile, y_positions, 'r-o', markersize=4, label='vy')
    ax3b.set_xlabel('vy (m/s)')
    ax3b.set_ylabel('y (mm)')

    # Use same scaling as ax3a for consistency
    ax3b.set_xlim(ax3a.get_xlim())

    # ax3b.set_title(f'Vertical velocity profile at t = {time[sample_frame]*1000:.2f} ms')
    ax3b.grid(True, alpha=0.3)
    
    piv.save_cfig(proc_path, 'disp2_velocity_profiles', test_mode=test_mode)

# Plot all velocities vy(vx)
fig4, ax4 = plt.subplots()
ax4.scatter(vel2[:, 0, 0, 1], vel2[:, 0, 0, 0], c='blue', s=4)
ax4.set_xlabel('vx (m/s)')
ax4.set_ylabel('vy (m/s)')
piv.save_cfig(proc_path, 'disp2_vy_vx', test_mode=test_mode)


# Set up video writer for velocity profiles
# if not test_mode and centres is not None and n_wins2[1] == 1:  # Only for 1D window arrays
fig_video, (ax_vx, ax_vy) = plt.subplots(1, 2, figsize=(12, 5))
writer = ani.FFMpegWriter(fps=10)

video_path = os.path.join(proc_path, 'disp2.mp4')
with writer.saving(fig_video, video_path, dpi=150):
    for i in range(n_corrs):
        # Clear both axes
        ax_vx.clear()
        ax_vy.clear()
        
        # Get y positions and velocity profiles for current frame
        y_positions = centres[:, 0, 0] * res_avg * 1000  # mm
        vx_profile = vel2[i, :, 0, 1]  # vx at centerline
        vy_profile = vel2[i, :, 0, 0]  # vy at centerline
        
        # Plot vx profile
        ax_vx.plot(vx_profile, y_positions, 'b-o', markersize=4, label='vx')
        ax_vx.set_xlabel('vx (m/s)')
        ax_vx.set_ylabel('y (mm)')
        ax_vx.grid(True, alpha=0.3)
        ax_vx.set_xlim([-5, 40])
        ax_vx.set_ylim([0, 21])
        
        # Plot vy profile  
        ax_vy.plot(vy_profile, y_positions, 'r-o', markersize=4, label='vy')
        ax_vy.set_xlabel('vy (m/s)')
        ax_vy.set_ylabel('y (mm)')
        ax_vy.grid(True, alpha=0.3)
        ax_vy.set_xlim([-5, 40])
        ax_vy.set_ylim([0, 21])
        
        # Set consistent title
        fig_video.suptitle(f'Velocity profiles at t = {time[i]*1000:.2f} ms')
        
        writer.grab_frame()
plt.close(fig_video)

# TODO: Advanced visualization and analysis
# - Video generation of velocity fields over time
# - Statistical analysis of turbulence properties
# - Profile fitting with turbulence models (Burgers equation)
# - Also later: combine autocorrs might not be best, rather fit profile with turbulence model from turbulence book (Burgers equation, max 3 params)

# Finally, show all figures
plt.show()
