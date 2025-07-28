import getpass
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm

import piv_functions as piv

# Add the functions directory to the path and import CVD check
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'functions'))
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

# File handling
current_dir = os.path.dirname(os.path.abspath(__file__))
cal_path = os.path.join(current_dir, "calibration",
                        "250624_calibration_PIV_500micron_res_std.txt")
user = getpass.getuser()
if user == "tommieverouden":
    # data_path = os.path.join("/Volumes/Data/Data/250623 PIV/", meas_name)
    data_path = os.path.join(
        "/Users/tommieverouden/Documents/Current data/250623 PIV/", meas_name)
elif user == "sikke":
    data_path = os.path.join("D:\\Experiments\\PIV\\", meas_name)

# Data saving settings
disp1_var_names = ['time', 'disp1', 'disp1_unf',
                   'disp1_spl', 'int1_unf', 'n_corrs']
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
ds_fac1 = 4             # Downsampling factor
sum_corrs1 = 21         # Number of correlation maps to sum
n_peaks1 = 10           # Number of peaks to find in correlation map
n_wins1 = (1, 1)        # Number of windows (rows, cols)
min_dist1 = 5           # Minimum distance between peaks
n_nbs1 = (41, 1, 1)     # Neighbourhood for local filtering
nbs_thr1 = 1            # Threshold for neighbour filtering
smooth_lam = 4e-7       # Smoothing lambda for splines

print("FIRST PASS: full frame correlation")
bckp1_loaded, loaded_vars = piv.backup(
    "load", proc_path, "pass1.npz", disp1_var_names, test_mode)

if bckp1_loaded:
    # Extract loaded variables using the same names as defined in disp1_var_names
    for var_name in disp1_var_names:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing backup data.")

if not bckp1_loaded:
    # LOADING & CORRELATION
    # Load images from disk
    imgs = piv.read_imgs(data_path, frame_nrs, format='tif', lead_0=5,
                         timing=True)

    # TODO: Pre-process images (background subtraction? thresholding?
    #  binarisation to reduce relative influence of bright particles?
    #  low-pass filter to remove camera noise?
    #  mind increase in measurement uncertainty -> PIV book page 140)

    # Number of correlation maps is shorter than the number of images
    n_corrs = len(imgs) - 1

    # Step 1: Calculate correlation maps (with downsampling, no windows, no shifts)
    corr1 = piv.calc_corrs(imgs, n_wins1,
                           shifts=None,
                           ds_fac=ds_fac1)

    # Step 2: Sum correlation maps with windowing
    corr1 = piv.sum_corrs(corr1,
                          shifts=np.zeros((n_corrs, 2)),
                          n_tosum=sum_corrs1,
                          n_wins=n_wins1)

    # Step 3: Find peaks in correlation maps
    disp1_unf, int1_unf = piv.find_disps(corr1,
                                         shifts=np.zeros((n_corrs, 2)),
                                         n_wins=n_wins1,
                                         n_peaks=n_peaks1,
                                         ds_fac=ds_fac1,
                                         min_dist=min_dist1)

    # Save unfiltered displacements
    disp1 = disp1_unf.copy()

# POST-PROCESSING
# Outlier removal
disp1 = piv.filter_outliers(
    'semicircle_rect', disp1_unf, a=d_max[0], b=d_max[1], verbose=True)
disp1 = piv.strip_peaks(disp1, axis=-2, verbose=True)

# Neighbour filtering
disp1_nbs = piv.filter_neighbours(
    disp1.copy(), thr=nbs_thr1, n_nbs=n_nbs1, verbose=True, mode='xy', replace=False)

# Define time arrays beforehand
time = np.linspace((frame_nrs[0] - 1) * dt,
                   (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

# Smooth the x displacement in time
disp1_spl = piv.smooth(time, disp1_nbs.copy(), lam=smooth_lam, type=int)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass1.npz", test_mode=test_mode,
           time=time, disp1=disp1, disp1_unf=disp1_unf, disp1_spl=disp1_spl,
           int1_unf=int1_unf, n_corrs=n_corrs)

# TODO: Move plot to function
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
ax.plot(1000 * time, vel1_nbs[:, 0, 0, 1], 'o', mfc='none',
        ms=4, c='black', label='Filtered neighbours')
ax.plot(1000 * time, vel1x_spl, c=cvd.get_color(1),
        label='Displacement to be used\n in 2nd pass (smoothed)')

ax.set_ylim(v_max[0] * -1.1, v_max[1] * 1.1)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('First pass')
ax.legend()
ax.grid()

piv.save_cfig(proc_path, "disp1", test_mode=test_mode)


# SECOND PASS: Split in 8 windows ==============================================
sum_corrs2 = 21         # Number of correlation maps to sum
n_peaks2 = 10           # Number of peaks to find in correlation map
n_wins2 = (8, 1)        # Number of windows (rows, cols)
min_dist2 = 3           # Minimum distance between peaks
pk_floor = 10           # Minimum peak intensity
n_nbs2 = (51, 3, 1)     # Neighbourhood for local filtering
nbs_thr2 = 5            # Threshold for neighbour filtering

print(f"SECOND PASS: {n_wins2} windows")
bckp2_loaded, loaded_vars2 = piv.backup(
    "load", proc_path, "pass2.npz", disp2_var_names, test_mode)

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
        imgs = piv.read_imgs(data_path, frame_nrs, format='tif', lead_0=5,
                             timing=True)

    # Prepare shifts from first pass (convert from smoothed displacements)
    shifts = disp1_spl[:, 0, 0, :]  # Shape: (n_corrs, 2)

    # Step 1: Calculate correlation maps (no downsampling, with windows and shifts)
    corr2 = piv.calc_corrs(imgs, n_wins2, shifts, overlap=0.2, ds_fac=1)

    # Step 2: Sum correlation maps with alignment and size expansion
    corr2 = piv.sum_corrs(corr2, shifts=shifts, n_tosum=sum_corrs2,
                          n_wins=n_wins2)

    # Step 3: Find peaks in summed correlation maps
    disp2_unf, int2_unf = piv.find_disps(corr2, shifts=shifts,
                                         n_wins=n_wins2,
                                         n_peaks=n_peaks2,
                                         floor=pk_floor,
                                         min_dist=min_dist2)

    # Save unfiltered displacements
    disp2 = disp2_unf.copy()

    # Extract centres from correlation maps for plotting and saving
    centres = np.zeros((n_wins2[0], n_wins2[1], 2))
    for j in range(n_wins2[0]):
        for k in range(n_wins2[1]):
            _, centre = corr2[(0, j, k)]  # Use first frame's centres
            centres[j, k] = centre

# POST-PROCESSING

# Basic global outlier removal of unreasonable displacements
disp2 = piv.filter_outliers(
    'semicircle_rect', disp2_unf, a=d_max[0], b=d_max[1], verbose=True)
disp2 = piv.strip_peaks(disp2, axis=-2, verbose=True)

# Very light neighbour filtering to remove extremes and replace missing values
disp2 = piv.filter_neighbours(disp2, thr=nbs_thr2, n_nbs=n_nbs2, verbose=True, mode='r', replace=True)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass2.npz", test_mode=test_mode,
           disp2=disp2, disp2_unf=disp2_unf, int2_unf=int2_unf, centres=centres)

# Calculate velocities for plots
vel2_unf = disp2_unf * res_avg / dt
vel2 = disp2 * res_avg / dt

# TODO: Move plots to functions

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
ax1.plot(time * 1000,
         np.nanmedian(vel2[:, :, :, 0], axis=(1, 2)), label='Median vy')
ax1.fill_between(time * 1000,
                 np.nanmin(vel2[:, :, :, 0], axis=(1, 2)),
                 np.nanmax(vel2[:, :, :, 0], axis=(1, 2)),
                 alpha=0.3, label='Min/Max vy')

# Plot vx (horizontal velocity)
ax1.plot(time * 1000,
         np.nanmedian(vel2[:, :, :, 1], axis=(1, 2)), label='Median vx')
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


# TODO: Move video creation to function

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
            ax_video.set_title(
                f'Velocity profiles at frame {i + 1} ({time[i] * 1000:.2f} ms)')
            ax_video.legend()
            ax_video.grid()

            writer.grab_frame()

    plt.close(fig_video)
    print(f"Video saved to {video_path}")


# THIRD PASS: Split in 24 windows ==============================================
# sum_corrs2 = 2
# n_peaks  





# TODO: fit profile with turbulence model from turbulence book (Burgers equation, with max 3 params)

# Finally, show all figures
plt.show()
