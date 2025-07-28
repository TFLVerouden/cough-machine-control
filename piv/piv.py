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
frame_nrs = list(range(4900, 5000)) if test_mode else list(range(1, 6000))
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

    # Step 1: Calculate correlation maps (with downsampling, no windows/shifts)
    corr1 = piv.calc_corrs(imgs, ds_fac=ds_fac1)

    # Step 2: Sum correlation maps
    corr1 = piv.sum_corrs(corr1, n_tosum=sum_corrs1)

    # Step 3: Find peaks in correlation maps
    disp1, int1 = piv.find_disps(corr1, n_peaks=n_peaks1,
                                 ds_fac=ds_fac1, min_dist=min_dist1)

    # Save unfiltered displacements
    disp1_unf = disp1.copy()

# POST-PROCESSING
# Outlier removal
disp1 = piv.filter_outliers('semicircle_rect', disp1, 
                            a=d_max[0], b=d_max[1], verbose=True)
disp1 = piv.strip_peaks(disp1, axis=-2, verbose=True)
disp1_glo = disp1.copy()

# Neighbour filtering
disp1 = piv.filter_neighbours(disp1, thr=nbs_thr1, n_nbs=n_nbs1, 
                              mode='xy', replace=False, verbose=True)
disp1_nbs = disp1.copy()

# Define time array
time = np.linspace((frame_nrs[0] - 1) * dt,
                   (frame_nrs[0] - 1 + n_corrs - 1) * dt, n_corrs)

# Smooth the x displacement in time
disp1 = piv.smooth(time, disp1, lam=smooth_lam, type=int)

# Save the displacements to a backup file
# TODO: adjust what's being saved
# piv.backup("save", proc_path, "pass1.npz", test_mode=test_mode,
#            time=time, disp1=disp1, disp1_unf=disp1_unf, disp1_spl=disp1_spl,
#            int1=int1, n_corrs=n_corrs)

# Plot a post-processing comparison of the x velocities in time
piv.plot_vel_comp(disp1_unf, disp1_glo, disp1_nbs, disp1, res_avg, frame_nrs, 
                 dt, ylim=(v_max[0] * -1.1, v_max[1] * 1.1),
                 proc_path=proc_path, file_name="disp1", test_mode=test_mode)


# SECOND PASS: Split in 8 windows ==============================================
sum_corrs2 = 21         # Number of correlation maps to sum
n_peaks2 = 10           # Number of peaks to find in correlation map
n_wins2 = (8, 1)        # Number of windows (rows, cols)
win_ov = 0.2            # Overlap between windows
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
    shifts = disp1[:, 0, 0, :]  # Shape: (n_corrs, 2)

    # Step 1: Calculate correlation maps (with windows and shifts)
    corr2 = piv.calc_corrs(imgs, n_wins2, shifts, overlap=win_ov)

    # Step 2: Sum correlation maps with alignment and size expansion
    corr2 = piv.sum_corrs(corr2, sum_corrs2, n_wins2, shifts=shifts)

    # Step 3: Find peaks in summed correlation maps
    disp2, int2 = piv.find_disps(corr2, n_wins2, n_peaks=n_peaks2,
                                         shifts=shifts, floor=pk_floor,
                                         min_dist=min_dist2)

    # Save unfiltered displacements
    disp2_unf = disp2.copy()

    # Extract centres from correlation maps for plotting and saving
    # TODO: Not correct
    centres = np.zeros((n_wins2[0], n_wins2[1], 2))
    for j in range(n_wins2[0]):
        for k in range(n_wins2[1]):
            _, centre = corr2[(0, j, k)]  # Use first frame's centres
            centres[j, k] = centre
            print(centre)

# POST-PROCESSING
# Outlier removal
disp2 = piv.filter_outliers('semicircle_rect', disp2,
                            a=d_max[0], b=d_max[1], verbose=True)
disp2 = piv.strip_peaks(disp2, axis=-2, verbose=True)
disp2_glo = disp2.copy()

# Very light neighbour filtering to remove extremes and replace missing values
disp2 = piv.filter_neighbours(disp2, thr=nbs_thr2, n_nbs=n_nbs2,
                              mode='r', replace=True, verbose=True)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass2.npz", test_mode=test_mode,
           disp2=disp2, disp2_unf=disp2_unf, int2_unf=int2, centres=centres)

# Plot some randomly selected velocity profiles
piv.plot_vel_prof(disp2, res_avg, frame_nrs, dt, centres,
                  mode='random')
# TODO: FIX CENTRES
plt.show()

# Plot the median, min and max velocity in time
piv.plot_vel_med(disp2, res_avg, frame_nrs, dt,
                 ylim=(v_max[0] * -1.1, v_max[1] * 1.1),
                 proc_path=proc_path, file_name="disp2_med", test_mode=test_mode)
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
            vy2 = vel2[i, :, 0, 0]

            # Plot using the same style as the random samples
            ax_video.plot(vx2, y_pos, '-o', c=cvd.get_color(1), label='vx')
            ax_video.plot(vy2, y_pos, '-o', c=cvd.get_color(0), label='vy')

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
