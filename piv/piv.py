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
test_mode = False
videos = False
rnd_plots = True
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
frames = list(range(650, 850)) if test_mode else list(range(1, 6000))
dt = 1 / 40000  # [s]

# Data processing settings
v_max = [5, 40]  # [m/s]

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
disp1_var_names = ['disp1_unf', 'int1_unf', 'disp1_glo', 'disp1_nbs',
                   'disp1', 'time']
disp2_var_names = ['disp2_unf', 'int2_unf', 'win_pos2', 'disp2_glo', 'disp2']
disp3_var_names = ['disp3_unf', 'int3_unf', 'win_pos3', 'disp3_glo', 'disp3']

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
n_tosum1 = 21           # Number of correlation maps to sum
n_peaks1 = 10           # Number of peaks to find in correlation map
n_wins1 = (1, 1)        # Number of windows (rows, cols)
min_dist1 = 5           # Minimum distance between peaks
n_nbs1 = (41, 1, 1)     # Neighbourhood for local filtering
nbs_thr1 = 1            # Threshold for neighbour filtering
smooth_lam = 4e-7       # Smoothing lambda for splines

print("FIRST PASS: full frame correlation")
bckp1_loaded, loaded_vars = piv.backup(
    "load", proc_path, "pass1.npz", disp1_var_names) # removed test_mode

if bckp1_loaded:
    # Extract loaded variables
    for var_name in disp1_var_names:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing backup data.")
else:

    # LOADING & CORRELATION
    # Load images from disk
    imgs = piv.read_imgs(data_path, frames, format='tif', lead_0=5,
                         timing=True)

    # TODO: Pre-processing images would happen here
    # Background subtraction? thresholding? binarisation to reduce relative influence of bright particles? low-pass filter to remove camera noise?  do mind an increase in measurement uncertainty -> PIV book page 140

    # Step 1: Calculate correlation maps (with downsampling, no windows/shifts)
    corr1 = piv.calc_corrs(imgs, ds_fac=ds_fac1)

    # Step 2: Sum correlation maps
    corr1_sum = piv.sum_corrs(corr1, n_tosum=n_tosum1)

    # Step 3: Find peaks in correlation maps
    disp1, int1_unf = piv.find_disps(corr1_sum, n_peaks=n_peaks1,
                                 ds_fac=ds_fac1, min_dist=min_dist1)
    disp1_unf = disp1.copy()

# POST-PROCESSING
# Outlier removal
disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, 
                            a=d_max[0], b=d_max[1], verbose=True)
disp1 = piv.strip_peaks(disp1, axis=-2, verbose=True)
disp1_glo = disp1.copy()

# Neighbour filtering
disp1 = piv.filter_neighbours(disp1, thr=nbs_thr1, n_nbs=n_nbs1, 
                            mode='xy', replace=False, verbose=True)
disp1_nbs = disp1.copy()

# Define time array
time = piv.get_time(frames, dt)

# Smooth the x displacement in time
disp1 = piv.smooth(time, disp1, lam=smooth_lam, type=int)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass1.npz", test_mode=test_mode,
        disp1_unf=disp1_unf, int1_unf=int1_unf,
        disp1_glo=disp1_glo, disp1_nbs=disp1_nbs,
        disp1=disp1, time=time)

# PLOTTING
# Plot a post-processing comparison of the x velocities in time
piv.plot_vel_comp(disp1_glo, disp1_nbs, disp1, res_avg, frames, 
                 dt, ylim=(v_max[0] * -1.1, v_max[1] * 1.1),
                 proc_path=proc_path, file_name="pass1_v-t", test_mode=test_mode)


# SECOND PASS: Split in 8 windows ==============================================
n_tosum2 = 21           # Number of correlation maps to sum
n_peaks2 = 10           # Number of peaks to find in correlation map
n_wins2 = (8, 1)        # Number of windows (rows, cols)
win_ov2 = 0.2            # Overlap between windows
min_dist2 = 3           # Minimum distance between peaks
pk_floor2 = 20           # Minimum peak intensity
n_nbs2 = (51, 3, 1)     # Neighbourhood for local filtering
nbs_thr2 = 5            # Threshold for neighbour filtering

print(f"SECOND PASS: {n_wins2} windows")
bckp2_loaded, loaded_vars2 = piv.backup(
    "load", proc_path, "pass2.npz", disp2_var_names) # removed test_mode

if bckp2_loaded:
    # Extract loaded variables
    for var_name in disp2_var_names:
        globals()[var_name] = loaded_vars2.get(var_name)
    print("Loaded existing second pass backup data.")
else:

    # LOADING & CORRELATION
    # Ensure we have the images loaded
    if 'imgs' not in globals():
        imgs = piv.read_imgs(data_path, frames, format='tif', lead_0=5,
                             timing=True)

    # Convert displacements from pass 1 to shifts for pass 2
    shifts2 = piv.disp2shift(n_wins2, disp1)

    # Step 1: Calculate correlation maps (with windows and shifts)
    corr2 = piv.calc_corrs(imgs, n_wins2, shifts=shifts2,
                           overlap=win_ov2)

    # Step 2: Sum correlation maps with alignment and size expansion
    corr2_sum = piv.sum_corrs(corr2, n_tosum2, n_wins2,
                              shifts=shifts2)

    # Step 3: Find peaks in summed correlation maps
    disp2, int2_unf = piv.find_disps(corr2_sum, n_wins2, n_peaks=n_peaks2, 
                                     shifts=shifts2,
                                     floor=pk_floor2, min_dist=min_dist2)

    # Save unfiltered displacements
    disp2_unf = disp2.copy()

    # Get physical window positions for plotting (from first frame)
    _, win_pos2 = piv.split_n_shift(imgs[0], n_wins2)

    # Note: The correlation map centers (used for displacement calculation) are stored separately in each correlation map as the second element of the tuple

# POST-PROCESSING
# Outlier removal
disp2 = piv.filter_outliers('semicircle_rect', disp2_unf,
                            a=d_max[0], b=d_max[1], verbose=True)
disp2 = piv.strip_peaks(disp2, axis=-2, verbose=True)
disp2_glo = disp2.copy()

# Very light neighbour filtering to remove extremes and replace missing values
disp2 = piv.filter_neighbours(disp2, thr=nbs_thr2, n_nbs=n_nbs2,
                            mode='r', replace=True, verbose=True)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass2.npz", test_mode=test_mode,
        disp2_unf=disp2_unf, int2_unf=int2_unf,
        win_pos2=win_pos2, disp2_glo=disp2_glo, disp2=disp2)

# PLOTTING
# Plot the median, min and max velocity in time
piv.plot_vel_med(disp2, res_avg, frames, dt,
                 ylim=(v_max[0] * -1.1, v_max[1] * 1.1),
                 title='Second pass',
                 proc_path=proc_path, file_name="pass2_v_med", test_mode=test_mode)

# Plot some randomly selected velocity profiles
piv.plot_vel_prof(disp2, res_avg, frames, dt, win_pos2,
                  mode='random', xlim=(-5, 40), ylim=(0, 21.12),
                  proc_path=proc_path, file_name="pass2_v",
                  subfolder='pass2', test_mode=test_mode) if rnd_plots else None

# Plot all velocity profiles in video
piv.plot_vel_prof(disp2, res_avg, frames, dt, win_pos2,
                  mode='video', xlim=(-5, 40), ylim=(0, 21.12),
                  proc_path=proc_path, file_name="pass2_v",
                  test_mode=test_mode) if videos else None


# THIRD PASS: Split in 24 windows ==============================================
n_tosum3 = 4             # Number of correlation maps to sum -> 1 ms mov.av.
n_peaks3 = 5             # Number of peaks to find in correlation map
n_wins3 = (24, 1)        # Number of windows (rows, cols)
win_ov3 = 0             # Overlap between windows
min_dist3 = 3            # Minimum distance between peaks
pk_floor3 = 20           # Minimum peak intensity
d_max[1] = 1.5 * d_max[1] # Less strict filtering in x-direction

n_nbs3 = (1, 3, 1)     # Neighbourhood for local filtering
nbs_thr3 = 3            # Threshold for neighbour filtering

print(f"THIRD PASS: {n_wins3} windows")
bckp3_loaded, loaded_vars3 = piv.backup(
    "load", proc_path, "pass3.npz", disp3_var_names, test_mode)

if bckp3_loaded:
    # Extract loaded variables
    for var_name in disp3_var_names:
        globals()[var_name] = loaded_vars3.get(var_name)
    print("Loaded existing third pass backup data.")
else:
    # LOADING & CORRELATION
    # Ensure we have the images loaded
    if 'imgs' not in globals():
        imgs = piv.read_imgs(data_path, frames, format='tif', lead_0=5,
                             timing=True)

    # Convert displacements from pass 2 to shifts for pass 3
    shifts3 = piv.disp2shift(n_wins3, disp2)

    # Step 1: Calculate correlation maps (with windows and shifts)
    corr3 = piv.calc_corrs(imgs, n_wins3, shifts=shifts3,
                           overlap=win_ov3)

    # Step 2: Sum correlation maps with alignment and size expansion
    corr3_sum = piv.sum_corrs(corr3, n_tosum3, n_wins3,
                              shifts=shifts3)

    # Step 3: Find peaks in summed correlation maps
    disp3, int3_unf = piv.find_disps(corr3_sum, n_wins3, n_peaks=n_peaks3,
                                     shifts=shifts3,
                                     floor=pk_floor2, min_dist=min_dist3)

    # Save unfiltered displacements
    disp3_unf = disp3.copy()

    # Get physical window positions for plotting (from first frame)
    _, win_pos3 = piv.split_n_shift(imgs[0], n_wins3)

# POST-PROCESSING
# Outlier removal
disp3 = piv.filter_outliers('semicircle_rect', disp3_unf,
                            a=d_max[0], b=d_max[1], verbose=True)
disp3 = piv.strip_peaks(disp3, axis=-2, verbose=True)
disp3_glo = disp3.copy()

# Neighbour filtering # TODO
disp3 = piv.filter_neighbours(disp3, thr=nbs_thr3, n_nbs=n_nbs3,
                            mode='x', replace=False, verbose=True)

# Save the displacements to a backup file
piv.backup("save", proc_path, "pass3.npz", test_mode=test_mode,
        disp3_unf=disp3_unf, int3_unf=int3_unf,
        win_pos3=win_pos3, disp3_glo=disp3_glo, disp3=disp3)

# PLOTTING
piv.plot_vel_med(disp3_glo, res_avg, frames, dt,
                    ylim=(v_max[0] * -1.1, v_max[1] * 1.1),
                    title='Third pass',
                    proc_path=proc_path, file_name="pass3_v_med", test_mode=test_mode)

piv.plot_vel_prof(disp3_glo, res_avg, frames, dt, win_pos3,
                    mode='random', xlim=(-5, 40), ylim=(0, 21.12),
                    proc_path=proc_path, file_name="pass3_v", subfolder='pass3', test_mode=test_mode) if rnd_plots else None

piv.plot_vel_prof(disp3_glo, res_avg, frames, dt, win_pos3,
                    mode='video', xlim=(-5, 40), ylim=(0, 21.12),
                    proc_path=proc_path, file_name="pass3_v",
                    test_mode=test_mode) if videos else None

# TODO: fit profile with turbulence model from turbulence book (Burgers equation, with max 3 params)

# Finally, show all figures
plt.show()

print('Done!')