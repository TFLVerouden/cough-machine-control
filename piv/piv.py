import getpass
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from datetime import datetime

import piv_functions as piv

print("\n\nStarting PIV analysis...")
# Add the functions directory to the path and import CVD check
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'functions'))
import cvd_check as cvd

# Set CVD-friendly colors
cvd.set_cvd_friendly_colors()

# Set experimental parameters
test_mode = False
videos = False
random_profiles = False
new_bckp = False
meas_series = 'PIV250723'
meas_name = 'PIV_1bar_80ms_closedtank'
cal_name = 'calibration_PIV_500micron_2025_07_23_C001H001S0001'
frames = list(range(500, 800)) if test_mode else "all"
dt = 1 / 40000  # [s]
depth = 0.01  # [m] Depth of the channel

# Set calibration parameters
cal_spacing = 0.0005  # [m] Calibration grid spacing
cal_roi = [45, 825, 225, 384]  # [px] Region of interest for calibration
cal_init_grid = (7, 5)  # Initial grid size for calibration
cal_bin_thr = 200  # Binarization threshold for calibration
cal_blur_ker = (5, 5)  # Blur kernel size for calibration
cal_open_ker = (3, 3)  # Opening kernel size for calibration

# Get current date and time for saving
run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# File handling
current_dir = os.path.dirname(os.path.abspath(__file__))
cal_path = os.path.join(current_dir, "calibration")

user = getpass.getuser()
if user == "tommieverouden":
    # data_path = os.path.join("/Volumes/Data/Data/250623 PIV/", meas_name)
    data_path = os.path.join(
        '/Users/tommieverouden/Documents/Current data', meas_series, meas_name)
elif user == "sikke":
    data_path = os.path.join('D:\\Experiments\\PIV\\', meas_series, meas_name)

# Data saving settings
var_names = [[],['disp1_unf', 'int1_unf', 'disp1_glo', 'disp1_nbs', 
              'disp1', 'time'], ['disp2_unf', 'int2_unf', 'win_pos2', 'disp2_glo', 'disp2'], ['disp3_unf', 'int3_unf', 'win_pos3', 'disp3_glo', 'disp3']]

# In the current directory, create a folder for processed data
# named the same as the final part of the data_path
proc_path = os.path.join(current_dir, 'processed', meas_series, meas_name)
if not os.path.exists(proc_path) and not test_mode:
    os.makedirs(proc_path)
    print(f"Created directory for processed data: {proc_path}")

# Read or create calibration data
data_cal = piv.load_backup(cal_path, cal_name)
if data_cal:
    res_avg = data_cal.get('resolution_avg_m_per_px')
    frame_w = data_cal.get('frame_size_m')[0]  # Already in meters
else:
    cal_im_path = os.path.join(cal_path, f"{cal_name}.tif")
    res_avg, _, frame_size = piv.calibrate_grid(cal_im_path, cal_spacing, roi=cal_roi, init_grid=cal_init_grid, binary_thr=cal_bin_thr, blur_ker=cal_blur_ker, open_ker=cal_open_ker, save=True, plot=True)
    frame_w = frame_size[0]  # Width in meters

# Count number of frames to be used
if frames == "all":
    nr_frames = piv.read_imgs(data_path, "all", format='tif',
                              lead_0=5, only_count=True, timing=False)
    frames = list(range(1, nr_frames + 1))

# FIRST PASS: Full frame correlation ===========================================
ds_fac1 = 4             # Downsampling factor
n_tosum1 = 20           # Number of correlation maps to sum
n_peaks1 = 10           # Number of peaks to find in correlation map
n_wins1 = (1, 1)        # Number of windows (rows, cols)
min_dist1 = 5           # Minimum distance between peaks
v_max1 = [10, 75]       # Global filter m/s
n_nbs1 = (41, 1, 1)     # Neighbourhood for local filtering
nbs_thr1 = 1            # Threshold for neighbour filtering
smooth_lam = 4e-7       # Smoothing lambda for splines

print("\nFIRST PASS: full frame correlation")
# Load existing backup data if available

loaded_vars = piv.load_backup(proc_path, "pass1.npz", var_names[1],
                              test_mode=(test_mode or new_bckp))
if loaded_vars:
    for var_name in var_names[1]:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing first pass backup data.")
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
d_max1 = np.array(v_max1) * dt / res_avg
disp1 = piv.filter_outliers('semicircle_rect', disp1_unf, 
                            a=d_max1[0], b=d_max1[1], verbose=True)
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
piv.save_backup(proc_path, "pass1.npz", test_mode=test_mode,
        ds_fac1=ds_fac1, n_tosum1=n_tosum1,
        n_peaks1=n_peaks1, n_wins1=n_wins1,
        min_dist1=min_dist1, d_max1=d_max1, n_nbs1=n_nbs1,
        nbs_thr1=nbs_thr1, smooth_lam=smooth_lam, disp1_unf=disp1_unf, int1_unf=int1_unf, disp1_glo=disp1_glo, disp1_nbs=disp1_nbs,
        disp1=disp1, time=time)

# PLOTTING
# Plot a post-processing comparison of the x velocities in time
piv.plot_vel_comp(disp1_glo, disp1_nbs, disp1, res_avg, frames, 
                 dt, ylim=(v_max1[0] * -1.1, v_max1[1] * 1.1),
                 disp_rejected=disp1_unf,
                 proc_path=proc_path, file_name="pass1_v-t",
                 title=f'First pass - {meas_name}', test_mode=test_mode)


# SECOND PASS: Split in 8 windows ==============================================
n_tosum2 = 20           # Number of correlation maps to sum
n_peaks2 = 10           # Number of peaks to find in correlation map
n_wins2 = (8, 1)        # Number of windows (rows, cols)
win_ov2 = 0.2           # Overlap between windows
v_max2 = [10, 75]       # Global filter m/s
n_nbs2 = (51, 3, 1)     # Neighbourhood for local filtering
nbs_thr2 = 5            # Threshold for neighbour filtering

# TODO: Plot v_center

print(f"\nSECOND PASS: {n_wins2} windows")
loaded_vars = piv.load_backup(proc_path, "pass2.npz", var_names[2],
                              test_mode=(test_mode or new_bckp))

if loaded_vars:
    # Extract loaded variables
    for var_name in var_names[2]:
        globals()[var_name] = loaded_vars.get(var_name)
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
                                     shifts=shifts2)

    # Save unfiltered displacements
    disp2_unf = disp2.copy()

    # Get physical window positions for plotting (from first frame)
    _, win_pos2 = piv.split_n_shift(imgs[0], n_wins2)

    # Note: The correlation map centers (used for displacement calculation) are stored separately in each correlation map as the second element of the tuple

# POST-PROCESSING
# Outlier removal
d_max2 = np.array(v_max2) * dt / res_avg
disp2 = piv.filter_outliers('semicircle_rect', disp2_unf,
                            a=d_max2[0], b=d_max2[1], verbose=True)
disp2 = piv.strip_peaks(disp2, axis=-2, verbose=True)
disp2_glo = disp2.copy()

# Very light neighbour filtering to remove extremes and replace missing values
disp2 = piv.filter_neighbours(disp2, thr=nbs_thr2, n_nbs=n_nbs2,
                            mode='r', replace=True, verbose=True, timing=True)

# Save the displacements to a backup file
piv.save_backup(proc_path, "pass2.npz", test_mode=test_mode,
        n_tosum2=n_tosum2, n_peaks2=n_peaks2, n_wins2=n_wins2,
                win_ov2=win_ov2, d_max2=d_max2,
                n_nbs2=n_nbs2,
                nbs_thr2=nbs_thr2, disp2_unf=disp2_unf, int2_unf=int2_unf,
        win_pos2=win_pos2, disp2_glo=disp2_glo, disp2=disp2)

# PLOTTING
# Plot the median, min and max velocity in time
piv.plot_vel_med(disp2, res_avg, frames, dt,
                 ylim=(v_max2[0] * -1.1, v_max2[1] * 1.1),
                 title=f'Second pass - {meas_name}',
                 proc_path=proc_path, file_name="pass2_v_med", test_mode=test_mode)

# Plot some randomly selected velocity profiles
piv.plot_vel_prof(disp2, res_avg, frames, dt, win_pos2,
                  mode='random', xlim=(v_max2[0] * -1.1, v_max2[1] * 1.1),
                  ylim=(0, frame_w * 1000),
                  disp_rejected=disp2_unf,
                  proc_path=proc_path, file_name="pass2_v",
                  subfolder='pass2', test_mode=not random_profiles)

# Plot all velocity profiles in video
piv.plot_vel_prof(disp2, res_avg, frames, dt, win_pos2,
                  mode='video', xlim=(v_max2[0] * -1.1, v_max2[1] * 1.1),
                  ylim=(0, frame_w * 1000),
                  disp_rejected=disp2_unf,
                  proc_path=proc_path, file_name="pass2_v",
                  test_mode=not videos)


# THIRD PASS: Split in 24 windows ==============================================
n_tosum3 = 20             # Number of correlation maps to sum -> 0.5 ms mov.av.
n_peaks3 = 5             # Number of peaks to find in correlation map
n_wins3 = (24, 1)        # Number of windows (rows, cols)
win_ov3 = 0             # Overlap between windows
v_max3 = [10, 75]       # Global filter m/s
n_nbs3 = (1, 3, 1)     # Neighbourhood for local filtering
nbs_thr3 = (4, 6)            # Threshold for neighbour filtering

print(f"\nTHIRD PASS: {n_wins3} windows")
loaded_vars = piv.load_backup(proc_path, "pass3.npz", var_names[3],
                              test_mode=(test_mode or new_bckp))

if loaded_vars:
    for var_name in var_names[3]:
        globals()[var_name] = loaded_vars.get(var_name)
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
                                        shifts=shifts3, subpx=True)

    # Save unfiltered displacements
    disp3_unf = disp3.copy()

    # Get physical window positions for plotting (from first frame)
    _, win_pos3 = piv.split_n_shift(imgs[0], n_wins3)

# POST-PROCESSING
# Outlier removal
d_max3 = np.array(v_max3) * dt / res_avg
disp3 = piv.filter_outliers('semicircle_rect', disp3_unf,
                            a=d_max3[0], b=d_max3[1], verbose=True)
disp3_glo = disp3.copy()

# Neighbour filtering
disp3 = piv.filter_neighbours(disp3, thr=nbs_thr3, thr_unit="pxs", n_nbs=n_nbs3, mode='xy', replace="closest", verbose=True, timing=True)
disp3 = piv.strip_peaks(disp3, axis=-2, verbose=True)
disp3_nbs = disp3.copy()

# Save the displacements to a backup file
piv.save_backup(proc_path, "pass3.npz", test_mode=test_mode,
        disp3_unf=disp3_unf, int3_unf=int3_unf,
        win_pos3=win_pos3, disp3_glo=disp3_glo, disp3=disp3,                 
                n_tosum3=n_tosum3,
                n_peaks3=n_peaks3, n_wins3=n_wins3,
                win_ov3=win_ov3,
                 d_max3=d_max3, n_nbs3=n_nbs3,
                nbs_thr3=nbs_thr3)

# PLOTTING
piv.plot_vel_med(disp3_nbs, res_avg, frames, dt,
                    ylim=(v_max3[0] * -1.1, v_max3[1] * 1.1),
                    title=f'Third pass - {meas_name}',                    proc_path=proc_path, file_name="pass3_v_med", test_mode=test_mode)

piv.plot_vel_prof(disp3_nbs, res_avg, frames, dt, win_pos3,
                    mode='random', xlim=(v_max3[0] * -1.1, v_max3[1] * 1.1), ylim=(0, frame_w * 1000),
                    disp_rejected=disp3_unf,
                    proc_path=proc_path, file_name="pass3_v", subfolder='pass3', test_mode=not random_profiles)

piv.plot_vel_prof(disp3_nbs, res_avg, frames, dt, win_pos3,
                    mode='video', xlim=(v_max3[0] * -1.1, v_max3[1] * 1.1), ylim=(0, frame_w * 1000),
                    disp_rejected=disp3_unf,
                    proc_path=proc_path, file_name="pass3_v",
                    test_mode=not videos)

# TODO: fit profile with turbulence model from turbulence book (Burgers equation, with max 3 params)

# FLOW RATE CALCULATION
# Example of how to calculate volumetric flow rates from velocity data:
# depth_m = 0.001  # Depth in meters (e.g., 1 mm)
# vel3 = disp3_nbs * res_avg / dt  # Convert displacement to velocity
# flow_m3s = piv.vel2flow(vel3, depth_m, frame_width_m)  # Flow rate in m³/s
# flow_Ls = flow_m3s * 1000  # Convert to L/s

# Calculate flow rate
q = piv.vel2flow(disp3_nbs, depth, frame_w)

# TODO: Import Gupta model data

# Plot flow rate in time, save to file
piv.plot_flow_rate(q, frames, dt, ylim=(0, np.nanmax(q) * 1100),
                    title=f'Flow rate - {meas_name}',
                    proc_path=proc_path, file_name="flow_rate",
                    test_mode=test_mode)
piv.save_backup(proc_path, "flow_rate.npz", test_mode=test_mode,
                flow_rate_Lps=q, time_s=time)

# Save all parameters to a backup file
piv.save_backup(proc_path, "params.npz", test_mode=test_mode,
                date_saved=run_date, meas_series=meas_series, meas_name=meas_name,
                cal_name=cal_name, dt=dt, frames_start=frames[0], frames_end=frames[-1], res_avg=res_avg)


# Finally, show all figures
print('Done!')
plt.show()

