import os
import getpass

import piv_functions as piv

# Set experimental parameters
meas_name = '250624_1333_80ms_whand'  # Name of the measurement
frame_nrs = [930, 931]
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


imgs = piv.load_images(data_path, frame_nrs, format='tif', lead_0=5,
                       timing=True)
imgs_ds = piv.downsample(imgs, ds_fac)

print()