from PIL import Image
import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import math
from scipy import signal,stats,spatial
import tifffile
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import time
import sys
import gc  # Garbage collector
from matplotlib_scalebar.scalebar import ScaleBar

folder_path = r"D:\Experiments\sideview_coughs\01_08_2025\PEO0dot25_1dot5bar_80ms_1dot5ml_7"
cal_path = os.path.dirname(folder_path)
savepath= r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\screenshotscamera"
cwd = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.dirname(cwd)
print(cwd)
#function_dir = os.path.join(parent_dir, 'cough-machine-control')
function_dir = os.path.join(parent_dir,'functions')
print(function_dir)
sys.path.append(function_dir)
import calibration

scale = calibration.get_calibration(cal_path)
print(f"{scale} mm/pix")
tif_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(".tif") and os.path.isfile(os.path.join(folder_path, f))
]

frame = 335
img = Image.open(tif_files[frame])
img = img.transpose(Image.FLIP_TOP_BOTTOM)
img = np.array(img)
img = img/np.max(img)


thresh = 0.78
img_rev=img
_, binary = cv.threshold(img_rev, thresh, 255, cv.THRESH_TOZERO_INV)

# plt.figure()
# plt.imshow(binary,cmap="grey")
# plt.show()
# #binary= 1- binary
# binary[binary == 0] = 1

# print(np.min(binary),np.max(binary))
# # Show
# plt.subplots(1,2,sharex=True,sharey=True)
# plt.subplot(1,2,1)

# plt.imshow(img,cmap="grey")
# plt.subplot(1,2,2)
# plt.imshow(binary,cmap="grey",vmin=0,vmax=1)

plt.show()
fig,ax = plt.subplots()
plt.imshow(img,cmap="grey",vmin=0,vmax=1)
scalebar = ScaleBar(scale, "mm")  
ax.add_artist(scalebar)
ax.set_xticks([])
ax.set_yticks([])
full_save_path = os.path.join(savepath, f"PEO0dot25original{frame}.svg")
plt.savefig(full_save_path)
plt.show()

