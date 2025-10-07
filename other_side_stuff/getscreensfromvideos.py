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
from matplotlib import cm
from matplotlib.colors import ListedColormap

from matplotlib_scalebar.scalebar import ScaleBar
base_cmap = cm.get_cmap('Blues_r', 256)

# Convert it to an array of RGBA values
colors = base_cmap(np.linspace(0, 1, 256))

# Force the last color (value = 1) to white
colors[-1] = [1, 1, 1, 1]  # RGBA = white
custom_cmap = ListedColormap(colors)
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

frame = 320
img = Image.open(tif_files[frame])
img = img.transpose(Image.FLIP_TOP_BOTTOM)
img = np.array(img)
img = img/np.max(img)


thresh = 0.72
img_rev=img
_, binary = cv.threshold(img_rev, thresh, 255, cv.THRESH_TOZERO_INV)

# plt.figure()
# plt.imshow(binary,cmap="grey")
# plt.show()
binary= 1- binary
binary[binary == 0] = 1

# print(np.min(binary),np.max(binary))
# # Show
# plt.subplots(1,2,sharex=True,sharey=True)
# plt.subplot(1,2,1)

# plt.imshow(img,cmap="grey")
# plt.subplot(1,2,2)
# plt.imshow(binary,cmap="grey",vmin=0,vmax=1)

# plt.show()
#fig,ax = plt.subplots()
# plt.imshow(img,cmap='grey',vmin=np.min(img),vmax=np.max(img))
# # scalebar = ScaleBar(scale, "mm")  
# # ax.add_artist(scalebar)
# # ax.set_xticks([])
# # ax.set_yticks([])
# ax.axis('off')
# full_save_path = os.path.join(savepath, f"sequencePEO0dot25{frame}.png")
# plt.savefig(full_save_path)
# plt.show()
frame = 320
img = Image.open(tif_files[frame])
img = img.transpose(Image.FLIP_TOP_BOTTOM)
img = np.array(img)
img = img/np.max(img)

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
import numpy as np
import os

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i in range(6):
    frame_idx = 320 + i*260
    img = Image.open(tif_files[frame_idx])
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = np.array(img)
    img = img / np.max(img)
    
    ax = axes[i]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_aspect('auto')  # <--- important to allow tighter spacing
    
    # Add time
    time_ms = (320 + i*260) / 20
    ax.text(0.10, 0.95, f"{time_ms:.1f} ms", color='white',
            fontsize=12, ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
    
    # Manual scalebar for first frame
    if i == 0:
        scalebar_length_mm = 10
        pixels_per_mm = 1 / scale
        ax.plot([img.shape[1]-50, img.shape[1]-50- scalebar_length_mm*pixels_per_mm],
                [img.shape[0]-20, img.shape[0]-20], color='white', lw=3)
        ax.text(img.shape[1]-50 - 0.5*scalebar_length_mm*pixels_per_mm, img.shape[0]-30,
                f'{scalebar_length_mm} mm', color='white', ha='center')
    
    ax.axis('off')

# Manually adjust positions for minimal whitespace
for r in range(2):
    for c in range(3):
        idx = r*3 + c
        pos = axes[idx].get_position()
        new_pos = [0.02 + c*0.32, 0.05 + (1-r)*0.47, 0.31, 0.45]  # [left, bottom, width, height]
        axes[idx].set_position(new_pos)

full_save_path = os.path.join(savepath, "sequencePEO0dot25_grid.png")
plt.savefig(full_save_path, dpi=300)
plt.show()
