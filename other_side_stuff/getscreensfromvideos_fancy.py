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

folder_path = r"D:\Experiments\sideview_coughs\05_09_25\camera_spraytec_positie\PEO_1percent_1dot5ml_camera_spraytec_positie_1"
cal_path = r"D:\Experiments\sideview_coughs\05_09_25\camera_spraytec_positie"
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

frame = 532
img = Image.open(tif_files[frame])
img = img.transpose(Image.FLIP_TOP_BOTTOM)
img = np.array(img)
img = img/np.max(img)*255
background = cv.GaussianBlur(img, (101, 101), 0)

# Subtract background
corrected = cv.subtract(img, background)
corrected = cv.normalize(corrected, None, 0, 255, cv.NORM_MINMAX)
# plt.figure()
# plt.imshow(corrected,cmap="gray")
# plt.show()
# # Optional: normalize to 0-255
img = corrected/np.max(corrected) 

thresh = 0.78
img_rev=img
_, binary = cv.threshold(img_rev, thresh, 255, cv.THRESH_TOZERO_INV)

plt.figure()
plt.imshow(binary,cmap="grey")
plt.show()
#binary= 1- binary
binary[binary == 0] = 1
binary_contour = np.array(255-binary*255,dtype= 'uint8')
# Suppose 'binary' is your 0/255 image
contours, hierarchy = cv.findContours(binary_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
areas = [cv.contourArea(c) for c in contours]
min_area = 25
filtered_contours = [c for c, a in zip(contours, areas) if a >= min_area]
img_uint8 = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
img_contours = cv.cvtColor(img_uint8, cv.COLOR_GRAY2BGR)
mask = np.zeros_like(img, dtype=np.uint8)  # same shape as original image

# 2️⃣ Fill the contours in the mask
cv.drawContours(mask, filtered_contours, -1, color=1, thickness=-1)

binary[mask!=1]=1
# Draw contours in red
cv.drawContours(img_contours, filtered_contours, -1, (0,0,255), -1)

# Show with matplotlib
# plt.figure(figsize=(6,6))
# plt.imshow(cv.cvtColor(img_contours, cv.COLOR_BGR2RGB))
# plt.title("Filament Contours")
# plt.axis("off")
# plt.show()

print(np.min(binary),np.max(binary))
# Show
plt.subplots(1,2,sharex=True,sharey=True)
plt.subplot(1,2,1)

plt.imshow(img,cmap="grey")
plt.subplot(1,2,2)
plt.imshow(binary,cmap="grey",vmin=0,vmax=1)

plt.show()
fig,ax = plt.subplots()
plt.imshow(binary,cmap="grey",vmin=0,vmax=1)
scalebar = ScaleBar(scale, "mm")  
ax.add_artist(scalebar)
ax.set_xticks([])
ax.set_yticks([])
full_save_path = os.path.join(savepath, f"PEO0dot25{frame}.svg")
plt.savefig(full_save_path)
plt.show()

fig,ax = plt.subplots()
plt.imshow(binary,cmap="grey",vmin=0,vmax=1)

ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
full_save_path = os.path.join(savepath, f"PEO0dot25{frame}_cover.svg")
plt.savefig(full_save_path)
plt.show()