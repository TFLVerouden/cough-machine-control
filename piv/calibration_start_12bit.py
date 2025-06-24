import tifffile
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy 
path = r"D:\Experiments\PIV\250624_calibration_PIV_500micron.tif"
img = cv.imread(path, cv.IMREAD_UNCHANGED)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

plt.imshow(gray,cmap='gray')
plt.show()