import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from skimage.draw import line
import networkx as nx
import glob
import os
import pandas as pd
import cv2 as cv
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from numpy.linalg import norm
from math import degrees, acos
import pickle
import re
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
import sys
plt.style.use('tableau-colorblind10')
colors = plt.cm.tab10.colors
markers = ["o","v","1","*","+","d","|","s","h","<","X"]
cwd = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.dirname(cwd)
print(cwd)
#function_dir = os.path.join(parent_dir, 'cough-machine-control')
save_path = r"C:\Users\sikke\Documents\GitHub\cough-machine-control\spraytec\results_spraytec\Serie_Averages\PEOjet"
filename = "waterjetcameradistribution"
full_save_path = os.path.join(save_path,filename)

function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
import calibration
PEO_jet = False
folder_path = r"D:\Experiments\sideview_coughs\jets"
scale = calibration.get_calibration(folder_path)
print(f"scale: {scale:.5f} mm/pix")
if PEO_jet:


    folder = r"D:\Experiments\sideview_coughs\jets\PEOjet_2"

    files = [f for f in glob.glob("*.tif", root_dir=folder) if "ximea" not in f.lower() and "calibration" not in f.lower()]

    cropped_value=200
    files = files[:1]
    for i,file in enumerate(files):    

        img= Image.open(folder +"\\" + file)  # Open TIFF
        img = np.array(img)
        if img.dtype != np.uint8:
            img = (img >> 8).astype(np.uint8)
            img = cv.flip(img, 0)

        #cropped = img[:,cropped_value:cropped_value+window_size]
        cropped = img[:,cropped_value:]
        # plt.figure()
        # plt.imshow(img,cmap='grey')
        # plt.show()
        #cropped = cv.blur(cropped,)
        ret,thresh = cv.threshold(255-cropped,170,255,cv.THRESH_TOZERO)
        # print(mean_val)
        # print(thresh_factor)
        
        # plt.figure()
        # plt.imshow(thresh,cmap='gray')
        # plt.show()

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_areas = np.array([cv.contourArea(contour) for contour in contours])
        min_area= 60

        mask = contour_areas > 0.5* np.max(contour_areas) # min_area
        print(mask)
        
        contours  = [contour for i,contour in enumerate(contours) if mask[i] ]
        print(len(contours))

        upper_contour = contours[1]
        lower_contour = contours[0]
        print(np.shape(upper_contour))


        
        # Convert contours to simpler Nx2 arrays
        upper = upper_contour[:,0,:]
        lower = lower_contour[:,0,:]

        # Sort by x (in case they are unordered)
        upper = upper[upper[:,0].argsort()]
        lower = lower[lower[:,0].argsort()]
        def extract_clean_edge(contour, mode="upper"):
            """
            contour: Nx1x2 array from cv2.findContours
            mode: "upper" (min y) or "lower" (max y)
            returns: Nx2 array with unique (x, y) edge points
            """

            # Flatten to Nx2
            pts = contour[:,0,:]

            # Group by x
            edge_dict = {}
            for x, y in pts:
                if x not in edge_dict:
                    edge_dict[x] = []
                edge_dict[x].append(y)

            clean_pts = []
            for x, ys in edge_dict.items():
                if mode == "upper":
                    y_sel = min(ys)  # top pixel
                else:
                    y_sel = max(ys)  # bottom pixel
                clean_pts.append([x, y_sel])

            clean_pts = np.array(sorted(clean_pts, key=lambda p: p[0]))  # sort by x
            return clean_pts


        upper = extract_clean_edge(upper_contour, mode="upper")
        lower = extract_clean_edge(lower_contour, mode="lower")
        print(np.shape(upper))
        # Interpolate so both contours have same x values
        x_vals = np.arange(min(upper[:,0].min(), lower[:,0].min()),
                        max(upper[:,0].max(), lower[:,0].max())+1)

        y_upper = np.interp(x_vals, upper[:,0], upper[:,1])
        y_lower = np.interp(x_vals, lower[:,0], lower[:,1])

        # Thickness profile
        thickness = (y_lower - y_upper +1) *scale

        # Show average thickness
        print("Average thickness (mm):", np.mean(thickness))

        # Plot profile

        plt.plot(x_vals, thickness)
        plt.xlabel("X position (pixels)")
        plt.ylabel("Thickness (mm)")
        plt.title("Jet Thickness Profile")
        plt.show()

        contour_image = np.zeros_like(thresh)

        # Draw contours as lines
        cv.drawContours(contour_image, contours, -1, color=255, thickness=1)

        ##Show result with matplotlib
        fig,ax =plt.subplots(1,2,figsize=(8, 6),sharex=True,sharey=True)
        ax[0].imshow(cropped, cmap='gray')
        plt.title("Contours Drawn with OpenCV")
        plt.axis('off')
        ax[1].imshow(thresh,cmap="grey")
        plt.show()
        print(len(contours))



def filament_file(
    folder,  
    cropped_value=55,
    mirror=True, 
    rotation=None,  
    scale = 1,
    selected_image=400,
    save_fig = False,
    thresh_factor = 25,
    plot= False,
    output_folder =".\\",
    skip_images=1,
    skip_first_files=250,
    window_size =200,
    bubble_value = 300,
    min_length=30,
    window=5,
    line_threshold=0.85,
    step_filament=1,
    target_y = 0.82,
    tolerance = 0.02

):
    """
    This one will add some blur or anything

    Parameters:
    - folder (str): Folder containing the TIFF file.
    - filename (str): TIFF filename.
    - output_folder (str): Folder to save output.
    - output_name (str): Name of output video file.
    - mirror (bool): If True, mirrors frames horizontally.
    - rotation (int or None): Rotation angle (90, 180, 270).
    - fps (int): Frames per second.
    - start_frame (int): First frame to include.
    - end_frame (int or None): Last frame to include (None = all frames).
    - calibration (bool): If True, uses a calibration TIFF to add a scale bar.
    - scalebar_length_mm (float): Length of the scale bar in mm (default 5 mm).
    - fps_camera (int): Camera's original FPS (default 20,000).
    """
    
   
    valid_rotations = {None, 90, 180, 270}
    if rotation not in valid_rotations:
        raise ValueError("Rotation must be None, 90, 180, or 270 degrees.")

    # Get calibration scale if needed
    

    total_frames = len(files)

    #preparing area and cnt arrays
   
    all_length_time = []
    first_file = True
    frame_number =0
    for i,file in enumerate(files):
           
        if i== selected_image or selected_image == -1:
            
            if i%skip_images==0 and i>skip_first_files:
                frame_number+=1
                img= Image.open(folder +"\\" + file)  # Open TIFF
                img = np.array(img)
                if img.dtype != np.uint8:
                    img = (img >> 8).astype(np.uint8)
                 
                if mirror:
                    img = cv.flip(img, 0)
                if rotation == 90:
                    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    img = cv.rotate(img, cv.ROTATE_180)
                elif rotation == 270:
                    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                if first_file == True:
                    cal_img = Image.open(folder +"\\" + files[0])
                    cal_img = np.array(cal_img)
                    
                    if cal_img.dtype != np.uint8:
                        cal_img = (cal_img >> 8).astype(np.uint8)
                    
                    mean_val = np.mean(np.mean(cal_img))
                    
                    thresh_ref = np.quantile(cal_img.flatten(),0.80)/255
                    first_file = False

                #cropped = img[:,cropped_value:cropped_value+window_size]
                cropped = img[:,cropped_value:]

                ret,thresh = cv.threshold(255-cropped,255-mean_val+thresh_factor,255,cv.THRESH_TOZERO)
                # print(mean_val)
                # print(thresh_factor)
               
                # plt.figure()
                # plt.imshow(thresh,cmap='gray')
                # plt.show()

                hierarchy,contours,areas = edges(thresh)

                
                # contour_image = np.zeros_like(thresh)

                # # Draw contours as lines
                # cv.drawContours(contour_image, contours, -1, color=255, thickness=5)

                # ##Show result with matplotlib
                # fig,ax =plt.subplots(1,2,figsize=(8, 6),sharex=True,sharey=True)
                # ax[0].imshow(contour_image, cmap='Reds')
                # plt.title("Contours Drawn with OpenCV")
                # plt.axis('off')
                # ax[1].imshow(thresh)
                # plt.show()
                # print(len(contours))
                
                if len(contours)>0:
                    contour_data= np.zeros_like(thresh)
                
                    skeleton = np.zeros_like(thresh)
                    dist_transform = np.zeros_like(thresh)
                    mask_bubbles = hierarchy[0,:,3]!=-1
                    areas = np.array(areas)
                    largest_contours = (areas >= 0.4* np.max(areas)) & (areas >= 100) 
                    contour_mask = np.zeros_like(contour_data)
                    for j,contour in enumerate(contours):
                    
                        #contour_mask[:] = 0  # reset instead of reallocate
                        if mask_bubbles[j]:
                            if areas[j]>bubble_value:
                                cv.fillPoly(contour_mask, [contour], -255)

                            # fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
                            # ax[0].imshow(thresh)
                            # ax[1].imshow(contour_mask)
                            # ax[2].imshow(overlay)
                            # ax[2].set_title("bubbles")
                            # plt.show()
                                continue                   
                        else: 
                            color,tag = shape_categorizer(contour)

                            if tag == "Filament":
                                #filament_contours.append(contour)
                                print("We found a filament")
                                
                                #cv.drawContours(contour_mask,[contour],-1,255,1)
               
                                cv.fillPoly(contour_mask, [contour], 255)
   

                    
   

                    dist = cv.distanceTransform(contour_mask,cv.DIST_L2,5)
                    mask_skeleton= skeletonize(dist>0)==1
                    dist[~mask_skeleton] =0
                    skeleton[dist!=0] = 255
                    fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
                    ax[0].imshow(skeleton,cmap='gray')
                    ax[1].imshow(img,cmap='gray')
                    plt.show()
                    dist_transform[dist!=0] = dist[dist!=0]
                    image  =cropped/255

                    merged = branch_combiner(skeleton,min_length=min_length)

                    all_lengths = []
                    for k in range(len(merged)):
                    #for i in range(1):
                        # fig,ax = plt.subplots(1,2)
                        # ax[1].imshow(image,cmap='gray')
                        branch = merged[k]
                        if branch[0][1] > branch[-1][1]:
                            branch = branch[::-1]

                        filament_len_array,dists_interpolating,x_branch,y_branch =branch_fitter(branch,image,dist,window=window, line_threshold=line_threshold,step_filament=step_filament,target_y =target_y, tolerance = tolerance)
                        all_lengths.append(dists_interpolating)
                    all_length_time.append(all_lengths)
                else: 
                    all_length_time.append([])
                if i % 10 == 0:
                    print(f"Processed {i}/{total_frames} frames...")


    
    return all_length_time
def edges(frame):
    """
    This functions finds the contours
    """

    contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv.contourArea(contour) for contour in contours]
    #contour_image = np.zeros_like(frame)

    # # Draw contours as lines
    # cv.drawContours(contour_image, contours, -1, color=255, thickness=-1)

    # # Show result with matplotlib
    # fig,ax =plt.subplots(1,2,figsize=(8, 6),sharex=True,sharey=True)
    # ax[0].imshow(contour_image, cmap='gray')
    # plt.title("Contours Drawn with OpenCV")
    # plt.axis('off')
    # ax[1].imshow(frame)
    # plt.show()
    

    return hierarchy,contours,contour_areas

folder = r"D:\Experiments\sideview_coughs\jets\waterjet_1"
files = [f for f in glob.glob("*.tif", root_dir=folder) if "ximea" not in f.lower() and "calibration" not in f.lower()]

cropped_value=800
files = files[:5000]
all_diam = np.array([])
for i,file in enumerate(files):    

    img= Image.open(folder +"\\" + file)  # Open TIFF
    img = np.array(img)
    if img.dtype != np.uint8:
        img = (img >> 8).astype(np.uint8)
        img = cv.flip(img, 0)

    #cropped = img[:,cropped_value:cropped_value+window_size]
    cropped = img[:,cropped_value:]
    # plt.figure()
    # plt.imshow(img,cmap='grey')
    # plt.show()
    #cropped = cv.blur(cropped,)
    ret,thresh = cv.threshold(255-cropped,170,255,cv.THRESH_TOZERO)
    # print(mean_val)
    # print(thresh_factor)
    
    # plt.figure()
    # plt.imshow(thresh,cmap='gray')
    # plt.show()
    # Fill small holes inside binary objects
    #thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)))

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_areas = np.array([cv.contourArea(contour) for contour in contours])
    min_area= 60
  
    mask_area = (contour_areas > 12) & (contour_areas < 400)   # min_area
    contours  = [contour for i,contour in enumerate(contours) if mask_area[i] ]
 
    mask_hierarchy = hierarchy[:,:,3] == -1

    mask_hierarchy = mask_hierarchy[0]
 
    contours  = [contour for i,contour in enumerate(contours) if mask_hierarchy[i] ]
    valid_contours = []

    for cnt in contours:
        # Compute center of contour
        M = cv.moments(cnt)
        if M["m00"] == 0:  # avoid division by zero
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

        # Check if center is inside any already kept contour
        inside = False
        for kept in valid_contours:
            if cv.pointPolygonTest(kept, center, False) >= 0:
                inside = True
                break

        if not inside:
            valid_contours.append(cnt)
    contour_areas = np.array([cv.contourArea(contour) for contour in valid_contours])
    contour_image = np.zeros_like(thresh)
    output = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)  # Convert to BGR so we can draw colored circles
    diameters = 2 * np.sqrt(contour_areas / np.pi)
    for cnt in valid_contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        #cv.circle(output, center, radius, (0, 255, 0), 1)  # Draw green circle
        diameter = 2 * np.sqrt(cv.contourArea(cnt) / np.pi)

        cv.circle(output, center, int(diameter/2), (255, 0, 0), 1)  # Draw green circle


    # Draw contours as lines
    cv.drawContours(contour_image, contours, -1, color=255, thickness=1)

    diameters = diameters*scale
    all_diam = np.concatenate((all_diam, diameters))
    ##Show result with matplotlib
    # fig,ax =plt.subplots(1,2,figsize=(8, 6),sharex=True,sharey=True)
    # ax[0].imshow(contour_image, cmap='gray')
    # plt.title("Contours Drawn with OpenCV")
    # plt.axis('off')
    # ax[1].imshow(cv.cvtColor(output,cv.COLOR_BGR2RGB))
    # plt.show()

all_diam =np.array(all_diam)

file = r"C:\Users\sikke\Documents\GitHub\cough-machine-control\spraytec\Averages\Unweighted\water_jet\average_waterjet_3.txt"
df = pd.read_table(file,delimiter=",",encoding='latin1')
df = df.replace('-', 0)
for col in df.columns:
    # Try converting each column to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='ignore')
columns_scattervalues = df.loc[:,"% V (0.100-0.117µm)":"% V (857.698-1000.002µm)"].columns.tolist()

bin_centers = np.array([])
for column in columns_scattervalues:
    match = re.search(r"\(([\d.]+)-([\d.]+)", column)
    if match:
        lower = float(match.group(1))
        upper = float(match.group(2))
        center = (lower + upper) / 2
        bin_centers = np.append(bin_centers,center)





diffs = np.diff(bin_centers)
bin_edges = np.zeros(len(bin_centers) + 1)

bin_edges[1:-1] = (bin_centers[:-1] + bin_centers[1:]) / 2
# First edge (extrapolate)
bin_edges[0] = bin_centers[0] - diffs[0] / 2

# Last edge (extrapolate)
bin_edges[-1] = bin_centers[-1] + diffs[-1] / 2
# Step 2: Calculate widths
bin_widths = np.diff(bin_edges)
all_diam = all_diam * 1000 #micrometers
print(np.mean(all_diam))


counts, _ = np.histogram(all_diam, bins=bin_edges)
counts = counts/ np.sum(counts)*100
plt.figure(figsize=(8,5))
plt.bar(bin_edges[:-1], counts, width=bin_widths, align='edge', edgecolor='black')
plt.xscale('log')

plt.xlabel(r"Diameter ($\mu$m)")
plt.ylabel("Number PDF (%)")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylim(1e-1,40)
plt.xlim(bin_edges[0],bin_edges[-1])
plt.savefig(full_save_path+".svg")
plt.show()
series_path  =r"C:\Users\sikke\Documents\GitHub\cough-machine-control\spraytec\results_spraytec\Serie_Averages\npz_files"

full_series_savepath = os.path.join(series_path,"waterjet_camera")
np.savez(full_series_savepath,n_percentages=counts,bins=bin_edges[:-1],bin_widths=bin_widths)