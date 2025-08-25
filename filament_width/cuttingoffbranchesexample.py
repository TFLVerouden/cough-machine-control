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
plt.style.use('tableau-colorblind10')
colors = plt.cm.tab10.colors
markers = ["o","v","1","*","+","d","|","s","h","<","X"]

# Build a graph representation of the skeleton
def build_graph(skel):
    """
    This function uses graph theory to make the skeleton into nodes and edges, which we can combine later on
    """
    G = nx.Graph()  # Make an empty graph

    rows, cols = np.nonzero(skel)  # Find coordinates of all white pixels

    # Go through every white pixel (r, c)
    for r, c in zip(rows, cols):
        for dr in [-1, 0, 1]:  # look in all directions
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # skip itself
                rr, cc = r + dr, c + dc
                if (0 <= rr < skel.shape[0] and  # check if inside image
                    0 <= cc < skel.shape[1] and
                    skel[rr, cc]):  # if neighbor is white too
                    G.add_edge((r, c), (rr, cc))  # connect them in graph
    return G

# Extract branches from the skeleton using the graph
def extract_branches(skel):
    """
    This functions gives all seperate loose branches
    """
    G = build_graph(skel)  # Build the graph from the skeleton
    nodes_of_interest = [n for n in G.nodes if G.degree[n] != 2]  # ends or junctions
    visited_edges = set()
    branches = []

    for node in nodes_of_interest:
        for neighbor in G.neighbors(node):
            edge = frozenset([node, neighbor])
            if edge in visited_edges:
                continue
            path = [node, neighbor]
            visited_edges.add(edge)

            current = neighbor
            prev = node

            while G.degree[current] == 2:  # keep walking if path is straight
                next_nodes = [n for n in G.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = frozenset([current, next_node])
                if edge in visited_edges:
                    break
                path.append(next_node)
                visited_edges.add(edge)
                prev, current = current, next_node

            branches.append(path)  # store the found branch
    return branches

# Filter out branches that are too short
def filter_branches(branches, min_length=20):
    """
    This function filters out all branches below a certain minimum length
    """
    return [b for b in branches if len(b) >= min_length]

# Sort branches by length (longest first)
def sort_branches(branches):
    """
    This function just sorts the branches.
    """
    return sorted(branches, key=len, reverse=True)

# Bresenham's line algorithm to draw a line between two points
def bresenham_line(p0, p1):
    """
    This function creates a pixelized line based on two endpoints
    """

    r0, c0 = p0
    r1, c1 = p1
    rr, cc = line(r0, c0, r1, c1)
    return list(zip(rr, cc))



def angle_between(v1, v2):
    """Return the angle in degrees between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def fit_line(points):
    """Fit a line to the last 5 points using linear regression and return the slope."""
    if len(points) < 5:
        return 0  # If fewer than 5 points, return 0 as the slope

    # Use only the last 5 points
    points = points[-5:]

    # Extract x and y coordinates
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Perform linear regression to get the slope (m)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m  # Return the slope of the fitted line

def fully_merge_branches(branches, max_angle=30):
    merged = []
    used = set()

    def are_touching(p1, p2):
        return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1

    def branches_touch(b1, b2):
        return any(are_touching(p1, p2) for p1 in [b1[0], b1[-1]] for p2 in [b2[0], b2[-1]])

    for i in range(len(branches)):
        if i in used:
            continue

        current = branches[i]
        used.add(i)
        extended = True

        while extended:
            extended = False
            best_j = -1
            best_merge = None
            best_length = len(current)

            for j in range(len(branches)):
                if j in used or j == i:
                    continue
                b = branches[j]
                if not branches_touch(current, b):
                    continue

                # Try all 4 connection directions
                candidates = [
                    (current + b, current[-1], current[-2], b[0], b[1]),
                    (current + b[::-1], current[-1], current[-2], b[-1], b[-2]),
                    (b + current, b[-1], b[-2], current[0], current[1]),
                    (b[::-1] + current, b[0], b[1], current[0], current[1]),
                ]

                for candidate, p1a, p1b, p2a, p2b in candidates:
                    if not are_touching(p1a, p2a):
                        continue



                    # Fit lines to the last 5 points of both branches
                    slope1 = fit_line(current)  # Slope of the current branch
                    slope2 = fit_line(b)        # Slope of the candidate branch

                    # Calculate the angle between the two fitted lines
                    line_angle = angle_between(np.array([1, slope1]), np.array([1, slope2]))
                    line_angle = np.min([line_angle,180-line_angle])
                    # If the angle between the fitted lines is below the threshold and the branch is longer, merge
                    if line_angle < max_angle  and len(candidate) > best_length:
                        best_merge = candidate
                        best_j = j
                        best_length = len(candidate)

            if best_merge is not None:
                current = best_merge
                used.add(best_j)
                extended = True

        merged.append(current)

    return merged



def branch_fitter(branch,image,dist,window=5, line_threshold=0.85,step_filament=1,target_y = 0.82, tolerance = 0.02,color='k',marker="o"):
    """
    This functions takes in the branch, raw image and dist_transformed picture.
    It fits the perpendicular line, and uses the pixel_values function to get the thickness for that

    soft parameters:
    window: How much points to consider for fitting the line on the dist transform.
    rolling_window: How much points to average on before calling that we are not in the filament anymore
    line_threshold:  How ligth is the outside threshold basically
    step_filament: Whether to skip steps or not

    """

  
    y,x = zip(*branch)

    max_length =50
    #len_filaments = int(len(branch) - 2* window)
    filament_len_array = np.arange(window,(len(branch)-window),step_filament)
    n_filaments = len(filament_len_array)
   
    dists_interpolating = np.zeros(n_filaments)
  
    adder =0

    for i in range(window, len(branch) - window, step_filament):
        if i< len(branch):

            # extract local neighborhood
            neighborhood = branch[i - window : i + window + 1]
            ys, xs = zip(*neighborhood)  # remember: y=row, x=col in image space

            # Fit a line: y = m*x + b
            A = np.vstack([xs, np.ones(len(xs))]).T
            m, b = np.linalg.lstsq(A, ys, rcond=None)[0]

            # Tangent vector direction
            if np.isclose(m, 0):  # horizontal line → vertical normal
                nx, ny = 0, 1
            elif np.isinf(m):  # vertical line → horizontal normal
                nx, ny = 1, 0
            else:
                # Normal slope = -1/m, so direction vector is (1, -1/m)
                nx = 1
                ny = -1 / m
                norm = np.hypot(nx, ny)
                nx /= norm
                ny /= norm

            y_current, x_current = branch[i]  # current point

            # Endpoints
            # Initialize point lists for both directions
            points_neg = []
            points_pos = []

            # Go in negative normal direction
            for step in range(1, max_length):
                xi = int(round(x_current - nx * step))
                yi = int(round(y_current - ny * step))
                if not (0 <= yi < image.shape[0] and 0 <= xi < image.shape[1]): #checks if the line is within the frame
                    break
                points_neg.append(image[yi, xi])

                if points_neg[-1] >= line_threshold:
                    break
            neg_endpoint = (int(round(x_current - nx * len(points_neg))), int(round(y_current - ny * len(points_neg))))

            # Go in positive normal direction
            for step in range(1, max_length):
                xi = int(round(x_current + nx * step))
                yi = int(round(y_current + ny * step))
                if not (0 <= yi < image.shape[0] and 0 <= xi < image.shape[1]):
                    break
                points_pos.append(image[yi, xi])

                if points_pos[-1] > line_threshold:
                        break
            pos_endpoint = (int(round(x_current + nx * len(points_pos))), int(round(y_current + ny * len(points_pos))))

            x0, y0 = neg_endpoint
            x1, y1 = pos_endpoint

            # # Use Bresenham line to plot
            pixels = bresenham_line((y0, x0), (y1, x1))

            dist_interpolating = pixel_values(pixels,image,dist,x_current,y_current,np.array([ny,nx]),target_y = target_y, tolerance = tolerance)
            
     
            dists_interpolating[adder] = dist_interpolating
   
            adder+=1
            # for r, c in pixels:
            #     plt.plot([x0, x1], [y0, y1], color='red', linewidth=1)        
    #arr=  distance_filter(filament_len_array,dists_interpolating)

    # plt.subplots(2,1)
    # plt.subplot(2,1,1)
    #plt.title(f"Filament_len 0 = {x[0],y[0]} ")
    #plt.plot(filament_len_array,dists_derivative,label="Coen's method of derivative")
    factor_outlier = 2


    #plt.subplot(1,3,3)
    #dx = abs(np.diff(dists_interpolating))
    #plt.plot(filament_len_array[:-1]+branch[0][1],dx,color= color)
    # plt.subplot(1,2,1)
    #mask = dx> factor_derivative * dists_interpolating[-1]
    #mask =np.append(False,mask)
    median = np.median(dists_interpolating)
    mean = np.mean(dists_interpolating)
    std = np.std(dists_interpolating)
    counting_mask = abs(dists_interpolating -mean) < std
    fraction = np.count_nonzero(counting_mask)/ dists_interpolating.shape[0]

    mask_toobig = (dists_interpolating> mean + factor_outlier*std) | (dists_interpolating<mean -factor_outlier*std)

    cancelled_points = np.zeros_like(dists_interpolating)
    #cancelled_points[mask] = dists_interpolating[mask]
    cancelled_points[mask_toobig]  = dists_interpolating[mask_toobig]
    cancelled_points = np.where(cancelled_points>0,cancelled_points,np.nan)
    #dists_interpolating[mask] = np.nan
    dists_interpolating[mask_toobig] = np.nan
    
    std_after = np.nanstd(dists_interpolating)
    mean_after = np.nanmean(dists_interpolating)
    median_after =np.nanmedian(dists_interpolating)
    # plt.scatter(filament_len_array+branch[0][1],dists_interpolating,marker= marker,color= color,s=10)
    # plt.suptitle(f'mean:{round(mean,2)}, mean after:{round(mean_after,2)}, median:{round(median,2)},'\
    #           f'median_after: {round(median_after,2)},stdbefore: {round(std,4)},std_after = {round(std_after,4)},fraction within one std: {fraction:.2f}',wrap=True)
    # plt.scatter(filament_len_array+branch[0][1],dists_interpolating,color= color)
    # plt.scatter(filament_len_array+branch[0][1],cancelled_points,color= 'k', marker= "x")
    






    # plt.subplot(2,1,2)
    # plt.imshow(image,cmap='gray')
    # plt.scatter(x[0], y[0] ,c= 'r')
    # plt.scatter(x[-1], y[-1] ,c= 'r',marker="x")
    #plt.show() 

    return filament_len_array,dists_interpolating,x,y
    
def pixel_values(pixels,image,dist,x,y,normal, target_y = 0.8, tolerance = 0.02):
    """
    Takes in the places of the pixels, raw image, dist transform, and the centre coordinate of dist transform
    target_y: used for where to define the boundary of filament
    tolerance:  For the spline fit with thresholding method
    Returns

    dist_deriv: based on a peak sweeping algorithm based on Coen MSC thesis
    dist_thresholding: Based on the spline fit
    dist_interpolating: Based on thresholding around certain threshold
    """

    pix_val = np.zeros(len(pixels))
    ###creating the image from the array

    for i, (py, px) in enumerate(pixels):
        if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
            pix_val[i] = image[py, px]
        else:
            pix_val[i] = np.nan  # or 0, depending on how you want to handle OOB
    
    dists  = cdist(pixels,np.array([[y,x]])).flatten()

    # Compute displacement vectors: pixel - center
    disp = np.array(pixels) - np.array([y, x])

    # Signed projection onto the normal
    signs = np.sign(np.dot(disp, normal))

    # Combine with distances
    signed_dists = dists * signs

    ####
    ####Method 2: Just linear interpolating
    #####
    non_exact_match_eps = 1e-6
    target_y = 0.8 + non_exact_match_eps

    dy = pix_val - target_y
    crossings = np.where(dy[:-1] * dy[1:] < 0)[0]

    x_interp = []

    for idx in crossings:
        x1, x2 = signed_dists[idx], signed_dists[idx+1]
        y1, y2 = pix_val[idx], pix_val[idx+1]
        if y1 != y2:
            x_cross = x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)
            x_interp.append(x_cross)

    x_interp = np.array(sorted(x_interp))  # sorted by x-position

    # Get value near zero
    zero_idx = np.argmin(np.abs(signed_dists))
    val_at_zero = pix_val[zero_idx]

    dist_interpolate = np.nan

    # Split crossings into negative and positive sides
    negs = [x for x in x_interp if x < 0]
    poss = [x for x in x_interp if x > 0]

    if val_at_zero > target_y:
        # Skip center: use next negative and positive beyond central region
        if len(negs) >= 2 and len(poss) >= 2:
            x_neg = negs[-2]
            x_pos = poss[1]
            dist_interpolate = abs(x_pos - x_neg)
        else:
            dist_interpolate = np.nan
    elif negs and poss:
        # Normal case: use closest on each side of zero
        x_neg = negs[-1]
        x_pos = poss[0]
        dist_interpolate = abs(x_pos - x_neg)
    else:
        dist_interpolate = np.nan


    return dist_interpolate


def filament_file(
    folder,  
    cropped_value=55,
    mirror=True, 
    rotation=None,  
    scale = 1,
    selected_image=400,
    save_fig = False,
    thresh_factor = 20,
    plot= False,
    output_folder =".\\",
    skip_images=1,
    skip_first_files=250,
    window_size =200,
    bubble_value = 300

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
    
    files = [f for f in glob.glob("*.tif", root_dir=folder) if "ximea" not in f.lower() and "calibration" not in f.lower()]

    valid_rotations = {None, 90, 180, 270}
    if rotation not in valid_rotations:
        raise ValueError("Rotation must be None, 90, 180, or 270 degrees.")

    # Get calibration scale if needed
    

    total_frames = len(files)

    #preparing area and cnt arrays

 
    first_file = True
    frame_number =0
    for i,file in enumerate(files):
           
        if i== selected_image or selected_image == -1:
            
            if i%skip_images==0 and i>skip_first_files:
                frame_number+=1
                img= Image.open(folder +"\\" + file)  # Open TIFF
                img = np.array(img)
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
                    mean_val = np.mean(np.mean(cal_img))
                    thresh_ref = np.quantile(cal_img.flatten(),0.80)/255
                    first_file = False

                cropped = img[:,cropped_value:cropped_value+window_size]
                
                ret,thresh = cv.threshold(255-cropped,255-mean_val+thresh_factor,255,cv.THRESH_TOZERO)
                
    
                hierarchy,contours,areas = edges(thresh)
                # contour_image = np.zeros_like(thresh)

                # # Draw contours as lines
                # cv.drawContours(contour_image, contours, -1, color=255, thickness=-1)

                # Show result with matplotlib
                # fig,ax =plt.subplots(1,2,figsize=(8, 6),sharex=True,sharey=True)
                # ax[0].imshow(contour_image, cmap='gray')
                # plt.title("Contours Drawn with OpenCV")
                # plt.axis('off')
                # ax[1].imshow(thresh)
                # plt.show()
                contour_data= np.zeros_like(thresh)
                filament_image = np.zeros_like(thresh)
              
                overlay = np.zeros_like(thresh)
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
                            filament_image  =cv.drawContours(filament_image,[contour],255,'blue',-1)

                            
                            #cv.drawContours(contour_mask,[contour],-1,255,1)
                         
                            cv.fillPoly(contour_mask, [contour], 255)
                            



                dist = cv.distanceTransform(contour_mask,cv.DIST_L2,5)
                mask_skeleton= skeletonize(dist>0)==1
                dist[~mask_skeleton] =0
                overlay[dist!=0] = 255
                dist_transform[dist!=0] = dist[dist!=0]
                # fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
                # ax[0].imshow(thresh)
                # ax[1].imshow(contour_mask)
                # ax[2].imshow(overlay)
                # plt.show()                
                if i % 10 == 0:
                    print(f"Processed {i}/{total_frames} frames...")


    
    return cropped,overlay,dist_transform,thresh_ref

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


def shape_categorizer(contour,largest=False):
    """
    Finds shapes of object:
    options:
    #['','Droplet,'Large_droplet','Filament','Small_filament','Cloud]
    """
    center,dim,angle =cv.minAreaRect(contour)
    area = cv.contourArea(contour)
    height = np.max(dim)
    width = np.min(dim)
    
    tag= "None"
    color = (0,0,255) #red
    if width==0:
            #tag = 'Droplet (0 width)'
            tag = ''
            color = (0,0,255) #red
    else:
        aspect_ratio = height/width

        enclosed_circle =cv.minEnclosingCircle(contour)
        r_enclosed = enclosed_circle[1]
        
        enclosed_circle_area =r_enclosed **2 *np.pi

        if area/enclosed_circle_area <0.85 :
            if aspect_ratio>=3:
                if area>40:
                    #large filament
                    tag ="Filament"
                    color = (0,255,0) #green

    

    return color,tag



def distance_visualizer(x, y, z, img):
    """
    Create a sparse image where each (x, y) position is set to z.

    Parameters:
        x, y, z (array-like): Pixel coordinates and their values.
        shape (tuple): Image shape (height, width).

    Returns:
        np.ndarray: 2D image with z-values at specified (x, y) locations.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    assert x.shape == y.shape == z.shape, "x, y, z must be the same shape"

    H,W = img.shape

    # Round and cast indices to int if needed
    x_idx = np.round(x).astype(int)
    y_idx = np.round(y).astype(int)

    # Clip to avoid indexing errors
    x_idx = np.clip(x_idx, 0, W - 1)
    y_idx = np.clip(y_idx, 0, H - 1)

    # Assign z values to (x, y) positions
    for xi, yi, zi in zip(x_idx, y_idx, z):
        img[yi, xi] = zi  # (row, col) = (y, x)

    return img

def branch_combiner(skeleton,min_length=30,max_angle=30):
        
    # Step 2: Re-extract branches from this new skeleton
    # combined_branches = extract_branches(skeleton)

    # # Step 3: Filter and sort the final branches again
    # final_branches = filter_branches(combined_branches, min_length=min_length)
    # final_branches = sort_branches(final_branches)




    # Start with filtered + sorted branches
    current_branches = sort_branches(filter_branches(extract_branches(skeleton), min_length=min_length))
    methods = np.zeros(shape=skeleton.shape)


    for i in range(len(current_branches)):
        for element in current_branches[i]:
            methods[element] =1


    
    # fig,(ax0,ax1) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    # ax0.imshow(skeleton,cmap= 'gray')
    # im = ax1.imshow(methods,cmap= 'viridis')
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # fig.colorbar(im, cax=cbar_ax)

    # plt.show()

    #merged = fully_merge_branches(current_branches,max_angle=max_angle)
    merged = filter_branches(current_branches, min_length=min_length)  # Filter short branches
   
    return merged


def main(folder,selected_image =1501,cropped_value=55, mirror=True, rotation=None,  
    scale = 1, save_fig = False, thresh_factor = 20, plot= False, output_folder =".\\",
    skip_images=1, skip_first_files=250, window_size =600,min_length = 30,window=5,max_angle=30, 
     line_threshold=0.85,step_filament=1,target_y = 0.82, tolerance = 0.02):
    # Load the horse silhouette and preprocess it
    
    image,skeleton,dist,thresh_ref = filament_file(folder,  
    cropped_value=cropped_value,
    mirror=True, 
    rotation=None,  
    scale = scale,
    selected_image = selected_image,
    save_fig = save_fig,
    thresh_factor = thresh_factor,
    plot= plot,
    output_folder =output_folder,
    skip_images=skip_images,
    skip_first_files=skip_first_files,
    window_size =window_size)

    image  =image/255

    # for i, b in enumerate(merged):
    #     print(f"Branch {i+1} (length: {len(b)}): {b}")
    methods = np.zeros(shape=image.shape)

    merged = branch_combiner(skeleton,min_length=min_length,max_angle=max_angle)

    all_lengths = []
    for i in range(len(merged)):
    #for i in range(1):
        # fig,ax = plt.subplots(1,2)
        # ax[1].imshow(image,cmap='gray')
        branch = merged[i]
        if branch[0][1] > branch[-1][1]:
            branch = branch[::-1]

        color = colors[i%len(colors)]
        marker = markers[i%len(markers)]
        filament_len_array,dists_interpolating,x_branch,y_branch =branch_fitter(branch,image,dist,window=window, line_threshold=line_threshold,step_filament=step_filament,target_y =target_y, tolerance = tolerance,color=color,marker=marker)
        all_lengths.append(dists_interpolating)
    return all_lengths
        # ax[1].scatter(x_branch[0], y_branch[0], marker=marker,color= color,s=50)
        # #ax[1].scatter(x_branch[-1], y_branch[-1], marker=marker,color= color,s=50)
        # ax[1].plot(x_branch, y_branch, linestyle="--",color= color,linewidth=1)
        # methods = distance_visualizer(x_branch[window:-window],y_branch[window:-window],dists_interpolating,methods)
   
        # plt.ylabel("Width (px)")
        # plt.xlabel("Filament_len (px)")
        # plt.show()

        # hist,bin_edges = np.histogram(dists_interpolating,40)
        # plt.hist
    
    # fig,(ax0,ax1) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    # ax0.imshow(image,cmap= 'gray')
    # im = ax1.imshow(methods,cmap= 'viridis')
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # fig.colorbar(im, cax=cbar_ax)
    # fig.suptitle('The raw image vs filament thickness picture')

    # plt.show()

    



folder = r"D:\Experiments\Preliminary_tests\3103_FirstSpraytec\1percentPEO1ml_2nd_Camera_4_C001H001S0001_C1S0001_20250331_164039"
def creating_pickles(folder,skip_first_files=250):
    numpy_array= []
    match = re.search(r'\\([^\\]+?)_Camera', folder)
    if match:
        result = match.group(1)
    else:
        print("No match found.")
    files = [f for f in glob.glob("*.tif", root_dir=folder) if "ximea" not in f.lower() and "calibration" not in f.lower()]
    amount_files = len(files)

    filament_val =main(folder,selected_image=-1,skip_first_files=skip_first_files)
    numpy_array.append(filament_val)

    savepath= r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays\\" + result +".pkl"
    with open(savepath, 'wb') as f:
        pickle.dump(numpy_array, f)


# with open(r"C:\Users\s2557762\Documents\filament_processing\Processed_arrays\test.pkl", 'rb') as f:
#     loaded = pickle.load(f)   

# print(loaded)
    # Optional display
creating_pickles(folder)