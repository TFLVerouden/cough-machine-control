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


cwd = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.dirname(cwd)
print(cwd)
#function_dir = os.path.join(parent_dir, 'cough-machine-control')


function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
import calibration

#folder = r"D:\Experiments\Preliminary_tests\PIV_propje\propje_Camera_4_C001H001S0001_C1S0001_20250324_171643"
fps = 20000
# scale =15.84 #pix/mm ->Propje
#scale = 15.67 #pix/mm  -> Measurements 31 March 2025
#scale = 15.79 #pix/mm -> Tamara en Nick Droplet atomization


path_dir = os.path.abspath(os.path.dirname(__file__))  # Get the absolute path of the script's directory





def movie_maker(folder,savefolder,start_frame = 0,output_name="animation",scale=1,fps=20000,cropped_value=0,delete_png=True,flip="vertical",process_png=True):
    files = [f for f in glob.glob("*.tif", root_dir=folder) if "ximea" not in f.lower() and "calibration" not in f.lower() and "leaked" not in f.lower()]

    temp_savefolder = os.path.join(savefolder,output_name,"processed_png")
    os.makedirs(savefolder, exist_ok=True)
    os.makedirs(temp_savefolder, exist_ok=True)
    start = time.time()
    frame_number =0
    
    if process_png:
        for i,file in enumerate(files):
            
            if i%10==0:
                print(f"{i}/{len(files)} processed, taking: {round(time.time()-start,4)} s")
                start = time.time()
            if (i>start_frame):
                frame_number +=1
                matplotlib.use("Agg")  # Use non-GUI backend

                fig,ax= plt.subplots()
                t_current = (i)/fps 
                img= Image.open(folder +"\\" + file)  # Open TIFF
                if flip==  "vertical":
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
                if flip == "horizontal":
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally

                img = np.array(img)
                img = img[:,cropped_value:]
                xsize = img.shape[1]
                ysize= img.shape[0]
                ax.imshow(img,cmap='gray',extent=[0,xsize*scale ,0,ysize*scale])
                if scale!=1:
                    ax.set_xlabel('x [mm]',labelpad=0)
                    ax.set_ylabel('y [mm]')
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                
                ax.set_title(f'Time: {round((t_current)*1000,1)} ms')
                plt.savefig(f"{temp_savefolder}\\frame_{frame_number:03d}.png",dpi=200)
                plt.close()
    print("Making video:")     
    os.system(f'ffmpeg -r 20 -i {temp_savefolder}\\frame_%03d.png -c:v libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 {savefolder}\\{output_name}.mp4')
    
    if delete_png ==True:
        for file in glob.glob(os.path.join(temp_savefolder, "*.png")):  # Finds all PNG files in the current directory
            os.remove(file)  # Deletes the file
        try:
            os.rmdir(temp_savefolder)
            print(f"Successfully removed {temp_savefolder}")
        except OSError as e:
            print(f"Error: {e}")

def movie_maker_folder_loop(folder_path,savefolder,fps=20000,cropped_value=0,delete_png=True,flip= "vertical",output_name="Video",process_png=True):
    scale = calibration.get_calibration(folder_path)
    print(f"scale: {scale:.5f} mm/pix")
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    filtered_folders = [f for f in folders if 'spraytec' not in f.lower() and 'Videos' not in f and 'skip' not in f]

    # Extract the first part of each filtered folder name (before '_Camera')
    output_names = [f.split('_Camera')[0] for f in filtered_folders]

    for folder_counter,folder in enumerate(filtered_folders):
        
        output_name = output_names[folder_counter]
        print("---------------------------------")
        print(f"We are at: {folder_counter}/{len(filtered_folders)} files, named: {output_name}")
        print("---------------------------------")
        print(folder)
        total_folder_path = os.path.join(folder_path, folder)
        
        print(total_folder_path)
        
        movie_maker(total_folder_path,savefolder,scale=scale,cropped_value=cropped_value, flip=flip,
                    fps=fps,delete_png=delete_png,output_name=output_name,process_png=process_png)
        gc.collect()  # Force memory cleanup

#folder  = r"D:\Experiments\Droplet_atomization\ETPoF_NickTamara\PEO2M_c1-1\0_75bar_12uL_c1-1_2_Camera_4_C001H001S0001_C1S0001_20250324_130907"
folder = r"D:\\Experiments\\sideview_coughs\\02_09_25"


savefolder =os.path.join(folder,"Videos")
output_name = folder.split('_Camera')[0]
output_name = output_name.split('\\')[-1]
roi = [500,950,100,550] #xstart,xend, ystart, yend
movie_maker_folder_loop(folder_path= folder,savefolder=savefolder,fps=20000,cropped_value=0,output_name=output_name,delete_png=True)
#movie_maker_folder_loop(folder,savefolder,fps=20000,cropped_value=0,output_name=output_name,delete_png=False)

#movie_maker_folder_loop(folder_path=folder_path,csv_file=csv_file,savefolder=savefolder,scale=scale,fps=20000,delete_png=False,process_png=False)
temp_savefolder = os.path.join(savefolder,output_name,"processed_png")
print(temp_savefolder)
print(savefolder,"\\",output_name)

#os.system(f"ffmpeg -r 20 -i {temp_savefolder}\\frame_%03d.png -vcodec libx264 {savefolder}\\{output_name}.mp4")
