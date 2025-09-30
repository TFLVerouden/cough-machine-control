import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import re
import os
from matplotlib.ticker import FixedLocator
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 16,             # Base font size for labels, ticks, etc.
    'axes.titlesize': 16,        # Title font size
    'axes.labelsize': 16,        # Axis label font size
    'xtick.labelsize': 14,       # X tick labels
    'ytick.labelsize': 14,       # Y tick labels
    'legend.fontsize': 14,       # Legend font size
    'axes.linewidth': 1.5,       # Thicker axis lines
    'xtick.major.size': 6,       # Major tick size (length)
    'ytick.major.size': 6,
    'xtick.major.width': 1.5,    # Major tick width
    'ytick.major.width': 1.5,
})
filename =r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays\\1percentPEO1ml_2nd.pkl"

plt.style.use('dark_background')

def contourmap_maker(filename,n_bins=100,max_width=1,fps=20000,scale=15.67,skip_first_files=250, filtered_length =200,filtered_std = 3,filtered_median_mean_diff =5):
    match = re.search(r'\\([^\\]+?)\.pkl$', filename)
    max_width_pix = np.ceil(max_width*scale)
    if match:
        save_filename = match.group(1)
    else:
        print("No match found.")
    save_path = r"D:\Experiments\Processed_Data\RemotePC\\heatmap_images\\"  + save_filename + "filter500.pdf" 
    if not os.path.exists(save_path):

        #filename = r"C:\Users\s2557762\Documents\filament_processing\Processed_arrays\test.pkl"
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)[0]

        length = len(loaded)
        time = np.arange(0,length,1)

        range_val = max_width_pix/ n_bins
        timesteps =length
        hist_data_all= np.zeros(shape=(n_bins,timesteps))
        hist_data_all_no_filter= np.zeros(shape=(n_bins,timesteps))

        time_len =(hist_data_all.shape[1]/fps)
        offset = skip_first_files/fps

        for i in range(timesteps):
            time_data= loaded[i]

            if len(time_data)>0:
                combined= []
                combined_no_filter = []
                for arr in time_data:
                    if ~np.isnan(arr).all():
                        len_arr  =len(arr)
                        mean = np.nanmean(arr)
                        std = np.nanstd(arr)
                        median = np.nanmedian(arr)

                        len_mask = len_arr > filtered_length
                        
                        std_mask = std < filtered_std
                        #median_mask = abs(median-mean)< filtered_median_mean_diff
                        mask = (std_mask) & (len_mask) 
                        #print(f"len:{len_arr},mean;{mean},std:{std},median:{median}" )

                        if mask:
                            combined.extend([val for val in arr if not np.isnan(val)])
                        combined_no_filter.extend([val for val in arr if not np.isnan(val)])
                if len(combined)>0:
                    combined = np.array(combined)
                    hist_data,bin_edges = np.histogram(combined,bins=n_bins,range=(0,max_width_pix))
                    hist_data_all[:,i] = hist_data 
                    combined_no_filter = np.array(combined_no_filter)
                    hist_data_no_filter,bin_edges_no_filter= np.histogram(combined_no_filter,bins=n_bins,range=(0,max_width_pix))
                    hist_data_all_no_filter[:,i] = hist_data_no_filter 


        fig,ax = plt.subplots(figsize=(9,9))
        vmin = 1
        print(save_filename,max(hist_data_all.flatten()))

        if "fullframe" in save_filename:
        # Do something if "fullframe" is in the filename
            vmax =558
        else:
            # Do something else if "fullframe" is not in the filename
            vmax = 147
        im= ax.imshow(hist_data_all,aspect='auto',vmin=vmin,vmax=vmax,
                    extent=[offset, offset+time_len, max_width_pix, 0],
                    cmap='gray_r')
        ax.set_ylabel('Width (mm)')
        ax.set_xlabel('Time (s)')
        yticks = ax.get_yticks()
        scaled_yticks = yticks / scale
        ax.set_yticks(yticks)  # This makes the labels match fixed tick locations
        ax.set_yticklabels([f"{tick:.1f}" for tick in scaled_yticks])
        #ax.set_title(f'{save_filename},\\'
        #                f'Min length:{filtered_length},max_Std:{filtered_std},max mean_median difference:{filtered_median_mean_diff}')
        
        cbar = fig.colorbar(im, ax=ax)
        tick_locs = cbar.get_ticks()
        cbar.locator = FixedLocator(tick_locs)
        cbar.set_ticklabels([f"{tick / scale:.2f}" for tick in tick_locs])
        cbar.set_label('Length (mm)')
        plt.gca().invert_yaxis()
        # for spine in ax.spines.values():
            # spine.set_visible(False)
        # im2= ax[1].imshow(hist_data_all_no_filter,aspect='auto',extent=[offset, offset+time_len, max_width, 0])
        
        # ax[1].set_ylabel('Width (mm)')
        # ax[1].set_xlabel('Time (s)')
        # yticks = ax[1].get_yticks()

        # # Divide by scale
        # scaled_yticks = yticks / scale

        # # Set the new ytick labels (optional: format to 1 decimal place)
        # ax[1].set_yticklabels([f"{tick:.1f}" for tick in scaled_yticks])
        
        # fig.colorbar(im2,ax=ax[1])  
        # plt.gca().invert_yaxis()
        plt.show()
        #plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        print(f"File already exists: {save_path}")


folder_path = r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays"
pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.endswith('.pkl') and os.path.isfile(os.path.join(folder_path, f))]
pkl_files = [f for f in pkl_files if 'fullframe' in os.path.basename(f)]
print(pkl_files)

for file in pkl_files:
    contourmap_maker(file,n_bins=100,filtered_length=200)


