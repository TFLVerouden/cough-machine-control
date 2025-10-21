import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import re
import os

filename =r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays\\1percentPEO1ml_2nd.pkl"


def contourmap_maker(filename,n_bins=200,max_width=50,fps=20000,scale=15,skip_first_files=250):
    match = re.search(r'\\([^\\]+?)\.pkl$', filename)

    if match:
        save_filename = match.group(1)
        print(save_filename)
    else:
        print("No match found.")
    save_path = r"D:\Experiments\Processed_Data\RemotePC\\heatmap_images\\"  + save_filename + "filtered.png" 

    #filename = r"C:\Users\s2557762\Documents\filament_processing\Processed_arrays\test.pkl"
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)[0]

    length = len(loaded)
    time = np.arange(0,length,1)

    scale = 15 #pix/mm, needs to be more accurate later on
    range_val = max_width/ n_bins
    timesteps =length
    hist_data_all= np.zeros(shape=(n_bins,timesteps))
    time_len =(hist_data_all.shape[1]/fps)
    offset = skip_first_files/fps
    for i in range(timesteps):
        time_data= loaded[i]
        if len(time_data)>0:
            combined = np.concatenate([c.reshape(-1) for c in time_data])
            hist_data,bin_edges = np.histogram(combined,bins=n_bins,range=(0,max_width))
            hist_data_all[:,i] = hist_data 


    plt.figure(figsize=(9,9))
    plt.imshow(hist_data_all,aspect='auto',extent=[offset, offset+time_len, max_width, 0])
    plt.ylabel('Width (mm)')
    plt.xlabel('Time (s)')
    plt.title(f'{save_filename}')
    cbar = plt.colorbar()
    cbar.set_label('Pixel Count')  
    plt.gca().invert_yaxis()
    plt.show()
    #plt.savefig(save_path)


folder_path = r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays"
pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.endswith('.pkl') and os.path.isfile(os.path.join(folder_path, f))]

for file in pkl_files:
    contourmap_maker(file,n_bins=100)


